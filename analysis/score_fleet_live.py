#!/usr/bin/env python3
"""Live scoreboard over the fleet archive (PREREG §V16-0) and the input-age
measurement (§V16-1).

    python analysis/score_fleet_live.py age   <archive_root>
    python analysis/score_fleet_live.py score <archive_root> --issued-from 2026-11-18 --out live_skill.json

`<archive_root>` holds `archive_fleet/` and `archive_fleet_meta/` (a checkout of
the `fleet-archive` branch).

`score` reports, per lead, for each arm against a reference (default
persistence_lag1): pooled MAE ratio with a basin-bootstrap 95% CI, the fraction
of basins where the arm's MAE is lower, the median relative bias
(pred-obs)/max(obs, 0.1) of both, and for the blend its 90% band coverage.
All days scored, no stride. Lead is reported two ways because they differ
(§V16 H-LAG): the archive's `horizon_day`, and the calendar lead
`target_date - issued_date`.

Targets dated today or later are dropped (a partial-day USGS mean is not an
observation -- the GATE 4 scorer's rule). `--issued-from` defaults to
2026-11-18: issue dates before it belong to §LIVE-1's window, which is read
once, from the m1 archive, on its own date.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.fleet_archive import MEMBERS  # noqa: E402

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
LIVE1_GUARD = "2026-11-18"
N_BOOT = 1000


def load(root: str, sub: str, issued_from: str, issued_to: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{root}/{sub}/**/*.csv.gz", recursive=True))
    files = [f for f in files if issued_from <= Path(f).name[:10] <= issued_to]
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f, dtype={"station_id": str}) for f in files),
                     ignore_index=True)


def parse_rdb(text: str) -> pd.DataFrame:
    """USGS RDB daily values -> long frame (station_id, date, obs). The first
    00060 mean column of each block is taken; a state response is one block
    per distinct column layout."""
    out = []
    block: list = []

    def flush():
        if len(block) < 3:
            return
        df = pd.read_csv(io.StringIO("\n".join([block[0]] + block[2:])), sep="\t", dtype=str)
        col = next((c for c in df.columns if c.endswith("_00060_00003")), None)
        if col and "site_no" in df and "datetime" in df:
            v = pd.to_numeric(df[col], errors="coerce")
            out.append(pd.DataFrame({"station_id": df["site_no"], "date": df["datetime"], "obs": v}))

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            block = []
        elif line.strip():
            block.append(line)
    flush()
    if not out:
        return pd.DataFrame(columns=["station_id", "date", "obs"])
    df = pd.concat(out, ignore_index=True).dropna(subset=["obs"])
    return df[df.obs >= 0].drop_duplicates(["station_id", "date"])


def fetch_obs(states, start: str, end: str, pause: float = 1.0) -> pd.DataFrame:
    frames = []
    for st in sorted(states):
        url = (f"{DV_URL}?format=rdb&stateCd={st.lower()}&parameterCd=00060&statCd=00003"
               f"&startDT={start}&endDT={end}&siteStatus=all")
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "riverwatch2/0.1"}), timeout=180) as r:
                    frames.append(parse_rdb(r.read().decode("utf-8", "replace")))
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  obs {st} attempt {attempt + 1} failed: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else parse_rdb("")


def _table(g: pd.DataFrame, arm: str, ref: str, rng) -> dict:
    g = g.dropna(subset=[arm, ref, "obs"])
    if g.empty:
        return {}
    ea, er = (g[arm] - g.obs).abs(), (g[ref] - g.obs).abs()
    pb = pd.DataFrame({"s": g.station_id, "ea": ea, "er": er}).groupby("s").sum()
    a, r = pb.ea.to_numpy(), pb.er.to_numpy()
    if r.sum() <= 0:
        return {}
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    den = r[idx].sum(1)
    boot = a[idx].sum(1)[den > 0] / den[den > 0]
    d = g.obs.clip(lower=0.1)
    out = {"rows": int(len(g)), "basins": int(len(pb)),
           "mae_ratio": round(float(a.sum() / r.sum()), 4),
           "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
           "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
           "frac_basins_better": round(float((a < r).mean()), 4),
           "bias_arm": round(float(((g[arm] - g.obs) / d).median()), 4),
           "bias_ref": round(float(((g[ref] - g.obs) / d).median()), 4)}
    if arm == "blend" and {"band_lo", "band_hi"} <= set(g.columns):
        b = g.dropna(subset=["band_lo", "band_hi"])
        if len(b):
            out["band90_coverage"] = round(float(((b.obs >= b.band_lo) & (b.obs <= b.band_hi)).mean()), 4)
    return out


def score(df: pd.DataFrame, obs: pd.DataFrame, pairs, today: str | None = None, seed: int = 0) -> dict:
    """`pairs` = [(arm, ref), ...]. Pure: unit-tested on synthetic rows."""
    today = today or date.today().isoformat()
    df = df[df.target_date < today].merge(
        obs.rename(columns={"date": "target_date"}), on=["station_id", "target_date"], how="inner")
    df["cal_lead"] = (pd.to_datetime(df.target_date) - pd.to_datetime(df.issued_date)).dt.days
    res = {"rows_scored": int(len(df)), "basins": int(df.station_id.nunique()),
           "issued": [df.issued_date.min(), df.issued_date.max()] if len(df) else [],
           "targets": [df.target_date.min(), df.target_date.max()] if len(df) else [],
           "pairs": {}}
    rng = np.random.default_rng(seed)
    for arm, ref in pairs:
        if arm not in df or ref not in df or df[arm].notna().sum() == 0:
            continue
        res["pairs"][f"{arm}|{ref}"] = {
            key: {int(k): t for k, g in df.groupby(key) if (t := _table(g, arm, ref, rng))}
            for key in ("horizon_day", "cal_lead")}
    return res


def input_age(meta: pd.DataFrame) -> dict:
    """§V16-1 statistic: issued_date - wx_fcst_start in days, per gauge-day."""
    out = {"gauge_days": int(len(meta))}
    iss = pd.to_datetime(meta.issued_date)
    for col in ("wx_fcst_start", "wx_hist_end", "nbm_start", "last_obs_date"):
        age = (iss - pd.to_datetime(meta[col], errors="coerce")).dt.days
        have = age.dropna()
        out[col] = {"missing_frac": round(float(age.isna().mean()), 4)}
        if len(have):
            out[col].update({"median_d": float(have.median()), "p10_d": float(have.quantile(.1)),
                             "p90_d": float(have.quantile(.9)),
                             "frac_ge_2d": round(float((have >= 2).mean()), 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["age", "score"])
    ap.add_argument("root")
    ap.add_argument("--issued-from", default=None)
    ap.add_argument("--issued-to", default="2100-01-01")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "age":
        meta = load(a.root, "archive_fleet_meta", a.issued_from or "2000-01-01", a.issued_to)
        if meta.empty:
            print("no provenance rows")
            return 1
        res = input_age(meta)
    else:
        frm = max(a.issued_from or LIVE1_GUARD, LIVE1_GUARD)
        if a.issued_from and a.issued_from < LIVE1_GUARD:
            print(f"--issued-from raised to {LIVE1_GUARD} (PREREG §LIVE-1 window)", file=sys.stderr)
        df = load(a.root, "archive_fleet", frm, a.issued_to)
        if df.empty:
            print(f"no fleet rows issued {frm}..{a.issued_to}; nothing to score")
            res = {"rows_scored": 0, "pairs": {}}
        else:
            stations = json.loads((ROOT / "data" / "stations_v15.json").read_text())
            stations = stations["stations"] if isinstance(stations, dict) else stations
            st_of = {s["id"]: s.get("state") for s in stations}
            states = {st_of.get(s) for s in df.station_id.unique()} - {None, ""}
            lo = (date.fromisoformat(df.target_date.min()) - timedelta(days=1)).isoformat()
            obs = fetch_obs(states, lo, date.today().isoformat())
            pairs = [("blend", "persistence_lag1")] + [(m, "persistence_lag1") for m in MEMBERS[1:]]
            pairs += [("blend", "nwm"), ("challenger", "blend")]
            res = score(df, obs, pairs)
        res["issued_from"] = frm
    res["generated_at"] = pd.Timestamp.utcnow().isoformat()
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt if a.cmd == "age" else json.dumps({k: v for k, v in res.items() if k != "pairs"}))
    if a.cmd == "score":
        for name, tabs in res.get("pairs", {}).items():
            print(f"\n{name}  (by calendar lead)\nlead rows basins ratio [ci] frac_better bias_arm bias_ref")
            for h, t in sorted(tabs["cal_lead"].items()):
                print(h, t["rows"], t["basins"], t["mae_ratio"], f"[{t['ci_lo']},{t['ci_hi']}]",
                      t["frac_basins_better"], t["bias_arm"], t["bias_ref"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
