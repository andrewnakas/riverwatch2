#!/usr/bin/env python
"""Does my UTC 00-23Z daily aggregation match CAMELS' NLDAS day definition?

I aggregate NLDAS hourly -> daily over UTC hours 00..23. USGS daily values use
LOCAL midnight-to-midnight, and CAMELS' own NLDAS forcing made some choice of
its own. A mismatch is a systematic sub-daily misregistration — the same class of
bug as the Daymet one-day precipitation offset, and it would be baked into all
12 years.

Test: compare my rebuilt overlap month (2010-01) against the FROZEN
camels_corpus_nldas_v2 at lags -2..+2, per variable. Lag 0 winning means the day
definitions agree. Anything else means they do not, and the aggregation window
must be shifted before the corpus is used.

Also splits by longitude: a UTC-vs-local mismatch grows westward, so if the
best lag or the correlation degrades toward the Pacific, that is the signature.
"""
from __future__ import annotations
import gzip, json, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
FROZEN = ROOT / "gpu1080/corpora671/camels_corpus_nldas_v2"
CKPT = ROOT / "data/modern_corpus/_nldas_monthly/nldas_201001.npz"
LATLON = json.load(open(ROOT / "data/modern_corpus/basin_latlon.json"))

COLS = {"precip": "precipitation_sum", "tmean": "temperature_2m_mean",
        "tmax": "temperature_2m_max", "tmin": "temperature_2m_min",
        "srad": "shortwave_radiation_sum", "vp": "vapor_pressure"}
LAGS = (-2, -1, 0, 1, 2)


def load_frozen(g, ym):
    p = FROZEN / f"{g}.csv.gz"
    if not p.exists():
        return None
    out = {}
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.strip().split(",")
            if not f[0].startswith(ym):
                continue
            rec = {}
            for k, col in COLS.items():
                try:
                    rec[k] = float(f[ci[col]])
                except (ValueError, KeyError):
                    rec[k] = np.nan
            out[f[0]] = rec
    return out


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import datetime as dt
    z = np.load(CKPT, allow_pickle=True)
    basins = [str(b) for b in z["basins"]]
    days = [str(d) for d in z["days"]]

    acc = {k: {l: [] for l in LAGS} for k in COLS}
    west, east = [], []
    for i, g in enumerate(basins):
        fr = load_frozen(g, "2010-01")
        if not fr or len(fr) < 25:
            continue
        for k in COLS:
            mine_by_day = {d: z[k][j][i] for j, d in enumerate(days)}
            for l in LAGS:
                A, B = [], []
                for d in days:
                    dd = (dt.date.fromisoformat(d) + dt.timedelta(days=l)).isoformat()
                    if dd in fr:
                        A.append(mine_by_day[d]); B.append(fr[dd][k])
                r = corr(A, B)
                if not math.isnan(r):
                    acc[k][l].append(r)
        # longitude split, for the UTC-vs-local signature
        ll = LATLON.get(g)
        if ll:
            A = [z["tmean"][j][i] for j, d in enumerate(days)]
            B = [fr[d]["tmean"] for d in days if d in fr]
            if len(A) == len(B):
                r0 = corr(A, B)
                (west if ll[1] < -100 else east).append(r0)

    print(f"NLDAS day-definition test — my UTC 00-23Z vs frozen CAMELS, 2010-01")
    print(f"basins compared: {len(acc['tmean'][0])}\n")
    allzero = True
    for k in COLS:
        med = {l: float(np.median(v)) for l, v in acc[k].items() if v}
        if not med:
            continue
        best = max(med, key=lambda l: med[l])
        allzero &= (best == 0)
        s = "  ".join(f"lag{l:+d}={med[l]:+.3f}" for l in sorted(med))
        flag = "" if best == 0 else "   <-- ⚠️ NOT lag 0"
        print(f"  {k:<7} {s}   -> best {best:+d}{flag}")

    print("\n--- UTC-vs-local signature (tmean corr at lag 0, by longitude) ---")
    if west and east:
        print(f"  EAST (lon > -100): median r = {np.median(east):.4f}  n={len(east)}")
        print(f"  WEST (lon < -100): median r = {np.median(west):.4f}  n={len(west)}")
        d = np.median(east) - np.median(west)
        print(f"  east-west gap: {d:+.4f}"
              + ("   <-- ⚠️ degrades westward = UTC/local mismatch"
                 if d > 0.02 else "   <-- no westward degradation"))

    print("\n--- VERDICT ---")
    if allzero:
        print("  ✅ every variable peaks at lag 0 — day definitions AGREE.")
    else:
        print("  ⛔ at least one variable does NOT peak at lag 0 — the aggregation")
        print("     window is misregistered and must be fixed before use.")


if __name__ == "__main__":
    main()
