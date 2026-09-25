#!/usr/bin/env python3
"""LEDGER 45 / T2d — HOW SOLID IS THE 3.94x PASS?

The conjunction rule is a HARD threshold (>=3x noise) read off a MEDIAN over a
near-median cohort of only ~40 basins. T3b showed a threshold can be met without
the underlying thing being real, so the honest question is: what is the
uncertainty on 3.94x, and does it clear 3.0 robustly?

Bootstrap the cohort median of |pcorr| (basin-level resample), and report the
same for the noise anchor so the RATIO is bootstrapped, not just the numerator.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path.home() / "riverwatch2"
sys.path.insert(0, str(H)); sys.path.insert(0, str(H / "analysis"))
import ledger45_satellite_screen as S                       # noqa: E402
from noq_harness import build_9stream_train, fit_global_invmse, score, split  # noqa: E402
from rescore9 import PREREG                                 # noqa: E402
from ivt_landing import partial_corr                        # noqa: E402

NOISE, BAND = S.NOISE, S.MEDIAN_BAND


def main():
    gids = sorted(json.load(open(H / "data/camels_gauge_ids.json"))["531"])
    forc = S.load_forcing(gids, ["prcp_daymet"])
    val, _ = S.load_product("cci_combined")

    tr, cols = build_9stream_train()
    fit, v0 = split(tr)
    w, _, _ = fit_global_invmse(fit, cols)
    base = score(v0, cols, w)
    anchor = float(base.median())
    print(f"ANCHOR {anchor:.6f} (prereg {PREREG})")
    assert abs(anchor - float(PREREG)) < 1e-5, "anchor mismatch — stop"
    v = v0.copy()
    v["resid"] = (v[cols].to_numpy(float) * w).sum(axis=1) - v.truth
    v["date"] = pd.to_datetime(v.date)
    med = base.median()
    near = set(base[(base - med).abs() <= BAND].index)

    rows = []
    for g in gids:
        if g not in forc or val[g].std(skipna=True) == 0:
            continue
        r = v[v.station_id == g][["date", "resid"]]
        if len(r) < 60:
            continue
        ch = pd.DataFrame({"C": val[g]})
        ch["prcp"] = forc[g]["prcp_daymet"].reindex(ch.index)
        m = r.merge(ch.reset_index().rename(columns={"index": "date"}),
                    on="date", how="inner").dropna()
        if len(m) < 60:
            continue
        ar = np.abs(m.resid.values)
        rng = np.random.default_rng(abs(hash(g)) % 2 ** 32)
        pc = partial_corr(m.C.values, ar, m.prcp.values)
        pn = partial_corr(rng.standard_normal(len(m)), ar, m.prcp.values)
        if np.isfinite(pc) and np.isfinite(pn):
            rows.append((g, abs(pc), abs(pn), len(m)))

    D = pd.DataFrame(rows, columns=["gid", "pc", "pn", "n"])
    Dn = D[D.gid.isin(near)]
    print(f"basins screened {len(D)}; near-median cohort {len(Dn)}; "
          f"median rows/basin {int(D.n.median())}")

    rng = np.random.default_rng(20260829)
    def boot(frame, k=6000):
        out = []
        idx = np.arange(len(frame))
        pc, pn = frame.pc.to_numpy(), frame.pn.to_numpy()
        for _ in range(k):
            s = rng.integers(0, len(frame), len(frame))
            out.append(np.median(pc[s]) / NOISE)
        return np.array(out)

    res = {"anchor": anchor, "noise_const": NOISE}
    for label, frame in (("near_median", Dn), ("all_basins", D)):
        b = boot(frame)
        pt = float(np.median(frame.pc) / NOISE)
        lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        p_pass = float((b >= 3.0).mean())
        emp_noise = float(np.median(frame.pn))
        res[label] = {"n_basins": int(len(frame)), "x_noise_point": pt,
                      "x_noise_boot95": [lo, hi], "P(>=3x)": p_pass,
                      "empirical_noise_median": emp_noise,
                      "x_empirical_noise": float(np.median(frame.pc) / emp_noise)}
        print(f"\n[{label}] n={len(frame)}")
        print(f"  x noise (const {NOISE}) = {pt:.2f}  boot95 [{lo:.2f}, {hi:.2f}]")
        print(f"  P(bootstrap >= 3.0x) = {p_pass:.3f}")
        print(f"  vs EMPIRICAL per-basin noise anchor ({emp_noise:.4f}): "
              f"{res[label]['x_empirical_noise']:.2f}x")

    p = H / "benchmarks/ledger45_cci_passrobust.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {p} md5 {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
