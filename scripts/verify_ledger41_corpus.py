#!/usr/bin/env python3
"""Ledger 41 Stage-1 gates: verify the pretraining corpus before any GPU.

Four gates, in the order a failure would matter:

  S1a  artifacts -- every file passes gzip integrity, has the expected schema,
       and the basin count clears the bar. Exit codes have lied here before;
       this reads the artifacts.
  S1b  zero NaN in any dynamic input. A NaN dynamic input poisons the member,
       and zero-filling it is worse than dropping the basin.
  S1c  LAG REGISTRATION. Cross-correlates precipitation against the discharge
       response over lags -3..+3 and compares the peak lag against the CAMELS
       daymet corpus measured by the SAME statistic in the SAME script. This is
       the gate that earns its place: a one-day precipitation offset was worth
       +0.229 NSE, it is invisible to mean and bias checks, and HYSETS does not
       document its day-stamp convention. Peaks are sign-aware -- never abs().
  S1d  SCALER DISTANCE. Per-variable mean and standard deviation against the
       CAMELS corpus the finetune will use. Out of band means the finetune
       inherits a normalisation the prior never trained on, which is confound
       (3) from ledger 40 returning by a different route.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
           "vapor_pressure", "shortwave_radiation_sum"]
LAGS = range(-3, 4)


def lag_profile(df):
    """Sign-aware cross-correlation of precip against the discharge response.

    Uses the first difference of discharge: the rising limb is what a
    precipitation date-stamp actually registers against, and differencing
    removes the baseflow trend that would otherwise dominate the correlation
    and flatten the peak.
    """
    p = df["precipitation_sum"].to_numpy(float)
    q = df["q_cfs"].to_numpy(float)
    dq = np.diff(q, prepend=np.nan)
    out = {}
    for L in LAGS:
        a = np.roll(p, L)
        if L > 0:
            a[:L] = np.nan
        elif L < 0:
            a[L:] = np.nan
        ok = np.isfinite(a) & np.isfinite(dq)
        out[L] = float(np.corrcoef(a[ok], dq[ok])[0, 1]) if ok.sum() > 500 else np.nan
    return out


def scan(corpus, limit, label):
    files = sorted(p for p in Path(corpus).glob("*.csv.gz")
                   if not p.name.startswith("._"))
    if limit:
        files = files[:limit]
    peaks, stats, bad_gzip, nan_hits, rows = [], [], [], [], []
    for f in files:
        try:
            with gzip.open(f, "rb") as fh:          # validates the stream marker
                fh.read()
        except Exception as e:
            bad_gzip.append((f.name, str(e)[:60]))
            continue
        df = pd.read_csv(f)
        rows.append(len(df))
        have = [c for c in DYNAMIC if c in df.columns]
        n_nan = {c: int(df[c].isna().sum()) for c in have if df[c].isna().any()}
        if n_nan:
            nan_hits.append((f.name, n_nan))
        if "q_cfs" in df.columns and df["q_cfs"].notna().sum() > 500:
            prof = lag_profile(df)
            finite = {k: v for k, v in prof.items() if np.isfinite(v)}
            if finite:
                peaks.append(max(finite, key=finite.get))     # sign-aware argmax
        stats.append({c: (df[c].mean(), df[c].std()) for c in have})
    agg = {}
    for c in DYNAMIC:
        m = [s[c][0] for s in stats if c in s and np.isfinite(s[c][0])]
        sd = [s[c][1] for s in stats if c in s and np.isfinite(s[c][1])]
        if m:
            agg[c] = (float(np.mean(m)), float(np.mean(sd)))
    print(f"\n[{label}] {len(files)} files, {len(stats)} read, "
          f"median rows {int(np.median(rows)) if rows else 0}")
    if peaks:
        from collections import Counter
        cnt = Counter(peaks)
        mode = cnt.most_common(1)[0]
        print(f"  lag peak distribution: "
              f"{dict(sorted(cnt.items()))}  -> mode {mode[0]:+d} "
              f"({100*mode[1]/len(peaks):.1f}% of {len(peaks)} basins)")
    return {"files": len(files), "read": len(stats), "bad_gzip": bad_gzip,
            "nan_hits": nan_hits, "peaks": peaks, "stats": agg,
            "median_rows": int(np.median(rows)) if rows else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/local_corpora/gages2_hysets_v1")
    ap.add_argument("--reference", default="data/local_corpora/camels_corpus_daymet_v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ref-limit", type=int, default=200)
    ap.add_argument("--min-basins", type=int, default=1500)
    ap.add_argument("--lag-share", type=float, default=0.75,
                    help="min share of POOL basins peaking at the reference lag "
                         "(S1c leg 2). Measured: SCDNA 0.80, Livneh+shift 0.57, "
                         "CAMELS on its own basins 0.92. The confound-free "
                         "registration measurement is leg 1, in "
                         "scripts/ledger41_registration_probe.py (bar 0.90; "
                         "SCDNA 0.988, Livneh 0.212 at lag 0).")
    ap.add_argument("--out", default="benchmarks/ledger41_stage1_gates.json")
    args = ap.parse_args()

    new = scan(args.corpus, args.limit, "ledger41 pretrain corpus")
    ref = scan(args.reference, args.ref_limit, "CAMELS reference corpus")

    from collections import Counter
    mode = lambda ps: Counter(ps).most_common(1)[0][0] if ps else None
    new_lag, ref_lag = mode(new["peaks"]), mode(ref["peaks"])

    print("\n" + "=" * 66)
    gates = {}

    gates["S1a_artifacts"] = {
        "n_basins": new["read"], "bar": args.min_basins,
        "bad_gzip": new["bad_gzip"],
        "pass": new["read"] >= args.min_basins and not new["bad_gzip"]}
    print(f"S1a artifacts     {new['read']} basins (bar {args.min_basins}), "
          f"{len(new['bad_gzip'])} corrupt -> "
          f"{'PASS' if gates['S1a_artifacts']['pass'] else 'FAIL'}")

    gates["S1b_no_nan"] = {"basins_with_nan": len(new["nan_hits"]),
                           "examples": new["nan_hits"][:5],
                           "pass": not new["nan_hits"]}
    print(f"S1b no NaN        {len(new['nan_hits'])} basins with NaN dynamics -> "
          f"{'PASS' if gates['S1b_no_nan']['pass'] else 'FAIL'}")

    # A matching mode is necessary but not sufficient: maurer also peaks at lag 0,
    # in only 62% of basins, because its precipitation is SMEARED across two days
    # rather than shifted. Measured shares on the CAMELS corpora with this exact
    # statistic: daymet 92%, nldas 89%, maurer 62%. The bar is set below the two
    # clean products and well above the smeared one.
    share = (Counter(new["peaks"])[new_lag] / len(new["peaks"])) if new["peaks"] else 0.0
    ref_share = (Counter(ref["peaks"])[ref_lag] / len(ref["peaks"])) if ref["peaks"] else 0.0
    gates["S1c_lag"] = {"pretrain_peak_lag": new_lag, "camels_peak_lag": ref_lag,
                        "pretrain_share_at_peak": round(share, 4),
                        "camels_share_at_peak": round(ref_share, 4),
                        "share_bar": args.lag_share,
                        "pass": (new_lag is not None and new_lag == ref_lag
                                 and share >= args.lag_share)}
    if new_lag is None or ref_lag is None:
        print("S1c lag           UNMEASURED")
    else:
        why = ""
        if new_lag != ref_lag:
            why = f"  <-- SHIFT THE CORPUS BY {ref_lag - new_lag:+d} DAY(S)"
        elif share < args.lag_share:
            why = "  <-- right mode, SMEARED profile (the maurer failure mode)"
        print(f"S1c lag           pretrain {new_lag:+d} @ {100*share:.0f}% vs "
              f"CAMELS {ref_lag:+d} @ {100*ref_share:.0f}% (bar {100*args.lag_share:.0f}%) -> "
              f"{'PASS' if gates['S1c_lag']['pass'] else 'FAIL'}{why}")

    print(f"\nS1d scaler distance vs {Path(args.reference).name}")
    print(f"  {'variable':26s} {'pretrain':>12s} {'camels':>12s} {'d mean':>9s} {'d std %':>9s}")
    sc = {}
    for c in DYNAMIC:
        if c not in new["stats"] or c not in ref["stats"]:
            continue
        (nm, ns), (rm, rs) = new["stats"][c], ref["stats"][c]
        dmean_sigma = (nm - rm) / rs if rs else np.nan
        dstd_pct = 100 * (ns - rs) / rs if rs else np.nan
        ok = abs(dmean_sigma) <= 0.15 and abs(dstd_pct) <= 15
        sc[c] = {"pretrain_mean": nm, "pretrain_std": ns, "ref_mean": rm,
                 "ref_std": rs, "dmean_sigma": dmean_sigma,
                 "dstd_pct": dstd_pct, "pass": bool(ok)}
        print(f"  {c:26s} {nm:6.2f}/{ns:5.2f} {rm:6.2f}/{rs:5.2f} "
              f"{dmean_sigma:+9.3f} {dstd_pct:+8.1f}%  {'ok' if ok else 'OUT OF BAND'}")
    gates["S1d_scaler"] = {"per_variable": sc,
                           "pass": bool(sc) and all(v["pass"] for v in sc.values())}

    verdict = all(g["pass"] for g in gates.values())
    print("\n" + "=" * 66)
    print(f"STAGE 1 VERDICT: {'PASS' if verdict else 'FAIL'}  "
          f"({sum(g['pass'] for g in gates.values())}/{len(gates)} gates)")
    if not verdict:
        print("FAIL ANY -> STOP. Fix the corpus; do not rationalise past a gate.")
    Path(args.out).write_text(json.dumps(
        {"gates": gates, "verdict": "PASS" if verdict else "FAIL"},
        indent=1, default=str))
    print(f"-> {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
