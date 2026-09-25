#!/usr/bin/env python3
"""LEDGER 50 — KNOWN-ANSWER TEST OF THE SENSITIVITY MAP.

The map (G0b/G0c/G0d) says a member's ensemble value is driven by two things:
its SKILL and its ERROR DECORRELATION, with measured coefficients

    Δ_ensemble ≈ k_skill * (solo_C − solo_I)  +  0.0584 * (corr_I − corr_C)

for a candidate C swapped in for incumbent I. Ledger 44 measured the actual held-out
h==1 swap delta for SEVEN already-rejected candidates, all swapped for `lstm_multi6`.
This checks the map against those seven known answers.

⚠️ This is an EXPLANATORY test, not an out-of-sample prediction: solo and corr are
computed on the same held-out rows the deltas were measured on. It asks "do the map's
two coefficients account for the spread of outcomes", not "could I have called them".
⛔ No candidate is shipped, reopened or re-seeded; ledger 44's NO-SHIP clause stands.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
import gate_eval as G                                          # noqa: E402
from phase_lever import load_allh                              # noqa: E402
from rescore9 import glob_seeds                                # noqa: E402
from ledger44_gate_transfer import per_basin_nse, TEST_END     # noqa: E402

FROZEN = Path.home() / "riverwatch2" / "frozen_base"
META = ("station_id", "date", "h", "truth")
INCUMBENT = "lstm_multi6"
K_SKILL = 0.188      # lstm_multi6's per-unit value at SMALL lifts (G0c shape applied to its +0.03 point)
K_CORR = 0.0584      # G0d, exactly linear


def main():
    TE = pd.read_pickle(FROZEN / "base_te.pkl")
    sc = TE[(TE.h == 1) & (TE.date <= TEST_END)].reset_index(drop=True)
    cols = [c for c in TE.columns if c not in META]
    others = [c for c in cols if c != INCUMBENT]
    y = sc["truth"].to_numpy(float)
    E = {c: sc[c].to_numpy(float) - y for c in cols}

    def meancorr(e):
        return float(np.mean([np.corrcoef(e, E[o])[0, 1] for o in others]))

    solo_I = float(per_basin_nse(sc, INCUMBENT).median())
    corr_I = meancorr(E[INCUMBENT])
    print(f"incumbent {INCUMBENT}: solo {solo_I:.4f}  mean err-corr {corr_I:.4f}")

    meas = {c["candidate"]: c["test_h1"]["paired"]
            for c in json.load(open("benchmarks/ledger44_gate_transfer.json"))["candidates"]}
    rows = []
    for tag, m in meas.items():
        ps = glob_seeds(f"camels531ls_{tag}_nhlstm_s*.csv.gz", G.L)
        if not ps:
            print(f"{tag}: no TEST dumps, skipped"); continue
        d = load_allh(ps)[list(META[:3]) + ["pred"]]
        j = sc[list(META[:3])].merge(d, on=list(META[:3]), how="left")
        if j["pred"].isna().any():
            print(f"{tag}: {int(j['pred'].isna().sum()):,} rows unmatched, skipped"); continue
        s2 = sc.copy(); s2[tag] = j["pred"].to_numpy(float)
        solo_C = float(per_basin_nse(s2, tag).median())
        corr_C = meancorr(s2[tag].to_numpy(float) - y)
        pred = K_SKILL * (solo_C - solo_I) + K_CORR * (corr_I - corr_C)
        rows.append({"cand": tag, "n_seeds": len(ps), "solo": solo_C, "corr": corr_C,
                     "d_solo": solo_C - solo_I, "d_corr": corr_I - corr_C,
                     "predicted": pred, "measured": m})
        print(f"  {tag:<10} solo {solo_C:.4f} ({solo_C-solo_I:+.4f})  corr {corr_C:.4f} "
              f"({corr_I-corr_C:+.4f})  pred {pred:+.6f}  measured {m:+.6f}")
    if len(rows) >= 3:
        P = np.array([r["predicted"] for r in rows]); M = np.array([r["measured"] for r in rows])
        r_p = float(np.corrcoef(P, M)[0, 1])
        r_s = float(pd.Series(P).corr(pd.Series(M), method="spearman"))
        print(f"\nn={len(rows)}  pearson(pred, measured) = {r_p:+.3f}   spearman = {r_s:+.3f}")
        print(f"  predicted range [{P.min():+.6f}, {P.max():+.6f}]  "
              f"measured range [{M.min():+.6f}, {M.max():+.6f}]")
        print(f"  ⇒ the map's SCALE is {'far too large' if abs(P).mean() > 5*abs(M).mean() else 'comparable'}"
              f" (mean |pred| {abs(P).mean():.6f} vs mean |meas| {abs(M).mean():.6f})")
        Path("benchmarks/ledger50_validate.json").write_text(json.dumps(
            {"incumbent": INCUMBENT, "solo_I": solo_I, "corr_I": corr_I,
             "k_skill": K_SKILL, "k_corr": K_CORR, "rows": rows,
             "pearson": r_p, "spearman": r_s}, indent=2))
        print("wrote benchmarks/ledger50_validate.json")


if __name__ == "__main__":
    main()
