#!/usr/bin/env python3
"""LEDGER 50 — THE ACHIEVABLE FRONTIER: skill vs decorrelation, for every member ever built.

G0c/G0d give the two coefficients, hence a BREAK-EVEN LINE for replacing `lstm_multi`:
    0.188 * d_solo + 0.0584 * d_corr_removed = 0   ⇒   d_solo = -0.31 * d_corr_removed
G0e showed the map's MAGNITUDE is ~10x too large outside its measurement regime (the MSE^-4
rule down-weights a weak candidate and absorbs the difference) — but the SIGN agreed on 5 of 7
candidates and the two disagreements were within noise of zero, so the BREAK-EVEN LINE itself
is usable even where the magnitudes are not: a uniform damping of both terms leaves the line.

This places every member with usable dumps on the (d_corr_removed, d_solo) plane against that
line, and asks whether ANY of them ever landed on the profitable side.

⛔ Descriptive/bounding only. Nothing is shipped or reopened; ledger 44's NO-SHIP clause stands.
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
REF = "lstm_multi"          # the incumbent a new multi-forcing member would replace
# ⚠️ BUG FIXED 2026-09-02: this was written 0.188/0.0584 = 3.22, the RECIPROCAL, which made the
# line 10x too permissive and returned '19 of 19 members profitable' — against 19 known failures.
# An implausible result is a bug until proven otherwise. Break-even is
#   0.188*d_solo + 0.0584*d_corr = 0  =>  d_solo = -(0.0584/0.188)*d_corr
SLOPE = 0.0584 / 0.188      # = 0.3106 of solo per 1.0 of correlation removed

CANDIDATES = ["multiL", "multi8", "multi9", "multi10", "multi11", "multi12", "multi5b",
              "multidrop", "multih128", "multih512", "multi730", "multi671", "multi14",
              "multi_eps05", "multi4", "multi9", "multibagb", "cmal", "gmm", "multi5swa"]


def main():
    TE = pd.read_pickle(FROZEN / "base_te.pkl")
    sc = TE[(TE.h == 1) & (TE.date <= TEST_END)].reset_index(drop=True)
    cols = [c for c in TE.columns if c not in META]
    y = sc["truth"].to_numpy(float)
    E = {c: sc[c].to_numpy(float) - y for c in cols}
    others = [c for c in cols if c != REF]        # the 8 a replacement would sit beside

    def meancorr(e):
        return float(np.mean([np.corrcoef(e, E[o])[0, 1] for o in others]))

    solo_R, corr_R = float(per_basin_nse(sc, REF).median()), meancorr(E[REF])
    print(f"reference {REF}: solo {solo_R:.4f}  mean err-corr {corr_R:.4f}")
    print(f"break-even: a member may lose {SLOPE:.2f} of solo per 1.0 of correlation removed "
          f"(= {SLOPE*0.10:.4f} solo per 0.10)\n")
    print(f"{'member':<14}{'seeds':>6}{'solo':>9}{'d_solo':>9}{'corr':>8}{'d_corr':>9}"
          f"{'needed':>9}{'margin':>10}  verdict")
    rows, seen = [], set()
    for tag in CANDIDATES:
        if tag in seen:
            continue
        seen.add(tag)
        ps = glob_seeds(f"camels531ls_{tag}_nhlstm_s*.csv.gz", G.L)
        if not ps:
            continue
        d = load_allh(ps)[list(META[:3]) + ["pred"]]
        j = sc[list(META[:3])].merge(d, on=list(META[:3]), how="left")
        if j["pred"].isna().any():
            print(f"{tag:<14} skipped (frame mismatch: {int(j['pred'].isna().sum()):,} rows)")
            continue
        s2 = sc.copy(); s2[tag] = j["pred"].to_numpy(float)
        solo = float(per_basin_nse(s2, tag).median())
        corr = meancorr(s2[tag].to_numpy(float) - y)
        d_solo, d_corr = solo - solo_R, corr_R - corr
        needed = -SLOPE * d_corr            # the most solo it may lose for that decorrelation
        margin = d_solo - needed            # > 0 ⇒ on the profitable side of the line
        rows.append({"member": tag, "n_seeds": len(ps), "solo": solo, "corr": corr,
                     "d_solo": d_solo, "d_corr_removed": d_corr,
                     "allowed_solo_loss": needed, "margin": margin})
        print(f"{tag:<14}{len(ps):>6}{solo:>9.4f}{d_solo:>+9.4f}{corr:>8.4f}{d_corr:>+9.4f}"
              f"{needed:>+9.4f}{margin:>+10.4f}  {'PROFITABLE' if margin > 0 else 'below the line'}")
    if rows:
        m = np.array([r["margin"] for r in rows])
        print(f"\nn={len(rows)} members with usable dumps.  margins: "
              f"best {m.max():+.4f} ({rows[int(np.argmax(m))]['member']}), "
              f"worst {m.min():+.4f}, median {np.median(m):+.4f}")
        print(f"members on the profitable side of the break-even line: {(m > 0).sum()} of {len(rows)}")
        Path("benchmarks/ledger50_frontier.json").write_text(json.dumps(
            {"reference": REF, "solo_ref": solo_R, "corr_ref": corr_R,
             "breakeven_slope_solo_per_corr": SLOPE, "members": rows}, indent=2))
        print("wrote benchmarks/ledger50_frontier.json")


if __name__ == "__main__":
    main()
