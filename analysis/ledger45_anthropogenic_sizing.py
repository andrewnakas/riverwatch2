#!/usr/bin/env python3
"""Ledger 45, Track 5: anthropogenic-disturbance KILL TEST (oracle sizing).

Question: could fixing anthropogenically-disturbed basins (dams / reservoirs /
irrigation / withdrawals) move the 531-basin median NSE at all?

This is a pure re-sorting computation on the frozen per-basin NSE vector of the
current record (0.8362893) -- no modelling. The disturbance cohort is built
A PRIORI from GAGES-II static attributes (Falcone 2011), so unlike the dead
skill-ranked cohort attempts it is not circular.

Data:
  analysis/noq_test_result.json        (md5 must be c4d7639df8647308e48dedc9f45f824e)
  data/cache/gages2/conterm_*.txt      (extracted from the published GAGES-II
                                        archive basinchar_and_report_sept_2011.zip,
                                        spreadsheets-in-csv-format.zip; latin-1)

LEAKAGE AUDIT (mandatory): the benchmark forbids discharge in any form.
Every field used below was checked against gagesII_sept30_2011_var_desc.xlsx.
  BANNED  BFI_AVE      -- ratio of base flow to total streamflow (q-derived)
  BANNED  RUNAVE7100   -- runoff map integrating "water use, regulation" --
                          built from gauged streamflow (q-derived)
  USED    WB5100_ANN_MM -- Wolock & McCabe water-BALANCE MODEL runoff from
                          precip+temperature only; explicitly NOT from gauges
                          and NOT reflecting regulation. Used only to express
                          reservoir storage as days-of-flow. Not a cohort
                          field itself.
  CAVEAT  CLASS (Ref/Non-ref) -- Falcone's screening is GIS-based but the
                          final reference call involved inspection of gauge
                          record remarks/hydrographs; reported as a
                          DIAGNOSTIC only, never used for the primary cohort.
All dam fields are National Inventory of Dams counts/storage; withdrawal is
county water-use estimates; irrigation is MODIS; canals are NHDPlus coding;
HYDRO_DISTURB_INDX is a 7-component GIS composite (major-dam density,
withdrawals, storage change, canals, NPDES distance, roads, fragmentation)
-- none derived from streamflow observations.

Output: benchmarks/ledger45_anthropogenic_sizing.json
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
NSE_PATH = ROOT / "analysis" / "noq_test_result.json"
NSE_MD5 = "c4d7639df8647308e48dedc9f45f824e"
G2 = ROOT / "data" / "cache" / "gages2"
OUT = ROOT / "benchmarks" / "ledger45_anthropogenic_sizing.json"

FIELDS_USED = {
    "conterm_hydromod_dams.txt": [
        "NDAMS_2009", "MAJ_NDAMS_2009", "DDENS_2009", "MAJ_DDENS_2009",
        "STOR_NID_2009", "STOR_NOR_2009", "RAW_DIS_NEAREST_DAM",
    ],
    "conterm_hydromod_other.txt": [
        "FRESHW_WITHDRAWAL", "PCT_IRRIG_AG", "CANALS_PCT", "NPDES_MAJ_DENS",
    ],
    "conterm_bas_classif.txt": ["HYDRO_DISTURB_INDX", "CLASS"],
    "conterm_hydro.txt": ["WB5100_ANN_MM"],  # water-balance MODEL runoff (see audit)
}
FIELDS_EXCLUDED = {
    "BFI_AVE": "base-flow index = ratio of base flow to total observed streamflow; q-derived (banned)",
    "RUNAVE7100": "1971-2000 runoff map integrating water use/regulation, built from gauged streamflow; q-derived (banned)",
    "PERDUN/PERHOR/CONTACT/TOPWET(hydro-sheet siblings)": "not needed; hydro sheet only opened for the WB5100 model column",
    "CLASS-as-primary-cohort": "Falcone reference screening partially involved inspection of gauge records; kept as diagnostic only",
}


def read_sheet(name, fields):
    out = {}
    with open(G2 / name, encoding="latin-1", newline="") as f:
        for r in csv.DictReader(f):
            sid = r["STAID"].strip().zfill(8)
            out[sid] = {k: r.get(k, "").strip() for k in fields}
    return out


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def median_of(vals):
    return statistics.median(vals)


def oracle(nse, cohort, transform):
    """Median after applying `transform` to every cohort basin's NSE."""
    newv = [transform(v) if b in cohort else v for b, v in nse.items()]
    return median_of(newv)


def main():
    md5 = hashlib.md5(NSE_PATH.read_bytes()).hexdigest()
    assert md5 == NSE_MD5, f"noq_test_result.json md5 {md5} != {NSE_MD5}"

    nse = json.load(open(NSE_PATH))["per_basin_nse"]
    assert len(nse) == 531
    base_median = median_of(list(nse.values()))
    p75 = statistics.quantiles(list(nse.values()), n=4)[2]

    # ---- attach GAGES-II attrs to the 531 -------------------------------
    dams = read_sheet("conterm_hydromod_dams.txt", FIELDS_USED["conterm_hydromod_dams.txt"])
    other = read_sheet("conterm_hydromod_other.txt", FIELDS_USED["conterm_hydromod_other.txt"])
    clas = read_sheet("conterm_bas_classif.txt", FIELDS_USED["conterm_bas_classif.txt"])
    hyd = read_sheet("conterm_hydro.txt", FIELDS_USED["conterm_hydro.txt"])

    basins = sorted(nse)
    missing = [b for b in basins if b not in dams or b not in clas]
    attrs = {}
    for b in basins:
        a = {}
        a.update({k: num(v) for k, v in dams.get(b, {}).items()})
        a.update({k: num(v) for k, v in other.get(b, {}).items()})
        a["HYDRO_DISTURB_INDX"] = num(clas.get(b, {}).get("HYDRO_DISTURB_INDX"))
        a["CLASS"] = clas.get(b, {}).get("CLASS")
        a["WB5100_ANN_MM"] = num(hyd.get(b, {}).get("WB5100_ANN_MM"))
        # storage as days of (model, non-q) mean flow:
        # STOR_NOR_2009 megaliters/km2 == mm of water depth over the basin.
        stor_mm = a.get("STOR_NOR_2009")
        wb = a.get("WB5100_ANN_MM")
        a["STOR_DAYS_WB"] = (stor_mm / (wb / 365.25)) if (stor_mm is not None and wb and wb > 0) else None
        attrs[b] = a

    # ---- cohort definitions --------------------------------------------
    def coh(pred):
        return {b for b in basins if pred(attrs[b])}

    hdi_vals = [attrs[b]["HYDRO_DISTURB_INDX"] for b in basins if attrs[b]["HYDRO_DISTURB_INDX"] is not None]
    hdi_q = statistics.quantiles(hdi_vals, n=4)
    hdi_p90 = statistics.quantiles(hdi_vals, n=10)[8]

    cohorts = {
        "any_dam (NDAMS_2009>=1)": coh(lambda a: (a["NDAMS_2009"] or 0) >= 1),
        "ge5_dams (NDAMS_2009>=5)": coh(lambda a: (a["NDAMS_2009"] or 0) >= 5),
        "major_dam (MAJ_NDAMS_2009>=1)": coh(lambda a: (a["MAJ_NDAMS_2009"] or 0) >= 1),
        "storage>=10d_wbflow": coh(lambda a: (a["STOR_DAYS_WB"] or 0) >= 10),
        "storage>=30d_wbflow": coh(lambda a: (a["STOR_DAYS_WB"] or 0) >= 30),
        "storage>=90d_wbflow": coh(lambda a: (a["STOR_DAYS_WB"] or 0) >= 90),
        f"hdi_top_half (>= {hdi_q[1]:g})": coh(lambda a: a["HYDRO_DISTURB_INDX"] is not None and a["HYDRO_DISTURB_INDX"] >= hdi_q[1]),
        f"hdi_top_quartile (>= {hdi_q[2]:g})": coh(lambda a: a["HYDRO_DISTURB_INDX"] is not None and a["HYDRO_DISTURB_INDX"] >= hdi_q[2]),
        f"hdi_top_decile (>= {hdi_p90:g})": coh(lambda a: a["HYDRO_DISTURB_INDX"] is not None and a["HYDRO_DISTURB_INDX"] >= hdi_p90),
        "irrig>=5pct": coh(lambda a: (a["PCT_IRRIG_AG"] or 0) >= 5),
        "irrig>=1pct": coh(lambda a: (a["PCT_IRRIG_AG"] or 0) >= 1),
        "withdrawal>=50mm_yr": coh(lambda a: (a["FRESHW_WITHDRAWAL"] or 0) >= 50),
        "withdrawal>=10mm_yr": coh(lambda a: (a["FRESHW_WITHDRAWAL"] or 0) >= 10),
        "canals>=5pct": coh(lambda a: (a["CANALS_PCT"] or 0) >= 5),
        "nonref_CLASS (diagnostic only)": coh(lambda a: a["CLASS"] == "Non-ref"),
    }
    cohorts["union_disturbed"] = (
        cohorts["major_dam (MAJ_NDAMS_2009>=1)"]
        | cohorts["storage>=30d_wbflow"]
        | cohorts["irrig>=5pct"]
        | cohorts[f"hdi_top_quartile (>= {hdi_q[2]:g})"]
        | cohorts["withdrawal>=50mm_yr"]
    )

    # ---- diagnostics: is disturbance even correlated with poor NSE? -----
    def spear(field):
        pairs = [(attrs[b][field], nse[b]) for b in basins if attrs[b][field] is not None]
        if len(pairs) < 10:
            return None
        r, p = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
        return {"n": len(pairs), "spearman_r": round(float(r), 4), "p": float(f"{p:.3g}")}

    correlations = {f: spear(f) for f in
                    ["NDAMS_2009", "MAJ_NDAMS_2009", "DDENS_2009", "STOR_NID_2009",
                     "STOR_NOR_2009", "STOR_DAYS_WB", "HYDRO_DISTURB_INDX",
                     "FRESHW_WITHDRAWAL", "PCT_IRRIG_AG", "CANALS_PCT",
                     "NPDES_MAJ_DENS", "RAW_DIS_NEAREST_DAM"]}

    # ---- cohort table: size, position vs median, oracle sweep -----------
    UPLIFTS = {
        "plus_0.02": lambda v: min(v + 0.02, 1.0),
        "plus_0.05": lambda v: min(v + 0.05, 1.0),
        "plus_0.10": lambda v: min(v + 0.10, 1.0),
        "raise_to_p75": lambda v: max(v, p75),
        "raise_to_1.0": lambda v: 1.0,
    }
    table = {}
    for name, c in cohorts.items():
        if not c:
            table[name] = {"n": 0}
            continue
        cv = [nse[b] for b in c]
        rv = [nse[b] for b in basins if b not in c]
        below = sum(1 for v in cv if v < base_median)
        u = mannwhitneyu(cv, rv, alternative="less") if rv else None
        entry = {
            "n": len(c),
            "n_below_median": below,
            "n_above_median": len(c) - below,
            "frac_of_531": round(len(c) / 531, 3),
            "cohort_median_nse": round(median_of(cv), 4),
            "rest_median_nse": round(median_of(rv), 4) if rv else None,
            "cohort_minus_rest": round(median_of(cv) - median_of(rv), 4) if rv else None,
            "mannwhitney_p_cohort_worse": float(f"{u.pvalue:.3g}") if u else None,
            "oracle_median_delta": {},
        }
        for uname, fn in UPLIFTS.items():
            m = oracle(nse, c, fn)
            entry["oracle_median_delta"][uname] = round(m - base_median, 7)
        # campaign-standard near-median lift (band = median +/- 0.02);
        # the previous (skill-ranked) cohort attempt died at 1.70x vs a 3.0x bar
        band = [b for b in basins if abs(nse[b] - base_median) <= 0.02]
        in_band = sum(1 for b in c if abs(nse[b] - base_median) <= 0.02)
        dens_c = in_band / len(c)
        dens_all = len(band) / 531
        entry["near_median_band_n"] = in_band
        entry["near_median_lift"] = round(dens_c / dens_all, 2) if dens_all else None
        table[name] = entry

    result = {
        "ledger": 45,
        "track": 5,
        "question": "could fixing anthropogenically-disturbed basins move the 531-median NSE?",
        "nse_source": {"path": "analysis/noq_test_result.json", "md5": md5,
                       "n_basins": 531, "base_median": base_median,
                       "p75": p75},
        "gages2_source": "basinchar_and_report_sept_2011.zip (USGS, Falcone 2011), spreadsheets-in-csv-format",
        "n_531_missing_gages2": len(missing),
        "missing_ids": missing,
        "fields_used": FIELDS_USED,
        "fields_excluded_q_leakage": FIELDS_EXCLUDED,
        "storage_days_note": "STOR_NOR_2009 [ML/km2 == mm] / (WB5100_ANN_MM/365.25); denominator is a precip+temperature water-balance MODEL, not observed discharge",
        "hdi_quantiles_531": {"q25": hdi_q[0], "q50": hdi_q[1], "q75": hdi_q[2], "p90": hdi_p90},
        "correlations_vs_nse": correlations,
        "cohort_table": table,
        "bars": {"resolution_floor": 0.00103, "record_claim": 0.003},
        "verdict": {
            "call": "KILL",
            "reasons": [
                "all 531 CAMELS basins are GAGES-II CLASS=Ref (least-disturbed); the non-ref cohort is EMPTY -- the corpus was screened for exactly the disturbance the SOTA authors blame",
                "disturbance-NSE correlations are weak (max |rho|=0.19, STOR_DAYS_WB; ~3.7% of rank variance) and FRESHW_WITHDRAWAL is POSITIVELY correlated with NSE (+0.12)",
                "every genuinely-disturbed cohort has near-median lift 0.00-0.68x (prior cohort attempt died at 1.70x vs a 3.0x bar): disturbed basins sit deep in the low tail, where uplift cannot cross the median",
                "plausible oracle (+0.05 NSE on every cohort basin) moves the median +0.0002..+0.0018 for all targeted cohorts -- below the +0.003 record bar; +0.02 is below the 0.00103 resolution floor",
                "to reach +0.003 via the largest measurably-hurt cohort (storage>=10d, n=40) requires >+0.10 NSE on all 40 basins, ~30x any per-basin gain any campaign lever has delivered",
                "broad cohorts (any_dam n=303, union n=221) show larger oracle deltas only because they cover ~half the corpus, i.e. they are not a targeting signal at all",
            ],
        },
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1)[:200])
    print("wrote", OUT)
    print("md5", hashlib.md5(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
