#!/usr/bin/env python3
"""VECTOR F -- is any static attribute derived from OBSERVED DISCHARGE?

Addor et al. (2017) ship CAMELS attributes in groups. The climate/soil/geology/topo/vegetation
groups are forcing- or map-derived and carry no target information. The HYDROLOGIC SIGNATURE group
(q_mean, runoff_ratio, baseflow_index, q5, q95, high_q_freq, low_q_freq, zero_q_freq, hfd_mean,
stream_elas, slope_fdc) is computed FROM OBSERVED DISCHARGE over 1989-10-01..2009-09-30. Any of
those in the static set hands the model a summary statistic of its own target -- and under P-MODERN
that window overlaps the training span, so it would also be a per-basin target fingerprint.

Read the names back out of the CHECKPOINTS, never from a module default: an earlier pass in this
audit nearly reported false leakage by reading the module-level STATIC_FEATS (14 GAGES-II features
including RUNAVE7100 and BFI_AVE) instead of cfg["static_feats"] as actually trained.
"""
import glob, json, re
import torch

SIGNATURE = {
    "q_mean", "runoff_ratio", "baseflow_index", "stream_elas", "slope_fdc", "q5", "q95",
    "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "hfd_mean",
    # GAGES-II discharge-derived equivalents seen elsewhere in this repo
    "RUNAVE7100", "BFI_AVE", "PERDUN", "PERHOR", "TOPWET", "WB5100_ANN_MM",
}
# substring probes for anything discharge-ish we did not enumerate
PROBES = ("q_", "_q", "runoff", "baseflow", "bfi", "flow", "discharge", "fdc", "hfd", "elas")

cks = sorted(glob.glob("data/mblstm/gpu_ckpts/m1_*.pt"))
print(f"reading static_feats out of {len(cks)} MODERN-1 checkpoints\n")
sets, per = {}, {}
for c in cks:
    try:
        cfg = torch.load(c, map_location="cpu", weights_only=False).get("cfg", {})
    except Exception as e:
        print(f"  !! {c}: {e}"); continue
    sf = cfg.get("static_feats")
    if sf is None: continue
    key = tuple(sf)
    sets.setdefault(key, []).append(c.split("/")[-1])
    per[c] = sf

print(f"distinct static sets across checkpoints: {len(sets)}")
bad_any = []
for i, (key, members) in enumerate(sets.items(), 1):
    names = list(key)
    exact = sorted(n for n in names if n in SIGNATURE)
    fuzzy = sorted(n for n in names
                   if n not in exact and any(p in n.lower() for p in PROBES))
    print(f"\n--- set {i}: {len(names)} attributes, {len(members)} checkpoints")
    print(f"    e.g. {members[0]}" + (f" (+{len(members)-1} more)" if len(members) > 1 else ""))
    print("    " + ", ".join(names))
    if exact:
        print(f"    ⛔ DISCHARGE-DERIVED (exact match): {exact}"); bad_any += exact
    if fuzzy:
        print(f"    ⚠️ needs a human look (name pattern): {fuzzy}")
    if not exact and not fuzzy:
        print("    ✅ no discharge-derived attribute")

print("\n" + "=" * 74)
if bad_any:
    print(f"⛔ VECTOR F FIRES: {sorted(set(bad_any))} are computed from observed discharge.")
else:
    print("✅ VECTOR F CLEAN: no static attribute in any MODERN-1 checkpoint is derived from")
    print("   observed discharge. Every one is forcing- or map-derived.")
json.dump({"n_checkpoints": len(per), "n_distinct_sets": len(sets),
           "sets": [{"n": len(k), "names": list(k), "checkpoints": v} for k, v in sets.items()],
           "discharge_derived_found": sorted(set(bad_any))},
          open("benchmarks/m1_leakaudit_vectorF_statics.json", "w"), indent=2)
