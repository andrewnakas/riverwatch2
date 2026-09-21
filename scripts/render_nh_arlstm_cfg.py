#!/usr/bin/env python3
"""LEDGER 54 — render a NeuralHydrology AR-LSTM config for one member/seed.

Replaces the sed-substituted template of LEDGER 53, which could not vary the
`dynamic_inputs` list (single-forcing members have 5 inputs, the multi-forcing
member 15). Everything not listed here is Nearing 2022's recipe verbatim.

usage: render_nh_arlstm_cfg.py --variant nhar0d --seed 501 --base /path/gpu1080 \
         --data-dir nh_data/daymet --forcings daymet --hidden 128 --holdout 0.0 \
         --epochs 30 --basin-file nh_data/daymet/basins.txt --out cfg.yml
"""
import argparse
from pathlib import Path

VARS = ["prcp", "tmax", "tmin", "vp", "srad"]
STATICS = ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq",
           "high_prec_dur", "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean",
           "area_gages2", "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
           "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",
           "frac_forest", "lai_max", "gvf_max", "gvf_diff", "root_depth_50",
           "carbonate_rocks_frac", "geol_permeability"]


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("variant", "base", "data-dir", "forcings", "basin-file", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--holdout", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    fs = [f.strip() for f in a.forcings.split(",") if f.strip()]
    # one forcing -> bare names (the single-forcing NH datasets); many -> suffixed
    dyn = VARS if len(fs) == 1 else [f"{v}_{f}" for f in fs for v in VARS]
    B = a.base.rstrip("/")
    L = [
        "# LEDGER 54 — NeuralHydrology AR-LSTM member, Nearing 2022 recipe",
        "# (HESS 26:5493): all inputs simultaneous, NSE loss, 365-d sequences,",
        "# batch 256, 30 epochs, lr 1e-3 -> 5e-4 @10 -> 1e-4 @25, lagged discharge",
        "# as an autoregressive input with a binary obs/sim flag.",
        f"# variant={a.variant} forcings={','.join(fs)} hidden={a.hidden} holdout={a.holdout}",
        f"experiment_name: l54_{a.variant}_s{a.seed}",
        f"run_dir: {B}/nh_runs",
        f"train_basin_file: {B}/{a.basin_file}",
        f"validation_basin_file: {B}/{a.basin_file}",
        f"test_basin_file: {B}/{a.basin_file}",
        f"data_dir: {B}/{a.data_dir}",
        "dataset: generic",
        '# Nearing 2022 split, identical to LEDGERS 51-53',
        'train_start_date: "01/10/1999"', 'train_end_date: "30/09/2008"',
        'validation_start_date: "01/10/1980"', 'validation_end_date: "30/09/1989"',
        'test_start_date: "01/10/1989"', 'test_end_date: "30/09/1999"',
        f"seed: {a.seed}", "device: cuda:0", "model: arlstm",
        f"hidden_size: {a.hidden}", "initial_forget_bias: 3", "output_dropout: 0.4",
        "head: regression", "output_activation: linear", "optimizer: Adam", "loss: NSE",
        "learning_rate:", "  0: 1e-3", "  10: 5e-4", "  25: 1e-4",
        "batch_size: 256", f"epochs: {a.epochs}", "seq_length: 365", "predict_last_n: 1",
        "clip_gradient_norm: 1", "num_workers: 2",
        "# in-training validation runs WITH the holdout and on a subsample: a health",
        "# curve only. The scored evaluation is scripts/nh_arlstm_eval.py (holdout off,",
        "# all basins).",
        "validate_every: 10", "validate_n_random_basins: 20", "save_validation_results: False",
        "metrics:", "  - NSE",
        "dynamic_inputs:", *[f"  - {v}" for v in dyn],
        "target_variables:", "  - q_mm",
        "lagged_features:", "  q_mm: [1]",
        "autoregressive_inputs:", "  - q_mm_shift1",
        "random_holdout_from_dynamic_features:", "  q_mm_shift1:",
        f"    missing_fraction: {a.holdout}", "    mean_missing_length: 5",
        "static_attributes:", *[f"  - {s}" for s in STATICS],
    ]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"rendered {a.out}: variant={a.variant} seed={a.seed} dyn={len(dyn)} "
          f"hidden={a.hidden} holdout={a.holdout} epochs={a.epochs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
