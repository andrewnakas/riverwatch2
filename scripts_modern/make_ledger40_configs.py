#!/usr/bin/env python
"""Write the three matched ledger-40 configs from the frozen daymet baseline.

A (control)  : CAMELS daymet, 5 dyn + 9 stat, RANDOM init
B (pretrain) : 4,634 Caravan basins, same 5 dyn + 9 stat
C (finetune) : CAMELS daymet, same everything as A, but initialised from B

Everything except `data_dir`, dates, epochs and the init is copied verbatim from
`rw2ls_daymet_lstm_mm_s222` so the comparison is matched by construction.

⚠️ A and C use the SAME 9 statics. Judging C against the existing 27-static
0.7542 baseline would confound initialisation with the attribute set.
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import yaml

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/base_cfg.yml")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ledger40_cfgs")

DYN = ["prcp", "tmax", "tmin", "srad", "vp"]
STAT = ["p_mean", "pet_mean", "aridity", "frac_snow", "high_prec_freq",
        "high_prec_dur", "low_prec_freq", "low_prec_dur", "p_seasonality"]


def main():
    cfg = yaml.safe_load(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    for k in ("run_dir", "train_dir", "img_log_dir", "experiment_name"):
        cfg.pop(k, None)
    cfg["dynamic_inputs"] = list(DYN)
    cfg["static_attributes"] = list(STAT)
    cfg["target_variables"] = ["q_mm"]

    # ---- A: control, random init, CAMELS daymet -------------------------
    a = copy.deepcopy(cfg)
    a["experiment_name"] = "l40a_control_daymet_9stat_s222"
    a["data_dir"] = "/home/nakas/riverwatch2/gpu1080/nh_data/daymet"
    for k in ("train_basin_file", "validation_basin_file", "test_basin_file"):
        a[k] = "/home/nakas/riverwatch2/gpu1080/nh_data/daymet/basins.txt"
    (OUT / "l40a_control.yml").write_text(yaml.safe_dump(a, sort_keys=False))

    # ---- B: pretrain on Caravan ----------------------------------------
    b = copy.deepcopy(cfg)
    b["experiment_name"] = "l40b_pretrain_caravan_s222"
    b["data_dir"] = "/home/nakas/riverwatch2/gpu4050/nh_data_caravan"
    for k in ("train_basin_file", "validation_basin_file", "test_basin_file"):
        b[k] = "/home/nakas/riverwatch2/gpu4050/nh_data_caravan/basins.txt"
    # Caravan spans 1981-2020. Hold out the tail so pretraining quality is
    # observable; the CAMELS split is irrelevant here and must be overridden.
    b["train_start_date"] = "02/01/1981"
    b["train_end_date"] = "31/12/2014"
    b["validation_start_date"] = "01/01/2015"
    b["validation_end_date"] = "31/12/2016"
    b["test_start_date"] = "01/01/2017"
    b["test_end_date"] = "31/12/2020"
    # ~23x the CAMELS sample volume per epoch => far fewer epochs, and the LR
    # schedule must be rewritten or it would never leave the first rung.
    b["epochs"] = 5
    b["learning_rate"] = {0: 0.001, 3: 0.0005, 4: 0.0001}
    b["validate_every"] = 5
    b["num_workers"] = 4
    (OUT / "l40b_pretrain.yml").write_text(yaml.safe_dump(b, sort_keys=False))

    # ---- C: finetune from B --------------------------------------------
    c = copy.deepcopy(a)
    c["experiment_name"] = "l40c_finetune_from_caravan_s222"
    # base_run_dir/checkpoint filled in at launch once B has produced weights
    (OUT / "l40c_finetune.yml").write_text(yaml.safe_dump(c, sort_keys=False))

    print(f"wrote 3 configs -> {OUT}")
    for f in sorted(OUT.glob("*.yml")):
        d = yaml.safe_load(f.read_text())
        print(f"  {f.name:<20} epochs={d['epochs']:<3} data={Path(d['data_dir']).name:<16} "
              f"dyn={len(d['dynamic_inputs'])} stat={len(d['static_attributes'])} "
              f"train={d['train_start_date']}..{d['train_end_date']}")


if __name__ == "__main__":
    main()
