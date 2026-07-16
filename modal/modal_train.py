"""Modal app: train the Kratzert/Li-Shen reference LSTM (neuralhydrology CudaLSTM)
on the CAMELS-531 no-q benchmark — one member per GPU call, fanned out over
3 forcings × 3 seeds → the LSTM¹²³ ensemble rung (paper Table D1 = 0.808), which
then folds into our existing δHBV members toward the 0.83 record.

WHY MODAL: free-tier serverless GPU we can drive non-interactively (no browser
notebook, no SSH babysitting). Each member is ~1 GPU-hr on a T4/L4; 9 members fit
the ~$30/mo free credit comfortably.

RECIPE (verified vs Kratzert 2021 + Li/Shen 2025 HESS 29:6829, this session):
  split  : train 1999-10-01→2008-09-30, val 1980-1989, test 1989-10-01→1999-09-30
           (Kratzert 2021 — the paper defers to it; == our prior split, confirmed)
  model  : CudaLSTM, hidden 256, seq 365, predict_last_n 1, output_dropout 0.4
  loss   : NSE ;  optimizer Adam, LR {0:1e-3, 20:5e-4, 25:1e-4}, batch 256, 30 epochs
  inputs : 5 forcings [prcp,tmax,tmin,vp,srad], target SPECIFIC DISCHARGE q_mm (mm/day)
  statics: 27 Addor attrs
  ensemble: per-forcing (daymet/nldas/maurer) × 3 seeds, simple mean of streamflow

DATA: the 3 forcing corpora + camels_attrs.json + camels_gauge_ids.json are uploaded
ONCE to the Modal Volume `riverwatch-corpora` by modal_launch.py (needs the SD card
mounted). Scripts (corpus_to_nh.py, nh_to_dump.py) are baked into the image from the
repo at deploy time.

RUN (from repo root, after `modal token new`):
  # 1. upload corpora (SD card must be mounted):
  ./.venv/bin/python modal/modal_launch.py upload
  # 2. train all 9 members + pull dumps:
  ./.venv/bin/python modal/modal_launch.py train
"""
from __future__ import annotations

import modal

APP_NAME = "riverwatch-nh-lstm"
VOLUME = "riverwatch-corpora"          # holds corpora + attrs (uploaded once)
OUT_VOLUME = "riverwatch-nh-out"       # holds per-member dumps to pull back

app = modal.App(APP_NAME)

# Image: neuralhydrology + our two scripts + the small json inputs baked in.
# torch/CUDA come from the NH dep tree; pin a CUDA-capable torch for the T4/L4.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "neuralhydrology==1.13.0",
        "torch==2.4.1",
        "numpy<2",
        "pandas",
        "xarray",
        "netCDF4",
        "scipy",
        "PyYAML",
        "tqdm",
    )
    # bake the repo scripts + json inputs into /root/rw
    .add_local_dir(
        local_path=".",
        remote_path="/root/rw",
        # only the bits the member job needs (scripts + the two json files);
        # the corpora come from the Volume, not the image.
        ignore=[
            "*.pt", "*.csv.gz", "*.p", ".git", ".venv", "data/nh/runs",
            "data/mblstm", "research", "benchmarks", "notebooks", "app",
            "*.png", "*.json.gz",
        ],
        copy=True,
    )
)

corpora_vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)

FORCINGS = ["daymet", "nldas", "maurer"]
SEEDS = [111, 222, 333]

NH_CONFIG_TMPL = """\
experiment_name: rw2_{forcing}_lstm_mm_s{seed}
run_dir: /out/runs
train_basin_file: /data/nh/{forcing}/basins.txt
validation_basin_file: /data/nh/{forcing}/basins.txt
test_basin_file: /data/nh/{forcing}/basins.txt
data_dir: /data/nh/{forcing}
dataset: generic
train_start_date: "01/10/1999"
train_end_date: "30/09/2008"
validation_start_date: "01/10/1980"
validation_end_date: "30/09/1989"
test_start_date: "01/10/1989"
test_end_date: "30/09/1999"
seed: {seed}
device: cuda:0
model: cudalstm
hidden_size: 256
initial_forget_bias: 3
output_dropout: 0.4
head: regression
output_activation: linear
optimizer: Adam
loss: NSE
learning_rate:
  0: 1e-3
  20: 5e-4
  25: 1e-4
batch_size: 256
epochs: 30
seq_length: 365
predict_last_n: 1
num_workers: 4
validate_every: 5
validate_n_random_basins: 531
metrics:
  - NSE
dynamic_inputs:
  - prcp
  - tmax
  - tmin
  - vp
  - srad
target_variables:
  - q_mm
static_attributes:
  - p_mean
  - pet_mean
  - aridity
  - p_seasonality
  - frac_snow
  - high_prec_freq
  - high_prec_dur
  - low_prec_freq
  - low_prec_dur
  - elev_mean
  - slope_mean
  - area_gages2
  - soil_depth_pelletier
  - soil_depth_statsgo
  - soil_porosity
  - soil_conductivity
  - max_water_content
  - sand_frac
  - silt_frac
  - clay_frac
  - frac_forest
  - lai_max
  - gvf_max
  - gvf_diff
  - root_depth_50
  - carbonate_rocks_frac
  - geol_permeability
"""


@app.function(
    image=image,
    gpu="L4",                       # ~cheapest CUDA GPU that trains this fast; T4 also fine
    volumes={"/data": corpora_vol, "/out": out_vol},
    timeout=60 * 60 * 3,            # 3h ceiling per member (trains in ~1h)
)
def train_member(forcing: str, seed: int) -> str:
    """One ensemble member: adapt corpus → NH data (cached on the volume) → train
    → evaluate → dump to the combine grid. Returns the dump path on /out."""
    import subprocess
    import sys
    from pathlib import Path

    rw = Path("/root/rw")
    py = sys.executable
    nh_data = Path(f"/data/nh/{forcing}")

    # 1) Build NH data ONCE per forcing (shared across seeds; cache on the volume).
    if not (nh_data / "basins.txt").exists():
        print(f"[{forcing}] building NH data via corpus_to_nh.py …", flush=True)
        subprocess.run(
            [py, str(rw / "scripts" / "corpus_to_nh.py"),
             "--forcing", forcing,
             "--corpus-dir", f"/data/corpora/camels_corpus_{forcing}_v2",
             "--out", str(nh_data)],
            cwd=str(rw), check=True,
        )
        corpora_vol.commit()
    else:
        print(f"[{forcing}] NH data already built (cached on volume)", flush=True)

    # 2) Write the config for this (forcing, seed).
    cfg = NH_CONFIG_TMPL.format(forcing=forcing, seed=seed)
    cfg_path = Path(f"/out/config_{forcing}_s{seed}.yml")
    cfg_path.write_text(cfg)

    # 3) Train.
    print(f"[{forcing} s{seed}] nh-run train …", flush=True)
    subprocess.run(["nh-run", "train", "--config-file", str(cfg_path)],
                   cwd=str(rw), check=True)

    # locate the run dir (nh-run appends a timestamp)
    runs = sorted(Path("/out/runs").glob(f"rw2_{forcing}_lstm_mm_s{seed}_*"),
                  key=lambda p: p.stat().st_mtime)
    run_dir = runs[-1]

    # 4) Evaluate on the test decade.
    print(f"[{forcing} s{seed}] nh-run evaluate …", flush=True)
    subprocess.run(["nh-run", "evaluate", "--run-dir", str(run_dir), "--period", "test"],
                   cwd=str(rw), check=True)

    # report the member's own median NSE (sanity)
    import pandas as pd
    mfile = next(run_dir.glob("test/model_epoch030/test_metrics.csv"))
    med = pd.read_csv(mfile)["NSE"].median()
    print(f"[{forcing} s{seed}] TEST median NSE = {med:.4f}", flush=True)

    # 5) Dump → combine grid (cfs, zero-padded station ids).
    dump = Path(f"/out/dumps/camels531_{forcing}_nhlstm_s{seed}.csv.gz")
    dump.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [py, str(rw / "scripts" / "nh_to_dump.py"),
         "--results", str(next(run_dir.glob("test/model_epoch030/test_results.p"))),
         "--forcing", forcing,
         "--out", str(dump)],
        cwd=str(rw), check=True,
    )
    out_vol.commit()
    print(f"[{forcing} s{seed}] DONE → {dump} (member NSE {med:.4f})", flush=True)
    return f"{forcing}_s{seed}: NSE={med:.4f} dump={dump.name}"


@app.function(image=image, volumes={"/out": out_vol})
def list_dumps() -> list[str]:
    from pathlib import Path
    return sorted(p.name for p in Path("/out/dumps").glob("*.csv.gz")) \
        if Path("/out/dumps").exists() else []
