#!/usr/bin/env python3
"""Generate the Kaggle GPU notebooks for the CAMELS-531 no-q push campaign.

Writes three .ipynb files into notebooks/:
  kaggle_train_dhbv.ipynb  — recipe-fidelity audit + combined-loss δHBV training
  kaggle_dump.ipynb        — GPU backtest dumps (asserts CAMELS static overlay)
  kaggle_eval_day1.ipynb   — full-531 exact-Kratzert day-1 eval

Design notes (see /Users/nakas/.claude/plans/warm-swimming-whale.md):
  * Train on Kaggle, combine locally. Dumps run on Kaggle (δHBV HBV loop is slow
    on local CPU/MPS). The day-1 eval runs on Kaggle for the full-531 δHBV headline.
  * Clone the repo from GitHub (public) so the code is always current + SHA-logged.
  * Corpora + statics arrive as Kaggle Datasets (see the data-packaging cell).
  * Save weights to /kaggle/working and "Save Version" to persist them as a Dataset
    (disk resets between sessions).

Run:  python notebooks/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/andrewnakas/riverwatch2.git"
BRANCH = "benchmark-competition-2026-07"

# The exact recipe-v2 δHBV launch (from scripts/run_gpu_dhbv.sh) PLUS the two
# fidelity fixes: --dhbv-loss combined (the decorrelation term never trained into
# shipped members) and --epochs 100 (vs the 50 used before).
TRAIN_FLAGS = (
    "--no-q-input --head dhbv --nmul 16 --dhbv-loss combined --forcing-correction "
    "--enc-vars camels1f --static-set camels --q-transform linear "
    "--hidden 256 --windows-per-station 1000 --batch 256 --val-stride 10 "
    "--train-start 1999-10-01 --train-end 2008-09-30 "
    "--val-start 1998-10-01 --val-end 1999-09-30 --epochs 100 --device cuda"
)


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP_CELL = code(
    "# --- Setup: clone repo (SHA-logged), install the few deps Kaggle lacks ---",
    "import os, subprocess, sys, textwrap",
    "os.chdir('/kaggle/working')",
    "if not os.path.exists('riverwatch2'):",
    f"    subprocess.run(['git','clone','--depth','1','--branch','{BRANCH}',",
    f"                    '{REPO_URL}'], check=True)",
    "os.chdir('/kaggle/working/riverwatch2')",
    "sha = subprocess.run(['git','rev-parse','--short','HEAD'],",
    "                     capture_output=True, text=True).stdout.strip()",
    "print('repo SHA:', sha)",
    "# Kaggle already has torch/pandas/numpy/scikit-learn; nothing else is needed.",
    "os.environ['RW2_ENABLE_MBLSTM'] = '1'",
    "os.environ['PYTHONUNBUFFERED'] = '1'",
)

# Wire the Kaggle Dataset inputs into the paths the scripts expect. Corpora live
# at /kaggle/input/<slug>/... ; statics are copied into data/ so the overlay finds
# them at the repo-relative path the scripts hardcode.
DATA_CELL = code(
    "# --- Wire Kaggle Dataset inputs to repo-relative paths ---",
    "import glob, shutil, os",
    "# Static attrs + gauge ids + station registry (load-bearing: without",
    "# camels_attrs.json the static overlay is all-NaN and NSE craters to 0.40).",
    "STATIC_DS = '/kaggle/input/rw2-camels-static'",
    "for f in ['camels_attrs.json','camels_gauge_ids.json','stations_40_enriched.json']:",
    "    src = os.path.join(STATIC_DS, f)",
    "    if os.path.exists(src):",
    "        shutil.copy(src, f'data/{f}')",
    "        print('staged', f)",
    "    else:",
    "        print('WARNING missing static input:', src)",
    "# Corpora: one Dataset per forcing. Resolve the dir that holds the 531 csv.gz.",
    "def corpus_dir(forcing):",
    "    cands = glob.glob(f'/kaggle/input/rw2-camels-corpus-{forcing}/**/camels_corpus_{forcing}_v2',",
    "                      recursive=True) or \\",
    "            glob.glob(f'/kaggle/input/rw2-camels-corpus-{forcing}/**/*.csv.gz', recursive=True)",
    "    if not cands: raise FileNotFoundError(f'no corpus for {forcing}')",
    "    d = cands[0]",
    "    return d if os.path.isdir(d) else os.path.dirname(d)",
    "for F in ['daymet','maurer','nldas']:",
    "    try: print(F, '->', corpus_dir(F), len(glob.glob(corpus_dir(F)+'/*.csv.gz')), 'basins')",
    "    except Exception as e: print(F, 'NOT MOUNTED', e)",
)


def build_train_nb() -> dict:
    return notebook([
        md("# RiverWatch2 — CAMELS-531 no-q δHBV training (Kaggle GPU)",
           "",
           "Trains **combined-loss δHBV** members (the decorrelation lever that was "
           "never actually trained into the shipped ckpts — `cfg.dhbv_loss=None`) at "
           "`nmul=16` + `--epochs 100`, the two recipe-fidelity fixes toward the "
           "Li/Shen 0.83 record.",
           "",
           "**Session policy:** ~3 seeds/forcing per 12-hr session; each seed's `.pt` "
           "is saved on val-improve, then *Save Version* persists `/kaggle/working` to "
           "a Dataset (disk resets otherwise). Set `FORCING` + `SEEDS` below."),
        SETUP_CELL,
        DATA_CELL,
        md("## Cell A — recipe-fidelity audit (NB0)",
           "Diffs each shipped ckpt's cfg vs the paper recipe so we know exactly what "
           "each retrain must change. Cheap; no GPU."),
        code(
            "# Audit: what did the shipped members actually train with?",
            "import torch, glob, json",
            "rows = []",
            "for f in sorted(glob.glob('/kaggle/input/**/camels531_*_dhbv_*.pt', recursive=True))[:6]:",
            "    try:",
            "        c = torch.load(f, map_location='cpu', weights_only=False)['cfg']",
            "        rows.append({'file': f.split('/')[-1], 'dhbv_loss': c.get('dhbv_loss'),",
            "                     'nmul': c.get('nmul'), 'q_transform': c.get('q_transform'),",
            "                     'n_static': len(c.get('static_feats', [])),",
            "                     'enc_vars': len(c.get('enc_vars', []))})",
            "    except Exception as e: rows.append({'file': f, 'err': str(e)})",
            "for r in rows: print(r)",
            "print('\\nTARGET: dhbv_loss=combined, nmul=16, epochs=100 (fidelity fixes)')",
        ),
        md("## Cell B — train (set FORCING + SEEDS)",
           "Runs the recipe-v2 δHBV launch **plus** `--dhbv-loss combined --epochs 100`. "
           "One job per seed; `--init-ckpt` resumes a warm start if a session timed out."),
        code(
            "FORCING = 'daymet'          # daymet | maurer | nldas",
            "SEEDS   = [971, 972, 973]   # 3 per 12-hr session is safe at 100 epochs",
            "",
            "import subprocess, os",
            "CORPUS = corpus_dir(FORCING)",
            "os.makedirs('/kaggle/working/ckpts', exist_ok=True)",
            f"FLAGS = {TRAIN_FLAGS!r}",
            "for s in SEEDS:",
            "    out = f'/kaggle/working/ckpts/camels531_{FORCING}_dhbv_combined100_s{s}.pt'",
            "    if os.path.exists(out):",
            "        print('skip (exists)', out); continue",
            "    cmd = (f'python scripts/train_mblstm.py --corpus-dir {CORPUS} '",
            "           f'{FLAGS} --seed {s} --out {out}')",
            "    print('>>', cmd, flush=True)",
            "    rc = subprocess.run(cmd, shell=True).returncode",
            "    print('seed', s, 'rc', rc, 'saved' if os.path.exists(out) else 'NO CKPT', flush=True)",
        ),
        md("## Cell C — persist weights",
           "List the trained `.pt`; then **File → Save Version** (with *Save output*) "
           "so `/kaggle/working/ckpts` becomes the `rw2-noq-ckpts` Dataset for the dump "
           "notebook + local download."),
        code(
            "import glob, os",
            "for f in sorted(glob.glob('/kaggle/working/ckpts/*.pt')):",
            "    print(f, round(os.path.getsize(f)/1e6, 2), 'MB')",
        ),
    ])


def build_dump_nb() -> dict:
    return notebook([
        md("# RiverWatch2 — GPU backtest dumps (Kaggle)",
           "",
           "Seed-ensembles the freshly trained δHBV members and writes `--dump-windows` "
           "csv.gz at the **same grid** as the existing v2r dumps (stride-14/ss3), so "
           "they inner-join in `combine_dumps.py`. **Asserts the CAMELS static overlay "
           "prints** — without it NSE craters to 0.40."),
        SETUP_CELL,
        DATA_CELL,
        code(
            "# Add the trained-ckpts Dataset as an input, then dump per forcing.",
            "import subprocess, glob, os",
            "FORCING = 'daymet'",
            "CORPUS = corpus_dir(FORCING)",
            "cks = sorted(glob.glob(f'/kaggle/input/rw2-noq-ckpts/**/camels531_{FORCING}_dhbv_combined100_s*.pt',",
            "                       recursive=True))",
            "assert cks, 'no trained ckpts mounted — add rw2-noq-ckpts as input'",
            "ckpt = ':'.join(cks)",
            "out = f'/kaggle/working/camels531_{FORCING}_combined100_ens_s14.csv.gz'",
            "cmd = (f'python scripts/backtest_mblstm.py --ckpt {ckpt} --corpus-dir {CORPUS} '",
            "       f'--start 1989-10-01 --end 1999-09-30 --stride 14 --stride-stations 3 '",
            "       f'--camels-subset 531 --label {FORCING}_combined100 --dump-windows {out}')",
            "log = subprocess.run(cmd, shell=True, capture_output=True, text=True)",
            "print(log.stdout[-2000:]); print(log.stderr[-500:])",
            "assert 'CAMELS static overlay' in log.stdout, 'OVERLAY MISSING — NSE will be ~0.40!'",
            "print('OK dump ->', out)",
        ),
        md("Download the `*_combined100_ens_s14.csv.gz` from the notebook output into "
           "`data/mblstm/gpu_dumps_s14/` locally, then combine with `combine_dumps.py`."),
    ])


def build_eval_nb() -> dict:
    return notebook([
        md("# RiverWatch2 — exact-Kratzert day-1 eval (Kaggle GPU, full-531)",
           "",
           "Runs `scripts/eval_day1_kratzert.py` (rolling lead-1, one prediction per "
           "calendar day) for the full-531 δHBV headline — the number directly "
           "comparable to the 0.83 record. LSTM members are cheap; δHBV needs the GPU. "
           "Validate against the ~0.808 LSTM-ensemble band first."),
        SETUP_CELL,
        DATA_CELL,
        md("**Cost note.** The evaluator uses single-basin `mblstm.forecast()` "
           "(~0.03-0.13s/call). Full-531 at `--stride-days 1` is ~hours even on GPU "
           "because the per-call pandas window rebuild doesn't GPU-accelerate. "
           "Practical settings: **LSTM** members → `--stride-days 1` full-531 (cheap, "
           "the protocol-validation run); **δHBV** headline → `--stride-days 3` "
           "(lead-1, one-third the issue dates, ~2-3 hr, fits a session). First "
           "validate against the ~0.808 LSTM band, then run δHBV."),
        code(
            "import subprocess, glob",
            "FORCING = 'daymet'",
            "CORPUS = corpus_dir(FORCING)",
            "# LSTM validation first (cheap, stride-1) → must land ~0.808 band.",
            "lstm = sorted(glob.glob(f'/kaggle/input/**/camels531_{FORCING}_v2r_s*.pt', recursive=True))[:3]",
            "if lstm:",
            "    cmd = (f'python scripts/eval_day1_kratzert.py --ckpt {\":\".join(lstm)} '",
            "           f'--corpus-dir {CORPUS} --start 1989-10-01 --end 1999-09-30 '",
            "           f'--stride-days 1 --camels-subset 531 --label {FORCING}_lstm_day1_full531')",
            "    print('>> LSTM validation:', cmd, flush=True); subprocess.run(cmd, shell=True)",
        ),
        code(
            "# δHBV headline at stride-days 3 (lead-1, ~1/3 cost).",
            "cks = sorted(glob.glob(f'/kaggle/input/rw2-noq-ckpts/**/camels531_{FORCING}_*combined100_s*.pt',",
            "                       recursive=True))[:3]",
            "if cks:",
            "    cmd = (f'python scripts/eval_day1_kratzert.py --ckpt {\":\".join(cks)} '",
            "           f'--corpus-dir {CORPUS} --start 1989-10-01 --end 1999-09-30 '",
            "           f'--stride-days 3 --camels-subset 531 --label {FORCING}_dhbv_day1_full531')",
            "    print('>> δHBV headline:', cmd, flush=True); subprocess.run(cmd, shell=True)",
        ),
    ])


def main() -> None:
    for name, nb in [("kaggle_train_dhbv", build_train_nb()),
                     ("kaggle_dump", build_dump_nb()),
                     ("kaggle_eval_day1", build_eval_nb())]:
        p = NB_DIR / f"{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1))
        print("wrote", p)


if __name__ == "__main__":
    main()
