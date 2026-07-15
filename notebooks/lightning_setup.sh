#!/bin/bash
# One-shot bootstrap for a Lightning AI Studio (or any fresh CUDA GPU box) to run
# the CAMELS no-q δHBV campaign. Unlike Kaggle (push-a-kernel API), Lightning is a
# persistent GPU IDE — you run scripts directly, so this just sets up the repo +
# data + deps once, then training is plain `python scripts/train_mblstm.py ...`.
#
# Lightning's T4/A10 are sm_75+ (compatible with modern torch) — NO torch pin
# needed (the P100 headache is gone).
#
# Usage (in the Lightning Studio terminal):
#   bash lightning_setup.sh
# then:
#   bash notebooks/run_campaign.sh          # trains all 3 forcings + dumps
set -euo pipefail
BRANCH=benchmark-competition-2026-07
REPO=https://github.com/andrewnakas/riverwatch2.git
ROOT=${ROOT:-$HOME/riverwatch2}

echo "=== 1. GPU check ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || { echo "NO GPU"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); \
  print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0), \
        'arch', torch.cuda.get_arch_list())"
# sanity: is this GPU's arch supported by the installed torch? (should be, sm_75+)
python - <<'PY'
import torch
cap=torch.cuda.get_device_capability(0); sm=f"sm_{cap[0]}{cap[1]}"
if sm not in torch.cuda.get_arch_list():
    print(f"WARNING: {sm} not in {torch.cuda.get_arch_list()} — pin a compatible torch:")
    print("  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121")
else:
    print(f"OK: torch supports {sm}")
PY

echo "=== 2. clone repo ==="
[ -d "$ROOT" ] || git clone --depth 1 --branch "$BRANCH" "$REPO" "$ROOT"
cd "$ROOT"
git pull --ff-only origin "$BRANCH" 2>/dev/null || true
echo "repo SHA: $(git rev-parse --short HEAD)"

echo "=== 3. deps (Lightning has torch/pandas/numpy; add scikit-learn/scipy if missing) ==="
python -c "import pandas, numpy, torch" 2>/dev/null || pip install -q pandas numpy torch
python -c "import scipy, sklearn" 2>/dev/null || pip install -q scipy scikit-learn

echo "=== 4. data ==="
mkdir -p data/mblstm/gpu_ckpts data/mblstm/gpu_dumps_s14 data/gpu_corpora
echo "Corpora: expected at data/gpu_corpora/camels_corpus_{daymet,maurer,nldas}_v2/*.csv.gz"
echo "Options to get them onto this box:"
echo "  A) Kaggle datasets (fast): pip install kaggle; configure ~/.kaggle/kaggle.json;"
echo "     kaggle datasets download andrewnakas/rw2-camels-corpus-daymet -p data/gpu_corpora --unzip"
echo "     (repeat for maurer, nldas) — then rename the flat dir to camels_corpus_<F>_v2/"
echo "  B) rsync/scp from your local /Volumes/STORAGE_SD/riverwatch2_data/"
echo "  C) the build script: python scripts/build_camels_corpus.py (needs raw CAMELS archive)"
echo "  camels_attrs.json + camels_gauge_ids.json + stations_40_enriched.json are IN the repo (data/)."

echo "=== setup done. Next: bash notebooks/run_campaign.sh ==="
