#!/bin/bash
# One-shot env bootstrap for the GTX 1080 (Pascal sm_61) box.
# Recent cu128 torch wheels dropped Pascal — pin the cu121 build that is
# proven on sm_60/61 (same pin the Kaggle P100 kernels used).
set -e
cd "$(dirname "$0")"

python3 -m venv .venv 2>/dev/null || true
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install neuralhydrology pandas numpy xarray netCDF4 scipy

echo "--- CUDA gate ---"
$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — do not train"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
x = torch.randn(512, 512, device="cuda:0")
y = (x @ x).sum().item()
print(f"OK: {name} sm_{cap[0]}{cap[1]}, matmul finite={y == y}")
EOF
echo "--- setup done: activate with 'source .venv/bin/activate' or use .venv/bin/python ---"
