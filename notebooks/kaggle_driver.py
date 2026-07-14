#!/usr/bin/env python3
"""Drive Kaggle GPU kernels for the CAMELS no-q campaign entirely via the API.

Generates a self-contained script-kernel, writes its kernel-metadata.json (with
enable_gpu + dataset_sources), pushes it, polls status to completion, and pulls the
output. No browser needed.

The kernel body (see build_kernel_src) does proper GPU init:
  * asserts torch.cuda.is_available() → fails LOUD instead of silently running on
    CPU (100-epoch δHBV on CPU = days);
  * prints nvidia-smi + torch CUDA device name / capability;
  * clones the repo, wires the corpus + static Datasets to repo-relative paths,
    runs the requested stage with --device cuda.

Subcommands:
  smoke     — nvidia-smi + a 5-epoch 1-seed δHBV train (proves GPU init end-to-end)
  train     — combined-loss δHBV, one forcing, N seeds
  dump      — GPU backtest dump for a forcing's trained members
  eval-day1 — exact-Kratzert day-1 (LSTM stride-1 validate + δHBV stride-3)
  status    — poll a pushed kernel to completion
  output    — pull a finished kernel's output files

Usage:
  python notebooks/kaggle_driver.py smoke
  python notebooks/kaggle_driver.py train --forcing daymet --seeds 971,972,973
  python notebooks/kaggle_driver.py status --slug andrewnakas/rw2-smoke
  python notebooks/kaggle_driver.py output --slug andrewnakas/rw2-train-daymet --dest data/mblstm/gpu_ckpts
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

USER = "andrewnakas"
REPO_URL = "https://github.com/andrewnakas/riverwatch2.git"
BRANCH = "benchmark-competition-2026-07"
NB_DIR = Path(__file__).resolve().parent
CORPUS_DS = {f: f"{USER}/rw2-camels-corpus-{f}" for f in ("daymet", "maurer", "nldas")}
STATIC_DS = f"{USER}/rw2-camels-static"
CKPTS_DS = f"{USER}/rw2-noq-ckpts"

# The recipe-v2 δHBV launch + the fidelity fix (--dhbv-loss combined = the
# decorrelation loss never trained into the shipped members). MEASURED on the
# Kaggle P100 (torch 2.4.1, sm_60): ~2s/step at batch 256 → windows-per-station
# 1000 would be ~70min/epoch (untenable). Cut to 150 (531×150/256≈311 steps ≈
# ~10min/epoch) so 50 epochs ≈ 8-9hr fits a 12hr session incl. the ~4min torch+
# gzip setup. epochs default is 50 (the shipped-member count; 100 was aspirational
# and won't fit the P100 budget).
TRAIN_FLAGS = (
    "--no-q-input --head dhbv --nmul 16 --dhbv-loss combined --forcing-correction "
    "--enc-vars camels1f --static-set camels --q-transform linear "
    "--hidden 256 --windows-per-station 150 --batch 256 --val-stride 10 "
    "--train-start 1999-10-01 --train-end 2008-09-30 "
    "--val-start 1998-10-01 --val-end 1999-09-30 --device cuda"
)

# -------------------------------------------------------------- kernel body ---

# NOTE: the KGAT_ granular token can push+run kernels but is DENIED kernels.get,
# so we can't read run logs via API. Every kernel therefore TEES all stdout/stderr
# to /kaggle/working/RESULT.txt (pullable via kernels output) and writes a final
# STATUS line, so failures are diagnosable without log access.
GPU_INIT = r'''
print("="*60, "\nGPU INIT", flush=True)
subprocess.run(["nvidia-smi","--query-gpu=name,compute_cap","--format=csv"], check=False)
# Kaggle's default torch (2.10/cu128) dropped sm_60, but Kaggle often assigns a
# P100 (sm_60). Probe the GPU arch WITHOUT committing the main import, and if the
# installed torch can't drive it, pin 2.4.1+cu121 (covers sm_60..sm_90) FIRST so
# the whole run uses a compatible torch.
import torch as _t
_cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else (0,0)
_sm = f"sm_{_cap[0]}{_cap[1]}"
print("device", _t.cuda.get_device_name(0) if _t.cuda.is_available() else "NONE",
      "| torch", _t.__version__, "| cap", _cap, "| arch", _t.cuda.get_arch_list(), flush=True)
assert _t.cuda.is_available(), "NO CUDA GPU — enable the GPU accelerator!"
if _sm not in _t.cuda.get_arch_list():
    print(f"{_sm} unsupported by torch {_t.__version__} — pinning torch 2.4.1+cu121", flush=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","torch==2.4.1",
        "torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121"], check=True)
    if os.environ.get("RW2_PINNED")!="1":
        os.environ["RW2_PINNED"]="1"
        os.execv(sys.executable, [sys.executable, os.path.abspath(sys.argv[0])])
import torch
assert _sm in torch.cuda.get_arch_list(), f"{_sm} still unsupported after pin"
print("EFFECTIVE torch", torch.__version__, "arch", torch.cuda.get_arch_list(),
      "| GPU usable:", torch.cuda.is_available(), flush=True)
# quick real GPU op to PROVE it works before the long run
_x = torch.randn(1000,1000,device="cuda"); print("GPU matmul ok:",
      float((_x@_x).sum().abs()>0), flush=True); del _x; torch.cuda.empty_cache()
print("="*60, flush=True)

# --- clone repo (pinned branch), log SHA ---
os.chdir("/kaggle/working")
if not os.path.exists("riverwatch2"):
    subprocess.run(["git","clone","--depth","1","--branch","%(branch)s","%(repo)s"], check=True)
os.chdir("/kaggle/working/riverwatch2")
sha = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
print("repo SHA:", sha, flush=True)
os.environ["RW2_ENABLE_MBLSTM"]="1"; os.environ["PYTHONUNBUFFERED"]="1"

# --- wire Kaggle Dataset inputs to repo-relative paths ---
for f in ["camels_attrs.json","camels_gauge_ids.json","stations_40_enriched.json"]:
    for src in glob.glob(f"/kaggle/input/**/{f}", recursive=True):
        shutil.copy(src, f"data/{f}"); print("staged", f); break
    else:
        print("WARNING missing static input:", f)

def corpus_dir(forcing):
    """Stage a writable dir of *.csv.gz (the trainer/backtest glob) from the Kaggle
    input, whatever form it arrived in. Kaggle auto-EXTRACTS uploaded archives, so
    the corpus lands as flat *.csv (uncompressed). We gzip those into the staged dir
    (fast, one-time/session). Also handles a .tar or pre-existing *.csv.gz."""
    import gzip, tarfile
    # Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>/ on this
    # account (not the classic /kaggle/input/<slug>/), so search ALL of /kaggle/input
    # for files whose path contains the slug.
    slug = f"rw2-camels-corpus-{forcing}"
    def find(ext):
        return [f for f in glob.glob(f"/kaggle/input/**/*.{ext}", recursive=True)
                if slug in f and not os.path.basename(f).startswith("._")]
    staged = f"/kaggle/working/corpus_{forcing}"
    os.makedirs(staged, exist_ok=True)
    if glob.glob(staged + "/*.csv.gz"):
        pass                                   # already staged this session
    elif find("csv.gz"):
        for f in find("csv.gz"):
            try: os.symlink(f, os.path.join(staged, os.path.basename(f)))
            except FileExistsError: pass
    elif find("tar"):
        with tarfile.open(find("tar")[0]) as t: t.extractall(staged)
        for f in glob.glob(staged + "/**/*.csv.gz", recursive=True):
            dst = os.path.join(staged, os.path.basename(f))
            if f != dst and not os.path.exists(dst): os.symlink(f, dst)
    else:                                      # flat uncompressed *.csv → gzip them
        csvs = find("csv")
        print(f"gzipping {len(csvs)} flat .csv → .csv.gz ...", flush=True)
        for f in csvs:
            dst = os.path.join(staged, os.path.basename(f) + ".gz")
            if not os.path.exists(dst):
                with open(f, "rb") as i, gzip.open(dst, "wb") as o:
                    shutil.copyfileobj(i, o)
    n = len(glob.glob(staged + "/*.csv.gz"))
    if not n:
        raise FileNotFoundError(f"no corpus files for {slug} under /kaggle/input")
    print(f"corpus {forcing}: {n} files -> {staged}", flush=True)
    return staged

def runlog(cmd):
    """Run a shell command, STREAM its combined stdout/stderr into RESULT.txt
    (the bare subprocess otherwise writes to the fd-level stdout the Tee misses),
    return the exit code."""
    print(">>", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    p.wait()
    print("rc", p.returncode, flush=True)
    return p.returncode
'''

def build_kernel_src(stage: str, forcing: str = "daymet", seeds="971,972,973",
                     epochs=100) -> str:
    head = GPU_INIT % {"branch": BRANCH, "repo": REPO_URL}
    if stage == "smoke":
        body = f'''
CORPUS = corpus_dir("daymet")
print("corpus:", CORPUS, len(glob.glob(CORPUS+"/*.csv.gz")), "basins", flush=True)
# 5-epoch 1-seed δHBV to prove GPU training wires up end-to-end
cmd = ("python scripts/train_mblstm.py --corpus-dir "+CORPUS+
       " --no-q-input --head dhbv --nmul 16 --dhbv-loss combined --forcing-correction "
       "--enc-vars camels1f --static-set camels --q-transform linear --hidden 256 "
       "--batch 256 --val-stride 10 --train-start 1999-10-01 --train-end 2008-09-30 "
       "--val-start 1998-10-01 --val-end 1999-09-30 --device cuda "
       "--epochs 5 --windows-per-station 100 --seed 999 "
       "--out /kaggle/working/smoke_daymet_s999.pt")
rc = runlog(cmd)
print("SMOKE rc", rc, "ckpt", os.path.exists("/kaggle/working/smoke_daymet_s999.pt"), flush=True)
'''
    elif stage == "train":
        body = f'''
FORCING = "{forcing}"; SEEDS = {[int(s) for s in seeds.split(",")]}
CORPUS = corpus_dir(FORCING)
os.makedirs("/kaggle/working/ckpts", exist_ok=True)
for s in SEEDS:
    out = f"/kaggle/working/ckpts/camels531_{{FORCING}}_dhbv_combined{epochs}_s{{s}}.pt"
    if os.path.exists(out): print("skip", out, flush=True); continue
    cmd = ("python scripts/train_mblstm.py --corpus-dir "+CORPUS+" {TRAIN_FLAGS} "
           f"--epochs {epochs} --seed {{s}} --out "+out)
    rc = runlog(cmd)
    print("seed", s, "rc", rc, "saved" if os.path.exists(out) else "NO CKPT", flush=True)
for f in sorted(glob.glob("/kaggle/working/ckpts/*.pt")):
    print("CKPT", f, round(os.path.getsize(f)/1e6,2),"MB", flush=True)
'''
    elif stage == "dump":
        body = f'''
FORCING = "{forcing}"; CORPUS = corpus_dir(FORCING)
cks = sorted(glob.glob(f"/kaggle/input/**/camels531_{{FORCING}}_dhbv_combined*_s*.pt", recursive=True))
assert cks, "no trained ckpts mounted — add rw2-noq-ckpts as a dataset source"
out = f"/kaggle/working/camels531_{{FORCING}}_combined{epochs}_ens_s14.csv.gz"
cmd = ("python scripts/backtest_mblstm.py --ckpt "+":".join(cks)+" --corpus-dir "+CORPUS+
       " --start 1989-10-01 --end 1999-09-30 --stride 14 --stride-stations 3 "
       f"--camels-subset 531 --label {{FORCING}}_combined --dump-windows "+out)
print(">>", cmd, flush=True)
log = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(log.stdout[-3000:]); print("STDERR", log.stderr[-500:], flush=True)
assert "CAMELS static overlay" in log.stdout, "OVERLAY MISSING — NSE would be ~0.40!"
print("OK dump", out, os.path.exists(out), flush=True)
'''
    elif stage == "eval-day1":
        body = f'''
FORCING = "{forcing}"; CORPUS = corpus_dir(FORCING)
# 1) LSTM validation (stride-1, cheap) — must land ~0.75-0.81 band
lstm = sorted(glob.glob(f"/kaggle/input/**/camels531_{{FORCING}}_v2r_s*.pt", recursive=True))[:3]
if lstm:
    cmd = ("python scripts/eval_day1_kratzert.py --ckpt "+":".join(lstm)+" --corpus-dir "+CORPUS+
           " --start 1989-10-01 --end 1999-09-30 --stride-days 1 --camels-subset 531 "
           f"--label {{FORCING}}_lstm_day1_full531")
    print(">> LSTM validate:", cmd, flush=True); subprocess.run(cmd, shell=True)
# 2) δHBV headline (stride-3, lead-1)
cks = sorted(glob.glob(f"/kaggle/input/**/camels531_{{FORCING}}_dhbv_combined*_s*.pt", recursive=True))[:3]
if cks:
    cmd = ("python scripts/eval_day1_kratzert.py --ckpt "+":".join(cks)+" --corpus-dir "+CORPUS+
           " --start 1989-10-01 --end 1999-09-30 --stride-days 3 --camels-subset 531 "
           f"--label {{FORCING}}_dhbv_day1_full531")
    print(">> δHBV headline:", cmd, flush=True); subprocess.run(cmd, shell=True)
# surface result jsons into the kernel output root
for j in glob.glob("benchmarks/day1kratzert_*full531.json"):
    shutil.copy(j, "/kaggle/working/"+os.path.basename(j)); print("OUT", j, flush=True)
'''
    else:
        raise SystemExit(f"unknown stage {stage}")
    body = textwrap.dedent(body).replace("{TRAIN_FLAGS}", TRAIN_FLAGS)
    # Flat prelude sets up a Tee to /kaggle/working/RESULT.txt (pullable via
    # kernels output — the KGAT token is DENIED kernels.get for run logs) and opens
    # a try:. GPU_INIT + the stage body are indented 4 under it; the footer records
    # a STATUS line so a pull of RESULT.txt shows exactly what happened.
    prelude = (
        "import os, subprocess, sys, glob, shutil, textwrap, traceback\n"
        "class _Tee:\n"
        "    def __init__(s, p): s.f=open(p,'w',buffering=1); s.o=sys.__stdout__\n"
        "    def write(s,x): s.f.write(x); s.o.write(x)\n"
        "    def flush(s): s.f.flush(); s.o.flush()\n"
        "sys.stdout = sys.stderr = _Tee('/kaggle/working/RESULT.txt')\n"
        "try:\n"
    )
    inner = head + body
    indented = "\n".join(("    " + ln if ln.strip() else ln)
                         for ln in inner.splitlines())
    footer = (
        "\n    print('STATUS: OK', flush=True)\n"
        "except Exception as _e:\n"
        "    traceback.print_exc()\n"
        "    print('STATUS: FAILED', type(_e).__name__, _e, flush=True)\n"
        "finally:\n"
        "    sys.stdout.flush()\n"
    )
    return prelude + indented + footer


# ------------------------------------------------------------- API plumbing ---

import textwrap  # noqa: E402


def push(slug: str, src: str, datasets: list[str], enable_gpu: bool = True) -> str:
    work = NB_DIR / ".kernels" / slug.split("/")[-1]
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernel.py").write_text(src)
    meta = {
        "id": slug,
        "title": slug.split("/")[-1],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",       # needed for the git clone
        # T4 is sm_75 (compatible with Kaggle's torch 2.10/cu128); the default P100
        # is sm_60 and torch REFUSES it ("not compatible... supports sm_70+"). Force
        # T4 or training silently can't use the GPU.
        "machine_shape": "nvidiaTeslaT4" if enable_gpu else "",
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"pushing {slug} (gpu={enable_gpu}, T4, datasets={datasets}) ...")
    cmd = ["kaggle", "kernels", "push", "-p", str(work)]
    # Kaggle caps concurrent GPU sessions at 2 and lags ~1-2 min releasing them;
    # retry on "Maximum batch GPU session count" until a slot frees.
    for attempt in range(15):
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if "session count" in out.lower():
            print(f"GPU sessions full, waiting 20s (attempt {attempt+1}) ...")
            time.sleep(20)
            continue
        print(out)
        break
    return slug


def poll(slug: str, every: int = 30, timeout: int = 43200) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        r = subprocess.run(["kaggle", "kernels", "status", slug],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if out != last:
            print(f"[{int(time.time()-t0)}s] {out}")
            last = out
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        time.sleep(every)
    return "timeout"


def pull_output(slug: str, dest: str):
    Path(dest).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", slug, "-p", dest],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("smoke", "train", "dump", "eval-day1"):
        s = sub.add_parser(c)
        s.add_argument("--forcing", default="daymet")
        s.add_argument("--seeds", default="971")   # 1 seed/session (~8-9hr on P100)
        s.add_argument("--epochs", type=int, default=50)   # P100 budget; shipped count
        s.add_argument("--slug", default="")
        s.add_argument("--wait", action="store_true", help="poll to completion")
    for c in ("status", "output"):
        s = sub.add_parser(c)
        s.add_argument("--slug", required=True)
        s.add_argument("--dest", default="data/mblstm/gpu_ckpts")
    a = ap.parse_args()

    if a.cmd == "status":
        print(poll(a.slug)); return
    if a.cmd == "output":
        pull_output(a.slug, a.dest); return

    slug = a.slug or f"{USER}/rw2-{a.cmd}" + (f"-{a.forcing}" if a.cmd != "smoke" else "")
    ds = [STATIC_DS]
    if a.cmd in ("train", "smoke", "dump", "eval-day1"):
        ds.append(CORPUS_DS[a.forcing if a.cmd != "smoke" else "daymet"])
    if a.cmd in ("dump", "eval-day1"):
        ds.append(CKPTS_DS)
    src = build_kernel_src(a.cmd, a.forcing, a.seeds, a.epochs)
    push(slug, src, ds)
    if a.wait or a.cmd == "smoke":
        st = poll(slug)
        print("FINAL:", st)
        if st == "complete":
            print("logs:")
            subprocess.run(["kaggle", "kernels", "output", slug, "-p",
                            str(NB_DIR / ".kernels" / slug.split("/")[-1] / "out")])


if __name__ == "__main__":
    main()
