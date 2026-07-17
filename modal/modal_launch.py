"""Driver for the Modal NH-LSTM run. Two subcommands:

  upload  — push the 3 forcing corpora + camels_attrs/gauge_ids to the Volume
            (run ONCE, needs the SD card mounted). Idempotent / resumable.
  train   — fan out the 9 members (3 forcings × 3 seeds), stream member NSEs,
            then pull the dumps back into data/mblstm/gpu_dumps_s14/.

Usage (from repo root, after `modal token set …`):
  ./.venv/bin/python modal/modal_launch.py upload   # SD card mounted
  ./.venv/bin/python modal/modal_launch.py train
  ./.venv/bin/python modal/modal_launch.py pull     # re-pull dumps only

The corpora live on the SD card; edit CORPUS_SRC if the mount path differs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOLUME = "riverwatch-corpora"
OUT_VOLUME = "riverwatch-nh-out"
CORPUS_SRC = Path("/Volumes/STORAGE_SD/riverwatch2_data")   # SD card
FORCINGS = ["daymet", "nldas", "maurer"]
SEEDS = [111, 222, 333]
MODAL = str(REPO / ".venv" / "bin" / "modal")
DUMP_DST = REPO / "data" / "mblstm" / "gpu_dumps_s14"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def upload() -> None:
    # sanity: SD mounted + corpora present
    missing = [f for f in FORCINGS
               if not (CORPUS_SRC / f"camels_corpus_{f}_v2").is_dir()]
    if missing or not CORPUS_SRC.is_dir():
        sys.exit(f"corpora not found (SD mounted?). missing={missing} under {CORPUS_SRC}")

    # the two small json inputs the adapter needs (baked in the image too, but the
    # adapter reads them relative to repo root — put them on the volume as well so a
    # future standalone run has them). We upload corpora under /corpora/.
    for f in FORCINGS:
        src = CORPUS_SRC / f"camels_corpus_{f}_v2"
        print(f"uploading {f} corpus ({sum(1 for _ in src.glob('*.csv.gz'))} basins) …",
              flush=True)
        _run([MODAL, "volume", "put", "-f", VOLUME, str(src),
              f"/corpora/camels_corpus_{f}_v2/"])
    print("UPLOAD DONE — corpora on volume", VOLUME)


def fetch() -> None:
    """Download the corpora straight from Kaggle onto the Modal Volume (cloud-side;
    no SD card, no laptop disk). Uses the DEPLOYED function + .spawn() so a local
    wifi blip can't kill the server-side job. Needs the `kaggle-creds` secret."""
    import time
    import modal
    fn = modal.Function.from_name("riverwatch-nh-lstm", "fetch_corpora_from_kaggle")
    call = fn.spawn()
    print(f"spawned fetch (call id {call.object_id}); running server-side, "
          f"safe across disconnects. polling …", flush=True)
    while True:
        try:
            print("RESULT:", call.get(timeout=60), flush=True)
            break
        except modal.exception.OutputExpiredError:
            print("  (result expired — check volume with `pull`/`ls`)", flush=True)
            break
        except TimeoutError:
            print("  … still fetching", flush=True)
            time.sleep(30)


def train() -> None:
    # import here so `upload`/`pull` don't require the app image locally
    sys.path.insert(0, str(REPO / "modal"))
    import modal
    from modal_train import app, train_member

    jobs = [(f, s) for f in FORCINGS for s in SEEDS]
    print(f"launching {len(jobs)} members: {FORCINGS} × seeds {SEEDS}", flush=True)
    with app.run():
        # starmap fans out across containers; results stream back as they finish
        for res in train_member.starmap(jobs, order_outputs=False):
            print("  ✓", res, flush=True)
    print("ALL MEMBERS DONE — pulling dumps")
    pull()


def pull() -> None:
    DUMP_DST.mkdir(parents=True, exist_ok=True)
    _run([MODAL, "volume", "get", "-f", OUT_VOLUME, "/dumps", str(DUMP_DST)])
    got = sorted(DUMP_DST.glob("camels531_*_nhlstm_s*.csv.gz"))
    print(f"pulled {len(got)} member dumps → {DUMP_DST}")
    for p in got:
        print("   ", p.name)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "upload":
        upload()
    elif cmd == "fetch":
        fetch()
    elif cmd == "train":
        train()
    elif cmd == "pull":
        pull()
    else:
        sys.exit("usage: modal_launch.py [upload|train|pull]")
