#!/usr/bin/env python3
"""LEDGER 53 — train a NeuralHydrology run with the fast (CUDA-graph) ARLSTM loop.

Drop-in for `nh-run train --config-file X`: same Config, same start_training,
same run directory layout and state_dict. The only difference is that
ARLSTM.forward is replaced in this process by scripts/nh_arlstm_fast.py (verified
numerically identical to NH's loop, see its equivalence_test). A marker file
FAST_LOOP.json is written into the run dir so the provenance is on disk.

usage: nh_arlstm_train.py --config-file cfg.yml [--no-graph] [--gpu 0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", help="fresh run: the NH config to train")
    ap.add_argument("--continue-run-dir", help="resume an interrupted run from its last saved epoch "
                    "(NH continue_training semantics: checkpoints go to a nested "
                    "continue_training_from_epochNNN/ dir; `epochs` means MORE epochs, so the remainder "
                    "is computed here from --target-epochs)")
    ap.add_argument("--target-epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None, help="override (does not affect results)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    assert bool(args.config_file) != bool(args.continue_run_dir), "give exactly one of --config-file / --continue-run-dir"

    import torch
    from neuralhydrology.utils.config import Config
    from neuralhydrology.training.train import start_training
    import nh_arlstm_fast

    resumed_from = None
    if args.config_file:
        cfg = Config(Path(args.config_file))
    else:
        run_dir = Path(args.continue_run_dir).resolve()
        cfg = Config(run_dir / "config.yml")
        ck = sorted(run_dir.glob("model_epoch*.pt"))
        assert ck, f"no checkpoints in {run_dir}"
        resumed_from = int(ck[-1].name[-6:-3])
        assert (run_dir / f"optimizer_state_epoch{resumed_from:03d}.pt").exists(), "no optimizer state"
        remaining = args.target_epochs - resumed_from
        assert remaining > 0, f"already at epoch {resumed_from}"
        cfg.update_config({"epochs": remaining})     # NH: epochs == how many MORE
        cfg.run_dir = run_dir
        cfg.is_continue_training = True
        print(f"[nh_arlstm_train] RESUME {run_dir.name} from epoch {resumed_from}: {remaining} more epochs "
              f"(lr schedule keys {dict(cfg.learning_rate)} are absolute epochs; optimizer state restores the current lr)",
              flush=True)
    if args.num_workers is not None:
        cfg.update_config({"num_workers": args.num_workers})
    assert cfg.model.lower() == "arlstm", cfg.model
    if args.gpu is not None:
        cfg.update_config({"device": f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"})
    nh_arlstm_fast.patch_arlstm(graph=not args.no_graph)
    t0 = time.time()
    start_training(cfg)
    run_dir = Path(cfg.run_dir)                      # nested dir when resumed
    marker = {"fast_loop": True, "cuda_graph": not args.no_graph, "captures": nh_arlstm_fast._STATE["captures"],
              "graphed_shapes": [list(k) for k in nh_arlstm_fast._STATE["graphs"]], "torch": torch.__version__,
              "seconds": round(time.time() - t0), "source": "scripts/nh_arlstm_fast.py",
              "resumed_from_epoch": resumed_from, "num_workers": cfg.num_workers,
              "fused_batches": nh_arlstm_fast._STATE.get("fused_batches", 0),
              "step_batches": nh_arlstm_fast._STATE.get("step_batches", 0)}
    (run_dir / "FAST_LOOP.json").write_text(json.dumps(marker, indent=1))
    if resumed_from is not None:
        base = Path(cfg.base_run_dir)
        prev = json.loads((base / "FAST_LOOP.json").read_text()) if (base / "FAST_LOOP.json").exists() else {}
        (base / "FAST_LOOP.json").write_text(json.dumps({**marker, "segments": prev.get("segments", []) + [marker]}, indent=1))
    print(f"[nh_arlstm_train] done in {marker['seconds']} s -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
