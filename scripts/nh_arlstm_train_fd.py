#!/usr/bin/env python3
"""LEDGER 55 TRACK 2 — nh_arlstm_train.py + forcing dropout.

Identical to scripts/nh_arlstm_train.py (same Config, same start_training, same
run directory layout and state_dict, same fused ARLSTM loop) with ONE addition:
scripts/nh_forcing_dropout.py masks one forcing product's 5 channels to zero on a
fraction p of training samples.

Nothing is added to the YAML, so the LEDGER 54 config guards
(`len(dynamic_inputs) == 5 * n_forcings`, model == arlstm, the six Nearing dates,
missing_fraction, ...) all still pass verbatim. Provenance goes into FAST_LOOP.json
in the run dir, which the queue guard already reads back — an artifact, not a
caller-side flag.

The end-of-run assertions run BEFORE the process exits 0, so a run whose mask
silently did nothing can never produce a dump.

usage: nh_arlstm_train_fd.py --config-file cfg.yml --forcing-dropout-p 0.30
                             [--continue-run-dir D --target-epochs 30]
                             [--no-graph] [--gpu 0]
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
    ap.add_argument("--continue-run-dir", help="resume an interrupted run from its last saved epoch")
    ap.add_argument("--target-epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--forcing-dropout-p", type=float, required=True,
                    help="probability that a training sample loses one forcing product")
    args = ap.parse_args()
    assert bool(args.config_file) != bool(args.continue_run_dir), \
        "give exactly one of --config-file / --continue-run-dir"
    assert args.forcing_dropout_p > 0, \
        "p must be > 0; the control is scripts/nh_arlstm_train.py, not p=0 here"

    import torch
    from neuralhydrology.utils.config import Config
    from neuralhydrology.training.train import start_training
    import nh_arlstm_fast
    import nh_forcing_dropout

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
        cfg.update_config({"epochs": remaining})
        cfg.run_dir = run_dir
        cfg.is_continue_training = True
        print(f"[nh_arlstm_train_fd] RESUME {run_dir.name} from epoch {resumed_from}: "
              f"{remaining} more epochs", flush=True)
    if args.num_workers is not None:
        cfg.update_config({"num_workers": args.num_workers})
    assert cfg.model.lower() == "arlstm", cfg.model
    if args.gpu is not None:
        cfg.update_config({"device": f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"})

    # order matters: the fast loop is installed first, then wrapped.
    nh_arlstm_fast.patch_arlstm(graph=not args.no_graph)
    nh_forcing_dropout.patch(cfg, args.forcing_dropout_p, torch.device(cfg.device))

    t0 = time.time()
    start_training(cfg)
    nh_forcing_dropout.log_epoch()

    # HARD GATE: assert the patch actually applied, before anything downstream
    # can treat this run as scorable. [[assert-the-patch-applied]]
    fd = nh_forcing_dropout.assert_applied()

    run_dir = Path(cfg.run_dir)
    marker = {"fast_loop": True, "cuda_graph": not args.no_graph,
              "captures": nh_arlstm_fast._STATE["captures"],
              "graphed_shapes": [list(k) for k in nh_arlstm_fast._STATE["graphs"]],
              "torch": torch.__version__, "seconds": round(time.time() - t0),
              "source": "scripts/nh_arlstm_fast.py + scripts/nh_forcing_dropout.py",
              "resumed_from_epoch": resumed_from, "num_workers": cfg.num_workers,
              "fused_batches": nh_arlstm_fast._STATE.get("fused_batches", 0),
              "step_batches": nh_arlstm_fast._STATE.get("step_batches", 0),
              "forcing_dropout": fd}
    (run_dir / "FAST_LOOP.json").write_text(json.dumps(marker, indent=1))
    if resumed_from is not None:
        base = Path(cfg.base_run_dir)
        prev = json.loads((base / "FAST_LOOP.json").read_text()) if (base / "FAST_LOOP.json").exists() else {}
        (base / "FAST_LOOP.json").write_text(
            json.dumps({**marker, "segments": prev.get("segments", []) + [marker]}, indent=1))

    # The fused path is the arm's own precondition, not just the queue's.
    assert marker["fused_batches"] > marker["step_batches"], (
        f"fused {marker['fused_batches']} <= step {marker['step_batches']}: the mask "
        "must not have flipped the AR dispatch — investigate before scoring")
    print(f"[nh_arlstm_train_fd] done in {marker['seconds']} s -> {run_dir} "
          f"(fused {marker['fused_batches']} > step {marker['step_batches']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
