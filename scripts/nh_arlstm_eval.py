#!/usr/bin/env python3
"""LEDGER 53 — evaluate a NeuralHydrology AR-LSTM run with the AR holdout OFF.

NH applies `random_holdout_from_dynamic_features` when the dataset is LOADED,
for every period (basedataset.py, no is_train guard). A plain
`nh-run evaluate` would therefore score the model with 50 % of the lagged
discharge withheld — the paper's benchmark number (0.879) is the
"no missing data during inference" case. This script loads the run's config,
removes the holdout, and evaluates one period at one epoch. Results land where
`nh-run evaluate` would put them:
  <run_dir>/<period>/model_epoch<NNN>/<period>_results.p

usage: nh_arlstm_eval.py --run-dir <dir> --period validation|test [--epoch 30]
"""
import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--period", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0, help="<0 runs on CPU")
    ap.add_argument("--fused", action="store_true",
                    help="use scripts/nh_arlstm_fast.py (verified max|dy|=0 vs NH's own loop). "
                         "Needed to make a CPU evaluation tractable: NH's stock 365-step loop is "
                         "~63x slower than the fused single cuDNN/oneDNN call.")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    args = ap.parse_args()

    import torch
    from neuralhydrology.utils.config import Config
    from neuralhydrology.evaluation.evaluate import start_evaluation
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.fused:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import nh_arlstm_fast
        # graph=False: CUDA graphs are GPU-only and the dispatch skips them off-GPU anyway.
        nh_arlstm_fast.patch_arlstm(graph=False, fused=True)

    run_dir = Path(args.run_dir).resolve()
    cfg = Config(run_dir / "config.yml")
    before = dict(cfg.random_holdout_from_dynamic_features)
    cfg.update_config({"random_holdout_from_dynamic_features": {}})
    assert not cfg.random_holdout_from_dynamic_features, "holdout still set"
    # tester.py: for period == "validation" NH evaluates only the FIRST
    # `validate_n_random_basins` basins (the in-training subsample, saved into
    # the run's config.yml). Lift it so the scored validation covers every basin.
    cfg.update_config({"validate_n_random_basins": 10**6})
    if args.gpu >= 0:
        cfg.update_config({"device": f"cuda:{args.gpu}"})
    else:
        cfg.update_config({"device": "cpu"})
    print(f"[nh_arlstm_eval] run={run_dir.name} period={args.period} epoch={args.epoch} "
          f"holdout {before} -> {{}} (inference with COMPLETE lagged discharge)", flush=True)
    t0 = time.time()
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=args.epoch, period=args.period)
    print(f"[nh_arlstm_eval] evaluation took {time.time()-t0:.0f}s", flush=True)
    if args.fused:
        import nh_arlstm_fast as _f
        print(f"[nh_arlstm_eval] fused batches={_f._STATE.get('fused_batches', 0)} "
              f"step batches={_f._STATE.get('step_batches', 0)}", flush=True)
    out = run_dir / args.period / f"model_epoch{args.epoch:03d}" / f"{args.period}_results.p"
    if not out.exists():
        print(f"ERROR: expected {out}", file=sys.stderr)
        return 1
    print(f"[nh_arlstm_eval] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
