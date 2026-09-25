#!/usr/bin/env python3
"""LEDGER 55 GATE 0 — are the MB-LSTM members still improving when the budget runs out?

The five MB-LSTM members hold 5/8 of the ensemble weight and are trained at
`--windows-per-station 300` (the default; no launcher overrides it), i.e. 18,660
optimizer steps and 2.74 presentations per training day, against 204,510 / 30.0
for the NH AR members. Checkpoint selection is `--select-by loss` on val1
(train_mblstm.py:482,1167-1174), so `best` is monotone and the SELECTED EPOCH is
a direct readout of whether the budget wall was reached.

The selected epoch is NOT in the .pt (train_mblstm.py:1172-1173 saves only
state_dict + cfg). The only source is the `  *saved*` marker on the epoch line.

Threshold pre-registered in PREREG_v2.md sLEDGER 55 BEFORE this was executed:
  PASS  >= 60% of admitted runs at best_epoch == EPOCHS, OR
        >= 80% at best_epoch >= EPOCHS-3 with mean dtail > 2*sd(dtail)
  FAIL  mass centred well below EPOCHS with a flat tail.
Prediction on record: PASS, with >= 20 of 25 runs at best_epoch == 30.
Falsifier on record: fewer than 15 of 25 at best_epoch == 30 refutes it.

Read-only. Zero GPU.

usage: l55_best_epoch.py [--members ...] [--seeds ...] [--epochs 30] [--out X.json]
"""
import argparse
import json
import re
import statistics
from pathlib import Path

LOGDIR = Path("logs")
EPOCH_RE = re.compile(
    r"^epoch (?P<ep>\d+)/(?P<tot>\d+)\s+.*?val_(?P<lname>\w+)=(?P<val>[0-9.]+).*?steps=(?P<steps>\d+)"
)


def parse(member: str, seed: int, epochs: int) -> dict:
    """Return one run's row, or {'refused': reason}. Guards mirror the prereg."""
    p = LOGDIR / f"l51_train_{member}_s{seed}.log"
    if not p.exists():
        return {"member": member, "seed": seed, "refused": "log missing"}
    txt = p.read_text(errors="replace")
    # GUARD: a warm start offsets the `best` initialisation (train_mblstm.py:1085-1087),
    # so its *saved* markers are not comparable with a fresh run's.
    if "init-ckpt val_" in txt:
        return {"member": member, "seed": seed, "refused": "warm start (init-ckpt)"}

    rows, saved = [], []
    for line in txt.splitlines():
        if not line.startswith("epoch "):
            continue
        m = EPOCH_RE.match(line)
        if not m:
            continue
        ep = int(m["ep"])
        rows.append((ep, float(m["val"]), int(m["steps"])))
        if "*saved*" in line:
            saved.append(ep)

    if len(rows) != epochs:
        return {"member": member, "seed": seed,
                "refused": f"{len(rows)} epoch lines, expected {epochs}"}
    if not saved:
        return {"member": member, "seed": seed, "refused": "no *saved* marker"}

    eps = [r[0] for r in rows]
    val = [r[1] for r in rows]
    steps = rows[-1][2]
    best_epoch = saved[-1]            # `best` is monotone under --select-by loss
    argmin_epoch = eps[val.index(min(val))]

    # tail slope over the last stretch, against its own epoch-to-epoch noise
    tail_lo = max(1, epochs - 10)
    d = [val[i - 1] - val[i] for i in range(tail_lo, epochs)]   # positive = still improving
    sd = statistics.pstdev(d) if len(d) > 1 else 0.0
    dtail = val[epochs - 4] - val[epochs - 1] if epochs >= 4 else 0.0

    return {
        "member": member, "seed": seed, "steps_last_epoch": steps,
        "best_epoch": best_epoch, "argmin_epoch": argmin_epoch,
        "val_final": val[-1], "val_min": min(val),
        "final_minus_min": val[-1] - min(val),
        "dtail": dtail, "sd_tail_delta": sd,
        "dtail_over_sd": (dtail / sd) if sd > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="fused3h1,daymeth1,nldash1,maurerh1,aorch1")
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    members = [m for m in a.members.split(",") if m]
    seeds = [int(s) for s in a.seeds.split(",") if s]

    rows = [parse(m, s, a.epochs) for m in members for s in seeds]
    ok = [r for r in rows if "refused" not in r]
    bad = [r for r in rows if "refused" in r]

    print(f"{'member':<12} {'seed':>5} {'steps':>6} {'best_ep':>8} {'argmin':>7} "
          f"{'final-min':>11} {'dtail/sd':>9}")
    for r in rows:
        if "refused" in r:
            print(f"{r['member']:<12} {r['seed']:>5}   REFUSED: {r['refused']}")
            continue
        dos = r["dtail_over_sd"]
        print(f"{r['member']:<12} {r['seed']:>5} {r['steps_last_epoch']:>6} "
              f"{r['best_epoch']:>8} {r['argmin_epoch']:>7} {r['final_minus_min']:>11.6f} "
              f"{(f'{dos:.2f}' if dos is not None else 'n/a'):>9}")

    n = len(ok)
    if n == 0:
        print("\nNO ADMITTED RUNS -- gate unrunnable. The selected epoch is not recoverable "
              "from the .pt; say so and fall back to the L52 S1 table.")
        return 2

    at_last = sum(r["best_epoch"] == a.epochs for r in ok)
    near_last = sum(r["best_epoch"] >= a.epochs - 3 for r in ok)
    mean_dtail = statistics.fmean(r["dtail"] for r in ok)
    mean_sd = statistics.fmean(r["sd_tail_delta"] for r in ok)

    clause_a = at_last / n >= 0.60
    clause_b = (near_last / n >= 0.80) and (mean_dtail > 2 * mean_sd)
    verdict = "PASS" if (clause_a or clause_b) else "FAIL"

    print(f"\nadmitted {n} / {len(rows)}   refused {len(bad)}")
    print(f"best_epoch == {a.epochs}      : {at_last}/{n} = {at_last/n:.2f}   "
          f"(clause A needs >= 0.60) -> {clause_a}")
    print(f"best_epoch >= {a.epochs-3}      : {near_last}/{n} = {near_last/n:.2f}   "
          f"(clause B needs >= 0.80 AND mean dtail > 2*sd)")
    print(f"mean dtail {mean_dtail:.6f} vs 2*sd {2*mean_sd:.6f} -> {mean_dtail > 2*mean_sd}"
          f"   clause B -> {clause_b}")
    print(f"\nGATE 0 VERDICT: {verdict}"
          f"   ({'Track 1 RUNS' if verdict == 'PASS' else 'Track 1 DIES HERE, zero GPU'})")
    print(f"PREREGISTERED PREDICTION was >= 20 of 25 at best_epoch == {a.epochs}: "
          f"measured {at_last} -> {'HIT' if at_last >= 20 else 'MISS'}"
          f"   (falsifier at < 15: {'TRIGGERED' if at_last < 15 else 'not triggered'})")

    out = {
        "gate": "LEDGER55_GATE0_best_epoch", "epochs": a.epochs,
        "members": members, "seeds": seeds,
        "rows": rows, "admitted": n, "refused": len(bad),
        "at_last_epoch": at_last, "frac_at_last": at_last / n,
        "near_last_epoch": near_last, "frac_near_last": near_last / n,
        "mean_dtail": mean_dtail, "mean_sd_tail_delta": mean_sd,
        "clause_a": clause_a, "clause_b": clause_b, "verdict": verdict,
        "prediction_hit": at_last >= 20, "falsifier_triggered": at_last < 15,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
