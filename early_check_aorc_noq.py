#!/usr/bin/env python3
"""Early check: is the AORC no-q member IN BAND with the existing LSTM streams?

The pre-registered prediction says the no-q gain hinges on own skill, not on
decorrelation. AORC's measured decorrelation (0.0398) is essentially the same as
the model-family axis (0.0394) that bought nothing; what separated the with-q
AORC success from the dHBV failure was that AORC's members were as good as their
peers while dHBV's were weaker.

Krogh-Vedelsby states it: ensemble MSE = mean member MSE − diversity. A member
that adds diversity but raises the mean by more is a net loss.

So the informative early signal is whether the AORC no-q member's training loss
sits in the same band as daymet/maurer/nldas at the SAME epoch. This runs before
the member finishes, so the prediction is testable well ahead of the ensemble
result — and if AORC is clearly weaker, the ~+0.003 projection should be revised
down again before the dumps even exist.

Compares at matched epochs only, since loss falls steeply and comparing epoch 21
against epoch 30 would be meaningless.
"""
import glob
import os
import re
import sys

import numpy as np

RUNS = "gpu1080/nh_runs"


def losses_from(path):
    if not os.path.exists(path):
        return {}
    txt = open(path, errors="ignore").read()
    return {int(m.group(1)): float(m.group(2)) for m in
            re.finditer(r"Epoch (\d+) average loss: avg_loss: ([0-9.]+)", txt)}


def find_log(pattern):
    """Prefer an explicit logs/ file, else the run dir's output.log."""
    for p in sorted(glob.glob(pattern), reverse=True):
        if os.path.isfile(p):
            return p
        cand = os.path.join(p, "output.log")
        if os.path.exists(cand):
            return cand
    return None


def main():
    new = losses_from("logs/aorcls_s111.log")
    if not new:
        sys.exit("no AORC no-q training log yet")
    latest = max(new)
    print(f"AORC no-q s111 is at epoch {latest}, loss {new[latest]:.5f}\n")

    peers = {}
    for f in ("daymet", "maurer", "nldas"):
        p = find_log(f"{RUNS}/rw2ls_{f}_lstm_mm_s111_*")
        d = losses_from(p) if p else {}
        if d:
            peers[f] = d
            print(f"  peer {f:7s} <- {os.path.basename(os.path.dirname(p))}")
    if not peers:
        sys.exit("no peer training logs found under the run dirs")

    print(f"\n=== training loss at matched epochs (same Li/Song split) ===")
    eps = [e for e in (5, 10, 15, 20, latest) if e in new
           and all(e in d for d in peers.values())]
    if not eps:
        sys.exit("no epochs in common with the peers yet")

    hdr = f"{'epoch':>6} {'AORC':>9}" + "".join(f" {k:>9}" for k in peers)
    print(hdr)
    for e in sorted(set(eps)):
        row = f"{e:>6} {new[e]:>9.5f}"
        for k, d in peers.items():
            row += f" {d[e]:>9.5f}"
        print(row)

    e = max(eps)
    peer_vals = np.array([d[e] for d in peers.values()])
    lo, hi = peer_vals.min(), peer_vals.max()
    a = new[e]
    print(f"\nat epoch {e}: peers span {lo:.5f}-{hi:.5f}, AORC {a:.5f}")
    print()
    if a <= hi:
        margin = (hi - a) / hi * 100
        print(f"=> AORC is IN BAND (at or below the worst peer by {margin:.1f}%).")
        print("   The pre-registered condition for a positive no-q gain is met on")
        print("   this evidence. Still expect a SMALL gain: decorrelation is only")
        print("   0.0398, so the member is a competent near-duplicate rather than")
        print("   a genuinely new view.")
    elif a <= hi * 1.05:
        print("=> AORC is MARGINALLY above the peer band (within 5%).")
        print("   Expect roughly nothing, possibly a very small positive.")
    else:
        print(f"=> AORC is WEAKER than every peer by {(a-hi)/hi*100:.1f}%.")
        print("   This is the dHBV pattern: diversity without comparable skill.")
        print("   Predict no gain, and revise the projection down before the")
        print("   dumps exist rather than after.")
    print("\nNOTE training loss is not test skill. This is an early directional")
    print("signal, and the binding test remains gate_eval on the dumps.")


if __name__ == "__main__":
    main()
