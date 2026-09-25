#!/usr/bin/env python3
"""Pull ledger-41 G3 treatment dumps to the Mac, namespaced, and REFUSE to
report their paths unless every integrity check passes.

Three checks, each guarding a failure that is silent downstream:

  gzip -t         a truncated dump reads as a short file, not as an error
  531 basins      a shrunken join is the classic silent failure on this frame
  PROVENANCE      distinct md5s prove the six files are different RUNS. They do
                  NOT prove the runs descend from two different PRIORS. If both
                  queues resolved the same prior -- an env var not taking, a
                  copied config, a generator pointing at the other box's dir --
                  you get six genuinely different runs, all six digests
                  distinct, every other check green, and the claim "two
                  independent priors agree" is FALSE. Since two priors are the
                  whole reason to separate "this artefact transfers" from "this
                  recipe transfers", that claim is machine-checked here rather
                  than inferred from which box a file came off. Checked against
                  the TRAINING LOG's "Starting training from checkpoint" line,
                  not only the config: the config records what was requested,
                  the log records which weights nh-run actually loaded.

  NOT A CONTROL   a treatment dump byte-identical to a CONTROL dump is the same
                  silent failure with a different pair -- G3 would compare a
                  control against itself. The treatment-vs-treatment check
                  cannot see it, since the two live in different name spaces.

  DISTINCT md5s   the ugly one. Both boxes write their arms as
                  `camels531ls_l41ft_nhlstm_TRAIN_s<SEED>.csv.gz` -- the SEED is
                  in the name but the PRIOR is not, so p222/s111 and p111/s111
                  are different runs with IDENTICAL filenames the moment they
                  leave their box. If a copy error puts one file in two slots,
                  G3 compares a prior against ITSELF and reports
                  delta 0.000000, breadth 0.0% -- a clean FAIL that is
                  INDISTINGUISHABLE from a genuine no-effect result. (Measured:
                  a deliberate identical-arms run reads exactly that.) So the
                  check has to sit upstream of the gate, not in an eyeball
                  downstream of it.
"""
import argparse
import hashlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dumps_l41_control"
ARMS = [("nakas@nakas4050ubuntu", "gpu4050", "222"),
        ("nakas-1080", "gpu1080", "111")]
SEEDS = ["111", "222", "333"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def provenance(host, box, prior, seeds, tag):
    """{seed: checkpoint-path-nh-run-actually-loaded} for one box's arms."""
    cmd = ""
    for sd in seeds:
        cmd += (f'echo "SEED {sd}"; '
                f'grep -ah "Starting training from checkpoint" '
                f'~/riverwatch2/logs/l41_{tag}_p{prior}_s{sd}.log 2>/dev/null | tail -1; ')
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, cmd],
                       capture_output=True, text=True)
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.startswith("SEED "):
            cur = line.split()[1]
        elif cur and "checkpoint" in line:
            out[cur] = line.split("checkpoint")[-1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l41ft", help="dump basename tag on the box")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--transfer-timeout", type=int, default=900,
                    help="hard ceiling per file, seconds")
    args = ap.parse_args()
    DEST.mkdir(exist_ok=True)

    pulled, missing = [], []
    for host, box, prior in ARMS:
        for s in SEEDS:
            remote = f"~/riverwatch2/{box}/dumps/camels531ls_{args.tag}_nhlstm_TRAIN_s{s}.csv.gz"
            local = DEST / f"treatment_PRIOR_p{prior}_TRAIN_s{s}.csv.gz"
            if not args.skip_pull:
                # Verify the SOURCE is complete before spending a transfer on it.
                # `nh_to_dump.py` writes the .gz in place, so the file EXISTS
                # long before it is finished -- and a partial .gz fails `gzip -t`
                # at this end, which reports CORRUPT when the truth is
                # INCOMPLETE. Measured 2026-08-21: pulled p111/s333 at 39.2 MB
                # of 49.6 MB because a wait loop checked existence, not
                # completion. Checking gzip integrity at the SOURCE removes the
                # ambiguity entirely: if it fails there, it is not ready yet.
                probe = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=10", host,
                     f"gzip -t {remote}"], capture_output=True)
                if probe.returncode != 0:
                    missing.append(f"p{prior}/s{s} (source not finished writing)")
                    continue
                # rsync, not scp: --partial resumes an interrupted transfer and
                # --timeout makes a STALLED link fail loudly instead of hanging.
                # Measured 2026-08-21: an scp from the 1080 stalled at 14.1 MB of
                # 47 MB with 0.03 s CPU and sat there indefinitely -- and because
                # subprocess.run had no timeout, a hung transfer was
                # indistinguishable from a slow one. A partial .gz also fails
                # `gzip -t`, so it reads as CORRUPT rather than INCOMPLETE.
                try:
                    rc = subprocess.run(
                        ["rsync", "-q", "--partial", "--timeout=60",
                         f"{host}:{remote}", str(local)],
                        capture_output=True,
                        timeout=args.transfer_timeout).returncode
                except subprocess.TimeoutExpired:
                    # a hard ceiling breach is NOT "not available yet" -- say so
                    print(f"  ⛔ p{prior}/s{s}: transfer exceeded "
                          f"{args.transfer_timeout}s and was aborted. Partial "
                          f"bytes kept for --partial resume; re-run to continue.")
                    missing.append(f"p{prior}/s{s} (TRANSFER TIMED OUT)")
                    continue
                if rc != 0:
                    missing.append(f"p{prior}/s{s} (not on {box} yet)")
                    continue
            if local.exists():
                pulled.append((prior, s, local))
            else:
                missing.append(f"p{prior}/s{s}")

    if missing:
        print(f"NOT YET AVAILABLE ({len(missing)}): " + ", ".join(missing))
    if not pulled:
        sys.exit("nothing to verify")

    print(f"\nverifying {len(pulled)} dump(s):")
    bad = []
    digests = defaultdict(list)
    for prior, s, p in pulled:
        ok_gzip = subprocess.run(["gzip", "-t", str(p)], capture_output=True).returncode == 0
        nb = None
        if ok_gzip:
            out = subprocess.run(
                f"gzcat {p} | awk -F, 'NR>1 && !seen[$1]++ {{n++}} END{{print n+0}}'",
                shell=True, capture_output=True, text=True).stdout.strip()
            nb = int(out) if out.isdigit() else None
        d = md5(p) if ok_gzip else None
        if d:
            digests[d].append(p.name)
        flag = ""
        if not ok_gzip:
            flag = "  <-- GZIP FAILED"
        elif nb != 531:
            flag = f"  <-- {nb} basins, expected 531"
        if flag:
            bad.append(p.name)
        print(f"  p{prior}/s{s}  {p.name:44s} {nb} basins  {d[:12] if d else '----'}{flag}")

    ctrl = {}
    for c in sorted(DEST.glob("camels531ls_daymet_nhlstm_TRAIN_s*.csv.gz")):
        ctrl[md5(c)] = c.name
    crossed = [(n, ctrl[d]) for d, names in digests.items() if d in ctrl for n in names]
    if crossed:
        print("\n⛔ A TREATMENT DUMP IS BYTE-IDENTICAL TO A CONTROL -- refusing:")
        for t, c in crossed:
            print(f"   {t}  ==  {c}")
        print("   G3 would compare a control against itself.")
        sys.exit(1)

    dupes = {d: names for d, names in digests.items() if len(names) > 1}
    if dupes:
        print("\n⛔ IDENTICAL FILES IN DIFFERENT SLOTS -- refusing to report paths:")
        for d, names in dupes.items():
            print(f"   md5 {d[:12]} shared by: {', '.join(names)}")
        print("   G3 would compare an arm against ITSELF and report a clean FAIL "
              "(delta 0.000000, breadth 0.0%) indistinguishable from a real null.")
        sys.exit(1)
    if bad:
        sys.exit(f"\n⛔ {len(bad)} dump(s) failed gzip/basin checks: {', '.join(bad)}")

    # --- provenance: do these runs actually descend from the prior we label them with?
    print("\nprovenance (checkpoint nh-run actually loaded, from the training log):")
    prov_bad = []
    for host, box, prior in ARMS:
        got = provenance(host, box, prior, SEEDS, "ft")
        for sd in SEEDS:
            if not any(pr == prior and s_ == sd for pr, s_, _ in pulled):
                continue
            ck = got.get(sd)
            want = f"l41_prior_s{prior}_ep"
            if ck is None:
                print(f"  p{prior}/s{sd}  <no checkpoint line in log>  <-- CANNOT VERIFY")
                prov_bad.append(f"p{prior}/s{sd} unverifiable")
            elif want not in ck:
                print(f"  p{prior}/s{sd}  {ck}  <-- EXPECTED {want}*")
                prov_bad.append(f"p{prior}/s{sd} loaded {ck}")
            else:
                print(f"  p{prior}/s{sd}  {ck}")
    if prov_bad:
        print("\n⛔ PROVENANCE MISMATCH -- refusing to report paths:")
        for b in prov_bad:
            print(f"   {b}")
        print("   Six distinct runs that descend from ONE prior would pass every "
              "other check here, and the two-priors claim would be false.")
        sys.exit(1)

    # A treatment dump identical to a CONTROL dump is the same silent failure
    # with a different pair: G3 would compare control against control and report
    # the identical clean FAIL. The treatment-vs-treatment check above cannot see
    # it, because both files are legitimately in different name spaces.
    print(f"\n✅ all {len(pulled)} distinct, gzip-clean, 531 basins")
    print("PATHS:")
    for _, _, p in pulled:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
