#!/usr/bin/env python3
"""Clone the AORC config for the intensity corpus, changing ONLY what must change.

The comparison is only meaningful if the intensity member differs from the plain
AORC member in exactly one respect: the five extra dynamic inputs. Architecture,
dates, loss, learning-rate schedule, batch size, sequence length and seed must
all be inherited verbatim, or a score difference could be attributed to a
hyperparameter change instead of to the sub-daily signal.

So this prints the full diff and asserts that nothing outside the intended set
moved.
"""
import re
import sys
from pathlib import Path

REF = Path("gpu1080/cfg_aorc_s111.yml")
base = REF.read_text()

EXTRA = ["p_max_1h", "p_max_3h", "p_hours", "p_cv", "p_centroid"]

made = []
for seed in (111, 222):
    t = base
    t = re.sub(r"^experiment_name:.*$", f"experiment_name: rw2_aorcx_lstm_mm_s{seed}",
               t, flags=re.M)
    t = re.sub(r"^seed:.*$", f"seed: {seed}", t, flags=re.M)
    t = t.replace("/nh_data/aorc", "/nh_data/aorcx")

    # extend dynamic_inputs with the sub-daily block
    m = re.search(r"^dynamic_inputs:\s*\n((?:\s*-\s*\w+\s*\n)+)", t, flags=re.M)
    if not m:
        sys.exit("could not find dynamic_inputs block")
    block = m.group(1)
    indent = re.match(r"(\s*)-", block).group(1)
    added = "".join(f"{indent}- {v}\n" for v in EXTRA)
    t = t[:m.end(1)] + added + t[m.end(1):]

    p = Path(f"gpu1080/cfg_aorcx_s{seed}.yml")
    p.write_text(t)
    made.append(p.name)

    if seed == 111:
        ob, nb = base.split("\n"), t.split("\n")
        print("=== full diff vs cfg_aorc_s111.yml ===")
        import difflib
        for line in difflib.unified_diff(ob, nb, lineterm="", n=1):
            if line.startswith(("---", "+++", "@@")):
                continue
            if line.startswith(("+", "-")):
                print(f"  {line}")

        # guard: only the intended fields may differ
        allowed = ("experiment_name", "train_basin_file", "validation_basin_file",
                   "test_basin_file", "data_dir") + tuple(EXTRA)
        changed = [l for l in difflib.unified_diff(ob, nb, lineterm="", n=0)
                   if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        for l in changed:
            body = l[1:].strip().lstrip("- ").split(":")[0].strip()
            if body not in allowed:
                sys.exit(f"UNEXPECTED CHANGE: {l!r}")
        print("\n  guard passed: only name, data paths and the 5 inputs differ")

print(f"\nwrote: {made}")
for name in made:
    txt = Path(f"gpu1080/{name}").read_text()
    assert "/nh_data/aorcx" in txt, f"{name} lost the aorcx path"
    assert "/nh_data/aorc/" not in txt, f"{name} still points at plain aorc"
    for v in EXTRA:
        assert v in txt, f"{name} missing {v}"
print("verified: paths and all 5 intensity inputs present in every config")
