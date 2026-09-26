import os, re, sys
cfg = sys.argv[1]
bad = 0
for ln in open(cfg):
    m = re.match(r"^(\w*(?:basin_file|data_dir)):\s*(\S+)", ln.strip())
    if not m:
        continue
    k, v = m.groups()
    ok = os.path.exists(v)
    status = "OK" if ok else "MISSING"
    print("  %-24s %-44s %s" % (k, v, status))
    if not ok:
        bad += 1
print("ALL PATHS RESOLVE" if bad == 0 else "%d MISSING PATHS - do not launch" % bad)
sys.exit(0 if bad == 0 else 1)
