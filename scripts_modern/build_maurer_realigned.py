#!/usr/bin/env python
"""Build a per-basin REALIGNED Maurer corpus for the frozen (Li/Song) task.

WHY. Registration audit of the frozen corpus: daymet precip peaks at k=0 vs the
flow rise in 93.3% of basins and nldas in 92.5%, but MAURER only in 69.2% — the
other ~31% peak at k=+1, i.e. maurer's precipitation is labelled ~1 day early
there (the COOP morning-observation convention). The modern experiment showed a
1-day precip offset is worth +0.2292 NSE when uncorrected, so this is the last
untested data-side lever for the 0.84 push.

⚠️⚠️ NO TARGET IS USED. The per-basin shift is chosen purely by agreement with
DAYMET precipitation — never by correlation with streamflow. Using the target to
preprocess inputs would be circular and could leak; aligning to an independently
verified forcing cannot. Daymet is the reference precisely because the audit
verified it at 93.3%.

For each basin, pick s in {0,1} maximising corr(maurer shifted LATER by s,
daymet). Shift only where s=1 helps by a clear margin.
"""
from __future__ import annotations
import argparse, gzip, json, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
SRC = ROOT / "gpu1080/corpora671/camels_corpus_maurer_v2"
REF = ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2"


def load(p, col="precipitation_sum"):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]
    dates = [r[0] for r in rows]
    v = np.empty(len(rows))
    for k, r in enumerate(rows):
        try:
            v[k] = float(r[ci[col]])
        except (ValueError, IndexError):
            v[k] = np.nan
    return hdr, ci, rows, dates, v


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 500 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu1080/corpora671/camels_corpus_maurer_realigned")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="require shift to beat no-shift by this much")
    ap.add_argument("--train-end", default="1995-09-30",
                    help="decide the shift using ONLY data up to here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = ROOT / a.out
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    b531 = set(ids["531"])

    shifted = kept = skipped = 0
    deltas = []
    for p in sorted(SRC.glob("*.csv.gz")):
        g = p.name.split(".")[0]
        rp = REF / p.name
        if not rp.exists():
            skipped += 1
            continue
        hdr, ci, rows, dm, vm = load(p)
        _, _, _, dd, vd = load(rp)

        # align on common dates, restricted to the TRAIN window for the decision
        idx = {d: i for i, d in enumerate(dd)}
        pairs = [(i, idx[d]) for i, d in enumerate(dm)
                 if d in idx and d <= a.train_end]
        if len(pairs) < 1000:
            skipped += 1
            continue
        mi = np.array([x for x, _ in pairs])
        di = np.array([y for _, y in pairs])
        m_now = vm[mi]
        d_now = vd[di]
        m_prev = np.where(mi >= 1, vm[np.maximum(mi - 1, 0)], np.nan)

        r0 = corr(m_now, d_now)          # maurer as-is vs daymet
        r1 = corr(m_prev, d_now)         # maurer shifted LATER by 1 day
        do_shift = (not math.isnan(r1)) and (not math.isnan(r0)) and (r1 > r0 + a.margin)
        if do_shift:
            shifted += 1
            deltas.append(r1 - r0)
        else:
            kept += 1

        if a.dry_run:
            continue
        # write, shifting EVERY forcing column (a day-convention offset is a
        # property of the product's day, not of one variable)
        cols = [c for c in hdr if c not in ("date", "q_cfs")]
        if do_shift:
            src = {c: [r[ci[c]] for r in rows] for c in cols}
            for t in range(len(rows) - 1, -1, -1):
                for c in cols:
                    rows[t][ci[c]] = src[c][t - 1] if t >= 1 else ""
        tmp = out / (p.name + ".tmp")
        with gzip.open(tmp, "wt") as fh:
            fh.write(",".join(hdr) + "\n")
            for r in rows:
                fh.write(",".join(r) + "\n")
        tmp.replace(out / p.name)

    print(f"maurer realignment (target-free, decided on data <= {a.train_end})")
    print(f"  shifted +1 day : {shifted}")
    print(f"  left as-is     : {kept}")
    print(f"  skipped        : {skipped}")
    if deltas:
        print(f"  median corr gain on shifted basins: +{np.median(deltas):.4f}")
    tot = shifted + kept
    if tot:
        print(f"  => {shifted/tot*100:.1f}% of basins realigned"
              f"   (audit predicted ~31%)")
    if a.dry_run:
        print("  (dry run — nothing written)")
    else:
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
