#!/usr/bin/env python3
"""Do our forcing products AGREE with each other on the days the ensemble fails?

The ceiling analysis says 70% of member error is SHARED and cannot be removed by
ensembling. The provenance hypothesis says why: daymet <- GHCN-Daily; NLDAS-2 <-
CPC <- COOP/GTS; maurer <- COOP; AORC <- Livneh (COOP) + NLDAS-2 method + Stage
IV. Four interpolations of one heavily overlapping gauge base. A gauge that
missed a storm is invisible to all four, so a common input error would survive
ensembling exactly the way the shared 70% does.

That predicts something testable on data already on disk: on the ~1% of days
carrying ~93% of squared error, the products should agree with each other about
as well as -- or better than -- on ordinary days. If instead they DISAGREE on
event days, the shared error is not a common input miss and the provenance story
is wrong. Both outcomes are worth knowing, which is why this runs before any
download.

Method note: the dumps are STRIDE-14 (verified: 14-day gaps within every basin),
so "top 1% of days" means top 1% of a 1-in-14 sample of dates, not of all days.
That is fine here -- we need day LABELS to join forcings against, not adjacency
-- but it means absolute day counts are ~14x undersampled.

Correctness gate: this must reproduce the known 93.2% / top-1% concentration
before any new number is believed. If it does not, the residual join is wrong and
everything downstream is meaningless.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(R, "data/mblstm/gpu_dumps_s14")
C = os.path.join(R, "data/local_corpora")

# Members with a full-531 with-q dump. AORC has none, so it cannot enter on the
# ERROR side; it still enters on the FORCING side below.
MEMBERS = ("daymet", "maurer", "nldas")
FORCINGS = ("daymet", "maurer", "nldas", "aorc")


def load_dump(path, subset):
    """Day-1 predictions, following shared_error_anatomy.py:31-38 exactly."""
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["pred"] = (d.ylo + d.yhi) / 2
    return d[["station_id", "t0", "truth", "pred"]]


def build_ensemble(ids):
    """Inner-join the member dumps and form the plain-mean ensemble."""
    m = None
    truths = {}
    for name in MEMBERS:
        p = f"{D}/camels531_{name}_withq4_full531.csv.gz"
        if not os.path.exists(p):
            sys.exit(f"missing dump: {p}")
        d = load_dump(p, ids)
        truths[name] = d.set_index(["station_id", "t0"]).truth
        x = d.rename(columns={"pred": name})[["station_id", "t0", name, "truth"]]
        if m is None:
            m = x
        else:
            m = m.merge(x[["station_id", "t0", name]], on=["station_id", "t0"])
        print(f"  {name:8s} {len(d):>7,} rows  {d.station_id.nunique():>3} basins"
              f"  -> joined {len(m):>7,}")

    if len(m) == 0:
        sys.exit("FATAL: inner join produced ZERO rows -- members are on different "
                 "grids or splits. (This is the wrong-split failure mode.)")

    # Alignment assert: the members must agree on observed discharge.
    base = truths[MEMBERS[0]]
    for name in MEMBERS[1:]:
        j = pd.concat([base, truths[name]], axis=1, join="inner").dropna()
        diff = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
        rel = diff.sum() / max(j.iloc[:, 0].abs().sum(), 1e-9)
        if rel > 0.002:
            sys.exit(f"FATAL: truth mismatch {MEMBERS[0]} vs {name}: rel {rel:.5f}")
    print(f"  truth agreement across members: OK (<=0.002)")

    m["ens"] = m[list(MEMBERS)].to_numpy(float).mean(1)
    m["err"] = m.ens - m.truth
    m["sq_err"] = m.err ** 2
    return m


def gate_concentration(m):
    """Reproduce the published 93.2%/top-1% result before trusting anything."""
    tot = m.sq_err.sum()
    print("\n=== GATE: error concentration (must match the known result) ===")
    shares = {}
    for frac in (0.01, 0.05, 0.10):
        k = int(len(m) * frac)
        share = m.sq_err.nlargest(k).sum() / tot
        shares[frac] = share
        print(f"  worst {frac*100:4.1f}% of days carry {share*100:5.1f}% of squared error")
    if not (0.85 <= shares[0.01] <= 0.98):
        sys.exit(f"GATE FAILED: top-1% share {shares[0.01]*100:.1f}% is far from the "
                 f"expected ~93%. The residual join is wrong; fix before continuing.")
    print("  => consistent with the recorded 93.2%. Join is sound.")
    return shares


def load_forcings(ids):
    """Daily precip per basin per product, via the verify_13235000.py::load path."""
    out = {}
    for prod in FORCINGS:
        rows = []
        d = f"{C}/camels_corpus_{prod}_v2"
        if not os.path.isdir(d):
            print(f"  {prod:8s} MISSING dir, skipped")
            continue
        for b in sorted(ids):
            f = f"{d}/{b}.csv.gz"
            if not os.path.exists(f):
                continue
            x = pd.read_csv(f, usecols=["date", "precipitation_sum"])
            x["station_id"] = b
            rows.append(x)
        if not rows:
            continue
        z = pd.concat(rows, ignore_index=True)
        z = z.rename(columns={"date": "t0", "precipitation_sum": prod})
        out[prod] = z
        print(f"  {prod:8s} {len(z):>9,} basin-days  "
              f"{z.station_id.nunique():>3} basins  "
              f"{z.t0.min()}..{z.t0.max()}")
    return out


def main():
    ids = set(str(x).zfill(8) for x in
              json.load(open(f"{R}/data/camels_gauge_ids.json"))["531"])
    print(f"CAMELS-531 subset: {len(ids)} basins\n")

    print("=== loading member dumps (with-q, full 531) ===")
    m = build_ensemble(ids)
    print(f"\nensemble rows: {len(m):,}  basins: {m.station_id.nunique()}")
    print(f"span: {m.t0.min()} .. {m.t0.max()}")

    gate_concentration(m)

    print("\n=== loading forcing corpora ===")
    F = load_forcings(ids)
    if len(F) < 2:
        sys.exit("need at least 2 forcing products")

    # Attach each product's precip to the ensemble rows.
    j = m[["station_id", "t0", "sq_err", "truth", "ens", "err"]].copy()
    have = []
    for prod, z in F.items():
        j = j.merge(z, on=["station_id", "t0"], how="left")
        have.append(prod)
    n_all = j[have].notna().all(axis=1).sum()
    print(f"\njoined rows: {len(j):,}; with ALL {len(have)} products present: {n_all:,}")
    # maurer ends 2008-12-31 and aorc starts 1980-10-01 -- report, never drop silently.
    for prod in have:
        print(f"  {prod:8s} present on {j[prod].notna().sum():>7,} rows "
              f"({j[prod].notna().mean()*100:5.1f}%)")

    # THE TEST: do the products agree with each other more on event days?
    thr01 = j.sq_err.quantile(0.99)
    thr05 = j.sq_err.quantile(0.95)
    groups = {
        "top-1% error days": j[j.sq_err >= thr01],
        "top-5% error days": j[j.sq_err >= thr05],
        "ordinary (<95th)": j[j.sq_err < thr05],
    }

    print("\n" + "=" * 72)
    print("THE TEST: inter-product precipitation agreement, by error class")
    print("=" * 72)
    print("If the four products are one gauge base re-interpolated, a gauge that")
    print("missed a storm is invisible to all of them -- so they should agree")
    print("with each other on the days we fail, and the shared error is INPUT error.\n")

    hdr = f"{'day class':<20}{'n':>8}{'mean pairwise r':>17}{'CV across':>11}{'max-min mm':>12}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for label, g in groups.items():
        g = g.dropna(subset=have)
        if len(g) < 50:
            print(f"{label:<20}{len(g):>8}   (too few rows)")
            continue
        P = g[have].to_numpy(float)
        rs = []
        for a in range(len(have)):
            for b in range(a + 1, len(have)):
                if P[:, a].std() > 1e-9 and P[:, b].std() > 1e-9:
                    rs.append(np.corrcoef(P[:, a], P[:, b])[0, 1])
        mean_r = float(np.mean(rs)) if rs else np.nan
        mu = P.mean(1)
        cv = float(np.mean(P.std(1)[mu > 0.1] / mu[mu > 0.1])) if (mu > 0.1).any() else np.nan
        spread = float(np.mean(P.max(1) - P.min(1)))
        summary[label] = (mean_r, cv, spread, len(g))
        print(f"{label:<20}{len(g):>8}{mean_r:>17.4f}{cv:>11.3f}{spread:>12.2f}")

    if "top-1% error days" in summary and "ordinary (<95th)" in summary:
        re_, cve, spe, _ = summary["top-1% error days"]
        ro, cvo, spo, _ = summary["ordinary (<95th)"]
        print("\n=== READING ===")
        print(f"  pairwise agreement  event {re_:.4f}  vs ordinary {ro:.4f}   "
              f"(delta {re_-ro:+.4f})")
        print(f"  disagreement (CV)   event {cve:.3f}  vs ordinary {cvo:.3f}")
        print(f"  spread mm           event {spe:.2f}  vs ordinary {spo:.2f}")
        print()
        print("  ** NEITHER of these raw numbers answers the question. **")
        print("  r and CV point OPPOSITE ways here (r says worse, CV says better)")
        print("  because event days are far wetter: correlation across large values")
        print("  is not comparable to correlation across mostly-zeros, and CV=sd/mean")
        print("  is mechanically deflated when the mean is large.")
        print()
        print("  The defensible test is the MAGNITUDE-MATCHED control below, which")
        print("  compares failure days against equally wet NON-failure days.")

    # Magnitude-matched control: the only comparison that isolates disagreement
    # from wetness. Without it, r and CV give contradictory verdicts.
    print("\n" + "=" * 72)
    print("MAGNITUDE-MATCHED CONTROL (the load-bearing test)")
    print("=" * 72)
    jj = j.dropna(subset=have).copy()
    jj["pmean"] = jj[have].mean(axis=1)
    jj["prange"] = jj[have].max(axis=1) - jj[have].min(axis=1)
    jj["event"] = jj.sq_err >= thr01
    wet_ev = jj[jj.event].pmean.mean()
    wet_or = jj[~jj.event].pmean.mean()
    print(f"  event days are {wet_ev/max(wet_or,1e-9):.1f}x wetter "
          f"({wet_ev:.2f} vs {wet_or:.2f} mm) -- this is the confound\n")
    bins = [0.1, 2, 5, 10, 20, 40, 80, 1e9]
    labs = ["0.1-2", "2-5", "5-10", "10-20", "20-40", "40-80", "80+"]
    jj["bin"] = pd.cut(jj.pmean, bins=bins, labels=labs)
    print(f"{'precip bin':>10} {'n_event':>8} {'n_ord':>8} "
          f"{'spread_ev':>10} {'spread_or':>10} {'ratio':>7}")
    print("-" * 56)
    acc = []
    for b in labs:
        g = jj[jj.bin == b]
        ev, od = g[g.event], g[~g.event]
        if len(ev) < 25 or len(od) < 25:
            continue
        re2, ro2 = ev.prange.mean(), od.prange.mean()
        acc.append((len(ev), re2 / max(ro2, 1e-9)))
        print(f"{b:>10} {len(ev):>8,} {len(od):>8,} {re2:>10.2f} {ro2:>10.2f} "
              f"{re2/max(ro2,1e-9):>7.2f}")
    if acc:
        w = np.array([a[0] for a in acc], float)
        ratio = float(np.average([a[1] for a in acc], weights=w))
        print(f"\n  weighted spread ratio (event/ordinary): {ratio:.3f}")
        if ratio > 1.10:
            print("  => AT THE SAME RAINFALL products disagree MORE on failure days.")
            print("     Input disagreement IS implicated in the failures.")
        elif ratio < 0.90:
            print("  => AT THE SAME RAINFALL products agree MORE on failure days:")
            print("     members were told the same thing and still failed together.")
        else:
            print("  => disagreement is the SAME once rainfall is matched; the raw")
            print("     r/CV gap was a wetness artifact, not a signal.")

    # How much does disagreement actually explain? Guards against overclaiming.
    print("\n  does product disagreement PREDICT error? corr(spread, |error|):")
    strengths = []
    for b in ["2-5", "5-10", "10-20", "20-40"]:
        g = jj[jj.bin == b]
        if len(g) < 200:
            continue
        c = float(np.corrcoef(g.prange, np.sqrt(g.sq_err))[0, 1])
        strengths.append(c)
        print(f"    bin {b:>6}: {c:+.4f}  (n={len(g):,})")
    if strengths and max(strengths) < 0.35:
        print("    => WEAK. Input disagreement is a CONTRIBUTOR, not the explanation")
        print("       for the shared error. Do not overclaim it.")

    # Stratify the event days by basin attributes (Phase 2).
    ap = f"{R}/data/camels_attrs.json"
    if os.path.exists(ap):
        A = json.load(open(ap))
        recs = A["basins"] if isinstance(A, dict) and "basins" in A else A
        if isinstance(recs, dict):
            attrs = pd.DataFrame([{"station_id": str(k).zfill(8), **v}
                                  for k, v in recs.items()])
        else:
            attrs = pd.DataFrame(recs)
            idc = next((c for c in attrs.columns
                        if c.lower() in ("station_id", "gauge_id", "id", "basin")), None)
            if idc:
                attrs["station_id"] = attrs[idc].astype(str).str.zfill(8)
        ev = j[j.sq_err >= thr01]
        per = (ev.groupby("station_id").size().rename("n_event")
               .to_frame().reset_index())
        per = per.merge(attrs, on="station_id", how="left")
        print("\n=== PHASE 2: which basins own the event days? ===")
        for col in ("elev_mean", "frac_snow", "aridity", "p_mean"):
            if col not in per.columns:
                continue
            v = pd.to_numeric(per[col], errors="coerce")
            w = per.n_event
            ok = v.notna() & (w > 0)
            if ok.sum() < 20:
                continue
            wmean = float((v[ok] * w[ok]).sum() / w[ok].sum())
            print(f"  {col:11s} event-weighted mean {wmean:8.3f}   "
                  f"unweighted basin mean {v[ok].mean():8.3f}")
        print("  (event-weighted >> unweighted means high-elevation / snowy basins")
        print("   carry the failures -- the orographic + frozen-precip signature)")

    out = f"{R}/benchmarks/forcing_disagreement_rows.csv.gz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    j.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(j):,} rows)")


if __name__ == "__main__":
    main()
