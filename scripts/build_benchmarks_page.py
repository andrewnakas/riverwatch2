#!/usr/bin/env python
"""Build dist/benchmarks.html — the public MODERN-1 evidence page.

Pure functions over the committed `benchmarks/m1_*` artifacts, so the whole
page is unit-testable without a build (see tests/test_benchmarks_page.py).
`scripts/build_static_site.py` calls `emit()` from shard 0 only, the same way
it writes stations.json.

EDITORIAL RULES, enforced in code rather than left to a reviewer:

 1. The OPERATIONAL number leads. 0.905829 is perfect-forcing; the measured
    cost of real 1-day-ahead GFS forcing is -0.048033, so anything a user would
    act on is ~0.86. Almost nobody publishes that discount; it is the most
    useful number here and it goes first.
 2. Every headline carries its forcing regime in the same sentence.
 3. Neither record is a first test read. `benchmarks/m1_test_reads.log` is
    append-only and holds two prior reads, two retractions and one withdrawal.
    It is rendered in full, not summarised away.
 4. Nearing 2022's 0.879 and the legacy 0.836289 appear only as context, never
    as a like-for-like comparison: different decade, cohort and product set.
 5. `_FORBIDDEN_CLAIMS` is rendered verbatim on the page. If a future edit wants
    to make one of those claims, it has to delete the sentence saying it cannot.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

# The frozen artifacts record `protocol_window` as the TRAINING window and carry
# no test window at all (only `frame: "test1"`), so the test window has to come
# from the protocol definition. Source of truth: gpu1080/queue_m1h_withq.sh:37
# (`TS=2018-10-01; TE=2025-09-30`) and its header comment at :23.
# tests/test_benchmarks_page.py asserts this pin still matches that launcher, so
# it cannot drift silently -- mislabelling a training window as a test window on
# a public page is exactly the error this guards.
TEST_WINDOWS = {
    "modern-v3-9yr": ("2018-10-01", "2025-09-30"),
    "modern-v3": ("2018-10-01", "2025-09-30"),
}

ARTIFACTS = {
    "withq": "m1_FINAL_withq_corrected.json",
    "noq": "m1_FINAL_noq_corrected.json",
    "withq_leads": "m1_withq_allleads_test14_FINAL.json",
    "noq_leads": "m1_noq_allleads_test14_FINAL.json",
    "decay": "m1_decay_day1_nldasm9.json",
    "sampling": "b0_sampling_verdict_conus404.json",
    "transfer": "river_transfer_study.json",
}

# Rendered verbatim on the page.
_FORBIDDEN_CLAIMS = [
    ("0.905829 is not operational skill.",
     "It is measured under perfect forcing — the model is handed the weather that "
     "actually occurred. Under real 1-day-ahead GFS forecast forcing the paired cost "
     "is −0.048033 (CI [−0.059808, −0.039718]). Lead with ≈0.86 for anything "
     "a user would act on."),
    ("Neither record is a first, virgin held-out read.",
     "Both are second test reads. The append-only log below records the earlier ones, "
     "including two retractions. Selection was on validation only, by a mechanical "
     "backward-elimination rule, which limits but does not remove the exposure."),
    ("This is not a like-for-like comparison with Nearing et al. 2022 (0.879) "
     "or with our own legacy 0.836289.",
     "Different decade, different basin cohort and a different forcing product set. "
     "Those numbers are context, not a scoreboard."),
    ("These are not four independent forcing products.",
     "Independent for precipitation and temperature only. gridMET carries NLDAS-2's "
     "radiation and humidity (srad slope 0.99941, r 0.9996), and nClimGrid's radiation "
     "and vapour pressure are byte-identical to gridMET's."),
    ("No “trimmed mean” figure is quoted for this roster.",
     "A trimmed mean is undefined for three members — dropping the max and min leaves "
     "one value — so it silently becomes the median, about +0.0016 higher. The record "
     "is the equal-weight mean."),
]


def load_artifacts(bench_dir: Path = BENCH) -> dict:
    """Read whatever is present. A missing artifact omits its section rather
    than failing the build — but `withq` and `decay` are required, because the
    page's two load-bearing claims come from them."""
    out = {}
    for key, name in ARTIFACTS.items():
        p = bench_dir / name
        if p.exists():
            try:
                out[key] = json.loads(p.read_text())
            except Exception:
                pass
    log = bench_dir / "m1_test_reads.log"
    out["test_reads"] = parse_test_reads(log.read_text()) if log.exists() else []
    return out


def parse_test_reads(text: str) -> list[dict]:
    """Parse the append-only test-read ledger into rows.

    Format is `<iso-timestamp> k=v k=v ...`, with RETRACTION / WITHDRAWN rows
    carrying a bare token instead of a leading key. Unparseable lines are kept
    as raw so nothing is silently dropped from the trail.
    """
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rec = {"ts": parts[0], "kind": "read", "raw": line}
        for tok in parts[1:]:
            if tok in ("RETRACTION", "WITHDRAWN"):
                rec["kind"] = tok.lower()
            elif "=" in tok:
                k, v = tok.split("=", 1)
                rec[k] = v
        rows.append(rec)
    return rows


def test_window(protocol: Optional[str]) -> Optional[tuple]:
    return TEST_WINDOWS.get(protocol or "")


def _f(x: Optional[float], d: int = 6) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _seed_count(rec: dict) -> str:
    """Seeds per member, read from the ship bar rather than assumed.

    The ship bar is {member: [seed, ...]}. Members must agree; if they do not,
    say so instead of quoting the first one.
    """
    bar = rec.get("ship_bar") or {}
    counts = {len(v) for v in bar.values() if isinstance(v, list)}
    if len(counts) == 1:
        return str(counts.pop())
    return "varying (" + ", ".join(str(c) for c in sorted(counts)) + ")" if counts else "—"


def _lead_table(withq: dict, noq: dict) -> str:
    wl = (withq or {}).get("by_lead") or {}
    nl = (noq or {}).get("by_lead") or {}
    leads = sorted({int(k) for k in list(wl) + list(nl)})
    if not leads:
        return ""
    rows = []
    for L in leads:
        w = (wl.get(str(L)) or wl.get(L) or {}).get("ensemble")
        n = (nl.get(str(L)) or nl.get(L) or {}).get("ensemble")
        rows.append(f"<tr><td>{L}</td><td class='n'>{_f(w)}</td><td class='n'>{_f(n)}</td></tr>")
    return (
        "<table><thead><tr><th>lead (days)</th>"
        "<th class='n'>with observed discharge</th>"
        "<th class='n'>without</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _reads_table(reads: list[dict]) -> str:
    if not reads:
        return "<p class='muted'>No test-read ledger found.</p>"
    rows = []
    for r in reads:
        kind = r.get("kind", "read")
        badge = {"read": "read", "retraction": "RETRACTED", "withdrawn": "WITHDRAWN"}.get(kind, kind)
        cls = {"retraction": "bad", "withdrawn": "bad"}.get(kind, "")
        head = r.get("headline")
        what = r.get("members") or r.get("target") or r.get("artifact") or ""
        why = r.get("reason") or r.get("note") or ""
        rows.append(
            f"<tr class='{cls}'><td>{_esc(r.get('ts', '')[:19])}</td>"
            f"<td><span class='badge {cls}'>{_esc(badge)}</span></td>"
            f"<td>{_esc(r.get('arm', ''))}</td>"
            f"<td class='n'>{_esc(head) if head else '—'}</td>"
            f"<td class='small'>{_esc(what)}</td>"
            f"<td class='small'>{_esc(why.replace('_', ' '))}</td></tr>")
    return ("<table><thead><tr><th>when (UTC)</th><th>event</th><th>arm</th>"
            "<th class='n'>headline</th><th>what</th><th>why</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _transfer_block(t: dict) -> str:
    """The click-anywhere transfer's measured skill.

    This is a user-facing capability, so the page states what it is (a
    neighbour's hydrograph rescaled by area) and what it is not (a model run at
    the clicked point), and shows the degradation with area mismatch rather than
    a single flattering headline.
    """
    if not t:
        return ""
    o = t.get("overall") or {}
    rows = "".join(
        f"<tr><td>within {_esc(b.get('label'))}</td><td class='n'>{b.get('n_pairs')}</td>"
        f"<td class='n'><strong>{_f(b.get('median_nse'), 4)}</strong></td>"
        f"<td class='n'>{_f(b.get('frac_above_0.5'), 3)}</td></tr>"
        for b in (t.get("bins") or []))
    headline_nse = _f((t.get("bins") or [{}])[0].get("median_nse"), 2) if t.get("bins") else "—"
    bins = t.get("bins") or []
    # The bin a reader should anchor on: the 2x band when present (it is the
    # realistic case for a clicked point), else the tightest available.
    headline_bin = bins[1] if len(bins) > 1 else (bins[0] if bins else {})
    byd = t.get("by_direction") or {}
    dirline = ""
    if len(byd) >= 2:
        dirline = ("<p class=\"muted\">By network position: " + " · ".join(
            f"{_esc(k)} median {_f(v.get('median_nse'), 4)} (n={v.get('n_pairs')})"
            for k, v in byd.items()) +
            ". ⛔ This refuted a heuristic of mine — the donor chooser used to prefer "
            "upstream gauges on physical reasoning. The contrast is <em>unpaired</em> "
            "(the two sets differ systematically in area ratio), so it does not justify "
            "preferring downstream either; the preference was simply removed and selection "
            "is on area ratio alone.</p>")
    return f"""
<h2>Clicking a point with no gauge</h2>
<p>The map lets you click anywhere near a river. There is no model run at that point: USGS NLDI
snaps the click to a mapped channel and delineates the upstream basin, then the nearest comparable
gauge's hydrograph is rescaled by the <strong>drainage-area ratio</strong>. The honest headline there
is runoff depth in <strong>mm/day</strong>, which is the quantity that transfers between catchments
of different size.</p>
<p>Measured by doing exactly that between <strong>{_esc(o.get('n_pairs', '—'))} real gauge pairs</strong>
on the same mainstem — donor's <em>observed</em> flow rescaled to the target, scored against the
target's own record over {_esc(' … '.join(t.get('window') or []))}:</p>
<table><thead><tr><th>area ratio</th><th class='n'>pairs</th>
<th class='n'>median NSE</th><th class='n'>frac &gt; 0.5</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">So a clicked point with a similarly-sized gauge on its river gets roughly
<strong>{_f(headline_bin.get('median_nse'), 2)}</strong>
NSE from the transfer alone, decaying as the areas diverge. That is well below what a gauge with its
own forecast gets, and well above nothing — which is why the panel always names the donor, the ratio
and this number. Beyond 10× area mismatch no number is shown at all.</p>
{dirline}
<p class="muted">⚠️ The mean is far below the median ({_f(o.get('mean_nse'), 3)} vs
{_f(o.get('median_nse'), 4)}): a minority of pairs transfer badly. This is a median statistic and the
tail is real. ⚠️ It is also a <em>neighbour-transfer</em> skill, not ungauged-basin model skill — it
needs a gauge on the same river.</p>"""


def build_html(a: dict) -> str:
    wq = a.get("withq") or {}
    nq = a.get("noq") or {}
    decay = a.get("decay") or {}
    wl, nl = a.get("withq_leads") or {}, a.get("noq_leads") or {}
    samp = a.get("sampling") or {}

    perfect = wq.get("headline")
    delta = decay.get("paired_delta")
    operational = (perfect + delta) if (perfect is not None and delta is not None) else None
    ci = decay.get("ci95") or [None, None]

    transfer_block = _transfer_block(a.get("transfer") or {})
    forbidden = "".join(
        f"<li><strong>{_esc(t)}</strong> {_esc(b)}</li>" for t, b in _FORBIDDEN_CLAIMS)

    cross = (wl.get("crosscheck") or {}).get("difference")
    cross_n = (nl.get("crosscheck") or {}).get("difference")

    samp_block = ""
    if samp:
        t = samp.get("slope_terciles") or {}
        tr = "".join(
            f"<tr><td>{_esc(k)}</td><td class='n'>{_f(v.get('paired_median_delta'))}</td>"
            f"<td class='n'>{_f((v.get('ci95') or [None])[0])} … {_f((v.get('ci95') or [None, None])[1])}</td>"
            f"<td class='n'>{_f(v.get('breadth'), 3)}</td></tr>"
            for k, v in t.items())
        samp_block = f"""
<h2>A side result: basin-average forcing beats a gauge-pixel sample</h2>
<p>The deployed system feeds each model a weather series sampled at the gauge's own
coordinate. Measured on a matched point/areal pair built from one source — so the only
thing that varies is the sampling — basin-average forcing carries more
precipitation–streamflow signal: paired median <strong>{_f(samp.get('paired_median_delta'))}</strong>
(CI [{_f((samp.get('ci95') or [None])[0])}, {_f((samp.get('ci95') or [None, None])[1])}]),
breadth {_f(samp.get('breadth'), 3)}, over {_esc(samp.get('frame', {}).get('n_basins', '—'))} basins.</p>
<table><thead><tr><th>terrain</th><th class='n'>paired Δ</th><th class='n'>CI95</th>
<th class='n'>breadth</th></tr></thead><tbody>{tr}</tbody></table>
<p class="muted">Two caveats that travel with it. This is a coupling screen, not a skill
measurement — the same screen fails a forcing product that ships with positive value, so
it must not be converted into an NSE claim. And the difference of medians read
{_f(samp.get('difference_of_medians'))} against the paired {_f(samp.get('paired_median_delta'))},
overstating it by about 1.9× — which is why every number on this page is a paired
per-basin difference with a bootstrap interval.</p>"""

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RiverWatch2 — streamflow benchmark evidence</title>
<style>
 :root {{ --bg:#0b1020; --panel:#131a2d; --line:#26314f; --fg:#e7ecf3; --muted:#aab7d4;
          --accent:#8ec5ff; --bad:#ff8a4c; }}
 html,body {{ margin:0; background:var(--bg); color:var(--fg);
   font-family:Inter,system-ui,-apple-system,sans-serif; line-height:1.55; }}
 .wrap {{ max-width:860px; margin:0 auto; padding:28px 16px 80px; }}
 h1 {{ font-size:26px; margin:0 0 4px; }}
 h2 {{ font-size:19px; margin:34px 0 8px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
 h3 {{ font-size:15px; margin:18px 0 6px; }}
 p, li {{ font-size:15px; }}
 .muted {{ color:var(--muted); font-size:13px; }}
 .lede {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:16px 18px; margin:18px 0; }}
 .big {{ font-size:34px; font-weight:700; letter-spacing:-0.5px; }}
 .big small {{ font-size:14px; font-weight:400; color:var(--muted); }}
 table {{ border-collapse:collapse; width:100%; margin:10px 0 4px; font-size:14px; }}
 th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
 th {{ color:var(--muted); font-weight:600; }}
 td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
 td.small {{ font-size:12px; color:var(--muted); max-width:300px; word-break:break-word; }}
 .badge {{ font-size:11px; padding:1px 6px; border-radius:999px; background:#1d2742;
           color:var(--accent); white-space:nowrap; }}
 .badge.bad {{ background:#3a2318; color:var(--bad); }}
 tr.bad td {{ background:rgba(255,138,76,0.05); }}
 ul.claims li {{ margin-bottom:10px; }}
 a {{ color:var(--accent); }}
 code {{ background:var(--panel); padding:1px 5px; border-radius:4px; font-size:13px; }}
</style></head><body><div class="wrap">

<h1>Streamflow forecasting — what was measured</h1>
<p class="muted">CAMELS basins, daily discharge. Frozen
{_esc((wq.get('frozen_utc') or '')[:10])} · git <code>{_esc((wq.get('git_commit') or '')[:8])}</code> ·
protocol <code>{_esc(wq.get('protocol', '—'))}</code></p>

<div class="lede">
 <p class="muted" style="margin:0 0 2px">Day-1 median NSE under <strong>real 1-day-ahead
 forecast weather</strong> — the number to use for anything operational</p>
 <div class="big">≈{_f(operational, 3)}
   <small>= {_f(perfect)} perfect-forcing {_f(delta)} measured forecast penalty</small></div>
 <p class="muted" style="margin:8px 0 0">The penalty is a paired per-basin measurement:
 <strong>{_f(delta)}</strong>, CI [{_f(ci[0])}, {_f(ci[1])}], over
 {_esc(decay.get('rows', '—'))} forecast–observation pairs at
 {_esc(decay.get('basins_scored', '—'))} basins, driven by archived GFS.
 Hardly anyone publishes this discount, and it is larger than every modelling
 improvement in the campaign that produced the model.</p>
</div>

<p class="muted">⚠️ The decay was measured on a sparse 14-day-stride frame, so only the
<em>difference</em> is reportable from it — the absolute value on that frame is not
comparable to the headline below. {_esc(decay.get('frame_caveat', ''))}</p>

<h2>Perfect-forcing skill, which is what benchmarks report</h2>
<p>Handed the weather that actually occurred, day-1 median NSE over
{_esc(wq.get('basins', '—'))} basins and {_esc(wq.get('rows', '—'))} rows, over the
held-out test window <strong>{_esc(' … '.join(test_window(wq.get('protocol')) or ['—']))}</strong>
(trained on {_esc(' … '.join(wq.get('protocol_window') or ['—']))}):</p>
<table><thead><tr><th>setting</th><th class='n'>day-1 median NSE</th><th>roster</th></tr></thead><tbody>
<tr><td>with observed discharge assimilated</td><td class='n'><strong>{_f(wq.get('headline'))}</strong></td>
    <td class='small'>{_esc(', '.join(wq.get('composition') or []))}</td></tr>
<tr><td>without observed discharge (rainfall–runoff only)</td><td class='n'><strong>{_f(nq.get('headline'))}</strong></td>
    <td class='small'>{_esc(', '.join(nq.get('composition') or []))}</td></tr>
</tbody></table>
<p class="muted">Equal-weight mean, zero fitted combination parameters,
{_esc(_seed_count(wq))} seeds per member. Point readout
<code>{_esc(wq.get('readout', '—'))}</code>.</p>

<h2>Every forecast lead, not just the one that gets reported</h2>
<p>Benchmarks score day 1. A forecast product has to stand up across the horizon, so here
is all of it — and the two settings decay in <strong>opposite directions</strong>:</p>
{_lead_table(wl, nl)}
<p class="muted">With discharge assimilated, skill falls by {_f(abs(wl.get('decay')) if wl.get('decay') is not None else None)} from lead 1 to
lead {_esc(len((wl.get('by_lead') or {})) or '—')} — assimilating a gauge reading is a
nowcast advantage that fades. Without it, skill <strong>rises</strong>
{_f(nl.get('decay'))}. That rise is real, measured, and <strong>unexplained</strong>: the
obvious reading ("longer leads see more antecedent weather") fails, because at lead 1
those days are already inside the model's input window. It is recorded as unexplained
rather than attributed.</p>
<p class="muted">Integrity check: the lead-1 row of the 14-lead output reproduces the
independently generated day-1 output to {cross if cross is not None else '—'} (with
discharge) and {cross_n if cross_n is not None else '—'} (without). Two separate code
paths, same answer — the cheapest proof that the headline and the table describe one model.</p>

{samp_block}

{transfer_block}

<h2>What this evidence does <em>not</em> support</h2>
<ul class="claims">{forbidden}</ul>

<h2>The retraction trail</h2>
<p>Every test-window read is appended to <code>benchmarks/m1_test_reads.log</code>, including
the ones that were withdrawn. Two earlier records were retracted after an audit found their
members were reading precipitation for the <em>next</em> day at the step being scored — a
privilege worth +0.021007 on a controlled pair. A verification of mine was also withdrawn:
it reported "no leak" on a case built to contain one, so it was void.</p>
{_reads_table(a.get('test_reads') or [])}

<h2>Method</h2>
<ul>
<li>Every difference is a <strong>paired per-basin</strong> delta with a bootstrap interval.
    A difference of medians has overstated an effect here more than ten times.</li>
<li>Selection happens on a validation window only; the test window is read once and
    every read is logged, including retracted ones.</li>
<li>Scores use every available day. A 14-day-stride evaluation frame flattered results by
    +0.036, so sparse frames are reported as differences only.</li>
<li>Nothing is admitted whose confidence interval includes zero.</li>
</ul>
<p class="muted">Artifacts behind this page:
<code>benchmarks/m1_FINAL_{{withq,noq}}_corrected.json</code>,
<code>m1_{{withq,noq}}_allleads_test14_FINAL.json</code>,
<code>m1_decay_day1_nldasm9.json</code>,
<code>b0_sampling_verdict_conus404.json</code>,
<code>m1_test_reads.log</code>, and a nine-vector leakage audit under
<code>analysis/m1_leak_*.py</code>.</p>
<p><a href="index.html">← back to the live forecast map</a></p>
</div></body></html>
"""


def emit(dist: Path) -> Path:
    """Write benchmarks.html into `dist`. Called by build_static_site (shard 0)."""
    a = load_artifacts()
    if "withq" not in a or "decay" not in a:
        raise RuntimeError(
            "benchmarks page needs m1_FINAL_withq_corrected.json and "
            "m1_decay_day1_nldasm9.json — the two load-bearing claims come from them")
    out = dist / "benchmarks.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(a))
    return out


def main() -> int:
    out = emit(ROOT / "dist")
    print(f"-> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
