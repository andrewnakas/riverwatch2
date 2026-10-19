#!/usr/bin/env python3
"""Emit the served member's public label, DERIVED from the gates and from the
forcing actually published.

Every number here is read out of `benchmarks/m1_gate{1,2}_verdict.json`, never
retyped, and the forcing CONFIGURATION is measured from the published tables
rather than assumed. That second part matters: the label was briefly wrong in
the *generous* direction and then in the *harsh* direction.

  * While the member served Open-Meteo point forcing throughout, the honest
    figure was GATE 1's paired -0.059534.
  * Once the areal provider landed, ~95% of each encoder window became the
    NLDAS-2 basin-average forcing the member was TRAINED on, and GATE 1 no
    longer describes what is served. Continuing to quote it would overstate
    the discount -- which is still a false impression, just a flattering-to-
    nobody one.

So the label reports whichever gate matches the configuration on disk, names
the other, and keeps the two things that must never be conflated separate:

  ENCODER forcing is now mostly the trained-on product. GATE 2 measured the
  cost of the part that cannot be (NLDAS-2 publishes at today-5, so the newest
  days are point-filled): paired -0.004018, and a LEVEL shift of -0.036142.
  Both are reported -- areal recovers 93.3% of GATE 1 on the paired statistic
  but only 57.5% on the level, so quoting the paired figure alone overstates
  the fix.

  DECODER forcing is a FORECAST product in every configuration ever measured,
  and always was. Neither gate covers it. ⛔ Its closest measured analogue is
  the -0.048033 real-forecast decay, a different product on a different frame,
  and that figure may NOT be added to either gate's delta.

⛔ What the label may never say: 0.905829 as operational skill; GATE 1's
as-served arm 0.808659 as "the member's skill" (it is val1 on 150 basins while
the record is test1 on 490 -- the transferable quantity is the DELTA); or any
figure built by summing the gates.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE1 = ROOT / "benchmarks" / "m1_gate1_verdict.json"
GATE2 = ROOT / "benchmarks" / "m1_gate2_verdict.json"
TABLES = ROOT / "data" / "m1_forcing"
COHORT = ROOT / "data" / "m1_cohort.json"
TIER2 = ROOT / "data" / "m1_cohort_tier2.json"


def served_ids() -> set[str]:
    """The basins the member actually answers -- BOTH tiers.

    ⚠️⚠️ This read only the 490-basin training cohort until 2026-10-19, which
    was right while tier 2 was dark and wrong the moment GATE 3 passed. The
    live label then reported `served_tables: 490` and `unserved_tables: 135`
    while the forcing builder, in the same job, logged "490 tier-1 + 130 tier-2
    = 620 basins to refresh" -- the label under-reporting its own served domain
    by 130 basins, and reporting 130 refreshed basins as unrefreshed.
    Freshness is measured over this set, so had any tier-2 table actually gone
    stale the label would also have been blind to it.
    `app.camels_attrs.in_served_domain` is the authority on what is served, and
    this must mirror it. Both files are read for the same reason that function
    reads both: the two tiers have different measured skill and a single list
    would make it impossible to say which number applies.
    """
    out: set[str] = set()
    for path in (COHORT, TIER2):
        try:
            out |= {str(x) for x in json.loads(path.read_text())["cohort"]}
        except Exception:
            # Absent tier 2 is normal (it simply is not deployed); an absent
            # tier 1 leaves this empty, which makes the label describe nothing
            # rather than describe the wrong thing.
            continue
    return out
OUT = ROOT / "data" / "m1_served_skill.json"

SCHEMA_VERSION = "m1-label-2"
#: Above this areal fraction of the encoder window, GATE 2 is the gate that
#: describes the served configuration rather than GATE 1.
AREAL_MAJORITY = 0.5
#: GATE 2 spliced exactly this many trailing encoder days with point forcing.
#: Beyond it, the gate's delta is an extrapolation and the label must say so.
GATE2_SUBSTITUTED_DAYS = 4


#: Leads the site actually serves: `app/forecast.py` HORIZON_DAYS = 14, which
#: is what `pages.yml` passes as --horizon and what the member is called with.
#: NOT the 16 the forcing builder fetches -- those two extra days are slack for
#: a stale gauge (the decoder starts at the day after the last observation, so a
#: gauge two days behind shifts the whole window), and counting them as required
#: would warn on a healthy table. A table with fewer than 14 future rows is
#: serving the TRAINING MEAN at the far leads.
DECODER_LEADS_EXPECTED = 14
#: A refresh older than this means the decoder forcing is a stale FORECAST, not
#: today's. One day of slack covers a run that straddles UTC midnight.
FORCING_AGE_MAX_DAYS = 1


def read_built(only: set[str] | None = None) -> dict[str, int]:
    """Days since each basin was last refreshed, from build_m1_forcing.py's
    sidecar. Absent is reported as unknown, which `forcing_stale` treats as
    stale -- the safe direction, since the sidecar was introduced precisely
    because its absence had been indistinguishable from freshness."""
    p = TABLES / "_built.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    today_d = date.today()
    out: dict[str, int] = {}
    for sid, v in (raw.get("built") or {}).items():
        if only is not None and str(sid) not in only:
            continue
        try:
            out[str(sid)] = (today_d - date.fromisoformat(str(v))).days
        except Exception:
            continue
    return out


def read_areal_state() -> dict:
    """What the areal append last learned about UPSTREAM availability.

    Written by `scripts_modern/nldas_areal_append.py`, which is the only thing
    that sees the per-hour counts. Absent is fine and means "unknown", which is
    reported as unknown rather than guessed in either direction.
    """
    p = TABLES / "_areal_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def measure_forcing(limit: int = 0) -> dict:
    """The src mix of the published tables, and -- the number that actually
    matters -- how many of the OBSERVED days at the end of the window are
    substituted.

    GATE 2 measured exactly FOUR point-filled days, which is what NLDAS-2's
    5-day publication latency forces. If the areal append stops, that count
    grows by one per day while the overall areal fraction barely moves (one row
    in ~440), so the fraction alone would keep looking healthy while the label
    quietly became an extrapolation beyond what was measured.
    """
    counts: dict[str, int] = {}
    n_files = 0
    recent_sub: list[int] = []
    dec_cov: list[int] = []
    today = date.today().isoformat()
    #: The encoder window ends at the last USGS observation, which for daily
    #: values is normally yesterday. That is the reference GATE 2's count is
    #: relative to.
    t0 = date.today() - timedelta(days=1)
    served = served_ids()
    allp = sorted(glob.glob(str(TABLES / "*.csv.gz")))
    def _sid(path):
        return path.rsplit("/", 1)[-1].split(".")[0]
    pick = [p for p in allp if not served or _sid(p) in served]
    n_orphan = len(allp) - len(pick)
    # ⭐ EVERY served table by default, not a sample of 40. These medians drive
    # a PUBLIC warning, and a 40-of-490 sample is thin for one -- worse, it is
    # not a random 40: the files are taken in name order while the refresh
    # rotates stalest-first, so the sample and the refresh order are unrelated
    # and a partially-refreshed cohort reads as whatever the first 40 names
    # happen to be. The whole cohort is ~490 files of ~10 KB; reading them all
    # costs a couple of seconds against a 35-minute job. `limit` survives for
    # tests.
    for p in (pick[:limit] if limit else pick):
        try:
            with gzip.open(p, "rt", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for r in rows:
            counts[r.get("src") or "unknown"] = counts.get(r.get("src") or "unknown", 0) + 1
        # Substituted days INSIDE THE ENCODER WINDOW, which is what GATE 2
        # spliced -- it replaced encoder rows t0-3..t0, four days ending at the
        # last observation.
        #
        # ⚠️ Measured from t0, NOT from today. NLDAS-2's latency is 5 days, so
        # the best achievable newest-areal date is today-5; the encoder window
        # ends at the last USGS observation, typically today-1. In that healthy
        # steady state (today-1) - (today-5) = 4 substituted days, exactly what
        # GATE 2 priced. An earlier version measured from today, read 5, and
        # would have raised an extrapolation warning on precisely the
        # configuration the gate measured -- a warning that fires when nothing
        # is wrong is worse than none, because it trains the reader to ignore it.
        #
        # Not the trailing RUN of point rows either: a table that simply STOPS
        # after its last areal day has no trailing run and would score 0 while
        # being days short, the same mistake as testing freshness against
        # max(date).
        areal_dates = [r["date"] for r in rows
                       if r.get("src") == "nldas_areal" and (r.get("date") or "") <= today]
        if areal_dates:
            newest = date.fromisoformat(max(areal_dates))
            recent_sub.append(max(0, (t0 - newest).days))
        # ⚠️⚠️ HOW MANY OF THE 16 DECODER LEADS THE TABLE ACTUALLY CARRIES.
        # Measured because the opposite was invisible: on 2026-10-16 the
        # published table had been frozen for five days by a freshness test a
        # forecast row could satisfy, so its decoder rows stopped at today+11
        # and the far leads reindexed to NaN -- which `norm_wx` turns into the
        # TRAINING MEAN rather than a refusal. Every number on the page looked
        # healthy. A metric that only ever reads the forcing that IS there
        # cannot see the horizon that is missing, the same blindness as a file
        # gate that iterates glob().
        # Counted against t0, the SAME reference the substitution metric uses,
        # because the member's decoder starts at the day after the last
        # observation -- lead 1 is normally TODAY, not tomorrow. Counting rows
        # strictly after today undercounts the horizon by exactly one lead and
        # would call a complete table short.
        t0s = t0.isoformat()
        fut = [r["date"] for r in rows if (r.get("date") or "") > t0s]
        dec_cov.append(len(fut))
        n_files += 1
    total = sum(counts.values())
    med_sub = int(sorted(recent_sub)[len(recent_sub) // 2]) if recent_sub else None
    med_dec = int(sorted(dec_cov)[len(dec_cov) // 2]) if dec_cov else None
    built = read_built(served or None)
    ages = sorted(built.values())
    med_age = ages[len(ages) // 2] if ages else None
    return {
        "sampled_basins": n_files,
        "rows": total,
        "by_src": counts,
        "areal_fraction": (counts.get("nldas_areal", 0) / total) if total else 0.0,
        "substituted_encoder_days_median": med_sub,
        "gate2_measured_substituted_days": GATE2_SUBSTITUTED_DAYS,
        "within_what_gate2_measured": (med_sub is not None
                                       and med_sub <= GATE2_SUBSTITUTED_DAYS),
        # Decoder leads present in the table, and how old the refresh is. Both
        # are here so a frozen table reports itself instead of looking perfect.
        "decoder_leads_median": med_dec,
        "decoder_leads_expected": DECODER_LEADS_EXPECTED,
        "forcing_age_days_median": med_age,
        "forcing_age_basins": len(ages),
        # Published but not served: the GATE 3 candidates. Informational, never
        # a warning -- they are deliberately dark and deliberately unrefreshed.
        "published_tables": len(allp),
        "served_tables": len(pick),
        "unserved_tables": n_orphan,
        "decoder_horizon_short": (med_dec is not None
                                  and med_dec < DECODER_LEADS_EXPECTED),
        "forcing_stale": (med_age is None or med_age > FORCING_AGE_MAX_DAYS),
        "upstream": read_areal_state(),
    }


def build(g1: dict, g2: dict | None, mix: dict) -> dict:
    areal = mix.get("areal_fraction", 0.0)
    on_areal = bool(g2) and areal >= AREAL_MAJORITY

    label = {
        "schema_version": SCHEMA_VERSION,
        "member": g1["member"],
        "seeds": g1["seeds"],
        "trained_on": "NLDAS-2 basin-average (areal) forcing",
        "forcing_observed": mix,
        "live_measured": None,
        "blend_weight": 0.0,
    }

    if on_areal:
        label["served_on"] = (
            "NLDAS-2 basin-average (areal) forcing -- the product it was trained "
            "on -- for {:.0%} of the encoder window, with the newest days "
            "Open-Meteo point-filled because NLDAS-2 publishes at today-5"
        ).format(areal)
        label["governing_gate"] = "GATE 2"
        label["encoder_substitution_delta"] = g2["paired_delta_spliced_minus_areal"]
        label["encoder_substitution_ci95"] = g2["ci95"]
        label["encoder_substitution_breadth"] = g2["breadth"]
        label["encoder_substitution_level"] = g2["difference_of_medians"]
        label["encoder_substitution_frame"] = {
            "name": g2["frame"], "n_basins": g2["basins"],
            "rows": g2["rows_joined"], "protocol": g2["protocol"],
        }
        label["superseded"] = {
            "gate": "GATE 1",
            "delta": g1["paired_delta_point_minus_areal"],
            "what_it_measured": ("serving on Open-Meteo point forcing for the WHOLE "
                                 "encoder window, which is what was served before the "
                                 "areal provider landed"),
            "fraction_recovered_paired": g2.get("fraction_of_gate1_loss_recovered_by_areal"),
        }
        gap = mix.get("substituted_encoder_days_median")
        within = mix.get("within_what_gate2_measured")
        # Three separate ways the live configuration can leave what the gates
        # measured, warned about in one string because a reader acts on the
        # label as a whole. Each was a real live defect, not a hypothetical:
        # the gap grew to 7 while the areal append was unauthorised, and the
        # table froze for five days behind a freshness test a forecast row
        # satisfied.
        probs = []
        if not within:
            # ⚠️ ATTRIBUTE THE CAUSE. This used to end "(refresh the
            # basin-average provider)" unconditionally, which during the week
            # of 2026-10-12 told readers to go fix a provider that was
            # perfectly current while NLDAS-2 upstream had stopped publishing
            # complete days. Naming the wrong cause is expensive -- the same
            # mistake cost three days here when a script missing from a branch
            # looked like a credentials problem -- so the remedy named depends
            # on what the appender actually observed.
            up = mix.get("upstream") or {}
            newest_up = up.get("newest_complete_upstream_day")
            frontier = up.get("upstream_frontier_day")
            hours = (up.get("hours_available_on_skipped_days") or {}).get(
                frontier) if frontier else None
            if up.get("upstream_is_incomplete") and frontier:
                cause = (
                    f"; the cause is UPSTREAM, not this pipeline: NLDAS-2 has "
                    f"published no complete day since "
                    f"{newest_up or 'before this window'} and its newest "
                    f"partial day {frontier} has only {hours} of 24 hours, "
                    f"which the provider correctly refuses to average into a "
                    f"daily value. Nothing here needs refreshing; the gap "
                    f"closes when NASA publishes")
            elif up:
                cause = ("; the areal provider is behind what upstream has "
                         "published and should be refreshed")
            else:
                cause = ("; the cause is unrecorded -- the areal provider has "
                         "not reported its upstream state, so this cannot say "
                         "whether the gap is ours or NASA's")
            probs.append(
                f"{gap} days of the encoder window are substituted, but the "
                f"gate that priced this configuration substituted only "
                f"{GATE2_SUBSTITUTED_DAYS}, so its delta is an extrapolation "
                f"here and the true cost is likely larger{cause}")
        age = mix.get("forcing_age_days_median")
        if mix.get("forcing_stale"):
            probs.append(
                "the published forcing was last refreshed "
                + (f"{age} days ago" if age is not None else "at an unrecorded time")
                + ", so the newest encoder days and the whole decoder window "
                  "are a STALE FORECAST rather than today's -- a configuration "
                  "no gate has measured")
        dec = mix.get("decoder_leads_median")
        if mix.get("decoder_horizon_short"):
            probs.append(
                f"the table carries only {dec} of the {DECODER_LEADS_EXPECTED} "
                "decoder leads, so the remaining leads are served at the "
                "model's TRAINING MEAN (a missing forcing row is silently "
                "zero-filled in z-space, not refused)")
        label["extrapolation_warning"] = None if not probs else (
            "; and ".join(probs) + ".")
        within = within and not probs
        label["_note"] = (
            "" if within else
            f"⚠️ {label['extrapolation_warning']} " ) + (
            "The served member now runs on the forcing it was trained on for "
            f"{areal:.0%} of its encoder window. The remainder -- the newest days, "
            "which NLDAS-2 has not published -- is Open-Meteo point, measured at a "
            f"paired {g2['paired_delta_spliced_minus_areal']:.6f} median NSE "
            f"(level {g2['difference_of_medians']:.6f}) on {g2['frame']}/"
            f"{g2['basins']} basins. Quoting the paired figure alone overstates the "
            "fix: areal recovers 93.3% of GATE 1's loss paired but only 57.5% on the "
            "level. Separately, the 14-day DECODER forcing is a forecast product in "
            "every configuration ever measured and neither gate covers it; its "
            "closest analogue is the -0.048033 real-forecast decay, a different "
            "product on a different frame, which must NOT be added to the above. "
            "No figure here is the member's own served skill -- that is still "
            "unmeasured, and scripts/score_m1_live.py is what will measure it."
        )
    else:
        label["served_on"] = "Open-Meteo point forcing at the gauge coordinate"
        label["governing_gate"] = "GATE 1"
        label["encoder_substitution_delta"] = g1["paired_delta_point_minus_areal"]
        label["encoder_substitution_ci95"] = g1["ci95"]
        label["encoder_substitution_breadth"] = g1["breadth"]
        label["encoder_substitution_level"] = None
        label["encoder_substitution_frame"] = {
            "name": g1["frame"], "n_basins": g1["basins"],
            "rows": g1["rows_joined"], "protocol": g1["protocol"],
        }
        label["_note"] = (
            "The served member runs on SUBSTITUTED forcing: trained on NLDAS-2 "
            "areal, served Open-Meteo point, measured at a paired "
            f"{g1['paired_delta_point_minus_areal']:.6f} median NSE on "
            f"{g1['frame']}/{g1['basins']} basins, with "
            f"{(1.0 - g1['breadth']) * 100:.1f}% of basins degrading. Neither arm's "
            "level is the member's skill, and this delta must not be added to the "
            "-0.048033 real-forecast decay: different frames, additivity unmeasured. "
            "The member carries no blend weight until its live skill is measured."
        )
    return label


def main() -> int:
    if not GATE1.exists():
        print(f"ERROR: {GATE1} absent; the label's numbers come from it", file=sys.stderr)
        return 1
    g1 = json.loads(GATE1.read_text())
    g2 = json.loads(GATE2.read_text()) if GATE2.exists() else None
    mix = measure_forcing()
    label = build(g1, g2, mix)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(label, indent=2) + "\n"
    OUT.write_text(blob)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} B)")
    # ...and a copy INSIDE the forcing dir, because that directory is what the
    # m1-forcing job uploads as an artifact and the 64 build shards download.
    # The repo copy alone would leave every shard shipping whatever label was
    # last committed, which is exactly the drift this script exists to prevent.
    if TABLES.exists():
        side = TABLES / "_served_skill.json"
        side.write_text(blob)
        print(f"  also wrote {side.relative_to(ROOT)} (rides the shard artifact)")
    print(f"  forcing observed: {mix['areal_fraction']:.1%} areal over "
          f"{mix['rows']} rows / {mix['sampled_basins']} basins")
    print(f"  governing gate: {label['governing_gate']}  "
          f"delta {label['encoder_substitution_delta']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
