"""Bias evaluation on SYNTHETIC digital-twin labs. No real data, no LLM.

Twins are generated from parameters we choose, so the truth is known and the system can be scored:
  1. transport     Does Meridian catch evidence that does NOT transfer from a majority elsewhere to a minority here?
  2. small_group   Can a small group get an answer through the network when its own lab must refuse? And does
                   anything below the minimum cell ever leave a lab?
Every query goes through the real LabNode gate and the real stats.py, exactly as the recipes do.

  venv/bin/python scripts/twin_eval.py [--trials 100]
Writes runs/twin_eval.json and demo/docs/bias_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envoy import stats                                   # noqa: E402
from envoy.catalog import LABS, _policy                   # noqa: E402
from envoy.lab import LabNode                             # noqa: E402
from envoy.planner_ml import evaluate as planner_evaluate  # noqa: E402
from envoy.schemas import LabQuery, LabResult             # noqa: E402

TWINS = {"TWIN_HOME": ["Minority", "Majority"], "TWIN_EXT": [], "TWIN_NET": []}


def register_twins():
    """Synthetic labs join the catalog with the standard policy, so the gate treats them like any lab. Call unregister after."""
    for tid, groups in TWINS.items():
        LABS[tid] = {**LABS["US_BRFSS_2023"], "name": f"{tid} (synthetic twin)", "country": tid, "groups": groups,
                     "policy": _policy("none"), "notes": ["SYNTHETIC twin: generated from chosen parameters, never from patient rows."]}


def unregister_twins():
    for tid in TWINS:
        LABS.pop(tid, None)


def benchmark(trials: int = 60, seed: int = 2026) -> dict:
    """Run the twin tests and return the summary (used by the API's live 'plant bias' button)."""
    register_twins()
    RELEASED.clear()
    try:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        tr = transport_test(rng, trials, minority_sizes=(500, 1000, 2000), ratios=(1.0, 2.0))
        sg = small_group_test(rng, max(20, trials // 3))
        return {"trials": trials, "synthetic": True, "transport": tr, "small_group": sg, "privacy": privacy_invariant(),
                "planner": planner_evaluate(), "seconds": round(time.perf_counter() - t0, 1)}
    finally:
        unregister_twins()

COLS = ["age", "male", "group", "hypertension", "stroke", "heart_attack", "sbp", "dbp", "weight"]


def _logit(p):
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------------------------------------ twin generators
def twin_events(rng, n, group, or_stroke, age_lo=19, age_hi=85, p_event=0.15, p_stroke_ref=0.4):
    """People with an event; is it a stroke (vs heart attack)? Odds of stroke scaled by or_stroke. Age shifts the
    odds too, so a younger minority is a confounder that Mantel-Haenszel must handle."""
    age = rng.integers(age_lo, age_hi + 1, n)
    male = rng.integers(0, 2, n)
    event = rng.random(n) < p_event
    p = _sig(_logit(p_stroke_ref) + 0.25 * (age - 50) / 20 + np.log(or_stroke))
    stroke = (event & (rng.random(n) < p)).astype(float)
    heart = (event & (stroke == 0)).astype(float)
    return pd.DataFrame({"age": age, "male": male, "group": group, "hypertension": 0.0, "stroke": stroke,
                         "heart_attack": heart, "sbp": np.nan, "dbp": np.nan, "weight": 1.0})[COLS]


def twin_young(rng, n, or_ht=3.0, p_ht=0.12, p_stroke_base=0.01):
    """Young adults 19-44: hypertension raises the odds of stroke by or_ht."""
    age = rng.integers(19, 45, n)
    male = rng.integers(0, 2, n)
    ht = (rng.random(n) < p_ht).astype(float)
    p = _sig(_logit(p_stroke_base) + 0.03 * (age - 30) + np.log(or_ht) * ht)
    stroke = (rng.random(n) < p).astype(float)
    return pd.DataFrame({"age": age, "male": male, "group": "all", "hypertension": ht, "stroke": stroke,
                         "heart_attack": 0.0, "sbp": np.nan, "dbp": np.nan, "weight": 1.0})[COLS]


# ------------------------------------------------------------------------------------------ gate round-trip
RELEASED: list[LabResult] = []   # every result that crossed a lab boundary, for the privacy invariant


def ask(node: LabNode, qid, op, pop, outcome=None) -> LabResult:
    q = LabQuery(query_id=qid, asker="twin-eval", operation=op, population=pop, outcome=outcome,
                 exposure="hypertension" if op == "logistic_or" else None)
    res = LabResult.model_validate_json(node.handle(q.model_dump_json()))
    RELEASED.append(res)
    return res


def strata(res):
    return {s.label: (s.a, s.b) for s in (res.strata or [])}


# ------------------------------------------------------------------------------------------ test 1: transport
def transport_test(rng, trials, minority_sizes=(500, 1000, 2000), ratios=(1.0, 1.5, 2.0)):
    """Country A: Majority (5000) + Minority (n). Country B: the minority's population as a majority (5000).
    Truth: OR_B / OR_minority = ratio. ratio 1 => evidence transfers; ratio > 1 => it does not.
    Meridian flags non-transfer when ceiling_for_transfer says not_supported."""
    out = []
    for n in minority_sizes:
        for r in ratios:
            flagged = refused = 0
            for t in range(trials):
                home = LabNode("TWIN_HOME", pd.concat([twin_events(rng, 5000, "Majority", 1.0),
                                                       twin_events(rng, n, "Minority", 1.5, age_hi=70)]))
                ext = LabNode("TWIN_EXT", twin_events(rng, 5000, "all", 1.5 * r))
                pop = lambda g: {"age_min": 19, "age_max": 120, "sex": "all", "group": g}
                rf = ask(home, f"t{t}-f", "event_split_strata", pop("Minority"))
                rr = ask(home, f"t{t}-r", "event_split_strata", pop("Majority"))
                re_ = ask(ext, f"t{t}-e", "event_split_strata", pop("all"))
                if "ok" != rf.status or "ok" != rr.status or "ok" != re_.status:
                    refused += 1
                    continue
                loc = stats.mantel_haenszel(strata(rf), strata(rr))
                mh = stats.mantel_haenszel(strata(re_), strata(rr))
                if loc.get("estimate") is None or mh.get("estimate") is None:
                    refused += 1
                    continue
                verdict, _ = stats.ceiling_for_transfer(stats.ratio(mh["log"], mh["se"], loc["log"], loc["se"]))
                flagged += verdict == "not_supported"
            answered = trials - refused
            out.append({"minority_n": n, "true_ratio": r, "trials": trials, "gate_refused": refused,
                        "flagged_not_transferable": flagged,
                        "rate": round(flagged / answered, 3) if answered else None,
                        "meaning": "false alarm rate" if r == 1.0 else "detection rate"})
            print(f"  transport n={n:5d} ratio={r:.1f}: flagged {flagged}/{answered} ({out[-1]['rate']}), gate refused {refused}")
    return out


# ------------------------------------------------------------------------------------------ test 2: small group
def small_group_test(rng, trials, home_n=2000, network_ns=(10000, 20000), true_or=3.0):
    """Hypertension -> stroke in young adults (the Q6 situation). The home lab has home_n young adults; the
    network adds two labs. Alone: does the home gate release an estimate? Network: do >= 2 labs release one, and
    does the pooled estimate contain the truth?"""
    alone_ok = net_ok = covered = 0
    pooled = []
    pop = {"age_min": 19, "age_max": 44, "sex": "all", "group": "all"}
    for t in range(trials):
        labs = [LabNode("TWIN_HOME", twin_young(rng, home_n, true_or))] + \
               [LabNode("TWIN_NET", twin_young(rng, n, true_or)) for n in network_ns]
        res = [ask(l, f"s{t}-{i}", "logistic_or", pop, outcome="stroke") for i, l in enumerate(labs)]
        alone_ok += res[0].status == "ok"
        ok = [r for r in res if r.status == "ok"]
        if len(ok) >= 2:
            net_ok += 1
            m = stats.dersimonian_laird([r.log_estimate for r in ok], [r.se for r in ok])
            pooled.append(m["estimate"])
            covered += m["ci_low"] <= true_or <= m["ci_high"]
    out = {"trials": trials, "home_young_n": home_n, "network_young_ns": list(network_ns), "true_or": true_or,
           "alone_answered": alone_ok, "network_answered": net_ok,
           "pooled_median_or": round(float(np.median(pooled)), 2) if pooled else None,
           "pooled_ci_covers_truth": round(covered / net_ok, 3) if net_ok else None}
    print(f"  small group: alone {alone_ok}/{trials}, network {net_ok}/{trials}, "
          f"pooled OR median {out['pooled_median_or']} (truth {true_or}), CI coverage {out['pooled_ci_covers_truth']}")
    return out


# ------------------------------------------------------------------------------------------ privacy invariant
def privacy_invariant():
    """Scan everything that left a lab: no stratum cell below min_cell, no model estimate with fewer than
    min_cases exposed cases, no raw n (only size buckets), no row-level fields."""
    from envoy.config import MIN_CASES_MODEL, MIN_CELL
    min_cell, min_cases = MIN_CELL, MIN_CASES_MODEL
    cells = [min(s.a, s.b) for r in RELEASED for s in (r.strata or [])]
    cases = [r.cases_exposed for r in RELEASED if r.status == "ok" and r.operation == "logistic_or"]
    fields = set().union(*(r.model_dump(exclude_none=True).keys() for r in RELEASED))
    return {"results_released": len(RELEASED), "ok": sum(r.status == "ok" for r in RELEASED),
            "refused": sum(r.status == "refused" for r in RELEASED),
            "cells_released": len(cells), "smallest_cell": min(cells) if cells else None, "min_cell_policy": min_cell,
            "cells_below_policy": sum(c < min_cell for c in cells),
            "estimates_released": len(cases), "fewest_exposed_cases": min(cases) if cases else None,
            "min_cases_policy": min_cases, "estimates_below_policy": sum(c < min_cases for c in cases),
            "fields_that_left_labs": sorted(fields)}


# ------------------------------------------------------------------------------------------ report
def report(res) -> str:
    t, s, p, pl = res["transport"], res["small_group"], res["privacy"], res["planner"]
    L = ["# Bias report (synthetic digital twins)", "",
         f"Generated {res['generated']} in {res['seconds']}s. Seed {res['seed']}. SYNTHETIC data only: every twin is generated "
         "from chosen parameters so the truth is known. The same gate (`envoy/lab.py`) and statistics (`envoy/stats.py`) "
         "as the live system; no language model is involved.", "",
         "## 1. Transport test: does evidence from a majority elsewhere apply to a minority here?", "",
         "Country A has a majority (5,000) and a minority (n). Country B is the minority's home population (5,000). "
         "We plant the truth: the effect in B is `ratio` times the effect in A's minority. ratio 1.0 = the evidence transfers; "
         "the flag rate there is the false-alarm rate. ratio > 1 = it does not transfer; the flag rate is the detection rate.", "",
         "| minority n | true ratio | flagged / answered | rate | gate refused |", "|---|---|---|---|---|"]
    for r in t:
        L.append(f"| {r['minority_n']} | {r['true_ratio']} | {r['flagged_not_transferable']} / {r['trials'] - r['gate_refused']} | "
                 f"{'' if r['rate'] is None else f'{r['rate']:.0%}'} ({r['meaning']}) | {r['gate_refused']} |")
    L += ["", "## 2. Small-group test: alone vs network (the Q6 situation)", "",
          f"Hypertension -> stroke among young adults, true OR {s['true_or']}. The home lab has {s['home_young_n']:,} young adults; "
          f"the network adds labs with {', '.join(f'{n:,}' for n in s['network_young_ns'])}.", "",
          f"- Home lab alone released an estimate: **{s['alone_answered']} / {s['trials']}** trials.",
          f"- Through the network (>= 2 labs): **{s['network_answered']} / {s['trials']}** trials.",
          f"- Pooled OR median {s['pooled_median_or']} (truth {s['true_or']}); the 95% CI contained the truth in "
          f"{'' if s['pooled_ci_covers_truth'] is None else f'{s['pooled_ci_covers_truth']:.0%}'} of answered trials.", "",
          "## 3. Privacy invariant: what left the labs", "",
          f"- {p['results_released']:,} results crossed a lab boundary ({p['ok']:,} answered, {p['refused']:,} refused).",
          f"- {p['cells_released']:,} stratum cells released; smallest {p['smallest_cell']} (policy minimum {p['min_cell_policy']}); "
          f"**{p['cells_below_policy']} below policy**.",
          f"- {p['estimates_released']:,} model estimates released; fewest exposed cases {p['fewest_exposed_cases']} "
          f"(policy minimum {p['min_cases_policy']}); **{p['estimates_below_policy']} below policy**.",
          f"- Fields that ever left a lab: `{', '.join(p['fields_that_left_labs'])}`. No row-level field exists in the schema.", "",
          "## 4. Planner: routing without a language model", "",
          f"A TF-IDF + logistic-regression classifier picks the analysis; the catalog fills every other slot (groups, local vs remote labs); "
          f"fixed rules screen the purpose. {pl['labelled_questions']} labelled questions, leave-one-out accuracy "
          f"**{pl['leave_one_out_accuracy']:.0%}**; held-out paraphrases **{pl['held_out_correct']} / {pl['held_out']}**. "
          f"Below confidence {pl['min_confidence']} it abstains (recipe 'unsupported') instead of guessing. Same question, same plan, every time.", "",
          "## Limits", "",
          "- Twins are simple: one confounder (age), no survey weights, no missingness. Real data is harder.",
          "- The planner's training set is small and written by us; accuracy on real researchers' phrasing is untested.",
          "- The transport test has power only when the minority is large enough for the gate to release cells.",
          "- This measures the deterministic layer. The language-model layer (deep mode) is blinded to lab names but not yet "
          "measured here: a name-swap test on it is the next step."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    register_twins()
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    print("transport test")
    tr = transport_test(rng, a.trials)
    print("small-group test")
    sg = small_group_test(rng, a.trials)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "seed": a.seed, "trials": a.trials, "synthetic": True,
           "transport": tr, "small_group": sg, "privacy": privacy_invariant(), "planner": planner_evaluate(), "seconds": round(time.perf_counter() - t0, 1)}
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "twin_eval.json").write_text(json.dumps(res, indent=1))
    md = report(res)
    (ROOT / "demo" / "docs" / "bias_report.md").write_text(md)
    print(f"\n{md}")
    print(f"wrote runs/twin_eval.json and demo/docs/bias_report.md ({res['seconds']}s)")


if __name__ == "__main__":
    main()
