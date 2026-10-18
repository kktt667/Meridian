"""Vector panel: deliberation as linear algebra, then ONE language call to narrate it.

  F  (claims x features)   built from gated aggregates only
  W  (features x lenses)   one column per reviewer lens (methodologist, clinician, ethics/regulatory)
  S = F @ W                caution score per claim per lens; above the lens threshold -> vote more cautious
  C[i, :, j] = F[i] * W[:, j]   contributions: why each vote came out the way it did (the audit trail)

Edge cases come from a fixed threat library triggered by features and each lab's published notes.
The narrator turns this audit into plain English; a number guard flags any number not present in the audit.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

from .catalog import LABS
from .llm import LLM, AgentFailed
from .schemas import Narrative

FEATURES = ["imprecision", "null_closeness", "inconsistency", "refused_share", "single_source",
            "cross_population", "small_counts", "weak_effect"]
FEATURE_MEANING = {
    "imprecision": "width of the 95% CI on the log scale (0 = very narrow, 1 = a 4-fold-wide interval or more)",
    "null_closeness": "how close the nearer CI bound is to 'no effect' (1 = touching or crossing it)",
    "inconsistency": "I^2 heterogeneity between labs (0-1)",
    "refused_share": "share of this claim's lab queries that the gates refused",
    "single_source": "1 if only one lab contributes",
    "cross_population": "1 if the claim compares or transfers across different populations",
    "small_counts": "how small the smallest released cell is (1 = at the minimum of 5, 0 = 20 or more)",
    "weak_effect": "1 - effect size (|log OR| scaled; 1 = no effect)",
}
LENSES = ["methodologist", "clinician", "ethics_regulatory"]
W = np.array([
    # methodologist, clinician, ethics/regulatory
    [0.5, 0.2, 0.2],   # imprecision (matters mostly when the interval is also near 'no effect')
    [1.2, 1.0, 0.6],   # null_closeness
    [1.2, 0.5, 0.6],   # inconsistency
    [0.6, 0.3, 1.0],   # refused_share
    [0.5, 0.2, 0.8],   # single_source
    [0.2, 0.3, 0.9],   # cross_population
    [0.6, 0.3, 0.4],   # small_counts
    [0.3, 1.0, 0.2],   # weak_effect
])
# A lens dissents (votes more cautiously than the statistics) only when the claim is near 'no effect' AND has at least
# one more serious weakness: near-null alone scores ~1.0-1.2, so thresholds sit just above one strong signal.
THRESHOLD = np.array([1.6, 1.6, 1.6])


def claim_features(claim: dict, ev: dict) -> tuple[np.ndarray, dict]:
    derived = [ev[i] for i in claim["evidence_ids"] if ev[i]["kind"] == "derived"]
    main = derived[-1] if derived else None
    labs = [ev[i] for i in claim["evidence_ids"] if ev[i]["kind"] == "lab_result"]
    f = dict.fromkeys(FEATURES, 0.0)
    if main and main.get("estimate"):
        lo, hi, e = main["ci_low"], main["ci_high"], main["estimate"]
        if (main.get("detail") or {}).get("scale") == "proportion":
            # proportions: work on the logit scale against the claim's own threshold (e.g. 0.5)
            lg = lambda x: math.log(max(min(x, 1 - 1e-6), 1e-6) / (1 - max(min(x, 1 - 1e-6), 1e-6)))
            null = lg(main["detail"].get("null", 0.5))
            lo, hi, e = math.exp(lg(lo) - null), math.exp(lg(hi) - null), math.exp(lg(e) - null)
        f["imprecision"] = min(1.0, math.log(hi / lo) / math.log(4)) if lo > 0 else 1.0
        margin = (math.log(lo) if e >= 1 else -math.log(hi)) if lo > 0 else 0
        f["null_closeness"] = float(max(0.0, min(1.0, 1 - margin / math.log(1.5))))
        f["weak_effect"] = float(max(0.0, 1 - abs(math.log(e)) / math.log(3)))
        f["inconsistency"] = float(main.get("i2") or 0.0)
    else:
        f.update(imprecision=1.0, null_closeness=1.0, weak_effect=1.0)
    if labs:
        f["refused_share"] = sum(l["status"] == "refused" for l in labs) / len(labs)
        ok_labs = {l["lab"] for l in labs if l["status"] == "ok"}
        all_labs = {l["lab"] for l in labs}
        # only penalise a cross-lab claim that LOST labs; a local claim is single-lab by design
        f["single_source"] = 1.0 if len(all_labs) >= 2 and len(ok_labs) <= 1 else 0.0
        mins = [l["detail"].get("min_cell") for l in labs if l["detail"].get("min_cell") is not None]
        f["small_counts"] = float(max(0.0, (20 - min(mins)) / 15)) if mins else 0.0  # 1 at a cell of 5, 0 at 20+
    f["cross_population"] = 1.0 if claim.get("cross_population") else 0.0
    return np.array([f[k] for k in FEATURES]), f


# ----------------------------------------------------------------------------- threat library
def threats(claims: list[dict], feats: dict, labs_used: list[str], recipe: str) -> list[dict]:
    out = []
    for c in claims:
        f, cid = feats[c["claim_id"]], c["claim_id"]
        out.append({"claim_id": cid, "category": "survivor_or_selection", "severity": "high",
                    "threat": "Surveys only include living people: if one event kills more often before it can be reported, the mix is biased.",
                    "how_to_check": "Compare with mortality or registry data for the same populations.", "trigger": "outcome = ever-diagnosed event"})
        if recipe != "undiagnosed_comparison":
            out.append({"claim_id": cid, "category": "access_to_diagnosis", "severity": "medium",
                        "threat": "Outcomes are doctor diagnoses, so groups with less access to care are under-counted.",
                        "how_to_check": "Repeat with measured outcomes (e.g. examination data) where available.", "trigger": "diagnosis-based variables"})
        else:
            out.append({"claim_id": cid, "category": "measurement", "severity": "medium",
                        "threat": "A single examination visit with a high reading is not a clinical diagnosis of hypertension.",
                        "how_to_check": "Use repeated measurements or ambulatory monitoring.", "trigger": "measured_high_bp from one visit"})
        if f["cross_population"]:
            out.append({"claim_id": cid, "category": "transportability", "severity": "high",
                        "threat": "Populations differ in survey year, mode (phone vs household) and health system, not only in ancestry.",
                        "how_to_check": "Adjust for more covariates or compare with a population-matched cohort.", "trigger": "cross_population = 1"})
        if f["inconsistency"] >= 0.5:
            out.append({"claim_id": cid, "category": "confounding", "severity": "high",
                        "threat": f"Labs disagree (I^2 = {f['inconsistency']:.2f}); a pooled number would hide real differences.",
                        "how_to_check": "Report each lab separately; investigate what differs between them.", "trigger": "inconsistency >= 0.5"})
        if f["single_source"] or f["refused_share"] > 0:
            out.append({"claim_id": cid, "category": "small_sample", "severity": "medium",
                        "threat": "Part of the intended evidence is missing (gate refusal or a single contributing lab).",
                        "how_to_check": "Recruit another lab with enough cases through the marketplace.", "trigger": "single_source or refused_share > 0"})
    for lab in labs_used:  # each lab's own published notes become explicit threats
        for note in LABS[lab]["notes"]:
            if any(k in note.lower() for k in ("only asked", "conditional", "top-coded", "imputed", "single survey year", "8 =")):
                out.append({"claim_id": "all", "category": "definition_mismatch", "severity": "low",
                            "threat": f"{lab}: {note}", "how_to_check": "Confirm the harmonised coding with the lab steward.",
                            "trigger": "lab notes"})
    return out


# ----------------------------------------------------------------------------- the panel
def run(claims: list[dict], evidence: list[dict], labs_used: list[str], recipe: str) -> dict:
    ev = {e["evidence_id"]: e for e in evidence}
    rows, feats = [], {}
    for c in claims:
        v, f = claim_features(c, ev)
        rows.append(v)
        feats[c["claim_id"]] = f
    F = np.vstack(rows)
    S = F @ W                                    # claims x lenses
    contrib = F[:, :, None] * W[None, :, :]      # claims x features x lenses
    ballots = {l: {"mode": "vector", "votes": [], "key_caveat": ""} for l in LENSES}
    for i, c in enumerate(claims):
        for j, lens in enumerate(LENSES):
            cautious = S[i, j] > THRESHOLD[j]
            if not cautious or c["ceiling"] in ("inconclusive", "inconsistent"):
                verdict = c["ceiling"]
            else:
                verdict = "inconsistent" if feats[c["claim_id"]]["inconsistency"] >= 0.5 else "inconclusive"
            top = sorted(((FEATURES[k], float(contrib[i, k, j])) for k in range(len(FEATURES))), key=lambda t: -t[1])[:3]
            ballots[lens]["votes"].append({
                "claim_id": c["claim_id"], "verdict": verdict, "confidence": "high" if abs(S[i, j] - THRESHOLD[j]) > 0.5 else "medium",
                "rationale": f"caution score {S[i, j]:.2f} vs threshold {THRESHOLD[j]:.2f}; top drivers: "
                             + ", ".join(f"{n} ({v:.2f})" for n, v in top),
                "evidence_ids": c["evidence_ids"][-1:], "score": round(float(S[i, j]), 3),
                "contributions": {FEATURES[k]: round(float(contrib[i, k, j]), 3) for k in range(len(FEATURES))}})
        # repropagate into for/against points straight from the numbers (no free text)
    pro, con = [], []
    for c in claims:
        f = feats[c["claim_id"]]
        main = c["evidence_ids"][-1:]
        if c["ceiling"] in ("supported", "not_supported"):
            pro.append({"claim": f"{c['claim_id']}: {c['ceiling_reason']}", "evidence_ids": main,
                        "strength": "strong" if f["null_closeness"] < 0.3 else "moderate"})
        bad = [k for k in ("imprecision", "inconsistency", "refused_share", "single_source", "cross_population", "small_counts") if f[k] >= 0.5]
        if bad:
            con.append({"claim": f"{c['claim_id']}: weakened by " + ", ".join(f"{k}={f[k]:.2f}" for k in bad), "evidence_ids": main,
                        "strength": "strong" if len(bad) >= 2 else "moderate"})
    return {
        "mode": "vector", "features": FEATURES, "feature_meaning": FEATURE_MEANING, "lenses": LENSES,
        "weights": W.round(3).tolist(), "thresholds": THRESHOLD.tolist(),
        "F": {c["claim_id"]: feats[c["claim_id"]] for c in claims}, "S": {c["claim_id"]: dict(zip(LENSES, S[i].round(3).tolist())) for i, c in enumerate(claims)},
        "for": {"position": "for", "summary": "Derived from the statistics (vector panel).", "points": pro},
        "against": {"position": "against", "summary": "Derived from caution features (vector panel).", "points": con},
        "edge_cases": {"edge_cases": threats(claims, feats, labs_used, recipe)},
        "ballots": ballots, "blind": True, "blind_map": {}, "modes": {"for": "vector", "against": "vector", "edge_cases": "threat_library"},
        "dropped_points": {},
    }


# ----------------------------------------------------------------------------- narrator: template first, LLM may only reword
NARRATOR_SYSTEM = """You rewrite a computed, audited research narrative into plain English for a clinician who is not a
statistician. You may change wording and sentence order only. You must keep EVERY number exactly, EVERY evidence ID
(E1, E2...), EVERY claim ID (C1, C2...) and EVERY verdict word (supported, not supported, inconclusive, inconsistent)
attached to the same claim. Do not add facts, numbers, causes, recommendations or opinions. Labs are named Lab A,
Lab B...: keep those labels. One summary sentence per claim, in order, and each one MUST start with its claim ID
("C1: ..."). Keep the statistical words as they are: say "odds", never "risk"; never say "cause", "causes", "leads to",
"prevent" or "should"."""

DRIFT_WORDS = ("risk", "cause", "causes", "caused", "leads to", "prevent", "should", "significant")

VERDICT_WORD = {"supported": "supported", "not_supported": "not supported", "inconclusive": "inconclusive", "inconsistent": "inconsistent"}


def template_narrative(question: str, verdicts: list[dict], evidence: list[dict], panel: dict) -> dict:
    """Deterministic narrative built only from the audit objects. Every number here is traceable by construction."""
    ev = {e["evidence_id"]: e for e in evidence}
    counts = Counter(v["final"] for v in verdicts)
    key = next((v for v in verdicts if "applied directly" in v["statement"] or "transfer" in v["statement"].lower()), None) or \
          next((v for v in verdicts if v["final"] in ("not_supported", "inconsistent")), verdicts[0] if verdicts else None)
    head = ", ".join(f"{n} {VERDICT_WORD[k]}" for k, n in counts.items())
    headline = f"{len(verdicts)} claims: {head}." + (f" Key result: {key['claim_id']} is {VERDICT_WORD[key['final']]}: {key['ceiling_reason']}" if key else "")
    summary = []
    for v in verdicts:
        main = next((ev[i] for i in reversed(v["evidence_ids"]) if ev.get(i, {}).get("kind") == "derived"), None)
        est = (f" ({main['evidence_id']}: {main['estimate']:.2f}, 95% CI {main['ci_low']:.2f}-{main['ci_high']:.2f})"
               if main and main.get("estimate") is not None else "")
        down = ""
        if v["final"] != v["ceiling"]:
            objs = [o for o in v.get("steward_objections", []) if o.get("severity") == "high"]
            f = (panel.get("F") or {}).get(v["claim_id"], {})
            top = ", ".join(f"{k.replace('_', ' ')} {x:.2f}" for k, x in sorted(f.items(), key=lambda t: -t[1])[:2] if x > 0)
            if objs:
                down = f" Downgraded from {VERDICT_WORD[v['ceiling']]}: the {objs[0].get('lab', 'data')} steward objected ({objs[0].get('reason', objs[0].get('issue', ''))[:140]})."
            elif top:
                down = f" The review panel downgraded it from {VERDICT_WORD[v['ceiling']]} because of {top}."
            else:
                down = f" The review panel voted more cautiously than the statistics and downgraded it from {VERDICT_WORD[v['ceiling']]}."
        summary.append(f"{v['claim_id']}: {v['statement']} Verdict: {VERDICT_WORD[v['final']]}{est}. {v['ceiling_reason']}{down}")
    high = [t for t in panel["edge_cases"]["edge_cases"] if t["severity"] == "high"]
    change = sorted({t["how_to_check"] for t in high})[:4] or ["More labs with enough cases."]
    caveats = sorted({t["threat"] for t in high})[:4] + ["All labs: self-reported, cross-sectional survey data; associations, not causes."]
    return {"headline": headline, "summary": summary, "what_would_change_it": change, "caveats": caveats}


def _numbers(obj) -> set[str]:
    return {f"{float(x):.2f}" for x in re.findall(r"\d+\.\d+", json.dumps(obj))}


def guard(template: dict, rewritten: dict, verdicts: list[dict]) -> list[str]:
    """Why a rewrite is rejected. Empty list = accepted."""
    problems = []
    extra = _numbers(rewritten) - _numbers(template)
    if extra:
        problems.append(f"numbers not in the audit: {sorted(extra)}")
    if len(rewritten["summary"]) != len(verdicts):
        problems.append(f"{len(rewritten['summary'])} summary sentences for {len(verdicts)} claims")
    for v, sent in zip(verdicts, rewritten["summary"]):
        if v["claim_id"] not in sent:
            problems.append(f"{v['claim_id']} missing from its sentence")
        w = VERDICT_WORD[v["final"]]
        if not re.search(rf"\b{w}\b", sent, re.I) or (w == "supported" and re.search(r"\bnot supported\b", sent, re.I)):
            problems.append(f"{v['claim_id']} verdict '{w}' changed or missing")
    tmpl_text = json.dumps(template).lower()
    drift = [d for d in DRIFT_WORDS if re.search(rf"\b{d}\b", json.dumps(rewritten).lower()) and not re.search(rf"\b{d}\b", tmpl_text)]
    if drift:
        problems.append(f"meaning drift, words added that are not in the audit: {drift}")
    return problems


def narrate(llm: LLM, question: str, verdicts: list[dict], evidence: list[dict], panel: dict, local_steps: list,
            blind_labs: list[str] | None = None) -> dict:
    from .agents import apply_map, blind_map
    template = template_narrative(question, verdicts, evidence, panel)
    out = {**template, "template": template, "mode": "template", "guard": [], "blind": bool(blind_labs), "unverified_numbers": []}
    m = blind_map(blind_labs) if blind_labs else {}
    blinded = json.loads(apply_map(json.dumps(template), m)) if m else template
    try:
        n = llm.structured("narrator", NARRATOR_SYSTEM, f"Narrative to reword (JSON):\n{json.dumps(blinded)}", Narrative).model_dump()
    except AgentFailed as e:
        llm.notes.append(f"narrator unavailable, template used: {e}")
        out["guard"] = [f"language model unavailable ({e})"]
        return out
    out["llm_blinded"] = n
    problems = guard(blinded, n, verdicts)
    if problems:
        out["guard"] = problems
        out["mode"] = "template (LLM rewrite rejected)"
        return out
    unmap = {v: k for k, v in m.items()}
    n = json.loads(apply_map(json.dumps(n), unmap)) if unmap else n
    out.update(n)
    out["mode"] = "llm rewrite of template (guard passed)"
    return out
