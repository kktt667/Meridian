"""Pre-registered research protocol and GRADE-inspired certainty.

The protocol is written and hashed BEFORE any lab is queried, so the analysis can't be changed after
seeing the data. Certainty follows GRADE logic for observational evidence (start low, downgrade for
risk of bias / inconsistency / indirectness / imprecision, upgrade for large effects). It is
GRADE-inspired, not a formal GRADE assessment.
"""
from __future__ import annotations

import hashlib
import json

MH = "OR_MH = sum_s(a_g,s * b_r,s / n_s) / sum_s(b_g,s * a_r,s / n_s) over age x sex strata s (Mantel-Haenszel)"
DL = "pooled log OR = sum_i w_i* y_i / sum_i w_i*, w_i* = 1/(se_i^2 + tau^2) (DerSimonian-Laird random effects)"
LOGIT = "logit P(outcome) = b0 + b1*hypertension + b2*age + b3*male; OR = exp(b1)"


def build(recipe: str, params: dict, home: str, question: str) -> dict:
    ext = params.get("external_labs", [])
    if recipe == "event_mix_comparison":
        fg, rg = params["focus_group"], params["reference_group"]
        pico = {"population": f"Adults 19+ who have had exactly one of stroke or heart attack",
                "exposure": f"Belonging to {fg} ({home}) or to an external population ({', '.join(ext)})",
                "comparison": f"{rg} adults in {home}", "outcome": "The event being a stroke rather than a heart attack"}
        facts = [
            {"fact_id": "F1", "statement": f"{home} can release enough {fg} events (each age x sex cell >= min cell)", "kind": "data_sufficiency",
             "how": "event_split_strata at home lab; gate suppresses small cells", "claim_id": None},
            {"fact_id": "F2", "statement": f"Within {home}, {fg} events are more often strokes than {rg} events", "kind": "association",
             "how": MH, "claim_id": "C1"},
            {"fact_id": "F3", "statement": "The external population's events are more often strokes than the home reference group's",
             "kind": "association", "how": MH, "claim_id": "C2"},
            {"fact_id": "F4", "statement": f"The external effect size transfers to {fg} patients in {home}", "kind": "transportability",
             "how": "ratio of F3 / F2 on the log scale; transfer is refuted if the 95% CI excludes 1", "claim_id": "C3"},
            {"fact_id": "F5", "statement": "Stroke and heart-attack definitions are comparable across labs", "kind": "definitions",
             "how": "compare published catalog definitions (all 'ever diagnosed by a doctor')", "claim_id": None},
        ]
        estimand = MH
    elif recipe == "age_gradient":
        pico = {"population": "Adults in each lab, split 19-44 and 65+", "exposure": "Diagnosed hypertension",
                "comparison": "No diagnosed hypertension (same age band)", "outcome": "Ever diagnosed stroke"}
        facts = [
            {"fact_id": "F1", "statement": "Each lab has >= 20 strokes among young hypertensives", "kind": "data_sufficiency",
             "how": "gate min_cases check inside each lab", "claim_id": None},
            {"fact_id": "F2", "statement": "In 19-44 year olds, hypertension is associated with stroke, consistently across labs",
             "kind": "association", "how": f"{LOGIT}; then {DL}", "claim_id": "C1"},
            {"fact_id": "F3", "statement": "The association is stronger at 19-44 than at 65+", "kind": "effect_modification",
             "how": "ratio of ORs (young/old) per lab, pooled by DerSimonian-Laird", "claim_id": "C2"},
            {"fact_id": "F4", "statement": "At 65+, labs agree on the strength of the association", "kind": "consistency",
             "how": "Cochran Q and I^2 across labs", "claim_id": "C3"},
        ]
        estimand = f"{LOGIT}; {DL}"
    else:
        pico = {"population": "Adults 20+ with a measured blood pressure >= 140/90", "exposure": f"Living in {home}'s population",
                "comparison": f"Population of {', '.join(ext)}", "outcome": "Never having been diagnosed with hypertension"}
        facts = [
            {"fact_id": "F1", "statement": "Both labs measured blood pressure at examination", "kind": "data_sufficiency",
             "how": "catalog flag measured_bp", "claim_id": None},
            {"fact_id": "F2", "statement": f"In {home}, most 20-44 year olds with a high reading are undiagnosed", "kind": "proportion",
             "how": "Wilson 95% interval for undiagnosed / high readings; established if the lower bound > 0.5", "claim_id": "C1"},
            {"fact_id": "F3", "statement": "Young adults with high readings are more often undiagnosed at home than externally",
             "kind": "association", "how": "2x2 odds ratio, ages 20-44", "claim_id": "C2"},
            {"fact_id": "F4", "statement": "The difference holds after adjusting for age", "kind": "association", "how": MH, "claim_id": "C3"},
        ]
        estimand = MH
    protocol = {"question": question, "home_lab": home, "recipe": recipe, "parameters": params, "pico": pico,
                "estimand": estimand, "facts": facts,
                "decision_rules": {"supported": "95% CI entirely on the hypothesised side of the null",
                                   "inconsistent": "pooled claims with I^2 >= 50% and Q p < 0.10",
                                   "inconclusive": "CI includes the null, a lab refused, or fewer than 2 labs contribute",
                                   "panel": "AI votes may downgrade a verdict, never upgrade it"}}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()
    return protocol


LEVELS = ["very low", "low", "moderate", "high"]


def certainty(final: str, est: dict | None, i2: float | None, n_labs: int, cross_population: bool) -> dict:
    if final not in ("supported", "not_supported") or not est or est.get("estimate") is None:
        return {"level": "not graded", "why": "Verdict is not definite.", "domains": {}}
    score, notes = 1, ["Start: low (observational survey data)."]
    e, lo, hi = est["estimate"], est["ci_low"], est["ci_high"]
    big = lo if e >= 1 else 1 / hi
    if big > 5:
        score += 2; notes.append("Upgrade x2: very large effect (CI bound beyond 5).")
    elif big > 2:
        score += 1; notes.append("Upgrade: large effect (CI bound beyond 2).")
    domains = {
        "risk_of_bias": "serious: self-reported diagnoses, cross-sectional, survivors only",
        "inconsistency": ("serious: I^2 >= 50%" if (i2 or 0) >= 0.5 else
                          ("not serious" if n_labs >= 2 else "not assessable: single source")),
        "indirectness": "serious: comparison across different populations / surveys" if cross_population else "not serious",
        "imprecision": "serious: wide or null-crossing CI" if (lo <= 1 <= hi or hi / lo > 3) else "not serious",
    }
    for k, v in domains.items():
        if v.startswith("serious"):
            score -= 1; notes.append(f"Downgrade: {k}.")
    level = LEVELS[max(0, min(3, score))]
    return {"level": level, "domains": domains, "why": " ".join(notes), "framework": "GRADE-inspired (not a formal GRADE assessment)"}
