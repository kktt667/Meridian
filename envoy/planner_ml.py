"""Deterministic planner: a small text classifier picks the recipe, code fills every other slot from the catalog.

No language model is needed to route a question. Local-vs-remote is a catalog lookup (which labs hold the groups,
variables and measured values the recipe needs), the groups are matched by name, and the purpose is screened by
fixed rules. The classifier returns a calibrated probability; below PLANNER_MIN_CONFIDENCE the planner abstains
and the LLM (if any) is asked instead. Same input -> same plan, every time, in milliseconds.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from .catalog import LABS
from .schemas import Plan

PLANNER_MIN_CONFIDENCE = 0.5

# ------------------------------------------------------------------------------------ training questions
# Labelled paraphrases of what researchers ask. Small on purpose: the classifier only has to route between
# the network's approved analyses, and every example is readable by a reviewer.
TRAIN: list[tuple[str, str]] = [
    # event_mix_comparison: is a group's event mix stroke-heavy / does evidence transfer across populations
    ("Should we prioritise stroke prevention for our East Asian patients? Is their event mix stroke-heavy?", "event_mix_comparison"),
    ("Can evidence from an East Asian population be applied to our Asian patients?", "event_mix_comparison"),
    ("Among our Black patients, are cardiovascular events more often strokes than heart attacks compared with White patients?", "event_mix_comparison"),
    ("Does the stroke to heart attack ratio differ between Hispanic and White adults?", "event_mix_comparison"),
    ("Is the proportion of strokes among cardiovascular events higher in Asian adults?", "event_mix_comparison"),
    ("Can we transfer findings from Korea to Korean-American patients in our clinic?", "event_mix_comparison"),
    ("Do our minority patients have a different mix of stroke versus myocardial infarction?", "event_mix_comparison"),
    ("Is stroke relatively more common than heart attack in our Asian subgroup, and does Korean data apply to them?", "event_mix_comparison"),
    ("Compare the share of events that are strokes between our Indigenous patients and the majority population.", "event_mix_comparison"),
    ("Does a guideline derived from Brazilian data transport to Brown patients here?", "event_mix_comparison"),
    ("Are cardiovascular events in our Asian population more often strokes, as in East Asia?", "event_mix_comparison"),
    ("We want to know whether external evidence on stroke burden applies to a small ethnic subgroup in our data.", "event_mix_comparison"),
    ("Stroke-heavy event profile: is it true for Asian adults in the US, and does it match Korea?", "event_mix_comparison"),
    ("Do events among our Hispanic patients skew toward stroke rather than infarction?", "event_mix_comparison"),
    ("Is our Asian subgroup too small to characterise, and can another country's population stand in for it?", "event_mix_comparison"),
    # age_gradient: hypertension -> stroke by age, young vs old, pooled across labs
    ("Is high blood pressure a stronger marker of stroke in young adults than in older adults?", "age_gradient"),
    ("We have too few young stroke cases to tell on our own whether hypertension matters more under 45.", "age_gradient"),
    ("Does the association between hypertension and stroke weaken with age?", "age_gradient"),
    ("How does the hypertension stroke odds ratio in people aged 19 to 44 compare with those over 65?", "age_gradient"),
    ("Is hypertension a bigger risk factor for stroke in the young?", "age_gradient"),
    ("Pool the young-adult hypertension to stroke estimate across countries; our own numbers are too small.", "age_gradient"),
    ("Does diagnosed high blood pressure predict stroke more strongly in younger adults than in the elderly?", "age_gradient"),
    ("Age gradient of the hypertension stroke relationship: is it steeper in early adulthood?", "age_gradient"),
    ("Among adults under 45 with hypertension, how much higher are the odds of stroke, and is that consistent across labs?", "age_gradient"),
    ("Is the stroke risk from high blood pressure larger at younger ages?", "age_gradient"),
    ("Do labs agree on the hypertension stroke odds ratio in people 65 and over?", "age_gradient"),
    ("We suspect hypertension is an unusually strong stroke signal in young adults; can the network confirm it?", "age_gradient"),
    ("Compare hypertension-related stroke odds in young versus older adults using several national surveys.", "age_gradient"),
    ("Young onset stroke and blood pressure: is the association stronger than in older people?", "age_gradient"),
    ("Is the effect of hypertension on stroke modified by age?", "age_gradient"),
    # undiagnosed_comparison: measured high BP but never diagnosed
    ("Most of our young adults with high blood pressure readings were never diagnosed. Is that unusually high compared with another country?", "undiagnosed_comparison"),
    ("What share of people with a measured blood pressure above 140/90 have never been told they have hypertension?", "undiagnosed_comparison"),
    ("How common is undiagnosed hypertension among young adults here versus the US?", "undiagnosed_comparison"),
    ("Among people whose examination shows high blood pressure, how many are unaware of it?", "undiagnosed_comparison"),
    ("Is the undiagnosed fraction of hypertension higher in our country than in countries that measure blood pressure?", "undiagnosed_comparison"),
    ("Compare awareness of hypertension: measured high readings without a prior diagnosis, by age band.", "undiagnosed_comparison"),
    ("How many adults with elevated readings have never received a hypertension diagnosis, and how does that compare internationally?", "undiagnosed_comparison"),
    ("Hidden hypertension: what proportion of high readings are in people never diagnosed?", "undiagnosed_comparison"),
    ("Is our screening gap for high blood pressure in young adults worse than elsewhere?", "undiagnosed_comparison"),
    ("Undiagnosed versus diagnosed hypertension among people with high measured blood pressure, compared across labs.", "undiagnosed_comparison"),
    ("Do young people with high blood pressure readings go undiagnosed more often in Korea than in the United States?", "undiagnosed_comparison"),
    ("What fraction of measured hypertension is unrecognised, and is that typical?", "undiagnosed_comparison"),
    ("Among adults with SBP at or above 140, how many were never told they had hypertension?", "undiagnosed_comparison"),
    ("Is lack of hypertension diagnosis despite high readings unusually common in our young adults?", "undiagnosed_comparison"),
    ("Awareness of high blood pressure by age: measured but never diagnosed, here versus a comparator country.", "undiagnosed_comparison"),
    # unsupported: outside the three approved analyses
    ("What is the best statin dose for secondary prevention after a heart attack?", "unsupported"),
    ("Predict which of our patients will have a stroke next year.", "unsupported"),
    ("How does diabetes affect kidney failure rates?", "unsupported"),
    ("What is the average cholesterol level in Brazil?", "unsupported"),
    ("Does air pollution cause asthma in children?", "unsupported"),
    ("Give me the list of patients with stroke in the Korean survey.", "unsupported"),
    ("What is the mortality rate after stroke within 30 days?", "unsupported"),
    ("How effective is aspirin for preventing a first heart attack?", "unsupported"),
    ("What proportion of adults smoke in each country?", "unsupported"),
    ("Is obesity rising faster in young adults than in older adults?", "unsupported"),
    ("Which hospital has the best outcomes for cardiac surgery?", "unsupported"),
    ("How much does a stroke cost the health system?", "unsupported"),
    ("Does salt intake explain the difference in blood pressure between countries?", "unsupported"),
    ("Can you summarise the latest hypertension guidelines?", "unsupported"),
    ("Rank the countries by cancer incidence.", "unsupported"),
]

# Group synonyms -> catalog group names. Matched as whole words, longest phrase first.
GROUP_SYNONYMS = {
    "Asian": ["east asian", "asian", "korean", "korean-american", "chinese", "japanese", "vietnamese", "filipino"],
    "Black": ["black", "african american", "african-american", "afro"],
    "White": ["white", "caucasian", "european"],
    "Hispanic": ["hispanic", "latino", "latina", "latinx"],
    "AIAN": ["aian", "native american", "american indian", "alaska native"],
    "Brown": ["brown", "pardo", "mixed race"],
    "Indigenous": ["indigenous", "indigena"],
    "Other": [],
}
FOCUS_REGION = {"Asian": "East Asia", "Brown": "South America", "Hispanic": "South America"}

# Purpose screening: fixed rules, because a refused purpose must never depend on a model's mood.
PURPOSE_RULES = [
    ("re_identification", r"\b(re-?identif|identify (the|a|which) (person|individual|patient)|who is the|names? of (the )?patients?|list of patients|row[- ]level)\b"),
    ("insurance_pricing", r"\b(insur|premium|underwrit|actuar)"),
    ("marketing", r"\b(market|advertis|target(ed)? (ads?|customers?)|sales lead)"),
    ("law_enforcement", r"\b(police|law enforcement|prosecut|immigration|deport)"),
    ("clinical_guideline_review", r"\b(guideline|protocol review|clinical practice)\b"),
]

_model: Pipeline | None = None


def model() -> Pipeline:
    global _model
    if _model is None:
        _model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english"),
                               LogisticRegression(C=5.0, max_iter=2000))
        _model.fit([q for q, _ in TRAIN], [r for _, r in TRAIN])
    return _model


def classify(question: str) -> tuple[str, float, dict[str, float]]:
    m = model()
    probs = m.predict_proba([question])[0]
    dist = {str(c): round(float(p), 3) for c, p in zip(m.classes_, probs)}
    best = max(dist, key=dist.get)
    return str(best), dist[best], dist


def explain(question: str) -> dict:
    """Why the classifier chose what it chose: per-class probability and the question words that pushed each class."""
    m = model()
    tf, lr = m.named_steps["tfidfvectorizer"], m.named_steps["logisticregression"]
    x = tf.transform([question]).toarray()[0]
    names = tf.get_feature_names_out()
    recipe, conf, dist = classify(question)
    words = {}
    for i, cls in enumerate(lr.classes_):
        contrib = x * lr.coef_[i]
        order = [int(j) for j in contrib.argsort()[::-1] if contrib[j] > 0][:6]
        words[str(cls)] = [{"word": str(names[j]), "weight": round(float(contrib[j]), 3)} for j in order]
    return {"chosen": recipe, "confidence": conf, "distribution": dist, "words": words,
            "purpose": purpose_category(question), "min_confidence": PLANNER_MIN_CONFIDENCE,
            "method": "TF-IDF (1-2 grams) + multinomial logistic regression; no language model"}


def find_groups(question: str, groups: list[str]) -> list[str]:
    q = question.lower()
    found = []
    for g in groups:
        for syn in sorted(GROUP_SYNONYMS.get(g, [g.lower()]) + [g.lower()], key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}\b", q):
                found.append((q.index(syn), g))
                break
    return [g for _, g in sorted(found)]


def purpose_category(question: str) -> str:
    q = question.lower()
    for cat, rx in PURPOSE_RULES:
        if re.search(rx, q):
            return cat
    return "public_health_research"


def ml_plan(question: str, home: str) -> tuple[Plan, float]:
    """Route the question and fill every slot from the catalog. Returns the plan and the recipe confidence."""
    recipe, conf, dist = classify(question)
    cat = purpose_category(question)
    others = [l for l in LABS if l != home]
    home_groups = LABS[home]["groups"]
    focus = ref = None
    ext: list[str] = []
    local: list[str] = []
    remote: list[str] = []
    assumptions = [f"ML planner: recipe '{recipe}' with confidence {conf:.2f} (distribution {dist}).",
                   "Local vs remote decided from the catalog: which labs publish the groups, variables and measured values the recipe needs."]
    scope: list[str] = []
    if conf < PLANNER_MIN_CONFIDENCE:
        recipe = "unsupported"
        assumptions.append(f"Confidence below {PLANNER_MIN_CONFIDENCE}: planner abstained.")
    if recipe == "event_mix_comparison":
        named = find_groups(question, home_groups)
        focus = next((g for g in named if g != "White"), named[0] if named else None)
        ref = next((g for g in named if g != focus), "White" if "White" in home_groups and focus != "White" else None)
        region = FOCUS_REGION.get(focus or "", None)
        ext = [l for l in others if LABS[l]["region"] == region] or [l for l in others if LABS[l]["region"] != LABS[home]["region"]][:1]
        local = [f"Stroke vs heart-attack mix of {focus} vs {ref} adults inside {home} (age- and sex-stratified, through the home gate)."]
        remote = [f"Whether that mix matches the {focus} home population: needs a lab in region '{region or 'another region'}' "
                  f"({', '.join(ext)}), then a transportability test (ratio of odds ratios)."]
        if not focus:
            scope.append("No focus group named in the question that the home lab publishes.")
    elif recipe == "age_gradient":
        ext = [l for l in ("US_BRFSS_2023", "BRAZIL_PNS_2019") if l != home] or others[:2]
        local = ["Hypertension -> stroke odds ratio in young adults (19-44) and at 65+ inside the home lab, if its gate releases them."]
        remote = [f"Pooled random-effects estimate and consistency check (I^2) need at least two labs: {', '.join(ext)} "
                  "(the largest interview surveys in the catalog)."]
    elif recipe == "undiagnosed_comparison":
        if LABS[home]["measured_bp"]:
            local = ["Share of adults with a measured high reading who were never diagnosed, by age band, inside the home lab."]
        else:
            scope.append(f"{home} has no measured blood pressure, so this recipe cannot run from it.")
        ext = [l for l in others if LABS[l]["measured_bp"]]
        remote = [f"A comparator population with measured blood pressure: {', '.join(ext) or 'none in the catalog'}."]
    else:
        scope.append("The question does not match any of the network's approved analyses.")
    return Plan(restated_question=question.strip(), purpose="Research question routed by the deterministic planner.",
                purpose_category=cat, recipe=recipe, focus_group=focus, reference_group=ref, external_labs=ext,
                can_establish_locally=local, needs_external_help=remote,
                document_insights=["Documents are not read by the ML planner; the pre-registered protocol and the evidence pack carry them."],
                assumptions=assumptions, out_of_scope=scope), conf


def evaluate() -> dict:
    """Leave-one-out accuracy on the labelled questions plus a held-out paraphrase set. Deterministic."""
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    X, y = [q for q, _ in TRAIN], [r for _, r in TRAIN]
    fresh = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"), LogisticRegression(C=5.0, max_iter=2000))
    loo = cross_val_score(fresh, X, y, cv=LeaveOneOut()).mean()
    held = [
        ("Our clinic's Asian patients: are their events mostly strokes, and does Korean evidence carry over?", "event_mix_comparison"),
        ("Does hypertension raise stroke odds more in people under 45 than over 65?", "age_gradient"),
        ("How many people with a high measured reading have never had a hypertension diagnosis, compared with the US?", "undiagnosed_comparison"),
        ("What is the effect of exercise on depression?", "unsupported"),
        ("Is the stroke share of cardiovascular events higher among Black than White adults?", "event_mix_comparison"),
        ("Is the blood pressure to stroke link weaker in the elderly?", "age_gradient"),
        ("Unrecognised hypertension among young adults: are we an outlier?", "undiagnosed_comparison"),
        ("Which drug lowers LDL the most?", "unsupported"),
    ]
    hits = sum(classify(q)[0] == r for q, r in held)
    return {"labelled_questions": len(TRAIN), "leave_one_out_accuracy": round(float(loo), 3),
            "held_out": len(held), "held_out_correct": hits, "min_confidence": PLANNER_MIN_CONFIDENCE}
