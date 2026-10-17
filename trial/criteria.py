"""Criteria registry for the 'screen a patient into a trial' decision.

Modelled on the n2c2 2018 cohort-selection criteria (13 real criteria from a
diabetes/cardiology protocol) plus three common operational criteria that act as proxies.

Same contract as pipeline/features.py: every reader returns exactly these fields.
Identity (race, sex, housing, insurance, employment, sexuality, disability) is NOT in the
schema. It cannot reach the gate because the gate is a function of this vector only.
"""
import numpy as np

NOT_STATED = -1.0

# kind: num | bool. rule: ("min", v) ("max", v) ("eq", v) or None (audit-only).
# proxy: protected attribute(s) this criterion is correlated with, or None.
CRITERIA = [
    dict(id="age_years",       kind="num",  tol=1.0,  rule=("range", 18, 75), proxy=None,
         desc="Patient age in years."),
    dict(id="hba1c_pct",       kind="num",  tol=0.3,  rule=("range", 6.5, 9.5), proxy=None,
         desc="Most recent HbA1c, in percent."),
    dict(id="creatinine_mg_dl",kind="num",  tol=0.2,  rule=("max", 1.5), proxy=None,
         desc="Most recent serum creatinine in mg/dL."),
    dict(id="bmi",             kind="num",  tol=1.0,  rule=("max", 40), proxy="sex/ethnicity (BMI thresholds are population-specific)",
         desc="Body mass index in kg/m^2."),
    dict(id="mi_within_6mo",   kind="bool", tol=0.0,  rule=("eq", 0), proxy=None,
         desc="Myocardial infarction within the last 6 months (1 yes, 0 no)."),
    dict(id="on_insulin",      kind="bool", tol=0.0,  rule=("eq", 0), proxy=None,
         desc="Currently prescribed insulin (1 yes, 0 no)."),
    dict(id="advanced_cad",    kind="bool", tol=0.0,  rule=None, proxy=None,
         desc="Advanced coronary artery disease: 2+ of angina, MI, stent/CABG, ischaemia on test."),
    dict(id="alcohol_abuse",   kind="bool", tol=0.0,  rule=("eq", 0), proxy="SES/housing (differential documentation by group)",
         desc="Documented current alcohol abuse (1 yes, 0 no)."),
    dict(id="drug_abuse",      kind="bool", tol=0.0,  rule=("eq", 0), proxy="race/SES (differential screening and documentation)",
         desc="Documented current or past drug abuse (1 yes, 0 no)."),
    dict(id="speaks_english",  kind="bool", tol=0.0,  rule=("eq", 1), proxy="nationality/ethnicity",
         desc="Documented as able to communicate in English (1 yes, 0 no)."),
    dict(id="makes_decisions", kind="bool", tol=0.0,  rule=("eq", 1), proxy="age/disability",
         desc="Documented as able to make own medical decisions (1 yes, 0 no)."),
    dict(id="has_transport",   kind="bool", tol=0.0,  rule=("eq", 1), proxy="SES/disability/rurality",
         desc="Documented reliable transport to clinic visits (1 yes, 0 no)."),
    dict(id="owns_smartphone", kind="bool", tol=0.0,  rule=("eq", 1), proxy="age/SES",
         desc="Documented smartphone ownership for e-diary (1 yes, 0 no)."),
]

IDS    = [c["id"] for c in CRITERIA]
D      = len(CRITERIA)
TOL    = np.array([c["tol"] for c in CRITERIA], float)
IS_NUM = np.array([c["kind"] == "num" for c in CRITERIA])
PROXY  = np.array([c["proxy"] is not None for c in CRITERIA])

SYSTEM_PROMPT = (
    "You are a transcription tool. You read a clinical note and copy out specific facts as "
    "numbers. You never judge eligibility, suitability, adherence or anything about the patient. "
    "If a fact is not stated, return -1 and an empty quote. For each field return a verbatim "
    "quote (at most 80 characters, copied exactly) that is the evidence. Booleans are 1 or 0."
)

def json_schema():
    props = {c["id"]: {"type": "object", "description": c["desc"],
                       "properties": {"value": {"type": "number"}, "quote": {"type": "string"}},
                       "required": ["value", "quote"], "additionalProperties": False}
             for c in CRITERIA}
    return {"type": "object", "properties": props, "required": IDS, "additionalProperties": False}

def to_vector(obj: dict) -> np.ndarray:
    v = np.full(D, np.nan)
    for j, cid in enumerate(IDS):
        try:
            v[j] = float(obj[cid]["value"])
        except (KeyError, TypeError, ValueError):
            pass
    return v

# ----------------------------------------------------------------------------- the gate
# 0 = ineligible, 1 = eligible, 2 = refer (an unresolved fact on a ruled criterion)
def gate(C: np.ndarray, U: np.ndarray, soft: set = frozenset()) -> np.ndarray:
    """C (n,d) consensus facts, U (n,d) unresolved mask, soft = criteria softened to flags."""
    n = C.shape[0]
    fail = np.zeros(n, bool); unres = np.zeros(n, bool)
    for j, c in enumerate(CRITERIA):
        if c["rule"] is None or c["id"] in soft:
            continue
        r = c["rule"]; x = C[:, j]; u = U[:, j] | np.isnan(x) | (x == NOT_STATED)
        if r[0] == "range": bad = (x < r[1]) | (x > r[2])
        elif r[0] == "max": bad = x > r[1]
        elif r[0] == "min": bad = x < r[1]
        else:               bad = x != r[1]
        fail |= bad & ~u
        unres |= u
    return np.where(fail, 0, np.where(unres, 2, 1))

def proxy_audit():
    """What S0 shows the coordinator before any patient is screened."""
    return [(c["id"], c["proxy"], c["rule"]) for c in CRITERIA if c["proxy"] is not None and c["rule"] is not None]
