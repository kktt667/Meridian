"""Data contracts.

Two kinds of model live here:
  * Wire models between coordinator and labs (LabQuery, LabResult). These are the security boundary:
    extra="forbid" everywhere, and a LabResult can only carry numbers, enumerated codes and IDs,
    so free text or row-level data cannot leave a lab even by accident.
  * LLM output models (Plan, Argument, EdgeCaseReport, Ballot). Every field is required (no defaults)
    so the structured-output schema is strict.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Strict = ConfigDict(extra="forbid")

Outcome = Literal["stroke", "heart_attack"]
Operation = Literal["logistic_or", "event_split_strata", "undiagnosed_strata"]
Tier = Literal["verdict", "evidence", "audit"]
RefusalCode = Literal[
    "min_population", "min_cases", "min_cell_all_strata", "budget_exhausted",
    "overlap_detected", "variable_unavailable", "group_unavailable", "sla_failure", "not_hired", "payment_failed",
]
CheckCode = Literal[
    "schema_valid", "min_population_ok", "min_cases_ok", "cells_suppressed", "budget_ok",
    "no_overlap", "covariates_age_sex", "unweighted",
]
Verdict = Literal["supported", "not_supported", "inconsistent", "inconclusive"]
STRATUM_LABELS = ("19-44|M", "19-44|F", "45-64|M", "45-64|F", "65+|M", "65+|F",
                  "19-44", "45-64", "65+", "20-44", "20-44|M", "20-44|F")


# ------------------------------------------------------------------ wire: coordinator -> lab
class Population(BaseModel):
    model_config = Strict
    age_min: int = Field(19, ge=18, le=100)
    age_max: int = Field(120, ge=18, le=120)
    sex: Literal["all", "male", "female"] = "all"
    group: str = Field("all", max_length=24, pattern=r"^[A-Za-z]+$")


class LabQuery(BaseModel):
    model_config = Strict
    query_id: str = Field(max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
    asker: str = Field(max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
    operation: Operation
    population: Population
    exposure: Optional[Literal["hypertension"]] = None
    outcome: Optional[Outcome] = None
    tier: Tier = "evidence"


# ------------------------------------------------------------------ wire: lab -> coordinator
class Stratum(BaseModel):
    model_config = Strict
    label: Literal[*STRATUM_LABELS]  # type: ignore[valid-type]
    a: int = Field(ge=0)  # first count (e.g. strokes, or undiagnosed)
    b: int = Field(ge=0)  # second count (e.g. heart attacks, or diagnosed)


TraceOp = Literal["validate_query", "check_budget", "check_overlap", "filter_population", "drop_missing",
                  "vector_mask", "stratify_count", "suppress_small_cells", "fit_logistic", "compute_ci", "hash_result"]
RowsBucket = Literal["<50", "50-499", "500-4999", "5000-49999", "50000+"]


class TraceStep(BaseModel):
    """One vectorised operation inside a lab. Codes and sizes only: the meaning is looked up by the coordinator."""
    model_config = Strict
    op: TraceOp
    rows: Optional[RowsBucket] = None
    ms: float = Field(ge=0)
    iterations: Optional[int] = Field(None, ge=0)


class LabResult(BaseModel):
    """The only thing a lab ever sends out. Allowlisted fields only."""
    model_config = Strict
    compute_trace: list[TraceStep] = []
    query_id: str
    lab_id: str
    operation: Operation
    status: Literal["ok", "refused"]
    refusal: Optional[RefusalCode] = None
    n_bucket: Optional[Literal["50-499", "500-4999", "5000-49999", "50000+"]] = None
    cases_exposed: Optional[int] = Field(None, ge=0)   # only released when >= MIN_CASES_MODEL
    log_estimate: Optional[float] = None
    se: Optional[float] = Field(None, ge=0)
    estimate: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    strata: Optional[list[Stratum]] = None
    strata_suppressed: int = 0
    checks: list[CheckCode] = []
    result_hash: str = ""


# ------------------------------------------------------------------ coordinator evidence
class Evidence(BaseModel):
    model_config = Strict
    evidence_id: str
    kind: Literal["lab_result", "derived"]
    lab: Optional[str] = None
    description: str
    estimate: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None
    i2: Optional[float] = None
    status: Literal["ok", "refused"] = "ok"
    refusal: Optional[str] = None
    detail: dict = {}


class Claim(BaseModel):
    model_config = Strict
    claim_id: str
    statement: str
    evidence_ids: list[str]
    ceiling: Verdict              # the strongest verdict the statistics allow
    ceiling_reason: str
    established_locally: bool     # could the asking lab establish this on its own?
    cross_population: bool = False  # compares or transfers across different populations (GRADE indirectness)


# ------------------------------------------------------------------ LLM outputs (all fields required)
class Plan(BaseModel):
    restated_question: str = Field(description="The research question in one precise sentence.")
    purpose: str = Field(description="Why the researcher needs this, in one sentence (stated to labs when hiring them).")
    purpose_category: Literal["public_health_research", "academic_research", "clinical_guideline_review",
                              "re_identification", "insurance_pricing", "marketing", "law_enforcement"] = Field(
        description="Honest category of the purpose. Labs refuse some categories.")
    recipe: Literal["event_mix_comparison", "age_gradient", "undiagnosed_comparison", "unsupported"] = Field(
        description="Which federated analysis answers the question.")
    focus_group: Optional[str] = Field(description="Subgroup inside the home lab to study (e.g. 'Asian'), or null.")
    reference_group: Optional[str] = Field(description="Comparison subgroup inside the home lab (e.g. 'White'), or null.")
    external_labs: list[str] = Field(description="Lab IDs from the catalog whose help is needed.")
    can_establish_locally: list[str] = Field(description="Parts of the question the home lab can answer alone.")
    needs_external_help: list[str] = Field(description="Parts that need other labs, and why.")
    document_insights: list[str] = Field(description="What the researcher's documents add (hypotheses, definitions, constraints).")
    assumptions: list[str] = Field(description="Assumptions the analysis relies on.")
    out_of_scope: list[str] = Field(description="Anything asked that the network cannot answer.")


class JobDecision(BaseModel):
    accept: bool = Field(description="True only if the job is in scope for this lab and its data can support it.")
    reason: str = Field(description="One or two sentences, factual, citing the policy or pre-flight result.")
    conditions: list[str] = Field(description="Conditions or limits on acceptance (empty if none).")


class Objection(BaseModel):
    evidence_id: str
    category: Literal["definition_mismatch", "misinterpretation", "data_quality", "out_of_scope_use", "other"]
    severity: Literal["high", "medium", "low"]
    issue: str


class StewardReview(BaseModel):
    confirmed: bool = Field(description="True if this lab's data was interpreted correctly in every claim that uses it.")
    objections: list[Objection]
    comment: str


class Narrative(BaseModel):
    headline: str = Field(description="One sentence answer to the researcher's question.")
    summary: list[str] = Field(description="One plain-English sentence per claim: verdict and why, citing evidence IDs.")
    what_would_change_it: list[str] = Field(description="Concrete evidence that would change the conclusion.")
    caveats: list[str] = Field(description="The most important limitations, from the audit.")


class Point(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(description="IDs from the evidence pack that support this point. Must not be empty.")
    strength: Literal["strong", "moderate", "weak"]


class Argument(BaseModel):
    position: Literal["for", "against"]
    summary: str
    points: list[Point]


class EdgeCase(BaseModel):
    threat: str
    category: Literal["confounding", "survivor_or_selection", "definition_mismatch", "transportability",
                      "small_sample", "measurement", "multiple_testing", "access_to_diagnosis", "other"]
    applies_to: list[str] = Field(description="Claim IDs or evidence IDs affected.")
    severity: Literal["high", "medium", "low"]
    how_to_check: str


class EdgeCaseReport(BaseModel):
    edge_cases: list[EdgeCase]


class ClaimVote(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: Literal["low", "medium", "high"]
    rationale: str
    evidence_ids: list[str]


class Ballot(BaseModel):
    votes: list[ClaimVote]
    key_caveat: str
