"""Gate and leak tests. Run: venv/bin/python -m pytest -q tests/"""
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from envoy.config import MIN_CELL, MIN_POPULATION
from envoy.lab import LabNode
from envoy.schemas import LabQuery, LabResult

CANARY_AGE = 99  # planted rows have an age no synthetic respondent has (19-89)


@pytest.fixture
def lab():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"age": rng.integers(19, 90, n), "male": rng.integers(0, 2, n), "group": rng.choice(["White", "Asian"], n),
                       "hypertension": rng.integers(0, 2, n).astype(float), "stroke": (rng.random(n) < 0.05).astype(float),
                       "heart_attack": (rng.random(n) < 0.05).astype(float), "sbp": rng.normal(130, 20, n),
                       "dbp": rng.normal(80, 10, n), "weight": 1.0})
    canary = df.iloc[:3].copy(); canary["age"] = CANARY_AGE; canary["stroke"] = 1.0
    node = LabNode(lab_id="US_NHANES_2017", df=pd.concat([df, canary], ignore_index=True))
    return node


def ask(node, **kw):
    q = {"query_id": "t1", "asker": "tester", "operation": "logistic_or", "population": {"age_min": 19, "age_max": 120},
         "exposure": "hypertension", "outcome": "stroke"}
    q.update(kw)
    return LabResult.model_validate_json(node.handle(json.dumps(q)))


def test_result_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        LabResult(query_id="x", lab_id="y", operation="logistic_or", status="ok", rows=[{"age": 40}])


def test_query_schema_rejects_free_text_injection():
    with pytest.raises(ValidationError):
        LabQuery(query_id="x; DROP", asker="a", operation="logistic_or", population={})
    with pytest.raises(ValidationError):
        LabQuery(query_id="x", asker="a", operation="raw_rows", population={})


def test_small_population_refused(lab):
    r = ask(lab, population={"age_min": CANARY_AGE, "age_max": 100})
    assert r.status == "refused" and r.refusal == "min_population"


def test_canary_rows_never_identifiable(lab):
    # Narrow queries around the canary are refused; nothing in any output reveals the planted rows.
    r = ask(lab, operation="event_split_strata", population={"age_min": 95, "age_max": 120}, outcome=None, exposure=None)
    assert r.status == "refused"


def test_differencing_attack_blocked(lab):
    ask(lab, query_id="a1", population={"age_min": 19, "age_max": 60})
    r = ask(lab, query_id="a2", population={"age_min": 19, "age_max": 61})
    assert r.status == "refused" and r.refusal == "overlap_detected"


def test_released_counts_respect_min_cell(lab):
    r = ask(lab, operation="event_split_strata", outcome=None, exposure=None)
    assert r.status == "ok"
    for s in r.strata:
        assert s.a >= MIN_CELL and s.b >= MIN_CELL


def test_output_contains_no_row_level_fields(lab):
    raw = lab.handle(json.dumps({"query_id": "t9", "asker": "z", "operation": "logistic_or",
                                 "population": {"age_min": 19, "age_max": 120}, "exposure": "hypertension", "outcome": "stroke"}))
    keys = set(json.loads(raw))
    assert keys <= set(LabResult.model_fields)
    assert len(raw) < 2000


def test_budget_exhausted(lab):
    from envoy.config import QUERY_BUDGET
    for i in range(QUERY_BUDGET):
        ask(lab, query_id=f"b{i}", population={"age_min": 19 + (i % 2) * 40, "age_max": 120})
    assert ask(lab, query_id="last").refusal == "budget_exhausted"


def test_min_population_constant_sane():
    assert MIN_POPULATION >= MIN_CELL
