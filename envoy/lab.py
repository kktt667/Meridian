"""A lab node. Holds ONE dataset, answers only allowlisted operations, and enforces the gate.

The coordinator talks to a lab only through `handle(json_str) -> json_str`, so the boundary is the
same whether the lab runs in this process (demo) or behind HTTP on another machine.
Every vectorised operation is recorded as a TraceStep (code, size bucket, milliseconds): no values.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .catalog import LABS
from .config import LAB_DATA_DIR, OVERLAP_WINDOW, QUERY_BUDGET
from .schemas import LabQuery, LabResult, Stratum, TraceStep

BANDS = [(19, 44, "19-44"), (45, 64, "45-64"), (65, 200, "65+")]

TRACE_MEANING = {
    "validate_query": "Checked the request against the allowlisted query schema",
    "check_budget": "Checked the asker's remaining query budget",
    "check_overlap": "Compared with the asker's earlier queries to block differencing attacks",
    "filter_population": "Boolean mask over the lab's rows: age window, sex, group",
    "drop_missing": "Dropped rows with missing exposure/outcome (vectorised notna)",
    "vector_mask": "Built 0/1 vectors for the events of interest (no row leaves the lab)",
    "stratify_count": "Summed the vectors within age x sex strata",
    "suppress_small_cells": "Removed any stratum with a count below the minimum cell size",
    "fit_logistic": "Fitted logistic regression (Newton/IRLS) on the masked arrays",
    "compute_ci": "Converted the coefficient and standard error into an odds ratio and 95% CI",
    "hash_result": "Hashed the outgoing aggregate so the coordinator can prove what it received",
}


def n_bucket(n: int) -> str:
    if n >= 50000:
        return "50000+"
    if n >= 5000:
        return "5000-49999"
    if n >= 500:
        return "500-4999"
    return "50-499" if n >= 50 else "<50"


@dataclass
class LabNode:
    lab_id: str
    df: pd.DataFrame = field(repr=False)
    history: dict = field(default_factory=dict)  # asker -> list[LabQuery]
    _trace: list = field(default_factory=list, repr=False)
    _t: float = 0.0

    @classmethod
    def load(cls, lab_id: str) -> "LabNode":
        df = pd.read_csv(LAB_DATA_DIR / f"{lab_id}.csv", low_memory=False)  # each node reads only its own file
        return cls(lab_id=lab_id, df=df)

    @property
    def policy(self) -> dict:
        return LABS[self.lab_id]["policy"]

    def preflight(self, request_json: str) -> dict:
        """Dry run of the gate: can this lab answer? Releases only ok/refusal code, never numbers, and does not
        count against the asker's budget. This is the lab's evidence-based self-evaluation."""
        q = LabQuery.model_validate_json(request_json)
        saved, self._trace, self._t = self.history.get(q.asker, []).copy(), [], time.perf_counter()
        try:
            res = self._run(q.model_copy(update={"asker": "__preflight__"}))
        finally:
            self.history.pop("__preflight__", None)
            self.history[q.asker] = saved
        return {"lab_id": self.lab_id, "operation": q.operation, "can_answer": res.status == "ok", "refusal": res.refusal}

    @property
    def size_bucket(self) -> str:
        return n_bucket(len(self.df))

    def _mark(self, op: str, rows: int | None = None, iterations: int | None = None):
        now = time.perf_counter()
        self._trace.append(TraceStep(op=op, rows=n_bucket(rows) if rows is not None else None,
                                     ms=round((now - self._t) * 1000, 3), iterations=iterations))
        self._t = now

    # -------------------------------------------------------------- boundary
    def handle(self, request_json: str) -> str:
        self._trace, self._t = [], time.perf_counter()
        q = LabQuery.model_validate_json(request_json)
        self._mark("validate_query")
        res = self._run(q)
        self._mark("hash_result")
        res.compute_trace = list(self._trace)
        res.result_hash = hashlib.sha256(res.model_dump_json(exclude={"result_hash", "compute_trace"}).encode()).hexdigest()[:16]
        return LabResult.model_validate_json(res.model_dump_json()).model_dump_json()  # re-validate before it leaves

    # -------------------------------------------------------------- gate + dispatch
    def _refuse(self, q: LabQuery, code: str, checks=None) -> LabResult:
        return LabResult(query_id=q.query_id, lab_id=self.lab_id, operation=q.operation, status="refused",
                         refusal=code, checks=checks or ["schema_valid"])

    def _overlaps(self, q: LabQuery) -> bool:
        p = q.population
        for old in self.history.get(q.asker, []):
            o = old.population
            same_shape = (old.operation == q.operation and old.outcome == q.outcome and o.sex == p.sex and o.group == p.group)
            d = abs(o.age_min - p.age_min) + abs(o.age_max - p.age_max)
            if same_shape and 0 < d <= OVERLAP_WINDOW:
                return True
        return False

    def _population(self, q: LabQuery) -> pd.DataFrame:
        p = q.population
        m = (self.df.age >= p.age_min) & (self.df.age <= p.age_max)
        if p.sex != "all":
            m &= self.df.male == (1 if p.sex == "male" else 0)
        if p.group != "all":
            m &= self.df.group == p.group
        return self.df[m]

    def _run(self, q: LabQuery) -> LabResult:
        hist = self.history.setdefault(q.asker, [])
        self._mark("check_budget")
        if len(hist) >= QUERY_BUDGET:
            return self._refuse(q, "budget_exhausted")
        overlap = self._overlaps(q)
        self._mark("check_overlap")
        hist.append(q)
        if overlap:
            return self._refuse(q, "overlap_detected", ["schema_valid", "budget_ok"])
        if q.population.group != "all" and q.population.group not in LABS[self.lab_id]["groups"]:
            return self._refuse(q, "group_unavailable")
        if q.operation == "undiagnosed_strata" and not LABS[self.lab_id]["measured_bp"]:
            return self._refuse(q, "variable_unavailable")
        d = self._population(q)
        self._mark("filter_population", len(self.df))
        if len(d) < self.policy["min_population"]:
            return self._refuse(q, "min_population", ["schema_valid", "budget_ok", "no_overlap"])
        return getattr(self, f"_op_{q.operation}")(q, d)

    # -------------------------------------------------------------- operations
    def _op_logistic_or(self, q: LabQuery, d: pd.DataFrame) -> LabResult:
        y, x = q.outcome, q.exposure or "hypertension"
        d = d[d[y].notna() & d[x].notna()]
        self._mark("drop_missing", len(d))
        checks = ["schema_valid", "budget_ok", "no_overlap", "min_population_ok"]
        yv, xv = d[y].to_numpy(float), d[x].to_numpy(float)
        cases_exposed = int((yv * xv).sum())
        self._mark("vector_mask", len(d))
        if len(d) < self.policy["min_population"]:
            return self._refuse(q, "min_population", checks[:3])
        if cases_exposed < self.policy["min_cases"] or int(yv.sum()) - cases_exposed < self.policy["min_cell"]:
            return self._refuse(q, "min_cases", checks)
        X = np.column_stack([np.ones(len(d)), xv, d.age.to_numpy(float), d.male.to_numpy(float)])
        m = sm.Logit(yv, X).fit(disp=0)
        self._mark("fit_logistic", len(d), int(m.mle_retvals.get("iterations", 0)))
        b, se = float(m.params[1]), float(m.bse[1])
        out = LabResult(query_id=q.query_id, lab_id=self.lab_id, operation=q.operation, status="ok",
                        n_bucket=n_bucket(len(d)), cases_exposed=cases_exposed,
                        log_estimate=round(b, 4), se=round(se, 4), estimate=round(float(np.exp(b)), 3),
                        ci_low=round(float(np.exp(b - 1.96 * se)), 3), ci_high=round(float(np.exp(b + 1.96 * se)), 3),
                        checks=checks + ["min_cases_ok", "covariates_age_sex", "unweighted"])
        self._mark("compute_ci")
        return out

    def _strata(self, q: LabQuery, d: pd.DataFrame, va: np.ndarray, vb: np.ndarray, by_sex: bool, bands=BANDS) -> LabResult:
        checks = ["schema_valid", "budget_ok", "no_overlap", "min_population_ok"]
        age, male = d.age.to_numpy(), d.male.to_numpy()
        cells = []
        for lo, hi, lab in bands:
            for sx in ([1, 0] if by_sex else [None]):
                m = (age >= lo) & (age <= hi)
                if sx is not None:
                    m &= male == sx
                cells.append((f"{lab}|{'M' if sx == 1 else 'F'}" if sx is not None else lab, int(va[m].sum()), int(vb[m].sum())))
        self._mark("stratify_count", len(d))
        out, suppressed = [], 0
        for label, a, b in cells:
            if a == 0 and b == 0:
                continue
            if a < self.policy["min_cell"] or b < self.policy["min_cell"]:
                suppressed += 1
                continue
            out.append(Stratum(label=label, a=a, b=b))
        self._mark("suppress_small_cells")
        if not out:
            return self._refuse(q, "min_cell_all_strata", checks)
        return LabResult(query_id=q.query_id, lab_id=self.lab_id, operation=q.operation, status="ok",
                         n_bucket=n_bucket(len(d)), strata=out, strata_suppressed=suppressed,
                         checks=checks + (["cells_suppressed"] if suppressed else []) + ["unweighted"])

    def _op_event_split_strata(self, q: LabQuery, d: pd.DataFrame) -> LabResult:
        # People who have had exactly one of stroke / heart attack: a = strokes, b = heart attacks.
        d = d[d.stroke.notna() & d.heart_attack.notna()]
        self._mark("drop_missing", len(d))
        s, h = d.stroke.to_numpy(), d.heart_attack.to_numpy()
        one = (s + h) == 1
        va, vb = (s == 1) & one, (h == 1) & one
        self._mark("vector_mask", len(d))
        return self._strata(q, d, va, vb, by_sex=True)

    def _op_undiagnosed_strata(self, q: LabQuery, d: pd.DataFrame) -> LabResult:
        # Among people with a high measured reading: a = never diagnosed, b = diagnosed.
        d = d[d.sbp.notna() & d.dbp.notna() & d.hypertension.notna()]
        self._mark("drop_missing", len(d))
        high = (d.sbp.to_numpy() >= 140) | (d.dbp.to_numpy() >= 90)
        dx = d.hypertension.to_numpy()
        va, vb = high & (dx == 0), high & (dx == 1)
        self._mark("vector_mask", len(d))
        bands = [(max(q.population.age_min, 20), 44, "20-44"), (45, 64, "45-64"), (65, 200, "65+")]
        return self._strata(q, d, va, vb, by_sex=False, bands=bands)


def load_all(lab_ids=None) -> dict[str, LabNode]:
    ids = lab_ids or list(LABS)
    return {i: LabNode.load(i) for i in ids if (LAB_DATA_DIR / f"{i}.csv").exists()}
