"""Federated analysis recipes. Deterministic: the AI picks a recipe and parameters, never the maths.

Every recipe first asks the HOME lab (free, still through its gate), records what it can establish alone,
then pays external labs only for what the home lab cannot answer.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import stats
from .audit import AuditLog
from .catalog import LABS
from .lab import TRACE_MEANING, LabNode
from .payments import BudgetExceeded, SimulatedX402
from .schemas import Claim, Evidence, LabQuery, LabResult


@dataclass
class Network:
    labs: dict[str, LabNode]
    home: str
    asker: str
    pay: SimulatedX402
    audit: AuditLog
    tier: str = "evidence"
    timeline: list = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    local_steps: list = field(default_factory=list)
    on_event: object = None
    _n: int = 0

    def _log(self, entry: dict):
        self.timeline.append(entry)
        if self.on_event:
            self.on_event(entry)

    # ------------------------------------------------------------ one round-trip to one lab
    def ask(self, lab_id: str, operation: str, population: dict, outcome=None, purpose="") -> LabResult:
        self._n += 1
        qid = f"q{self._n:02d}-{lab_id.split('_')[0].lower()}"
        local = lab_id == self.home
        node = self.labs[lab_id]
        remote = getattr(node, "pays_itself", False)  # real x402: the lab's HTTP endpoint takes payment itself
        receipt = None
        if not local and not remote:
            req = self.pay.quote(lab_id, qid, self.tier)
            self._log({"step": "payment_required", "lab": lab_id, "query_id": qid, "x402": req.__dict__})
            try:
                receipt = self.pay.escrow(req, lab_id, qid)
            except BudgetExceeded as e:
                self._log({"step": "budget_blocked", "lab": lab_id, "query_id": qid, "reason": str(e)})
                self.audit.add("budget_blocked", {"lab": lab_id, "query_id": qid})
                return LabResult(query_id=qid, lab_id=lab_id, operation=operation, status="refused", refusal="budget_exhausted")
            self._log({"step": "escrow_held", "lab": lab_id, "query_id": qid, "receipt": receipt.receipt_id})
        q = LabQuery(query_id=qid, asker=self.asker, operation=operation, population=population,
                     exposure="hypertension" if operation == "logistic_or" else None, outcome=outcome, tier=self.tier)
        self.audit.add("query_sent", {"lab": lab_id, "local": local, "purpose": purpose, "query": q.model_dump()})
        t0 = time.perf_counter()
        sla = LABS[lab_id]["policy"]["sla_seconds"]
        try:
            if os.environ.get("ENVOY_INJECT_FAILURE") == lab_id and not local:
                raise TimeoutError("injected failure for demo")
            res = LabResult.model_validate_json(node.handle(q.model_dump_json()))
            if time.perf_counter() - t0 > sla:
                raise TimeoutError(f"exceeded SLA of {sla}s")
        except RuntimeError as e:
            if "payment failed" not in str(e):
                raise
            res = LabResult(query_id=qid, lab_id=lab_id, operation=operation, status="refused", refusal="payment_failed")
            self.audit.add("x402_payment_failed", {"lab": lab_id, "query_id": qid, "error": str(e)[:200],
                                                   "attempts": getattr(node, "attempts", None)})
        except Exception as e:  # any lab-side failure is a verifiable SLA breach, never silently ignored
            res = LabResult(query_id=qid, lab_id=lab_id, operation=operation, status="refused", refusal="sla_failure")
            self.audit.add("sla_failure", {"lab": lab_id, "query_id": qid, "error": str(e)[:200]})
        self.audit.add("result_received", res.model_dump())
        if remote and not local and node.last_payment:
            from .catalog import price_for
            r = self.pay.record_x402(lab_id, qid, price_for(lab_id, self.tier), node.last_payment)
            self.audit.add("x402_payment", {"lab": lab_id, "query_id": qid, **node.last_payment})
            self._log({"step": "x402_" + r.status.split(" ")[0], "lab": lab_id, "query_id": qid,
                                  "transaction": node.last_payment.get("transaction"), "network": node.last_payment.get("network")})
            receipt_status = r.status
        if receipt:
            if res.refusal == "sla_failure":
                pen = self.pay.penalise(receipt, LABS[lab_id]["policy"]["stake_usd"])
                self.audit.add("penalty_applied", pen)
            else:
                (self.pay.release if res.status == "ok" else self.pay.refund)(receipt)
            self.audit.add("payment_" + receipt.status, {"receipt": receipt.receipt_id, "lab": lab_id, "usd": receipt.amount_usd})
        trace = [{"op": t.op, "meaning": TRACE_MEANING[t.op], "rows": t.rows, "ms": t.ms, "iterations": t.iterations}
                 for t in res.compute_trace]
        self._log({"step": "gate_" + res.status, "lab": lab_id, "query_id": qid, "local": local,
                              "purpose": purpose, "refusal": res.refusal,
                              "payment": receipt.status if receipt else ("local (free)" if local else (node.last_payment or {}).get("status", "?")),
                              "compute_ms": round(sum(t["ms"] for t in trace), 2), "compute_trace": trace})
        return res

    # ------------------------------------------------------------ evidence bookkeeping
    def add(self, kind, description, lab=None, est=None, status="ok", refusal=None, detail=None, i2=None) -> str:
        eid = f"E{len(self.evidence) + 1}"
        e = Evidence(evidence_id=eid, kind=kind, lab=lab, description=description, status=status, refusal=refusal,
                     estimate=_r(est, "estimate"), ci_low=_r(est, "ci_low"), ci_high=_r(est, "ci_high"),
                     p_value=_r(est, "p", 4), i2=i2, detail=detail or {})
        self.evidence.append(e)
        return eid

    def add_result(self, res: LabResult, description: str) -> str:
        detail = {"n_bucket": res.n_bucket, "cases_exposed": res.cases_exposed, "strata_released": len(res.strata or []),
                  "strata_suppressed": res.strata_suppressed, "result_hash": res.result_hash, "checks": res.checks,
                  "min_cell": min((min(s.a, s.b) for s in res.strata), default=None) if res.strata else None}
        est = {"estimate": res.estimate, "ci_low": res.ci_low, "ci_high": res.ci_high} if res.estimate else None
        if res.estimate is not None:
            est["p"] = stats.p_from_z(res.log_estimate, res.se)
        return self.add("lab_result", description, lab=res.lab_id, est=est, status=res.status,
                        refusal=res.refusal, detail=detail)

    def local(self, what: str, status: str, detail: str):
        self.local_steps.append({"question_part": what, "status": status, "detail": detail})


def _r(d, k, nd=3):
    if not d or d.get(k) is None:
        return None
    return round(float(d[k]), nd)


def _strata(res: LabResult) -> dict:
    return {s.label: (s.a, s.b) for s in (res.strata or [])}


def name(lab_id: str) -> str:
    return LABS[lab_id]["country"]


# ======================================================================== recipe 1
def event_mix_comparison(net: Network, focus_group: str, reference_group: str, external_labs: list[str]) -> list[Claim]:
    """Among people who had a stroke OR a heart attack (not both), are events more often strokes?"""
    claims = []
    pop = lambda g: {"age_min": 19, "age_max": 120, "sex": "all", "group": g}
    r_focus = net.ask(net.home, "event_split_strata", pop(focus_group), purpose=f"event mix, {focus_group} (home)")
    r_ref = net.ask(net.home, "event_split_strata", pop(reference_group), purpose=f"event mix, {reference_group} (home)")
    e_focus = net.add_result(r_focus, f"{net.home}: strokes (a) vs heart attacks (b) among {focus_group} adults, by age band x sex")
    e_ref = net.add_result(r_ref, f"{net.home}: strokes (a) vs heart attacks (b) among {reference_group} adults, by age band x sex")

    local_mh = None
    if r_focus.status == "ok" and r_ref.status == "ok":
        local_mh = stats.mantel_haenszel(_strata(r_focus), _strata(r_ref))
    e_loc = net.add("derived", f"Age/sex-adjusted odds that an event is a stroke: {focus_group} vs {reference_group} within {net.home} "
                    f"(Mantel-Haenszel, strata {', '.join(local_mh['strata_used']) if local_mh else 'none'})", est=local_mh,
                    status="ok" if local_mh and local_mh.get("estimate") else "refused")
    c, why = stats.ceiling_for_effect(local_mh)
    claims.append(Claim(claim_id="C1", statement=f"In {name(net.home)}, cardiovascular events among {focus_group} adults are more often strokes "
                        f"than among {reference_group} adults.", evidence_ids=[e_focus, e_ref, e_loc], ceiling=c, ceiling_reason=why,
                        established_locally=True))
    net.local(f"Is the {focus_group} event mix more stroke-heavy than {reference_group}?", c, why)
    focus_n = sum(a + b for a, b in _strata(r_focus).values())
    net.local(f"Can {net.home} characterise a specific East Asian population on its own?", "needs_help",
              f"Its {focus_group} category mixes many origins and has only {focus_n} released events; "
              f"{r_focus.strata_suppressed} strata were suppressed by the home gate.")

    k = 2
    for ext in external_labs:
        r_ext = net.ask(ext, "event_split_strata", pop("all"), purpose="event mix, whole population (external)")
        e_ext = net.add_result(r_ext, f"{ext}: strokes (a) vs heart attacks (b), all adults, by age band x sex")
        mh = stats.mantel_haenszel(_strata(r_ext), _strata(r_ref)) if r_ext.status == "ok" and r_ref.status == "ok" else None
        e_mh = net.add("derived", f"Odds that an event is a stroke: {ext} population vs {net.home} {reference_group} (Mantel-Haenszel)",
                       est=mh, status="ok" if mh and mh.get("estimate") else "refused")
        c, why = stats.ceiling_for_effect(mh)
        claims.append(Claim(claim_id=f"C{k}", statement=f"Cardiovascular events in {name(ext)} are more often strokes than among "
                            f"{reference_group} adults in {name(net.home)}.", evidence_ids=[e_ext, e_ref, e_mh], ceiling=c,
                            ceiling_reason=why, established_locally=False, cross_population=True))
        k += 1
        if local_mh and local_mh.get("estimate") and mh and mh.get("estimate"):
            rr = stats.ratio(mh["log"], mh["se"], local_mh["log"], local_mh["se"])
            e_rr = net.add("derived", f"Transportability: ({ext} vs {reference_group}) / ({focus_group} vs {reference_group}); "
                           "1 = same size of effect (conservative variance)", est=rr)
            c, why = stats.ceiling_for_transfer(rr)
            claims.append(Claim(claim_id=f"C{k}", statement=f"The size of the {name(ext)} effect can be applied directly to {focus_group} "
                                f"patients in {name(net.home)}.", evidence_ids=[e_mh, e_loc, e_rr], ceiling=c, ceiling_reason=why,
                                established_locally=False, cross_population=True))
            k += 1
    return claims


# ======================================================================== recipe 2
def age_gradient(net: Network, external_labs: list[str], outcome: str = "stroke") -> list[Claim]:
    """Is hypertension a stronger marker of the outcome in young adults (19-44) than at 65+?"""
    young, old, ratios = {}, {}, {}
    ev_y, ev_o = [], []
    for lab in [net.home] + external_labs:
        ry = net.ask(lab, "logistic_or", {"age_min": 19, "age_max": 44}, outcome=outcome, purpose="young adults 19-44")
        ro = net.ask(lab, "logistic_or", {"age_min": 65, "age_max": 120}, outcome=outcome, purpose="adults 65+")
        ev_y.append(net.add_result(ry, f"{lab}: odds ratio hypertension -> {outcome}, ages 19-44 (adj. age, sex)"))
        ev_o.append(net.add_result(ro, f"{lab}: odds ratio hypertension -> {outcome}, ages 65+ (adj. age, sex)"))
        if ry.status == "ok":
            young[lab] = (ry.log_estimate, ry.se)
        if ro.status == "ok":
            old[lab] = (ro.log_estimate, ro.se)
        if ry.status == "ok" and ro.status == "ok":
            ratios[lab] = stats.ratio(ry.log_estimate, ry.se, ro.log_estimate, ro.se)
        if lab == net.home:
            st_y = "established" if ry.status == "ok" else "needs_help"
            net.local(f"Hypertension -> {outcome} odds ratio in young adults", st_y,
                      "Computed locally." if ry.status == "ok" else f"Home gate refused ({ry.refusal}): too few young cases.")
            net.local(f"Hypertension -> {outcome} odds ratio at 65+", "established" if ro.status == "ok" else "needs_help",
                      "Computed locally." if ro.status == "ok" else f"Home gate refused ({ro.refusal}).")
    claims = []
    my = stats.dersimonian_laird([v[0] for v in young.values()], [v[1] for v in young.values()])
    e_my = net.add("derived", f"Random-effects pooled OR, ages 19-44, labs: {', '.join(young) or 'none'}", est=my or None,
                   i2=round(my.get("I2", 0), 3) if my else None, detail={"Q_p": round(my.get("Q_p", 1), 4)} if my else {})
    c, why = stats.ceiling_for_pooled(my, len(young)) if my else ("inconclusive", "No lab could release a young-adult estimate.")
    claims.append(Claim(claim_id="C1", statement=f"In adults aged 19-44, diagnosed hypertension is associated with higher odds of {outcome}, "
                        "consistently across labs.", evidence_ids=ev_y + [e_my], ceiling=c, ceiling_reason=why,
                        established_locally=net.home in young and len(young) == 1))
    mr = stats.dersimonian_laird([r["log"] for r in ratios.values()], [r["se"] for r in ratios.values()])
    e_mr = net.add("derived", f"Pooled ratio of ORs (19-44 / 65+), labs: {', '.join(ratios) or 'none'}; >1 = stronger in young",
                   est=mr or None, i2=round(mr.get("I2", 0), 3) if mr else None)
    c, why = stats.ceiling_for_pooled(mr, len(ratios)) if mr else ("inconclusive", "No lab had both age groups.")
    claims.append(Claim(claim_id="C2", statement=f"Hypertension is a stronger marker of {outcome} in young adults (19-44) than in adults 65+.",
                        evidence_ids=ev_y + ev_o + [e_mr], ceiling=c, ceiling_reason=why, established_locally=False))
    mo = stats.dersimonian_laird([v[0] for v in old.values()], [v[1] for v in old.values()])
    e_mo = net.add("derived", f"Random-effects pooled OR, ages 65+, labs: {', '.join(old)}", est=mo or None,
                   i2=round(mo.get("I2", 0), 3) if mo else None, detail={"Q_p": round(mo.get("Q_p", 1), 4)} if mo else {})
    c, why = stats.ceiling_for_pooled(mo, len(old)) if mo else ("inconclusive", "No 65+ estimates.")
    claims.append(Claim(claim_id="C3", statement=f"In adults 65+, labs agree on how strongly hypertension is associated with {outcome}.",
                        evidence_ids=ev_o + [e_mo], ceiling=c, ceiling_reason=why, established_locally=False))
    return claims


# ======================================================================== recipe 3
def undiagnosed_comparison(net: Network, external_labs: list[str]) -> list[Claim]:
    """Among adults with a high measured blood pressure, how many were never diagnosed?"""
    pop = {"age_min": 20, "age_max": 120}
    res = {}
    for lab in [net.home] + external_labs:
        r = net.ask(lab, "undiagnosed_strata", pop, purpose="high reading: undiagnosed (a) vs diagnosed (b)")
        res[lab] = r
        eid = net.add_result(r, f"{lab}: among adults with measured BP >= 140/90, never diagnosed (a) vs diagnosed (b), by age band")
        res[lab + "_eid"] = eid
    claims = []
    h = res[net.home]
    hs = _strata(h)
    if "20-44" in hs:
        u, d = hs["20-44"]
        p, lo, hi = stats.wilson(u, u + d)
        e = net.add("derived", f"{net.home}: share of 20-44-year-olds with a high reading who were never diagnosed (Wilson CI)",
                    est={"estimate": p, "ci_low": lo, "ci_high": hi}, detail={"undiagnosed": u, "total_high": u + d,
                                                                              "scale": "proportion", "null": 0.5})
        c = "supported" if lo > 0.5 else ("not_supported" if hi < 0.5 else "inconclusive")
        why = f"Share {p:.0%} (95% CI {lo:.0%}-{hi:.0%}) {'exceeds' if lo > 0.5 else 'does not clearly exceed'} one half."
        claims.append(Claim(claim_id="C1", statement=f"In {name(net.home)}, most young adults (20-44) with a high blood pressure reading have "
                            "never been diagnosed.", evidence_ids=[res[net.home + '_eid'], e], ceiling=c, ceiling_reason=why,
                            established_locally=True))
        net.local("Share of young adults with high readings who are undiagnosed", c, why)
    net.local("Is that unusually high? (needs a comparison population with measured BP)", "needs_help",
              "Requires another lab with examination blood pressure.")
    k = 2
    for ext in external_labs:
        r = res[ext]
        if r.status != "ok" or h.status != "ok":
            claims.append(Claim(claim_id=f"C{k}", statement=f"High readings are more often undiagnosed in {name(net.home)} than in {name(ext)}.",
                                evidence_ids=[res[ext + '_eid']], ceiling="inconclusive", ceiling_reason="A lab refused.",
                                established_locally=False, cross_population=True)); k += 1
            continue
        es = _strata(r)
        young = stats.mantel_haenszel({"20-44": hs["20-44"]}, {"20-44": es["20-44"]}) if "20-44" in hs and "20-44" in es else None
        e_y = net.add("derived", f"Odds of being undiagnosed, ages 20-44: {net.home} vs {ext}", est=young)
        c, why = stats.ceiling_for_effect(young)
        claims.append(Claim(claim_id=f"C{k}", statement=f"Among young adults (20-44) with high readings, being undiagnosed is more common in "
                            f"{name(net.home)} than in {name(ext)}.", evidence_ids=[res[net.home + '_eid'], res[ext + '_eid'], e_y],
                            ceiling=c, ceiling_reason=why, established_locally=False, cross_population=True)); k += 1
        mh = stats.mantel_haenszel(hs, es)
        e_mh = net.add("derived", f"Age-adjusted odds of being undiagnosed (Mantel-Haenszel over age bands): {net.home} vs {ext}", est=mh)
        c, why = stats.ceiling_for_effect(mh)
        claims.append(Claim(claim_id=f"C{k}", statement=f"Across ages, high readings are more often undiagnosed in {name(net.home)} than in {name(ext)}.",
                            evidence_ids=[res[net.home + '_eid'], res[ext + '_eid'], e_mh], ceiling=c, ceiling_reason=why,
                            established_locally=False, cross_population=True)); k += 1
    return claims


RECIPES = {"event_mix_comparison": event_mix_comparison, "age_gradient": age_gradient,
           "undiagnosed_comparison": undiagnosed_comparison}
