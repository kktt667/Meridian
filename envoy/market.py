"""Agent marketplace: discovery, pre-flight self-evaluation, job acceptance, reputation, stewards.

Design rules (anti-persuasion):
  * A lab's "pitch" is facts: its published policy, a pre-flight gate dry run on the actual job, and a
    reputation computed by the platform from the ledger of past runs. Nothing self-described is scored.
  * Policy is checked in code first. The lab's own agent may then reject a job; it can never accept one
    that the policy rejects.
  * Labs are paid per query whatever the answer is. Penalties apply only to verifiable failures (SLA).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

from .catalog import LABS, price_for
from .config import RUNS_DIR
from .lab import LabNode
from .llm import LLM, AgentFailed
from .schemas import JobDecision, LabQuery, StewardReview

LEDGER = RUNS_DIR / "ledger.jsonl"

REQUIREMENTS = {
    "event_mix_comparison": [("event_split_strata", {"age_min": 19, "age_max": 120}, None, "event mix, all adults")],
    "age_gradient": [("logistic_or", {"age_min": 19, "age_max": 44}, "stroke", "hypertension -> stroke, ages 19-44"),
                     ("logistic_or", {"age_min": 65, "age_max": 120}, "stroke", "hypertension -> stroke, ages 65+")],
    "undiagnosed_comparison": [("undiagnosed_strata", {"age_min": 20, "age_max": 120}, None, "undiagnosed among high readings")],
}


# ---------------------------------------------------------------- reputation (computed, not claimed)
def reputation(lab_id: str) -> dict:
    stats = Counter()
    refusals = Counter()
    runs = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            e = json.loads(line)
            if e["lab"] != lab_id:
                continue
            runs.add(e["run_id"])
            stats.update({k: e.get(k, 0) for k in ("queries", "ok", "sla_failures", "objections_raised", "jobs_accepted", "jobs_declined")})
            refusals.update(e.get("refusals", {}))
    q = stats["queries"]
    return {"runs": len(runs), "queries_served": q, "answered": stats["ok"],
            "answer_rate": round(stats["ok"] / q, 2) if q else None, "refusals_by_reason": dict(refusals),
            "sla_failures": stats["sla_failures"], "jobs_accepted": stats["jobs_accepted"], "jobs_declined": stats["jobs_declined"],
            "steward_objections_raised": stats["objections_raised"], "source": "platform ledger of past runs (not self-reported)"}


def record(run_id: str, net, decisions: dict, stewards: dict):
    by_lab = defaultdict(lambda: {"queries": 0, "ok": 0, "refusals": Counter(), "sla_failures": 0})
    for t in net.timeline:
        if t["step"].startswith("gate_"):
            b = by_lab[t["lab"]]
            b["queries"] += 1
            if t["step"] == "gate_ok":
                b["ok"] += 1
            elif t.get("refusal"):
                b["refusals"][t["refusal"]] += 1
                b["sla_failures"] += t["refusal"] == "sla_failure"
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        for lab in set(by_lab) | set(decisions):
            b = by_lab[lab]
            d = decisions.get(lab, {})
            f.write(json.dumps({"ts": time.time(), "run_id": run_id, "lab": lab, "queries": b["queries"], "ok": b["ok"],
                                "refusals": dict(b["refusals"]), "sla_failures": b["sla_failures"],
                                "jobs_accepted": int(d.get("accept", False)), "jobs_declined": int(d.get("accept") is False),
                                "objections_raised": len(stewards.get(lab, {}).get("objections", []))}) + "\n")


# ---------------------------------------------------------------- discovery + pre-flight
def discover(labs: dict[str, LabNode], home: str, recipe: str, purpose_category: str, asker: str) -> list[dict]:
    out = []
    for lab_id, node in labs.items():
        if lab_id == home:
            continue
        pol = LABS[lab_id]["policy"]
        policy_ok = purpose_category in pol["allowed_purposes"] and purpose_category not in pol["refused_purposes"]
        checks = []
        for i, (op, pop, outcome, label) in enumerate(REQUIREMENTS[recipe]):
            q = LabQuery(query_id=f"pf{i}", asker=asker, operation=op, population=pop,
                         exposure="hypertension" if op == "logistic_or" else None, outcome=outcome)
            r = node.preflight(q.model_dump_json())
            checks.append({"requirement": label, "can_answer": r["can_answer"], "refusal": r["refusal"]})
        n_ok = sum(c["can_answer"] for c in checks)
        out.append({"lab_id": lab_id, "country": LABS[lab_id]["country"], "region": LABS[lab_id]["region"],
                    "agent_model": pol["agent_model"], "policy_ok": policy_ok,
                    "policy_reason": "purpose allowed" if policy_ok else f"purpose '{purpose_category}' refused by lab policy",
                    "preflight": checks, "capability": "full" if n_ok == len(checks) else ("partial" if n_ok else "none"),
                    "price_per_query_usd": None, "reputation": reputation(lab_id)})
    return out


# ---------------------------------------------------------------- sealed bids + auditable selection (code, no LLM)
CAPABILITY_RANK = {"full": 2, "partial": 1, "none": 0}


def bid(card: dict, decision: dict, recipe: str, tier: str) -> dict:
    """A lab's sealed bid is facts only: its published per-query price at this tier, the job total for the
    queries this recipe needs, its pre-flight capability and its ledger reputation. Nothing self-described."""
    n_q = len(REQUIREMENTS[recipe])
    per = price_for(card["lab_id"], tier)
    rep = card["reputation"]
    return {"lab_id": card["lab_id"], "accepted": bool(decision["accept"]), "capability": card["capability"],
            "price_per_query_usd": per, "job_quote_usd": round(per * n_q, 2), "queries": n_q,
            "rate_multiplier": LABS[card["lab_id"]]["policy"]["rate_multiplier"],
            "answer_rate": rep["answer_rate"], "queries_served": rep["queries_served"], "sla_failures": rep["sla_failures"],
            "sla_seconds": LABS[card["lab_id"]]["policy"]["sla_seconds"], "stake_usd": LABS[card["lab_id"]]["policy"]["stake_usd"],
            "conditions": decision.get("conditions", [])}


def select(bids: list[dict], preferred: list[str], budget_usd: float, k: int = 2) -> tuple[list[str], dict]:
    """Rank accepted bids: capability first, then track record (answer rate; new labs count as 0.5), then price.
    Take up to k within budget. The researcher's own preferred labs win if they bid. Every outcome has a reason."""
    reasons: dict[str, str] = {}
    pool = []
    for b in bids:
        if not b["accepted"]:
            reasons[b["lab_id"]] = "declined or not offered"
        elif b["capability"] == "none":
            reasons[b["lab_id"]] = "pre-flight: cannot answer any required analysis"
        else:
            pool.append(b)
    key = lambda b: (-CAPABILITY_RANK[b["capability"]], -(b["answer_rate"] if b["answer_rate"] is not None else 0.5),
                     b["sla_failures"], b["job_quote_usd"])
    ranked = sorted(pool, key=key)
    pref = [b for b in ranked if b["lab_id"] in preferred]
    order = pref + [b for b in ranked if b["lab_id"] not in preferred]
    hired, spent = [], 0.0
    for i, b in enumerate(order):
        rec = "new lab" if b["answer_rate"] is None else f"answer rate {b['answer_rate']:.0%}"
        facts = f"{b['capability']} capability, {rec}, ${b['job_quote_usd']:.2f} for {b['queries']} queries"
        if len(hired) >= k:
            better = order[k - 1]
            reasons[b["lab_id"]] = f"lost: {facts}; outranked by {better['lab_id']} (" + (
                "higher capability" if CAPABILITY_RANK[better["capability"]] > CAPABILITY_RANK[b["capability"]] else
                "better track record" if (better["answer_rate"] or 0.5) > (b["answer_rate"] or 0.5) else
                "cheaper" if better["job_quote_usd"] < b["job_quote_usd"] else "tie, earlier in catalog") + ")"
        elif spent + b["job_quote_usd"] > budget_usd + 1e-9:
            reasons[b["lab_id"]] = f"lost: {facts}; would exceed the remaining budget ${budget_usd - spent:.2f}"
        else:
            hired.append(b["lab_id"]); spent += b["job_quote_usd"]
            reasons[b["lab_id"]] = ("won (requested): " if b["lab_id"] in preferred else "won: ") + facts
    return hired, {"hired": hired, "reasons": reasons, "rule": "capability > track record > SLA failures > price; "
                   f"up to {k} labs within budget; the researcher's preferred labs are taken first", "committed_usd": round(spent, 2)}


# ---------------------------------------------------------------- the lab's own agent decides
LAB_AGENT_SYSTEM = """You are the AI agent that represents one research lab's dataset in a federated network.
Another lab wants to hire you. Decide whether to ACCEPT the job. Rules:
- Your published policy has already been checked by code and passed. The pre-flight results show which parts of
  the job your gate can actually answer on your real data.
- Accept only if the purpose is legitimate research, the job is within your data's scope, and at least one
  required analysis can be answered. Decline if the request would be misleading given your data (e.g. a
  population your survey doesn't cover) and say why.
- Be factual. Do not advertise or persuade; base your reason on the policy, notes and pre-flight results."""


def lab_decides(llm: LLM, card: dict, job: dict) -> tuple[dict, str]:
    lab_id = card["lab_id"]
    if not card["policy_ok"]:
        return {"accept": False, "reason": card["policy_reason"], "conditions": [], "by": "policy (code)"}, "policy"
    if card["capability"] == "none":
        reasons = ", ".join(sorted({c["refusal"] for c in card["preflight"] if c["refusal"]}))
        return {"accept": False, "reason": f"Pre-flight: the gate cannot answer any required analysis ({reasons}).",
                "conditions": [], "by": "pre-flight (code)"}, "preflight"
    if llm is None:  # fast mode: the decision is exactly what policy + pre-flight say (no language model)
        partial = [c["requirement"] for c in card["preflight"] if not c["can_answer"]]
        return {"accept": True, "reason": f"Policy passed; pre-flight: {card['capability']} capability on this job.",
                "conditions": [f"Cannot answer: {p}" for p in partial], "by": "policy + pre-flight (code)"}, "rule"
    content = (f"Your lab: {lab_id} ({LABS[lab_id]['name']}).\nYour data notes: {json.dumps(LABS[lab_id]['notes'])}\n"
               f"Your definitions: {json.dumps(LABS[lab_id]['definitions'])}\nPre-flight on this job: {json.dumps(card['preflight'])}\n"
               f"Job offered: {json.dumps(job)}")
    try:
        d = llm.structured(f"lab_agent_{lab_id}", LAB_AGENT_SYSTEM, content, JobDecision)
        return {**d.model_dump(), "by": f"lab agent ({llm.model_for('lab_agent_' + lab_id) if llm.backend == 'ollama' else llm.backend})"}, "llm"
    except AgentFailed as e:
        llm.notes.append(f"lab agent {lab_id} offline fallback: {e}")
        partial = [c["requirement"] for c in card["preflight"] if not c["can_answer"]]
        return {"accept": True, "reason": "Policy passed and pre-flight shows the gate can answer the job.",
                "conditions": [f"Cannot answer: {p}" for p in partial], "by": "offline rule"}, "offline"


# ---------------------------------------------------------------- lab stewards review their own evidence
STEWARD_SYSTEM = """You are the data steward agent for one lab. Your lab's aggregate results were used by another
lab's analysis. Check ONLY how YOUR lab's evidence was interpreted: definitions, coding, population covered,
and whether any claim uses your data beyond what it can support. Use your lab's notes. Confirm if correct.
Object (with the evidence ID) only for concrete problems. You cannot vote on whether a claim is true."""


def steward_review(llm: LLM, lab_id: str, evidence: list[dict], claims: list[dict]) -> tuple[dict, str]:
    mine = [e for e in evidence if e.get("lab") == lab_id]
    ids = {e["evidence_id"] for e in mine}
    used = [c for c in claims if ids & set(c["evidence_ids"])]
    content = (f"Your lab: {lab_id}\nNotes: {json.dumps(LABS[lab_id]['notes'])}\nDefinitions: {json.dumps(LABS[lab_id]['definitions'])}\n"
               f"Your evidence: {json.dumps(mine)}\nClaims that use it: {json.dumps(used)}")
    try:
        r = llm.structured(f"steward_{lab_id}", STEWARD_SYSTEM, content, StewardReview)
        r = r.model_dump()
        r["objections"] = [o for o in r["objections"] if o["evidence_id"] in ids]  # may only object to its own evidence
        return r, "llm"
    except AgentFailed as e:
        llm.notes.append(f"steward {lab_id} offline fallback: {e}")
        return {"confirmed": True, "objections": [], "comment": "Offline steward: no automated objection; notes attached for the human reviewer: "
                + " ".join(LABS[lab_id]["notes"])}, "offline"
