"""End-to-end run: plan -> local analysis -> paid federated queries -> evidence -> debate -> vote -> human sign-off."""
from __future__ import annotations

import base64
import json
import time
import uuid
from collections import Counter
from pathlib import Path

from . import agents, market, protocol, vector_panel
from .audit import AuditLog
from .catalog import LABS
from .config import DEFAULT_BUDGET_USD, RUNS_DIR
from .lab import load_all
from .llm import LLM
from .payments import SimulatedX402
from .recipes import RECIPES, Network

_LABS_CACHE: dict = {}


def get_labs(home: str | None = None, tier: str = "evidence"):
    """inprocess: every lab node in this process (each still holds only its own data).
    x402: only the researcher's HOME lab is loaded here; every other lab is a separate HTTP server reached via x402."""
    from .config import TRANSPORT
    if TRANSPORT == "x402":
        from .remote import RemoteLab
        key = ("x402", home)
        if key not in _LABS_CACHE:
            _LABS_CACHE[key] = {l: (load_all([l])[l] if l == home else RemoteLab(l, tier)) for l in LABS}
        labs = _LABS_CACHE[key]
    else:
        if "inprocess" not in _LABS_CACHE:
            _LABS_CACHE["inprocess"] = load_all()
        labs = _LABS_CACHE["inprocess"]
    for node in labs.values():
        node.history.clear()  # fresh query budget per run
    return labs


def read_documents(paths: list[str]) -> list[dict]:
    docs = []
    for p in map(Path, paths or []):
        if p.suffix.lower() == ".pdf":
            docs.append({"name": p.name, "type": "pdf", "data": base64.standard_b64encode(p.read_bytes()).decode()})
        else:
            docs.append({"name": p.name, "type": "text", "data": p.read_text(errors="replace")})
    return docs


def run(question: str, home: str, documents: list[dict] | None = None, tier: str = "evidence",
        budget_usd: float = DEFAULT_BUDGET_USD, blind: bool = True, asker: str = "researcher",
        offline: bool | None = None, progress=print, mode: str | None = None, preferred_labs: list[str] | None = None) -> dict:
    from .config import RUN_MODE
    mode = mode or RUN_MODE
    deep = mode == "deep"
    if home not in LABS:
        raise ValueError(f"Unknown home lab {home}; choose from {list(LABS)}")
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    audit = AuditLog(run_dir / "audit.jsonl")
    llm = LLM(offline=offline)
    llm.on_call = lambda c: emit("agent_call", **c)
    t0 = time.time()
    timeline: list = []

    def emit(kind: str, **data):
        ev = {"t": round(time.time() - t0, 3), "kind": kind, **data}
        timeline.append(ev)
        progress(ev)

    def stage(name, **kw):
        emit("stage", stage=name, **kw)
    audit.add("run_started", {"question": question, "home": home, "tier": tier, "budget_usd": budget_usd, "blind": blind,
                              "documents": [d["name"] for d in documents or []], "ai_mode": "offline" if llm.offline else "online"})

    # 1. Plan
    stage("planning")
    p, plan_mode = agents.plan(llm, question, home, documents or [])
    params, fixes = agents.validate_plan(p, home)
    chosen = [l for l in (preferred_labs or []) if l in LABS and l != home]
    if chosen and params["recipe"] != "unsupported":  # the researcher's own choice of partner labs wins over the planner's
        fixes.append(f"Researcher selected partner labs: {', '.join(chosen)}")
        params["external_labs"] = chosen
    audit.add("plan", {"mode": plan_mode, "plan": p.model_dump(), "validated": params, "corrections": fixes})
    stage("plan_ready", recipe=params["recipe"], mode=plan_mode)
    if params["recipe"] == "unsupported":
        result = {"run_id": run_id, "status": "unsupported", "question": question, "home": home, "plan": p.model_dump(),
                  "plan_corrections": fixes, "message": "The network's approved analyses cannot answer this question.",
                  "ai": llm.cost_summary(), "timeline": timeline}
        _save(run_dir, result)
        return result

    recipe = params.pop("recipe")
    labs = get_labs(home, tier)
    net = Network(labs=labs, home=home, asker=asker, pay=SimulatedX402(budget_usd), audit=audit, tier=tier,
                  on_event=lambda e: emit("lab", **e))

    # 2. Marketplace: discover labs, pre-flight (yes/no only), offer the job, each lab's own agent decides
    stage("discovering_labs")
    job = {"requester_lab": home, "purpose": p.purpose, "purpose_category": p.purpose_category, "recipe": recipe,
           "requirements": [r[3] for r in market.REQUIREMENTS[recipe]], "returns": "gated aggregates only", "retention": "none"}
    cards = market.discover(labs, home, recipe, p.purpose_category, asker)
    decisions, deposits = {}, {}
    for card in cards:
        offered = card["policy_ok"] and card["capability"] != "none"
        if offered:  # commitment deposit, refunded if the lab declines or is not hired
            req = net.pay.quote(card["lab_id"], f"job-{card['lab_id'][:6].lower()}", "verdict")
            deposits[card["lab_id"]] = net.pay.escrow(req, card["lab_id"], "job-offer")
        d, dmode = market.lab_decides(llm if deep else None, card, job)
        decisions[card["lab_id"]] = {**d, "offered": offered, "mode": dmode}
        stage("lab_decision", lab=card["lab_id"], accept=d["accept"], capability=card["capability"])
    hired = market.choose(cards, decisions, params.get("external_labs", []), recipe)
    for lab_id, r in deposits.items():
        (net.pay.release if lab_id in hired else net.pay.refund)(r)
    params["external_labs"] = hired
    audit.add("hiring", {"job": job, "cards": cards, "decisions": decisions, "hired": hired})
    stage("hired", labs=hired)

    # 3. Pre-register the protocol (facts, estimand, decision rules) BEFORE any lab releases data
    proto = protocol.build(recipe, params, home, question)
    audit.add("protocol_preregistered", {"protocol_hash": proto["protocol_hash"], "protocol": proto})
    stage("protocol_preregistered", protocol_hash=proto["protocol_hash"][:16])

    # 4. Local analysis, then paid external queries (inside the recipe)
    stage("querying_labs")
    t_recipe = time.perf_counter()
    claims = RECIPES[recipe](net, **params)
    recipe_ms = (time.perf_counter() - t_recipe) * 1000
    labs_used = [home] + [l for l in params.get("external_labs", []) if l != home]
    stage("evidence_ready", n_evidence=len(net.evidence), n_claims=len(claims))

    # 3. Debate and vote (agents see aggregates only; lab names blinded by default)
    pack = agents.build_pack(question, home, claims, net.evidence, net.local_steps, p, labs_used)
    audit.add("evidence_pack", {"claims": [c.model_dump() for c in claims], "n_evidence": len(net.evidence)})
    stage("deliberating", blind=blind, mode=mode)
    if deep:
        debate = agents.run_debate(llm, pack, blind, labs_used)
    else:
        t_vec = time.perf_counter()
        debate = vector_panel.run([c.model_dump() for c in claims], [e.model_dump() for e in net.evidence], labs_used, recipe)
        debate["panel_ms"] = round((time.perf_counter() - t_vec) * 1000, 2)
        emit("vector_panel", ms=debate["panel_ms"], claims=len(claims), lenses=len(debate["lenses"]), features=len(debate["features"]))
    audit.add("debate", {"modes": debate["modes"], "dropped_points": debate["dropped_points"],
                         "ballots": {k: v["votes"] for k, v in debate["ballots"].items()}})

    # 4. Verdicts: statistics ceiling + panel vote, most cautious wins
    verdicts = []
    for c in claims:
        votes = [v["verdict"] for b in debate["ballots"].values() for v in b["votes"] if v["claim_id"] == c.claim_id]
        final, review, why = agents.combine(c.ceiling, votes)
        verdicts.append({"claim_id": c.claim_id, "statement": c.statement, "ceiling": c.ceiling, "ceiling_reason": c.ceiling_reason,
                         "votes": dict(Counter(votes)), "final": final, "human_review_required": review,
                         "rule": why, "established_locally": c.established_locally, "evidence_ids": c.evidence_ids})
    # 5. Lab stewards check how their OWN data was used; objections can only make verdicts more cautious
    stage("steward_review")
    stewards = {}
    ev_dump, cl_dump = [e.model_dump() for e in net.evidence], [c.model_dump() for c in claims]
    for lab_id in labs_used:
        if deep:
            rev, smode = market.steward_review(llm, lab_id, ev_dump, cl_dump)
        else:  # fast mode: steward attaches the lab's published notes for the human reviewer; no automated objection
            rev, smode = {"confirmed": True, "objections": [], "comment": "Notes attached: " + " ".join(LABS[lab_id]["notes"])}, "rule"
        stewards[lab_id] = {**rev, "mode": smode}
    for v in verdicts:
        objs = [dict(o, lab=l) for l, r in stewards.items() for o in r["objections"] if o["evidence_id"] in v["evidence_ids"]]
        v["steward_objections"] = objs
        if any(o["severity"] == "high" for o in objs) and v["final"] in ("supported", "not_supported"):
            v["final"], v["human_review_required"] = "inconclusive", True
            v["rule"] += " A data steward raised a high-severity objection: downgraded to inconclusive."
        elif objs:
            v["human_review_required"] = True
    audit.add("steward_reviews", stewards)

    ev = {e.evidence_id: e for e in net.evidence}
    for v, c in zip(verdicts, claims):
        derived = [ev[i] for i in c.evidence_ids if ev[i].kind == "derived"]
        main = derived[-1] if derived else None
        labs_ok = {ev[i].lab for i in c.evidence_ids if ev[i].kind == "lab_result" and ev[i].status == "ok"}
        v["certainty"] = protocol.certainty(v["final"], main.model_dump() if main else None, main.i2 if main else None,
                                            len(labs_ok), c.cross_population)
    by_claim = {v["claim_id"]: v for v in verdicts}
    refused = [e.lab for e in net.evidence if e.kind == "lab_result" and e.status == "refused"]
    for f in proto["facts"]:
        if f["claim_id"]:
            f["status"] = by_claim[f["claim_id"]]["final"] if f["claim_id"] in by_claim else "not evaluated"
        elif f["kind"] == "data_sufficiency":
            f["status"] = "met" if not refused else f"partly met: gate refused at {', '.join(sorted(set(refused)))}"
        else:
            f["status"] = "met: all labs use 'ever diagnosed by a doctor' items"
    stage("narrating")
    narrative = vector_panel.narrate(llm, question, verdicts, ev_dump, debate, net.local_steps)
    audit.add("narrative", narrative)
    audit.add("verdicts", {"verdicts": verdicts, "facts": proto["facts"]})
    market.record(run_id, net, decisions, stewards)
    stage("awaiting_signoff")

    result = {
        "run_id": run_id, "status": "awaiting_signoff", "question": question, "home": home, "tier": tier,
        "plan": p.model_dump(), "plan_mode": plan_mode, "plan_corrections": fixes, "recipe": recipe, "protocol": proto,
        "marketplace": {"job": job, "cards": cards, "decisions": decisions, "hired": hired}, "stewards": stewards,
        "local_breakdown": net.local_steps, "evidence": [e.model_dump() for e in net.evidence],
        "mode": mode, "narrative": narrative, "verdicts": verdicts, "debate": debate, "payments": net.pay.summary(), "lab_traffic": net.timeline,
        "ai": llm.cost_summary(), "timeline": timeline,
        "performance": _performance(net, recipe_ms, llm),
        "limits": pack["standing_caveats"] + (["Payments are SIMULATED (x402-style flow, test values only)."] if net.pay.summary()["simulated"]
                                              else ["Payments are real x402 settlements on the Base Sepolia TESTNET (test USDC); job deposits and penalties are simulated."]),
        "audit_file": str(run_dir / "audit.jsonl"),
    }
    _save(run_dir, result)
    return result


def signoff(run_id: str, decision: str, reviewer: str, note: str = "") -> dict:
    assert decision in ("accept", "challenge", "reject")
    run_dir = RUNS_DIR / run_id
    result = json.loads((run_dir / "result.json").read_text())
    audit = AuditLog(run_dir / "audit.jsonl")
    audit.prev = json.loads((run_dir / "audit.jsonl").read_text().splitlines()[-1])["hash"]
    audit.add("human_signoff", {"decision": decision, "reviewer": reviewer, "note": note})
    result["status"] = {"accept": "accepted", "challenge": "challenged", "reject": "rejected"}[decision]
    result["signoff"] = {"decision": decision, "reviewer": reviewer, "note": note, "ts": time.time()}
    ok, n = AuditLog.verify(run_dir / "audit.jsonl")
    result["audit_chain"] = {"valid": ok, "entries": n}
    _save(run_dir, result)
    return result


def _performance(net, recipe_ms: float, llm) -> dict:
    lab_ms = sum(t.get("compute_ms", 0) for t in net.timeline if t["step"].startswith("gate_"))
    ops = sum(len(t.get("compute_trace", [])) for t in net.timeline if t["step"].startswith("gate_"))
    llm_s = sum(c["seconds"] for c in llm.calls)
    return {"lab_compute_ms": round(lab_ms, 1), "vector_ops": ops, "lab_queries": sum(1 for t in net.timeline if t["step"].startswith("gate_")),
            "coordinator_stats_ms": round(max(0.0, recipe_ms - lab_ms), 1), "llm_seconds": round(llm_s, 1), "llm_calls": len(llm.calls),
            "llm_output_tokens": sum(c["output_tokens"] for c in llm.calls),
            "note": "All maths runs on numeric arrays inside the labs (milliseconds); language models only plan and review."}


def _save(run_dir: Path, result: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
