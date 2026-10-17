"""HTTP API for the frontend.  Run:  venv/bin/uvicorn envoy.api:app --reload --port 8000

GET  /catalog                  marketplace cards
GET  /demos                    the three demo questions
POST /runs                     {question, home, documents:[{name,text}], tier, budget_usd, blind}  -> full result
POST /runs/{run_id}/signoff    {decision: accept|challenge|reject, reviewer, note}
GET  /runs/{run_id}            saved result
GET  /runs/{run_id}/audit      audit chain entries + validity
"""
from __future__ import annotations

import json
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import pipeline
from .audit import AuditLog
from .catalog import public_catalog
from .cli import DEMOS
from .config import RUNS_DIR, ROOT

app = FastAPI(title="Envoy federated research network")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Doc(BaseModel):
    name: str
    text: str


class RunRequest(BaseModel):
    question: str
    home: str
    documents: list[Doc] = []
    tier: str = "evidence"
    budget_usd: float = 10.0
    blind: bool = True
    offline: bool | None = None
    mode: str = "fast"
    preferred_labs: list[str] = []


class SignoffRequest(BaseModel):
    decision: str
    reviewer: str = "researcher"
    note: str = ""


@app.get("/")
def index():
    return FileResponse(ROOT / "envoy" / "static" / "meridian.html")


@app.get("/control")
def control_room():
    return FileResponse(ROOT / "envoy" / "static" / "index.html")


@app.get("/info")
def info():
    from .catalog import LABS
    from .config import TRANSPORT
    from .llm import LLM
    return {"backend": LLM().backend, "transport": TRANSPORT, "labs": list(LABS)}


@app.get("/runs")
def list_runs(limit: int = 30):
    out = []
    for p in sorted(RUNS_DIR.glob("*/result.json"), reverse=True)[:limit]:
        try:
            r = json.loads(p.read_text())
            out.append({"run_id": r["run_id"], "home": r.get("home"), "mode": r.get("mode", "?"), "status": r.get("status"),
                        "question": r.get("question", "")[:120]})
        except Exception:
            continue
    return out


@app.get("/catalog")
def catalog():
    labs = pipeline.get_labs()
    return public_catalog({k: v.size_bucket for k, v in labs.items()})


@app.get("/demos")
def demos():
    out = []
    for k, (home, q, docs) in DEMOS.items():
        out.append({"id": k, "home": home, "question": q,
                    "documents": [{"name": d.split("/")[-1], "text": (ROOT / d).read_text()} for d in docs]})
    return out


@app.post("/runs")
def create_run(req: RunRequest):
    docs = [{"name": d.name, "type": "text", "data": d.text} for d in req.documents]
    try:
        return pipeline.run(req.question, req.home, docs, req.tier, req.budget_usd, req.blind, offline=req.offline, progress=lambda *_: None, mode=req.mode, preferred_labs=req.preferred_labs)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/runs/{run_id}/signoff")
def signoff(run_id: str, req: SignoffRequest):
    if req.decision not in ("accept", "challenge", "reject"):
        raise HTTPException(400, "decision must be accept, challenge or reject")
    return pipeline.signoff(run_id, req.decision, req.reviewer, req.note)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    p = RUNS_DIR / run_id / "result.json"
    if not p.exists():
        raise HTTPException(404, "unknown run")
    return json.loads(p.read_text())


@app.get("/runs/{run_id}/audit")
def get_audit(run_id: str):
    p = RUNS_DIR / run_id / "audit.jsonl"
    if not p.exists():
        raise HTTPException(404, "unknown run")
    ok, n = AuditLog.verify(p)
    return {"valid": ok, "entries": n, "chain": [json.loads(l) for l in p.read_text().splitlines()]}


@app.post("/runs/stream")
def create_run_stream(req: RunRequest):
    """Server-Sent Events: every stage, lab round-trip (with vector-op traces) and agent call as it happens,
    then a final 'result' event with the full result."""
    q: queue.Queue = queue.Queue()
    docs = [{"name": d.name, "type": "text", "data": d.text} for d in req.documents]

    def work():
        try:
            r = pipeline.run(req.question, req.home, docs, req.tier, req.budget_usd, req.blind, offline=req.offline, progress=q.put, mode=req.mode, preferred_labs=req.preferred_labs)
            q.put({"kind": "result", "result": r})
        except Exception as e:  # surface errors to the UI instead of hanging the stream
            q.put({"kind": "error", "error": str(e)})
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while (ev := q.get()) is not None:
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
