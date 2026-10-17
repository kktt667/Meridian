"""Command line runner.

  venv/bin/python -m envoy.cli demo 2            # run one of the three demo questions (2, 6, 5)
  venv/bin/python -m envoy.cli ask "question" --home US_BRFSS_2023 --docs protocol.md
  venv/bin/python -m envoy.cli catalog
  venv/bin/python -m envoy.cli verify RUN_ID      # check the audit hash chain
"""
from __future__ import annotations

import argparse
import json

from . import pipeline
from .audit import AuditLog
from .catalog import public_catalog
from .config import RUNS_DIR, ROOT

DEMOS = {
    "2": ("US_BRFSS_2023", "Should we prioritise stroke prevention for our East Asian patients? Our Asian subgroup is small and mixes "
          "many origins. Is their cardiovascular event mix stroke-heavy, and can evidence from an East Asian population be applied to them?",
          ["demo/docs/q2_us_clinic_memo.md"]),
    "6": ("KOREA_KNHANES_2024", "Is high blood pressure a stronger marker of stroke in young adults than in older adults? We have too few "
          "young stroke cases to tell on our own.", ["demo/docs/q6_korea_protocol.md"]),
    "5": ("KOREA_KNHANES_2024", "Most of our young adults with high blood pressure readings were never diagnosed. Is that unusually high "
          "compared with another country that measures blood pressure?", ["demo/docs/q5_korea_screening_note.md"]),
}

C = {"supported": "\033[32m", "not_supported": "\033[31m", "inconsistent": "\033[33m", "inconclusive": "\033[90m", "end": "\033[0m"}


def progress(ev: dict):
    k = ev["kind"]
    if k == "stage":
        extra = {x: y for x, y in ev.items() if x not in ("t", "kind", "stage")}
        print(f"[{ev['t']:6.1f}s] {ev['stage']} {extra if extra else ''}", flush=True)
    elif k == "vector_panel":
        print(f"[{ev['t']:6.1f}s]    vector panel: {ev['claims']} claims x {ev['features']} features x {ev['lenses']} lenses in {ev['ms']} ms", flush=True)
    elif k == "agent_call":
        print(f"[{ev['t']:6.1f}s]    agent {ev['role']:28s} {ev['model']:14s} {ev['seconds']:5.1f}s  {ev['output_tokens']} tok", flush=True)
    elif k == "lab" and ev.get("step", "").startswith("gate_"):
        print(f"[{ev['t']:6.1f}s]    lab   {ev['lab']:20s} {ev['step']:13s} {ev.get('refusal') or '':16s} "
              f"{ev.get('compute_ms', 0):6.1f} ms  {len(ev.get('compute_trace', []))} ops  pay: {ev.get('payment')}", flush=True)


def show(r: dict):
    print("\n" + "=" * 100)
    print(f"RUN {r['run_id']}   status: {r['status']}   AI: {r['ai']['mode']}   cost: ${r['ai']['cost_usd']}")
    print(f"Question: {r['question']}")
    if r["status"] == "unsupported":
        print(r["message"]); return
    p = r["plan"]
    print(f"\nPLAN ({r['plan_mode']}): {p['restated_question']}\n  recipe: {r['recipe']}")
    for x in p["can_establish_locally"]:
        print(f"  [local]    {x}")
    for x in p["needs_external_help"]:
        print(f"  [needs]    {x}")
    for x in p["out_of_scope"]:
        print(f"  [out]      {x}")
    if r["plan_corrections"]:
        print(f"  corrections by validator: {r['plan_corrections']}")
    print("\nLOCAL-FIRST BREAKDOWN (computed)")
    for s in r["local_breakdown"]:
        print(f"  - {s['question_part']}: {s['status']}  ({s['detail']})")
    print("\nLAB TRAFFIC (gate + payment)")
    for t in r["lab_traffic"]:
        if t["step"].startswith("gate_"):
            print(f"  {t['step']:13s} {t['lab']:20s} {t['purpose'][:45]:45s} {t.get('refusal') or '':20s} {t['payment']}")
    pay = r["payments"]
    kind = "SIMULATED x402" if pay.get("simulated", True) else f"REAL x402 on Base Sepolia: {pay.get('real_x402_payments', 0)} payments"
    print(f"  spent ${pay['spent_usd']} of ${pay['budget_usd']} ({kind})")
    for rc in pay.get("receipts", []):
        if not rc.get("simulated", True) and rc.get("signature"):
            print(f"    tx {rc['signature']}  ({rc['lab_id']}, {rc['status']})  https://sepolia.basescan.org/tx/{rc['signature']}")
    print("\nEVIDENCE (derived)")
    for e in r["evidence"]:
        if e["kind"] == "derived" and e["estimate"] is not None:
            print(f"  {e['evidence_id']:4s} {e['estimate']:.2f} ({e['ci_low']:.2f}-{e['ci_high']:.2f})  {e['description'][:95]}")
    d = r["debate"]
    print(f"\nDEBATE (blind={d['blind']}; modes {d['modes']}; dropped uncited points {d['dropped_points']})")
    for side in ("for", "against"):
        print(f"  {side.upper()}: {d[side]['summary'][:200]}")
        for pt in d[side]["points"][:4]:
            print(f"     - [{','.join(pt['evidence_ids'])}] {pt['claim'][:150]}")
    print("  EDGE CASES:")
    for ec in d["edge_cases"]["edge_cases"][:5]:
        print(f"     - ({ec['severity']}, {ec['category']}) {ec['threat'][:140]}")
    print("\nVERDICTS (statistics ceiling x panel vote; most cautious wins)")
    for v in r["verdicts"]:
        col = C.get(v["final"], "")
        print(f"  {v['claim_id']} {col}{v['final'].upper():14s}{C['end']} {v['statement']}")
        print(f"       ceiling={v['ceiling']} ({v['ceiling_reason']})  votes={v['votes']}  review={'YES' if v['human_review_required'] else 'no'}")
    n = r.get("narrative") or {}
    if n:
        print(f"\nNARRATIVE ({n.get('mode')}; unverified numbers: {n.get('unverified_numbers') or 'none'})")
        print(f"  {n.get('headline', '')}")
        for x in n.get("summary", []):
            print(f"   - {x}")
    print(f"\nLIMITS: " + " | ".join(r["limits"]))
    print(f"Audit trail: {r['audit_file']}")


def main():
    ap = argparse.ArgumentParser(prog="envoy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("which", choices=list(DEMOS)); d.add_argument("--offline", action="store_true")
    d.add_argument("--deep", action="store_true", help="full LLM debate instead of the vector panel")
    d.add_argument("--no-blind", action="store_true"); d.add_argument("--signoff", choices=["accept", "challenge", "reject"])
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--home", required=True); a.add_argument("--docs", nargs="*")
    a.add_argument("--tier", default="evidence"); a.add_argument("--budget", type=float, default=10.0); a.add_argument("--offline", action="store_true")
    a.add_argument("--no-blind", action="store_true"); a.add_argument("--json", action="store_true")
    sub.add_parser("catalog")
    v = sub.add_parser("verify"); v.add_argument("run_id")
    s = sub.add_parser("signoff"); s.add_argument("run_id"); s.add_argument("decision", choices=["accept", "challenge", "reject"])
    s.add_argument("--reviewer", default="researcher"); s.add_argument("--note", default="")
    args = ap.parse_args()

    if args.cmd == "catalog":
        print(json.dumps(public_catalog(), indent=1)); return
    if args.cmd == "verify":
        print(AuditLog.verify(RUNS_DIR / args.run_id / "audit.jsonl")); return
    if args.cmd == "signoff":
        r = pipeline.signoff(args.run_id, args.decision, args.reviewer, args.note)
        print(r["status"], r["audit_chain"]); return
    if args.cmd == "demo":
        home, q, docs = DEMOS[args.which]
        r = pipeline.run(q, home, pipeline.read_documents([str(ROOT / x) for x in docs]), blind=not args.no_blind,
                         offline=True if args.offline else None, progress=progress, mode="deep" if args.deep else None)
    else:
        r = pipeline.run(args.question, args.home, pipeline.read_documents(args.docs), tier=args.tier, budget_usd=args.budget,
                         blind=not args.no_blind, offline=True if args.offline else None, progress=progress)
    if getattr(args, "json", False):
        print(json.dumps(r, indent=1, default=str))
    else:
        show(r)
    if getattr(args, "signoff", None) and r["status"] == "awaiting_signoff":
        s = pipeline.signoff(r["run_id"], args.signoff, "demo-reviewer")
        print(f"\nHuman sign-off: {s['status']}  audit chain valid={s['audit_chain']['valid']} ({s['audit_chain']['entries']} entries)")


if __name__ == "__main__":
    main()
