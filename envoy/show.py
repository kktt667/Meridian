"""Console demo: the real pipeline, stage by stage, showing the actual objects as they are produced.

  venv/bin/python -m envoy.show 2            # demo question 2 (6, 5 likewise)
  venv/bin/python -m envoy.show 2 --step     # pause after each stage (for recording)
  venv/bin/python -m envoy.show --question "..." --home KOREA_KNHANES_2024 --prefer US_NHANES_2017

Nothing here is mocked: every table is rendered from the event the pipeline emits at that moment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import pipeline
from .audit import AuditLog
from .catalog import LABS, public_catalog
from .cli import DEMOS
from .config import RUNS_DIR, TRANSPORT
from .market import reputation
from .planner_ml import PLANNER_MIN_CONFIDENCE, classify, model, purpose_category

con = Console()
OPTS = {"step": False, "clear": False}
STATE: dict = {"cards": [], "gate_rows": [], "tx": []}
VCOL = {"supported": "green", "not_supported": "red", "inconsistent": "yellow", "inconclusive": "bright_black"}


def pause(label: str):
    if OPTS["step"]:
        con.input(f"\n[dim]-- {label} -- press Enter --[/dim]")
    if OPTS["clear"]:
        con.clear()


def head(n: int, title: str, sub: str = ""):
    con.rule(f"[bold]STAGE {n}: {title}[/bold]" + (f"  [dim]{sub}[/dim]" if sub else ""))


def short(lab: str) -> str:
    return LABS[lab]["country"] if lab in LABS else lab


def vcol(v: str) -> str:
    return f"[{VCOL.get(v, 'white')}]{v.replace('_', ' ')}[/]"


# ------------------------------------------------------------------------------------ stage 0: the market
def stage_catalog(home: str):
    head(0, "THE MARKET", "what each lab publishes: metadata and rules, never data")
    labs = pipeline.get_labs(home, "evidence")
    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    for c in ("lab", "country", "size", "groups", "measured BP", "gate", "rate", "track record (ledger)"):
        t.add_column(c)
    for c in public_catalog({k: v.size_bucket for k, v in labs.items()}):
        rep = reputation(c["lab_id"])
        pol = c["policy"]
        t.add_row(("[bold]" if c["lab_id"] == home else "") + c["lab_id"] + (" (home)[/bold]" if c["lab_id"] == home else ""),
                  c["country"], c["size"], ", ".join(c["groups"]) or "-", "yes" if c["measured_bp"] else "no",
                  f"n>={pol['min_population']} cell>={pol['min_cell']} cases>={pol['min_cases']}",
                  f"x{pol['rate_multiplier']} (${c['prices_usd']['evidence']:.2f}/query)",
                  f"{rep['queries_served']} served, answer rate {rep['answer_rate'] if rep['answer_rate'] is not None else '-'}, SLA fails {rep['sla_failures']}")
    con.print(t)
    con.print("[dim]Refused purposes everywhere: re-identification, insurance pricing, marketing, law enforcement. "
              f"Transport: {TRANSPORT} (x402 = real USDC settlements on Base Sepolia testnet).[/dim]")
    pause("market")


# ------------------------------------------------------------------------------------ stage 1: the question -> plan (ML)
def stage_plan(question: str, home: str, ev: dict):
    head(1, "QUESTION -> PLAN", "machine learning + catalog lookup; no language model")
    con.print(Panel(question, title=f"research question from {home}", border_style="cyan"))
    recipe, conf, dist = classify(question)
    m = model()
    tf, lr = m.named_steps["tfidfvectorizer"], m.named_steps["logisticregression"]
    x = tf.transform([question]).toarray()[0]
    names = tf.get_feature_names_out()
    t = Table(title="recipe classifier (TF-IDF + logistic regression)", box=box.SIMPLE_HEAD)
    t.add_column("recipe"); t.add_column("probability"); t.add_column("words in the question that drove it")
    for i, cls in enumerate(lr.classes_):
        contrib = x * lr.coef_[i]
        top = [names[j] for j in np.argsort(-contrib)[:5] if contrib[j] > 0]
        bar = "#" * int(dist[cls] * 30)
        t.add_row(("[bold]" if cls == recipe else "") + cls, f"{bar} {dist[cls]:.2f}", ", ".join(top))
    con.print(t)
    con.print(f"chosen: [bold]{recipe}[/bold] at confidence {conf:.2f} (abstains below {PLANNER_MIN_CONFIDENCE}) · "
              f"purpose screen (fixed rules): [bold]{purpose_category(question)}[/bold]")
    p = ev["plan"]
    t = Table(title="decomposition: what our own data can establish vs what needs other labs", box=box.SIMPLE_HEAD)
    t.add_column("local (home lab, free, still gated)"); t.add_column("remote (hire through the market)")
    for a, b in zip(p["can_establish_locally"] + [""] * 3, p["needs_external_help"] + [""] * 3):
        if a or b:
            t.add_row(a, b)
    con.print(t)
    con.print(f"parameters from the catalog: focus={p['focus_group']} reference={p['reference_group']} external={p['external_labs']}")
    if ev["corrections"]:
        con.print("[yellow]validate_plan corrections:[/yellow] " + " | ".join(ev["corrections"]))
    else:
        con.print("[green]validate_plan: no corrections needed[/green]")
    pause("plan")


# ------------------------------------------------------------------------------------ stage 2: data classification + bids
def stage_market(ev: dict):
    head(2, "DATA CLASSIFICATION + SEALED BIDS", "pre-flight on real data (yes/no only), then bids are facts")
    cards = STATE["cards"]
    reqs = [c["requirement"] for c in cards[0]["card"]["preflight"]] if cards else []
    t = Table(title="pre-flight: can each lab's gate answer each required analysis? (no numbers released)", box=box.SIMPLE_HEAD)
    t.add_column("lab"); t.add_column("policy")
    for r in reqs:
        t.add_column(r)
    t.add_column("capability"); t.add_column("lab agent")
    for c in cards:
        card, d = c["card"], c["decision"]
        cells = ["[green]yes[/green]" if x["can_answer"] else f"[red]no: {x['refusal']}[/red]" for x in card["preflight"]]
        t.add_row(card["lab_id"], "ok" if card["policy_ok"] else f"[red]{card['policy_reason']}[/red]", *cells, card["capability"],
                  ("[green]accept[/green]" if d["accept"] else "[red]decline[/red]") + f" ({d['by']})")
    con.print(t)
    sel = ev["selection"]
    t = Table(title="sealed bids -> selection", box=box.SIMPLE_HEAD)
    for c in ("lab", "capability", "track record", "quote", "outcome"):
        t.add_column(c)
    for b in ev["bids"]:
        r = sel["reasons"].get(b["lab_id"], "")
        rec = "new lab" if b["answer_rate"] is None else f"{b['answer_rate']:.0%} answered, {b['sla_failures']} SLA fails"
        t.add_row(b["lab_id"], b["capability"], rec, f"${b['job_quote_usd']:.2f} ({b['queries']} x ${b['price_per_query_usd']:.2f})",
                  ("[green]" if r.startswith("won") else "[red]") + r + "[/]")
    con.print(t)
    con.print(f"rule: {sel['rule']} · committed ${sel['committed_usd']:.2f} · deposits of the others refunded")
    pause("market")


# ------------------------------------------------------------------------------------ stage 3: protocol
def stage_protocol(ev: dict):
    head(3, "PRE-REGISTERED PROTOCOL", "hashed into the audit log before any lab releases a number")
    pr = ev["protocol"]
    t = Table(box=box.SIMPLE_HEAD, title="PICO")
    t.add_column("population"); t.add_column("exposure"); t.add_column("comparison"); t.add_column("outcome")
    t.add_row(*[pr["pico"][k] for k in ("population", "exposure", "comparison", "outcome")])
    con.print(t)
    t = Table(box=box.SIMPLE_HEAD, title="facts to establish and how (decided before seeing data)")
    t.add_column("fact"); t.add_column("statement"); t.add_column("kind"); t.add_column("how"); t.add_column("claim")
    for f in pr["facts"]:
        t.add_row(f["fact_id"], f["statement"], f["kind"], f["how"], f["claim_id"] or "-")
    con.print(t)
    con.print(f"estimand: {pr.get('estimand', '')}")
    con.print(f"[bold]protocol sha256 {pr['protocol_hash']}[/bold]")
    pause("protocol")


# ------------------------------------------------------------------------------------ stage 4: gate + payments (live rows)
def gate_row(ev: dict):
    ok = ev["step"] == "gate_ok"
    ops = ev.get("compute_trace") or []
    STATE["gate_rows"].append(ev)
    con.print(f"  {'[green]OK     [/green]' if ok else '[red]REFUSED[/red]'} {short(ev['lab']):14s} {ev.get('purpose', ''):52s} "
              f"{'local' if ev.get('local') else 'remote':6s} {len(ops):2d} ops {ev.get('compute_ms', 0):7.1f} ms  "
              f"pay: {ev.get('payment', '')}" + (f"  [red]{ev.get('refusal')}[/red]" if not ok else ""))
    if ops and len(STATE["gate_rows"]) == 1:
        con.print("           trace: " + " > ".join(o["op"] + (f"[{o['rows']}]" if o.get("rows") else "") for o in ops))


def x402_row(ev: dict):
    STATE["tx"].append(ev)
    con.print(f"  [magenta]x402[/magenta]    {short(ev['lab']):14s} {ev['step']:20s} tx {str(ev.get('transaction'))[:18]}... on {ev.get('network')}")


def stage_evidence(ev: dict):
    con.print()
    t = Table(title="evidence pack: what actually crossed the boundary", box=box.SIMPLE_HEAD)
    for c in ("id", "kind", "lab", "estimate (95% CI)", "status", "description"):
        t.add_column(c)
    for e in ev["evidence"]:
        est = f"{e['estimate']:.2f} ({e['ci_low']:.2f}-{e['ci_high']:.2f})" if e.get("estimate") is not None else "-"
        if e.get("i2") is not None:
            est += f" I2={e['i2']:.0%}"
        t.add_row(e["evidence_id"], e["kind"], short(e["lab"]) if e.get("lab") else "-", est,
                  "[green]ok[/green]" if e["status"] == "ok" else f"[red]{e['status']} {e.get('refusal') or ''}[/red]", e["description"][:90])
    con.print(t)
    t = Table(title="statistics ceiling per claim (fixed rules; the panel can only lower it)", box=box.SIMPLE_HEAD)
    t.add_column("claim"); t.add_column("statement"); t.add_column("ceiling"); t.add_column("why")
    for c in ev["claims"]:
        t.add_row(c["claim_id"], c["statement"][:80], vcol(c["ceiling"]), c["ceiling_reason"])
    con.print(t)
    con.print(f"[dim]fields that left the labs: counts >= min cell, coefficients, size buckets, hashes. Rows moved: 0.[/dim]")
    pause("evidence")


# ------------------------------------------------------------------------------------ stage 5: bias-aware panel
def stage_panel(ev: dict):
    head(5, "BIAS-AWARE REVIEW PANEL", "S = F x W : features from the aggregates, one column per reviewer lens; no language model")
    F, S, W, th, feats, lenses = ev["F"], ev["S"], ev["W"], ev["thresholds"], ev["feature_names"], ev["lens_names"]
    t = Table(title="F: bias features per claim (0 = fine, 1 = worst)", box=box.SIMPLE_HEAD)
    t.add_column("claim")
    for f in feats:
        t.add_column(f.replace("_", "\n"), justify="right")
    for cid, row in F.items():
        cells = []
        for f in feats:
            v = row[f]
            cells.append(f"[red]{v:.2f}[/red]" if v >= 0.8 else f"[yellow]{v:.2f}[/yellow]" if v >= 0.5 else f"[dim]{v:.2f}[/dim]")
        t.add_row(cid, *cells)
    con.print(t)
    t = Table(title="W: how much each lens cares about each feature", box=box.SIMPLE_HEAD)
    t.add_column("feature")
    for l in lenses:
        t.add_column(l, justify="right")
    for f, w in zip(feats, W):
        t.add_row(f, *[f"{x:.1f}" for x in w])
    con.print(t)
    t = Table(title=f"S = F x W: caution score per lens (dissent above {th[0]})  ->  vote", box=box.SIMPLE_HEAD)
    t.add_column("claim")
    for l in lenses:
        t.add_column(l)
    for cid, row in S.items():
        cells = []
        for l in lenses:
            v = next(x for x in ev["ballots"][l]["votes"] if x["claim_id"] == cid)
            top = sorted(v["contributions"].items(), key=lambda kv: -kv[1])[:2]
            cells.append(f"{row[l]:.2f} -> {vcol(v['verdict'])}  [dim]" + ", ".join(f"{k} {x:.2f}" for k, x in top) + "[/dim]")
        t.add_row(cid, *cells)
    con.print(t)
    high: dict = {}
    for x in ev["threats"]:
        if x["severity"] == "high":
            high.setdefault(x["category"], {**x, "claims": []})["claims"].append(x["claim_id"])
    if high:
        con.print("[bold]threat library (edge cases, triggered by features and lab notes):[/bold]")
        for x in high.values():
            con.print(f"  [red]{x['category']}[/red] ({', '.join(x['claims'])}): {x['threat']}  [dim]check: {x['how_to_check']}[/dim]")
    con.print(f"[dim]panel computed in {ev['ms']} ms. Every vote has its contribution vector: that is the audit trail.[/dim]")
    pause("panel")


# ------------------------------------------------------------------------------------ stage 6-8 from the result
def stage_verdicts(r: dict):
    head(6, "VERDICTS", "statistics ceiling x panel votes x steward objections; most cautious wins")
    t = Table(box=box.SIMPLE_HEAD)
    for c in ("claim", "statement", "ceiling", "votes", "final", "certainty", "rule"):
        t.add_column(c)
    for v in r["verdicts"]:
        t.add_row(v["claim_id"], v["statement"][:85], vcol(v["ceiling"]), ", ".join(f"{k} {n}" for k, n in v["votes"].items()),
                  vcol(v["final"]) + (" [yellow]human review[/yellow]" if v["human_review_required"] else ""),
                  (v.get("certainty") or {}).get("level", ""), v["rule"][:80])
    con.print(t)
    pause("verdicts")


def stage_narrative(r: dict):
    head(7, "NARRATIVE", "code writes it from the audit; the language model may only reword it, blinded; a guard checks")
    n = r["narrative"]
    con.print(Panel("\n".join([n["template"]["headline"]] + ["- " + s for s in n["template"]["summary"]]),
                    title="1. template narrative (deterministic, every number from the audit)", border_style="green"))
    if n.get("llm_blinded"):
        con.print(Panel(n["llm_blinded"]["headline"] + "\n" + "\n".join("- " + s for s in n["llm_blinded"]["summary"]),
                        title="2. what the language model saw and wrote (labs blinded: Lab A, Lab B...)", border_style="blue"))
    g = n.get("guard") or []
    con.print(Panel(("[green]guard passed: same numbers, same verdict per claim, same IDs[/green]" if not g else
                     "[red]guard rejected the rewrite: " + "; ".join(g) + "[/red]") + f"\nmode: {n['mode']}",
                    title="3. guard", border_style="red" if g else "green"))
    con.print(Panel(n["headline"] + "\n" + "\n".join("- " + s for s in n["summary"]) + "\n\n[dim]would change it: " +
                    " · ".join(n["what_would_change_it"]) + "[/dim]", title="4. narrative shown to the researcher (unblinded by code)", border_style="white"))
    pause("narrative")


def stage_audit(r: dict):
    head(8, "AUDIT + HUMAN SIGN-OFF", "hash-chained log; nothing becomes a conclusion until a human signs")
    ok, n = AuditLog.verify(RUNS_DIR / r["run_id"] / "audit.jsonl")
    pay = r["payments"]
    settled = [x for x in pay.get("receipts", []) if not x.get("simulated") and x.get("status") == "settled"]
    t = Table(box=box.SIMPLE_HEAD)
    t.add_column("item"); t.add_column("value")
    t.add_row("run", r["run_id"])
    t.add_row("protocol hash", r["protocol"]["protocol_hash"])
    t.add_row("audit chain", f"[green]valid[/green] ({n} entries)" if ok else f"[red]BROKEN[/red] ({n} entries)")
    t.add_row("rows of patient data moved", "0")
    t.add_row("gate decisions", f"{sum(1 for g in STATE['gate_rows'] if g['step'] == 'gate_ok')} answered, "
                                f"{sum(1 for g in STATE['gate_rows'] if g['step'] != 'gate_ok')} refused (not charged)")
    t.add_row("paid to labs", f"${pay['spent_usd']}" + (f" · {len(settled)} real x402 settlements" if settled else " (simulated)"))
    t.add_row("language model", f"{r['performance'].get('llm_calls')} call(s), {r['performance'].get('llm_seconds')} s, used only to reword the narrative")
    t.add_row("status", r["status"])
    con.print(t)
    con.print(f"sign off: [bold]venv/bin/python -m envoy.cli signoff {r['run_id']} accept|challenge|reject[/bold]   "
              f"verify: [bold]venv/bin/python -m envoy.cli verify {r['run_id']}[/bold]")


# ------------------------------------------------------------------------------------ event dispatch
def on_event(ev: dict):
    k = ev["kind"]
    if k == "stage":
        s = ev["stage"]
        if s == "plan_ready":
            stage_plan(STATE["question"], STATE["home"], ev)
        elif s == "lab_decision":
            STATE["cards"].append(ev)
        elif s == "bids":
            stage_market(ev)
        elif s == "protocol_preregistered":
            stage_protocol(ev)
        elif s == "querying_labs":
            head(4, "GATE + PAYMENT", "each query passes the lab's own gate; paid per answer; refusals cost nothing")
        elif s == "evidence_ready":
            stage_evidence(ev)
        elif s == "narrating":
            con.print("[dim]rewording the template narrative with the local language model (blinded)...[/dim]")
    elif k == "lab":
        if ev.get("step", "").startswith("gate_"):
            gate_row(ev)
        elif ev.get("step", "").startswith("x402_"):
            x402_row(ev)
    elif k == "vector_panel":
        stage_panel(ev)
    elif k == "agent_call":
        con.print(f"  [dim]model {ev.get('model')} ({ev.get('role')}): {ev.get('seconds')} s[/dim]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", nargs="?", choices=list(DEMOS))
    ap.add_argument("--question"); ap.add_argument("--home"); ap.add_argument("--prefer", nargs="*", default=None)
    ap.add_argument("--tier", default="evidence"); ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--mode", default="fast"); ap.add_argument("--offline", action="store_true")
    ap.add_argument("--step", action="store_true", help="pause after each stage"); ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    OPTS.update(step=a.step, clear=a.clear)
    if a.demo:
        home, question, _ = DEMOS[a.demo]
    elif a.question and a.home:
        home, question = a.home, a.question
    else:
        ap.error("give a demo number or --question and --home")
    STATE.update(question=question, home=home)
    if OPTS["clear"]:
        con.clear()
    con.print(Panel("[bold]MERIDIAN[/bold]  federated evidence marketplace\nthe question travels; the data stays home; "
                    "code decides, ML flags bias, the language model only rewords, a human signs", border_style="cyan"))
    stage_catalog(home)
    r = pipeline.run(question, home, [], tier=a.tier, budget_usd=a.budget, blind=True, offline=a.offline,
                     progress=on_event, mode=a.mode, preferred_labs=a.prefer)
    if r["status"] == "unsupported":
        con.print(Panel(f"[red]unsupported[/red]: {r['message']}\n{json.dumps(r['plan']['assumptions'], indent=1)}"))
        return
    stage_verdicts(r)
    stage_narrative(r)
    stage_audit(r)


if __name__ == "__main__":
    main()
