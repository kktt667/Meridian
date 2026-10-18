# Envoy: the AI layer (rename freely)

Labs keep data local. A researcher's question is broken into facts, answered locally where possible, and the
rest is hired out to other labs' AI agents over x402. Statistics cap what can be claimed, a diverse panel of
open-weight models argues and votes, lab stewards can object, a human signs off, and everything is audited.

## Run it

```bash
venv/bin/python -m pytest -q tests/                      # gate + leak tests
venv/bin/python -m envoy.cli demo 2 --signoff accept      # US asks Korea (transportability)
venv/bin/python -m envoy.cli demo 6                       # Korea refused locally, US + Brazil hired
venv/bin/python -m envoy.cli demo 5                       # undiagnosed young adults; US/Brazil declined by pre-flight
venv/bin/uvicorn envoy.api:app --port 8000                # API for the frontend (POST /runs/stream = live SSE feed)
```

### AI backend (`ENVOY_BACKEND`)
- `ollama` (default): local open-weight models, nothing leaves the laptop. Per-role families in `config.ROLE_MODELS`
  (edge: qwen2.5:7b, FOR: llama3.1:8b, AGAINST: mistral:7b, voters: one of each; the planner is ML, not an LLM); each lab's own agent and
  steward run on that lab's model (`catalog.LABS[...]["policy"]["agent_model"]`). Override: `ENVOY_MODEL_<ROLE>=...`.
- `claude`: Anthropic API (`ANTHROPIC_API_KEY` in `.env`).
- `offline` / `--offline`: deterministic agents, labelled as such.

### Real x402 payments (`ENVOY_TRANSPORT=x402`)
```bash
venv/bin/python scripts/make_wallets.py     # creates testnet wallets in .env (git-ignored)
# fund the RESEARCHER address with Base Sepolia USDC at https://faucet.circle.com (no ETH needed)
bash scripts/run_labs.sh                     # each lab = its own HTTP server, ports 4101-4104, x402 on POST /query/{tier}
ENVOY_TRANSPORT=x402 venv/bin/python -m envoy.cli demo 6
```
In x402 mode only the researcher's home lab is loaded in-process; every other lab is reached over HTTP.
A gate refusal returns 409, and the x402 middleware never settles payment on responses >= 400, so refusals cost nothing.
Job deposits and SLA penalties are still simulated (labelled), since x402 "exact" has no escrow contract.

## Live view in the browser (`http://localhost:8000/`)
`envoy/static/live.html` renders the same stages as the console demo from the SSE stream (`POST /runs/stream`): classifier
bars and driving words, decomposition, pre-flight matrix, bid table, protocol and hash, gate rows with basescan links,
forest plot of the evidence, the F heatmap, W, S = F x W bars against the lens thresholds, verdicts, the four-step
narrative (template, blinded LLM rewrite, guard, final) and the audit KPIs with sign-off buttons. "Replay a recorded run"
plays back a saved run's timeline. The older market/chat UI is at `/meridian`, the dev page at `/control`.

## Console demo (the backend, stage by stage)
`venv/bin/python -m envoy.show 6 --step` runs the real pipeline and renders each stage from the event it emits: market
catalog -> ML plan (classifier probabilities and the words that drove it, local vs remote) -> pre-flight matrix and sealed
bids -> pre-registered protocol -> gate traffic with x402 transactions -> evidence and ceilings -> the bias panel
(F, W, S = F x W, votes with contributions) -> verdicts -> narrative (template, blinded LLM rewrite, guard) -> audit.
`--step` pauses after each stage, `--clear` clears the screen between stages, `--offline` skips the language model.

## One run, step by step (each step emitted live as an event)
1. **Plan** (`planner_ml.py`, no LLM): a TF-IDF + logistic-regression classifier picks the recipe with a confidence score
   (abstains below 0.5); the catalog decides local vs remote (which labs publish the needed groups, variables and measured
   values); fixed rules screen the purpose (re-identification, insurance, marketing, law enforcement are refused categories).
   `validate_plan` then checks and repairs parameters within the catalog, logging every repair. Same question, same plan.
   Measured in `demo/docs/bias_report.md` (`scripts/twin_eval.py`), alongside the synthetic-twin bias tests.
2. **Marketplace** (`market.py`): discover labs -> policy check (code) -> **pre-flight** gate dry run (yes/no + refusal code,
   no numbers) -> the lab's own agent accepts/declines -> **sealed bid** (published per-query rate x queries needed, pre-flight
   capability, ledger reputation) -> `market.select`: capability > track record > SLA failures > price, up to 2 labs within
   budget, researcher's preferred labs first; every win and loss gets a written reason -> commitment deposit refunded unless
   hired. Each lab's x402 server charges exactly its published price (`catalog.price_for`). Reputation is computed from the
   platform ledger (`runs/ledger.jsonl`), never self-described.
3. **Pre-registered protocol** (`protocol.py`): PICO, facts F1..Fn, estimand formulas, decision rules; SHA-256 hash written to the
   audit log before any lab releases data.
4. **Local first, then paid queries** (`recipes.py`): each lab runs allowlisted operations; every vectorised op is traced
   (code, size bucket, ms, iterations; meaning looked up coordinator-side). SLA breaches -> refund + compensation from stake.
5. **Statistics** (`stats.py`): Mantel-Haenszel, DerSimonian-Laird, I², ratios -> a verdict **ceiling** per claim.
6. **Debate** (`agents.run_debate`): FOR / AGAINST / EDGE-CASE, then 3 voters (methodologist, clinician, ethics/regulatory) on
   different model families; lab and country names blinded; uncited points dropped and counted.
7. **Stewards** (`market.steward_review`): each contributing lab checks how ITS data was used; high-severity objection -> inconclusive.
8. **Verdict + certainty**: final = ceiling unless the panel/stewards are more cautious (never upgraded); GRADE-inspired certainty.
9. **Human sign-off** + hash-chained audit (`envoy.cli verify <run_id>`).

Performance block in every result: lab compute ms, number of vector ops, coordinator stats ms, LLM seconds/tokens.
