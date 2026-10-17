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
  (planner/edge: qwen2.5:7b, FOR: llama3.1:8b, AGAINST: mistral:7b, voters: one of each); each lab's own agent and
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

## One run, step by step (each step emitted live as an event)
1. **Plan** (`agents.plan`): question + researcher documents + catalog -> recipe, purpose, local vs needs-help. Code validates
   and repairs parameters within the catalog, logging every repair.
2. **Marketplace** (`market.py`): discover labs -> policy check (code) -> **pre-flight** gate dry run (yes/no + refusal code,
   no numbers) -> the lab's own agent accepts/declines -> commitment deposit refunded unless hired. Reputation is computed
   from the platform ledger (`runs/ledger.jsonl`), never self-described.
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
