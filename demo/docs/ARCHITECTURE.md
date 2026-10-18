# Meridian: how it works, end to end

Federated clinical-evidence marketplace. The question travels; the data stays home. Code decides, machine learning
flags bias, a language model only rewords, a human signs. Built for SotonShip 2026 (Responsible AI track).

## 1. The problem and the idea

Health data cannot move. KNHANES (Korea) and IFLS forbid redistribution; hospitals are bound by law; national surveys sit
in silos. So most evidence comes from majority populations, and a clinician treating a Korean-American patient is
using data mostly about white patients while the relevant data sits in Seoul.

Meridian inverts the flow. Each lab runs a node that holds its own data and answers a small set of pre-approved
statistical questions through a privacy gate. A researcher's agent plans the analysis, hires labs on a market, pays
per answer over x402, combines the released aggregates with fixed statistics, has a bias-aware panel review the
result, and hands it to a human to sign. Every step is hash-chained.

## 2. Stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy; harmonised CSV per lab (`age, male, group, hypertension, stroke, heart_attack, sbp, dbp, weight`) built by `scripts/prepare_labs.py` from BRFSS 2023 (US, telephone), NHANES 2017-18 (US, examination), KNHANES 2024 (Korea, examination), PNS 2019 (Brazil, household). Raw data is git-ignored because of licences. |
| Lab node | `envoy/lab.py`: statsmodels `Logit`, vectorised numpy masks; strict pydantic wire schemas (`extra="forbid"`) |
| Statistics | `envoy/stats.py`: scipy for chi-square and normal tails; everything else hand-written (Mantel-Haenszel, DerSimonian-Laird, Wilson) |
| Planner | scikit-learn TF-IDF (1-2 grams, English stop words) + multinomial logistic regression, `envoy/planner_ml.py` |
| Panel | numpy matrix product, `envoy/vector_panel.py` |
| Language models | Ollama, local open weights: qwen2.5:7b, llama3.1:8b, mistral:7b, qwen2.5:3b; JSON-schema structured output, temperature 0, `num_ctx` 8192; one model loaded at a time (16 GB machine). Optional Claude backend. |
| Payments | x402 Python SDK v2.25.0 (Coinbase): `PaymentMiddlewareASGI` on each lab server, `x402ClientSync` + `EthAccountSigner` on the researcher side, facilitator `x402.org`, USDC on Base Sepolia (`eip155:84532`) |
| API / transport | FastAPI + uvicorn; labs on ports 4101-4104; coordinator on 8000; Server-Sent Events for the live run |
| Audit | SHA-256 hash chain (`envoy/audit.py`), HMAC receipts, protocol hash |
| UI | `envoy/static/live.html`: vanilla HTML + canvas swarm, no framework; `envoy/show.py`: rich console demo |
| Tests | pytest (`tests/test_gate.py`), `scripts/twin_eval.py` synthetic-twin evaluation |

## 3. One run, step by step

### 3.1 Plan (`planner_ml.py`, `agents.validate_plan`)
A 60-example labelled set of researcher questions trains a TF-IDF + logistic-regression classifier over four classes:
`event_mix_comparison`, `age_gradient`, `undiagnosed_comparison`, `unsupported`. The classifier returns a probability
per class; below 0.5 the planner abstains (recipe `unsupported`) instead of guessing. The page shows the per-class
probabilities and the question n-grams whose TF-IDF x coefficient contribution pushed the chosen class.

Everything else is catalog lookup: focus and reference groups are matched by name and synonym against the home lab's
published groups; local vs remote is decided by which labs publish the variables and measured values the recipe needs
(e.g. `undiagnosed_comparison` needs `measured_bp`); purpose is screened by fixed regex rules (re-identification,
insurance pricing, marketing, law enforcement are refused categories). `validate_plan` then checks and repairs
parameters inside the catalog and logs every repair. A researcher's preferred labs override the planner's choice but
are filtered by the same catalog rule. No language model is involved; same question, same plan, in milliseconds.

### 3.2 Market (`market.py`)
`discover` builds a card per lab: policy check in code (purpose allowed?), then a **pre-flight**: a dry run of the lab's
gate on its real data for each requirement of the recipe, returning only yes/no plus a refusal code, never a number,
and not counted against the asker's budget. `lab_decides` lets the lab's own agent accept or decline (fast mode:
policy + pre-flight result; deep mode: the lab's own local model, which can never accept what policy rejected).
`bid` turns each accepted card into a sealed bid of facts: published per-query price (`base x rate_multiplier`), the job
total, pre-flight capability, and reputation computed from the platform ledger (`runs/ledger.jsonl`: queries served,
answer rate, SLA failures). `select` ranks capability > answer rate > SLA failures > price, takes up to two labs within
budget, honours the researcher's requested labs first, and writes a reason for every win and loss. A commitment
deposit (simulated) is escrowed per offered lab and refunded unless hired.

### 3.3 Pre-registered protocol (`protocol.py`)
Before any lab releases a number, the recipe's PICO, the facts to establish, the estimand formula and the decision
rules are written as JSON, SHA-256 hashed, and appended to the audit log. Results are later matched to the facts by
claim ID, so the goalposts cannot move after seeing data.

### 3.4 Gate + payment (`lab.py`, `recipes.py`, `payments.py`, `lab_server.py`, `remote.py`)
The coordinator speaks to a lab only through `handle(json) -> json`, the same whether the node is in-process or
behind HTTP. Inside `LabNode._run`, in order: schema validation (pydantic, allowlisted operations and fields),
**query budget** (40 per asker), **differencing check** (an earlier query of the same shape whose age window differs
by 1-4 years is refused as an overlap attack), group and variable availability, **minimum population** (50; 30 for
Korea). Then one of three operations:

- `logistic_or`: logistic regression of outcome on exposure with age and sex covariates; needs >= 20 outcome cases
  among the exposed and >= 5 among the unexposed; releases log OR, SE, OR, CI, a size bucket and the case count.
- `event_split_strata`: among people with exactly one of stroke / heart attack, counts strokes (a) vs heart attacks
  (b) per age band x sex; any cell below 5 is suppressed.
- `undiagnosed_strata`: among people with measured SBP >= 140 or DBP >= 90, counts never-diagnosed (a) vs diagnosed
  (b) per age band; same suppression.

Every vectorised step is recorded as a `TraceStep` (operation code, row-count bucket, milliseconds), never a value.
The outgoing aggregate is hashed, re-validated against the strict schema, and only then leaves. Rows never leave.

With `ENVOY_TRANSPORT=x402`, each lab runs its own FastAPI server whose `POST /query/{tier}` route is wrapped by
`PaymentMiddlewareASGI` at the lab's published price. The researcher's client signs a USDC payment with its wallet;
the facilitator settles it on Base Sepolia and the transaction hash goes into the audit log and the UI (basescan link).
A gate refusal returns 409 before payment, so a refusal costs nothing. The home lab is queried free, through the same
gate. SLA breaches (timeout, malformed output) are the only penalised events, from a simulated stake; a null result
is never penalised.

### 3.5 Statistics and ceilings (`stats.py`)
Deterministic, on aggregates only:
- Mantel-Haenszel odds ratio across age x sex strata with Robins-Breslow-Greenland variance;
- DerSimonian-Laird random-effects pooling with Cochran's Q, I-squared and tau-squared;
- ratio of two log estimates with independent variance (conservative) for transportability and effect modification;
- Wilson intervals for proportions.

Fixed rules then set a **ceiling** per claim: `supported` if the CI excludes the null in the stated direction,
`not_supported` if it excludes it the other way, `inconclusive` if it includes it; a pooled claim is `inconsistent`
when Q p < 0.10 and I-squared >= 50%, and `inconclusive` with fewer than two contributing labs; a transfer claim can only
be refuted (ratio CI excludes 1), never proven.

### 3.6 Bias-aware panel (`vector_panel.py`)
For each claim, eight features are computed from the gated aggregates, each scaled 0 (fine) to 1 (worst):
imprecision (CI width on the log scale), null closeness, inconsistency (I-squared), refused share (how many of the
claim's lab queries were refused), single source, cross population, small counts (smallest released cell), weak
effect. `F` is claims x features. `W` is features x three reviewer lenses (methodologist, clinician,
ethics/regulatory) with hand-set, published weights. `S = F @ W` is the caution score; a lens votes more cautiously
than the statistics when its score exceeds 1.6. The elementwise product `F[i] * W[:, j]` is the contribution vector of
every vote, which is the audit trail. A threat library turns features and each lab's published notes into explicit
edge cases (survivor bias, access to diagnosis, transportability, confounding, small sample, definition mismatch).
The panel runs in well under a millisecond.

Deep mode replaces this with the original multi-model debate: For, Against and Edge-case agents on three different
model families argue citing evidence IDs (uncited points are dropped and counted), three lens voters vote, and each
lab's steward model checks how its own data was interpreted and may object. Lab and country names are blinded to
Lab A / Lab B throughout.

### 3.7 Verdict (`agents.combine`, `protocol.certainty`)
Ceiling x majority of the three votes, most cautious wins; the panel can only lower a verdict, never raise it; a
downgrade or any steward objection flags the claim for human review; a high-severity steward objection downgrades
to inconclusive. A GRADE-inspired certainty rating (start low for observational survey data, up for large effects,
down for risk of bias, inconsistency, indirectness, imprecision) is attached to definite verdicts.

### 3.8 Narrative (`vector_panel.template_narrative`, `narrate`, `guard`)
Code writes the narrative first, from the audit objects only, so every number is traceable by construction. The
language model receives that text with labs blinded and may only reword it. A guard then rejects the rewrite if it
introduces any number not in the audit, drops a claim ID, changes a verdict word, or adds meaning-drift words
("risk", "cause", "significant", "should"...). On rejection the template stands and the rejection reason is shown.
Code unblinds the accepted text.

### 3.9 Audit and sign-off (`audit.py`, `pipeline.signoff`)
Every event (run start, plan, hiring with bids and reasons, protocol, each query and result, each payment, debate,
verdicts, narrative, ledger update) is appended as a JSON line whose hash includes the previous hash. The ledger
updates each lab's reputation. `cli verify` or `GET /runs/{id}/audit` re-checks the chain. Nothing becomes a
conclusion until a qualified human accepts, challenges or rejects it, which is itself appended to the chain.

## 4. Evidence that the bias defences work: synthetic digital twins (`scripts/twin_eval.py`)

Real data cannot be used to score the system because the truth is unknown. So the evaluation generates
**synthetic twin labs** from parameters we choose, registers them in the catalog with the standard policy, and runs
every query through the real `LabNode` gate and the real `stats.py`, with no language model. 200 trials per cell,
seed 2026, about 7 seconds. Results are in `demo/docs/bias_report.md` and at `GET /bias_report`.

1. **Transport test.** Country A has a majority (5,000) and a minority (n = 500 / 1,000 / 2,000, younger on average,
   so age confounds). Country B is the minority's home population (5,000). We plant the truth: B's stroke odds are
   `ratio` times A's minority odds. Meridian flags non-transfer when the ratio-of-ORs CI excludes 1.
   - ratio 1.0 (evidence transfers): flagged 0.5-2% (false alarms).
   - ratio 2.0: flagged 59% at n=500, **94% at n=1,000, 98% at n=2,000**.
   - ratio 1.5: 24% / 55% / 70%: subtler differences need bigger minorities.
2. **Small-group test** (the Q6 situation). Hypertension -> stroke in young adults, true OR 3.0. The home lab has 2,000
   young adults; the network adds labs with 10,000 and 20,000. Alone, the home gate released an estimate in
   **1 of 200** trials (too few cases). Through the network, **200 of 200**, pooled OR median 3.0, 95% CI covering the
   truth 96% of the time.
3. **Privacy invariant.** Across all 6,000 results that crossed a lab boundary, **0** stratum cells below the minimum
   cell of 5 and **0** model estimates below 20 exposed cases; the only fields that ever left are counts, coefficients,
   size buckets, status codes, trace codes and hashes.
4. **Planner.** 60 labelled questions, leave-one-out accuracy 78%, held-out paraphrases 7/8, abstains below 0.5.

Twins are for testing only, never evidence, and are labelled SYNTHETIC everywhere. Limits: one confounder, no survey
weights, no missingness; the planner's training set is small and ours; deep mode's language-model layer is blinded but
not yet name-swap tested.

## 5. Why this design

- **Legal and practical.** Data that cannot be shared can still answer questions. The gate is the lab's own policy,
  enforced in its own process; the coordinator never sees rows.
- **Incentives that scale.** Labs are paid per answer at their own published rate, refusals are free, reputation is
  computed from the ledger rather than claimed, and penalties apply only to verifiable failures. A lab gains nothing
  from persuasion and loses nothing from an honest null.
- **Bias is measured, not promised.** Transportability is tested, not assumed; small groups get answers by pooling
  instead of being silenced by privacy rules; measured versus diagnosed exposes label bias; the panel's features come
  from numbers, not wording; routing is deterministic; the language model cannot change a number or a verdict.
- **Accountable.** Pre-registration before data, hash chain, contribution vectors for every vote, human sign-off.
- **Cheap.** A fast-mode run is about a second of compute plus one short local model call; the panel is a matrix
  product. Deep mode exists for when model diversity in words is wanted.

## 6. What is real and what is simulated

Real: the four national datasets, the gate, the statistics, the vector panel, local open-weight models, x402
settlements on the Base Sepolia testnet (test USDC), the audit chain, the synthetic-twin evaluation.
Simulated: commitment deposits, SLA stakes and penalties, subscriptions. Estimates are unweighted and
cross-sectional, from self-reported diagnoses among survivors: associations, not causes. Health use requires qualified
human review.

## 7. Demo questions

- **Q2** (US asks): is the US Asian subgroup's event mix stroke-heavy, and does Korean evidence transfer? Locally OR
  1.32 (1.07-1.62); Korea vs US White 3.02; ratio of ORs 2.29 (1.55-3.40): **transfer not supported**. The panel
  downgraded the local claim to inconclusive (near-null, small cells).
- **Q6** (Korea asks): hypertension -> stroke in young vs old. Korea's own gate refuses the young-adult query
  (`min_cases`); the network answers: pooled young OR 4.05, gradient 1.89, 65+ **inconsistent** (I-squared 71%).
- **Q5** (Korea asks): undiagnosed hypertension among young adults with high readings, 90% in Korea vs 55% in the US
  NHANES comparator.
