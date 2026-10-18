# Technical and strategic briefing

## The thesis in three sentences
Clinical evidence is mostly about majority populations because the data that would fix that cannot legally move.
We move the question instead: each lab answers a small set of pre-approved statistical queries through its own
privacy gate, is paid per answer, and the network tests whether the evidence transfers before anyone uses it.
Every step is deterministic where it can be, measured where it cannot, and written to a tamper-evident log a human signs.

## Architecture in one line each
- **Lab node** (`envoy/lab.py`): holds one dataset; answers 3 allowlisted operations via `handle(json) -> json`; enforces the gate; emits only aggregates and op traces.
- **Planner** (`envoy/planner_ml.py`): TF-IDF + logistic regression chooses the analysis; catalog lookup fills everything else; regex screens purpose; abstains under 0.5.
- **Market** (`envoy/market.py`): policy check, pre-flight dry run, lab accept/decline, sealed bids, deterministic selection with written reasons, ledger reputation.
- **Protocol** (`envoy/protocol.py`): PICO + facts + estimand + decision rules, SHA-256 hashed before any data moves.
- **Recipes** (`envoy/recipes.py`): three federated analyses; home lab first (free), then paid partners.
- **Statistics** (`envoy/stats.py`): Mantel-Haenszel (RBG variance), DerSimonian-Laird, I-squared, ratio of ORs, Wilson; fixed verdict ceilings.
- **Panel** (`envoy/vector_panel.py`): 8 bias features x 3 lenses, S = F x W, threshold 1.6, contribution vectors; threat library; deep mode = LLM debate.
- **Narrator**: template from the audit, blinded LLM reword, guard (numbers, IDs, verdict words, drift words).
- **Payments** (`envoy/payments.py`, `envoy/lab_server.py`, `envoy/remote.py`): x402 v2 middleware on each lab; USDC on Base Sepolia; 409 before payment on refusal.
- **Audit** (`envoy/audit.py`): hash chain per run; ledger; human sign-off appended.
- **Evaluation** (`scripts/twin_eval.py`): synthetic twins with planted truth through the real gate and stats.
- **Surfaces**: FastAPI + SSE (`envoy/api.py`), swarm UI (`envoy/static/live.html`), console demo (`envoy/show.py`), CLI.

## Every design decision and why
| Decision | Why | What we rejected |
|---|---|---|
| Question travels, data stays | Licences (KNHANES, IFLS forbid redistribution) and law; also the only way labs will join | Central data lake; synthetic-data sharing (inherits bias, leaks on small groups) |
| Allowlisted operations only | A gate can be proven over 3 operations; it cannot be proven over arbitrary SQL or free text | "Ask anything" agents reading rows |
| Gate inside the lab's process | The lab's policy is enforced by the lab, not trusted to the coordinator | Coordinator-side filtering |
| Strict wire schema (`extra="forbid"`) | Nothing undeclared can leave; the schema is the privacy contract | Loose JSON |
| Differencing check (age window differs by 1-4 years) | Two overlapping queries can be subtracted to isolate a few people | Budget only |
| Query budget (40 per asker) | Bounds cumulative leakage per session | Unlimited |
| Pre-flight dry run | Labs self-evaluate on evidence (can my gate answer?) not on claims; releases no numbers and costs no budget | Labs describing their own capability |
| Reputation from the ledger | Nothing self-described is scored; persuasion earns nothing | Ratings, self-reported stats |
| Sealed bids, capability > track record > price | Quality first, then reliability, then cost; every outcome has a written reason | Lowest price wins (races to the bottom) |
| Paid per answer, refusal free, null never penalised | Removes the incentive to over-release or to over-claim | Success fees (bias toward positive findings) |
| Penalties only for verifiable failures (SLA) | Anything else is arguable | Penalising "bad" answers |
| x402 per query on testnet | Machine-to-machine payment with an on-chain receipt per query; no invoices; refusals cost nothing | Subscriptions only; off-chain ledger |
| Protocol hash before data | Pre-registration: the goalposts are fixed; p-hacking by re-running is visible | Post-hoc analysis |
| Home lab first, free | Establish locally what can be; pay only for the gap | Always hire |
| Mantel-Haenszel on stratum counts | Age/sex adjustment from 2x2 tables only, so labs release counts, not rows; RBG variance works with sparse strata | Pooled individual-level regression (needs rows) |
| Random effects + I-squared rule | Labs that disagree must not be averaged into a false consensus | Fixed-effect pooling |
| Transfer can only be refuted | Similarity is never proven from two surveys; a ratio CI including 1 is "no evidence of difference", not "same" | Declaring transfer |
| Verdict ceiling from statistics | The AI can never make a claim stronger than the numbers allow | Model-decided verdicts |
| Vector panel (fast) | Deterministic, auditable (contribution vectors), microseconds, no wording bias | LLM-only review |
| LLM debate (deep) kept | Model diversity in words when wanted; blinded; uncited points dropped | Single-model review |
| Panel can only downgrade | Caution is the only direction a reviewer can move a verdict | Upgrades |
| Steward objection downgrades | The lab that released the data has the last word on how it was interpreted | No lab voice after release |
| GRADE-inspired certainty | Direction and trust are separate; observational survey data starts low | Reporting "supported" as "strong" |
| ML planner, no LLM fallback | Same question, same plan; measurable; abstains honestly | LLM planner (unmeasurable, wording-sensitive) |
| Template-first narrative + guard | The model cannot add a number, change a verdict or introduce "risk"/"cause"/"significant" | Free-form LLM summaries |
| Blinding (Lab A / Lab B) | Country names carry priors; the model never sees them | Named labs in prompts |
| Name-swap invariance check per run | Proves the deterministic layer does not depend on who the lab is | Assumed |
| Naive counterfactual per verdict | Shows what an ordinary analysis of the same numbers would conclude | Hidden value-add |
| Synthetic twins for evaluation | Real data has unknown truth; twins have planted truth; no sensitive data in the demo | Claims without numbers |
| Hash-chained audit + human sign-off | Nobody has to trust us; they can check; nothing is a conclusion until a human signs | Logs you can edit |
| Local open-weight models (Ollama) | No confidential text leaves the machine; cost ~0 | Hosted LLM APIs |

## The gate, exactly
Order inside `LabNode._run`: schema -> budget (40/asker) -> overlap (same shape, age window differs by 1-4 years -> refuse) ->
group/variable available -> min population (50; Korea 30) -> operation. `logistic_or` needs >= 20 outcome cases among the
exposed and >= 5 among the unexposed; `event_split_strata` and `undiagnosed_strata` suppress any age x sex cell under 5.
Output: counts >= 5, coefficients, size buckets (`<50, 50-499, 500-4999, 5000-49999, 50000+`), op trace (code, bucket, ms),
hash. Never: rows, exact n, free text.

## The statistics, exactly
- MH: `OR = sum_s(a_s d_s / n_s) / sum_s(b_s c_s / n_s)`; variance Robins-Breslow-Greenland; CI = exp(log OR +/- 1.96 SE).
- Random effects: weights 1/(SE^2 + tau^2), tau^2 from DerSimonian-Laird, Q = sum w (y - y_fe)^2, I^2 = (Q - df)/Q.
- Ratio of effects: log a - log b, SE = sqrt(SE_a^2 + SE_b^2), conservative when both share a reference.
- Ceilings: supported (CI excludes null, stated direction), not_supported (other direction), inconclusive (includes null),
  inconsistent (Q p < 0.10 and I^2 >= 50%), inconclusive if fewer than 2 labs for a pooled claim.
- Wilson interval for proportions; claim threshold 0.5 for "most".

## The panel, exactly
Features per claim (0-1): imprecision = log(hi/lo)/log 4; null_closeness = 1 - margin/log 1.5; inconsistency = I^2;
refused_share; single_source (cross-lab claim that lost labs); cross_population; small_counts = (20 - min cell)/15;
weak_effect = 1 - |log OR|/log 3. W is 8x3, published. S = F W; lens dissents when S > 1.6 and the claim is definite;
dissent -> inconsistent if I^2 >= 0.5 else inconclusive. Final = ceiling unless the majority of 3 is more cautious.
Contribution C[i, :, j] = F[i] * W[:, j] is stored with every vote.

## The numbers to quote
- Q2: local OR 1.32 (1.07-1.62); Korea vs US White 3.02 (2.16-4.22); ratio 2.29 (1.55-3.40): transfer not supported.
  Naive would have applied 3.02. Panel downgraded C1 (null closeness 0.83, small counts 0.67).
- Q6: Korea refused its own young-adult query (min_cases); pooled young OR 4.05 (3.61-4.54); gradient 1.89 (1.68-2.14);
  65+ inconsistent, I^2 71%, Q p 0.032. Naive fixed-effect would say supported.
- Q5: 90% of young Koreans with high readings undiagnosed vs 55% US NHANES.
- Twins (200 trials): 2x non-transfer caught 94% (n=1,000) / 98% (n=2,000), false alarms 0.5-2%; 1.5x: 55-70%.
  Small group: alone 1/200, network 200/200, pooled OR 3.0 vs truth 3.0, CI coverage 96%. Privacy: 0 of 6,000 below policy.
- Planner: 60 labelled questions, LOO 78%, held-out 7/8.
- Fast run: ~1 s compute + one local model call (20-40 s); panel 0.2 ms. Deep: ~5-8 min, ~10 model calls.
- Payments: real USDC settlements on Base Sepolia, lab rates x0.8-x1.4 on $0.10 / $0.50 / $2.00 tiers.

## Questions you will get, and the answers
- **"Isn't this just federated learning?"** No model is trained. It is federated *analysis*: fixed statistical queries over
  aggregates, with a market and an audit layer. Simpler, provable, and no gradients to leak.
- **"Differential privacy?"** Not yet. We use k-anonymity-style thresholds (cell >= 5, cases >= 20), a differencing check
  and a budget. DP noise on released counts is the natural next step; the gate is the place to add it.
- **"Why can an LLM be in the loop at all?"** Only to reword a text that code wrote, blinded, under a guard that rejects
  new numbers, changed verdicts or drift words. It was rejected twice in live runs for exactly those reasons.
- **"How do you know the bias detection works?"** Synthetic twins with planted truth, through the real gate and stats:
  94% detection, under 2% false alarms. The button runs it live.
- **"Unweighted survey data, self-report, survivors."** Yes, stated on screen and in every narrative; associations, not
  causes; certainty starts low. Weights are a planned addition to the gate operations.
- **"Who pays?"** Pharma, public health bodies, hospital researchers per question; labs earn from data they already hold
  and cannot otherwise use. Subscriptions and deposits are simulated today; per-query payment is real on testnet.
- **"Why x402?"** Machine-to-machine, per-request, on-chain receipt, refusal free. An agent can hire another agent
  without an account, an invoice or a human.
- **"What stops a malicious researcher?"** Purpose screen, lab policy, pre-flight, budget, differencing check, cell
  suppression, and a ledger that records every query against the asker.
- **"What stops a lying lab?"** It cannot describe itself; capability is pre-flight on its real data, reputation is the
  ledger, output is schema-checked and hashed, SLA breaches are penalised from stake.
- **"Can the panel be gamed?"** The weights are published and fixed; votes have contribution vectors; the name-swap test
  proves the deterministic layer ignores identity.
- **"Why only three analyses?"** Because the gate can be reasoned about over three. More recipes are added by writing
  the operation, its gate checks and its protocol template, not by loosening the gate.
- **"What's simulated?"** Deposits, SLA stakes/penalties, subscriptions. Everything else runs.

## Roadmap (say it, don't promise it)
- **Environmental digital twins.** Build a twin of the minority group that simulates the environmental factors behind
  the population difference (diet, salt, smoking, screening access, health system), then reweight the external
  population to match the local one (inverse-odds transport weighting) and estimate the expected transfer error and its
  uncertainty before any real query is paid for. The twin tells you *how wrong* borrowing would be, not just *whether*.
- Differential-privacy noise at the gate; survey weights inside operations; more recipes (time-to-event, dose-response).
- Name-swap and planted-bias tests on the deep-mode LLM layer.
- Real deposits and stakes on x402; subscription tiers; a coverage fund that pays new labs in under-represented
  populations to join (the flywheel).

## Glossary: every term, in plain words

### Data and privacy
- **Federated analysis**: the computation goes to the data; each lab computes on its own rows and sends back summaries. Nothing is pooled centrally.
- **Aggregate**: a summary number (a count, a rate, a coefficient). The opposite of a row, which is one person's record.
- **Gate**: the code inside each lab that decides whether a query may be answered and what may leave. It is the lab's own policy, enforced by the lab.
- **Minimum cell (5)**: no released count may be smaller than 5 people. Small counts identify people (k-anonymity: every released group has at least k people in it).
- **Minimum cases (20)**: a model estimate is only released if at least 20 people with the outcome are behind it. Below that the estimate is unstable and revealing.
- **Minimum population (50; 30 for Korea)**: the filtered group must have at least this many people before anything is computed.
- **Stratum (plural strata)**: a slice of the population, here an age band x sex (e.g. women 45-64). Releasing counts per stratum lets us adjust for age and sex without rows.
- **2x2 table**: four counts: group A with/without the outcome, group B with/without. Everything in Mantel-Haenszel starts from one of these per stratum.
- **Differencing attack**: ask for ages 40-64, then ages 41-64, subtract, and you have learned about the 40-year-olds. We refuse the second query when windows differ by 1-4 years.
- **Query budget (40)**: each asker may send at most 40 queries per lab per session, so leakage cannot accumulate without limit.
- **Pre-flight**: a dry run of the gate that answers only "could I answer this?" (yes, or a refusal code). It releases no numbers and does not count against the budget.
- **Size bucket**: instead of the exact number of rows, the lab reports a range (500-4999). Exact sizes can identify small datasets.
- **Op trace**: the list of vectorised steps the lab ran (filter, mask, count, suppress...) with timing, never values. It shows what was done, not what the data said.
- **Differential privacy**: adding calibrated random noise to released numbers so that no single person's presence changes the output much. We do not do it yet; it is the next layer at the gate.
- **Synthetic data / digital twin**: fake rows generated from chosen parameters or from released summaries. Useful to test a system (we know the truth); never evidence.

### Statistics
- **Odds**: the chance of an event divided by the chance of it not happening. 25% probability = odds of 1 to 3.
- **Odds ratio (OR)**: the odds in one group divided by the odds in another. OR 1 = no difference; OR 2 = twice the odds.
- **Log odds ratio**: the natural log of the OR. Statistics are done on the log scale because it is symmetric (OR 2 and OR 0.5 are equally far from 1).
- **Standard error (SE)**: how uncertain an estimate is. Smaller SE = more precise.
- **95% confidence interval (CI)**: the range of values compatible with the data. If it excludes 1 for an odds ratio, the effect is "statistically clear" in that direction.
- **Null / no effect**: OR = 1 (or a share = 0.5 for "most"). Our ceilings are defined by whether the CI includes the null.
- **p-value**: the probability of seeing an effect this big if there were none. We report it but decide by CIs.
- **Confounding**: a third factor that drives both the exposure and the outcome (age drives both ethnicity mix and stroke). Comparing without adjusting for it gives a false answer.
- **Adjustment**: removing a confounder's influence, either by comparing within strata (MH) or by including it as a covariate in a regression.
- **Mantel-Haenszel (MH)**: a way to combine 2x2 tables across strata into one age- and sex-adjusted odds ratio. Formula: sum over strata of (a*d/n) divided by sum of (b*c/n). Needs only counts.
- **Robins-Breslow-Greenland variance**: the standard way to compute the uncertainty of an MH odds ratio; it stays valid when strata are sparse.
- **Logistic regression**: a model for a yes/no outcome; its coefficient for the exposure, exponentiated, is an adjusted odds ratio. Run inside the lab with age and sex as covariates.
- **Covariate**: a variable included in a model to adjust for it (age, sex).
- **Wilson interval**: a confidence interval for a proportion that behaves well for small counts and near 0% or 100%.
- **Meta-analysis / pooling**: combining estimates from several labs into one. Each lab's estimate is weighted by its precision.
- **Fixed effect**: pooling that assumes every lab measures the same true effect. Hides disagreement.
- **Random effects**: pooling that allows the true effect to differ between labs and widens the interval accordingly. We use this.
- **DerSimonian-Laird**: the standard method to estimate how much true effects vary between labs (tau-squared) and pool with random effects.
- **Tau-squared**: the estimated variance of the true effect across labs. 0 = labs agree.
- **Cochran's Q**: a test statistic for "do the labs disagree more than chance?" Its p-value is Q p.
- **I-squared (I^2)**: the share of the variation between labs that is real disagreement rather than chance. 0% = agree, 71% = mostly real disagreement.
- **Heterogeneity / inconsistency**: labs disagreeing. Our rule: Q p < 0.10 and I^2 >= 50% means "inconsistent: do not pool."
- **Effect modification**: when an effect is different in different groups (hypertension matters more in the young). Measured as a ratio of odds ratios.
- **Ratio of odds ratios**: one OR divided by another, with their uncertainties added. Used for transportability (external effect / local effect) and effect modification (young / old).
- **Transportability**: whether an effect measured in one population applies to another. Can be refuted (ratio CI excludes 1) but never proven.
- **Verdict ceiling**: the strongest verdict the statistics allow for a claim: supported, not supported, inconclusive, inconsistent. The panel can only lower it.
- **Supported / not supported / inconclusive / inconsistent**: CI excludes the null in the stated direction / in the opposite direction / includes the null / labs disagree too much to pool.
- **Certainty (GRADE)**: a separate rating of how much the evidence can be trusted. GRADE is the medical standard: observational data starts "low"; large effects raise it; risk of bias, inconsistency, indirectness and imprecision lower it.
- **Risk of bias**: features of the data that could distort the answer (self-report, survivors only).
- **Indirectness**: the evidence is about a different population or measure than the question (comparing Koreans with Korean-Americans).
- **Imprecision**: a wide confidence interval.
- **Cross-sectional**: everyone measured at one point in time; shows association, not what came first.
- **Self-report / "ever told by a doctor"**: the outcome is what the person says a doctor told them; misses the undiagnosed and the dead.
- **Survivor bias**: surveys only include the living; if one event kills more often, it is under-counted.
- **Label bias**: the outcome label ("diagnosed") is itself biased because diagnosis depends on access to care. Q5 measures this directly (90% undiagnosed).
- **Survey weights**: numbers that make a sample represent the population. Our estimates are unweighted (stated as a limit).
- **Inverse-odds transport weighting**: reweighting an external population so its mix of characteristics matches the local one, to estimate what the effect would be here. Roadmap.

### Machine learning and the panel
- **Deterministic**: same input, same output, every time. Code and our ML are; language models are not.
- **TF-IDF**: turns text into numbers by counting words, giving rarer words more weight.
- **n-gram**: a run of n words ("blood pressure" is a 2-gram). We use 1- and 2-grams.
- **Stop words**: common words ("the", "that") removed before counting.
- **Logistic regression classifier**: a model that gives a probability for each class (here, each analysis recipe) from the TF-IDF features.
- **Confidence / abstain**: the probability of the chosen class; below 0.5 the planner says "unsupported" instead of guessing.
- **Leave-one-out accuracy**: train on all examples but one, test on that one, repeat for every example. 78%.
- **Held-out**: examples never used in training. 7/8.
- **Feature (panel)**: a number between 0 and 1 describing one weakness of a claim (imprecision, closeness to null, inconsistency, refused share, single source, cross population, small counts, weak effect).
- **F, W, S**: F = claims x features; W = features x lenses (published weights); S = F x W = caution score per claim per lens.
- **Lens**: a reviewer viewpoint with its own weights: methodologist (cares about precision and consistency), clinician (effect size), ethics/regulatory (refusals, cross-population use).
- **Threshold (1.6)**: the caution score above which a lens votes more cautiously than the statistics.
- **Contribution vector**: feature-by-feature breakdown of a lens's score; the explanation of each vote.
- **Threat library**: fixed list of edge cases (survivor bias, access to diagnosis, transportability, confounding, small sample, definition mismatch) triggered by features and lab notes.
- **Combine rule**: final verdict = ceiling unless the majority of the three votes is more cautious; ties go to caution; any downgrade requires human review.
- **Steward**: the lab's own reviewer of how its data was interpreted; may object to its own evidence only; a high-severity objection downgrades.
- **Blinding**: replacing lab and country names with Lab A / Lab B before any language model sees text.
- **Name-swap invariance**: rename all labs, rerun the panel, verdicts must be identical. Proves identity is not used.
- **Naive counterfactual**: what an ordinary analysis of the same numbers would conclude (assume transfer, pool despite disagreement, no panel). Shown next to our verdict.
- **Template narrative**: the summary written by code from the audit, so every number is traceable by construction.
- **Guard**: the check on the language model's rewrite: no new numbers, same claim IDs, same verdict words, no drift words.
- **Drift words**: words that change meaning without changing numbers: "risk" (we measured odds), "cause", "significant", "should".
- **Language model (LLM)**: a text generator. Here used only to reword, blinded, under the guard (fast mode) or to argue and vote (deep mode).
- **Open weights / Ollama**: models whose files run locally (qwen, llama, mistral). Nothing leaves the laptop.
- **Structured output / JSON schema**: forcing the model to answer in a fixed shape so code can check it.
- **Temperature 0**: the model always picks its most likely word; as repeatable as a language model gets.
- **Planted truth / planted bias**: in synthetic twins we set the real effect ourselves, so we can score whether the system finds it.
- **Detection rate**: share of planted non-transfers the system flagged. **False alarm rate**: share of true transfers wrongly flagged.
- **CI coverage**: how often the interval contains the true value; should be ~95%.

### Market, payments, audit
- **Catalog**: what each lab publishes: groups, variables, measured values, gate policy, rates, notes. Metadata only.
- **Recipe**: one of the three approved federated analyses, each with fixed queries, statistics and claims.
- **Sealed bid**: each lab's offer, made of facts only: price per query x queries needed, pre-flight capability, ledger track record.
- **Ledger**: the platform's record of every lab's past jobs (queries, answers, refusals, SLA failures). Reputation is computed from it.
- **Reputation / track record**: answer rate and failures from the ledger; never self-described.
- **SLA**: service-level agreement: answer within 30 s with valid output. A breach is the only penalised event.
- **Stake / penalty**: money a lab puts up that can be deducted for an SLA breach (simulated today).
- **Escrow / deposit**: money held during a job and released or refunded depending on outcome (simulated today).
- **Tier**: verdict ($0.10), evidence ($0.50), audit ($2.00): how much detail the researcher buys per query.
- **Rate multiplier**: a lab's published markup on the base tier price (x0.8 to x1.4).
- **x402**: an HTTP payment standard: a server answers 402 Payment Required with a price; the client signs a payment; a facilitator settles it on chain; the server then answers. Machine-to-machine, per request.
- **402 / 409**: HTTP codes. 402 = pay first. We return 409 (refused) before payment when the gate says no, so refusals are free.
- **Facilitator**: the x402 service that verifies the signed payment and submits it to the chain.
- **USDC**: a dollar-pegged token. **Test USDC**: the free testnet version, no real value.
- **Base Sepolia / testnet**: a test blockchain (chain id eip155:84532). Transactions are real and public, money is not.
- **Transaction hash**: the on-chain receipt for a payment; viewable on basescan.
- **Gas**: the fee to write to a blockchain. With x402's exact scheme the facilitator pays it, so the researcher needs no ETH.
- **Wallet**: a key pair; the researcher's address receives test USDC and signs payments.
- **Pre-registration**: writing down the analysis plan and decision rules before seeing results, so they cannot be changed to fit.
- **PICO**: Population, Intervention/exposure, Comparison, Outcome: the standard way to state a clinical question.
- **Estimand**: the exact quantity being estimated, as a formula (e.g. the MH odds ratio).
- **Protocol hash**: SHA-256 fingerprint of the pre-registered protocol, stored before any data moves.
- **SHA-256**: a function that turns any text into a fixed 64-character fingerprint; any change produces a different fingerprint.
- **Hash chain**: each log entry includes the fingerprint of the previous entry; editing one breaks every link after it.
- **Tamper-evident**: you cannot prevent edits, but any edit is detectable.
- **Audit trail**: the ordered, hash-chained record of everything the run did.
- **Human sign-off**: accept, challenge or reject, appended to the chain; nothing is a conclusion before it.
- **Challenge**: freezes the run as disputed with a note; to change a conclusion you re-run with a new protocol.
- **SSE (server-sent events)**: how the browser receives each step live as it happens.
- **Fast mode / deep mode**: vector panel (code and ML, one optional reword call) / LLM debate (agents argue and vote).
