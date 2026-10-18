# Bias report (synthetic digital twins)

Generated 2026-10-18 02:13 in 6.1s. Seed 2026. SYNTHETIC data only: every twin is generated from chosen parameters so the truth is known. The same gate (`envoy/lab.py`) and statistics (`envoy/stats.py`) as the live system; no language model is involved.

## 1. Transport test: does evidence from a majority elsewhere apply to a minority here?

Country A has a majority (5,000) and a minority (n). Country B is the minority's home population (5,000). We plant the truth: the effect in B is `ratio` times the effect in A's minority. ratio 1.0 = the evidence transfers; the flag rate there is the false-alarm rate. ratio > 1 = it does not transfer; the flag rate is the detection rate.

| minority n | true ratio | flagged / answered | rate | gate refused |
|---|---|---|---|---|
| 500 | 1.0 | 4 / 200 | 2% (false alarm rate) | 0 |
| 500 | 1.5 | 47 / 200 | 24% (detection rate) | 0 |
| 500 | 2.0 | 117 / 200 | 58% (detection rate) | 0 |
| 1000 | 1.0 | 1 / 200 | 0% (false alarm rate) | 0 |
| 1000 | 1.5 | 110 / 200 | 55% (detection rate) | 0 |
| 1000 | 2.0 | 188 / 200 | 94% (detection rate) | 0 |
| 2000 | 1.0 | 4 / 200 | 2% (false alarm rate) | 0 |
| 2000 | 1.5 | 139 / 200 | 70% (detection rate) | 0 |
| 2000 | 2.0 | 196 / 200 | 98% (detection rate) | 0 |

## 2. Small-group test: alone vs network (the Q6 situation)

Hypertension -> stroke among young adults, true OR 3.0. The home lab has 2,000 young adults; the network adds labs with 10,000, 20,000.

- Home lab alone released an estimate: **1 / 200** trials.
- Through the network (>= 2 labs): **200 / 200** trials.
- Pooled OR median 3.0 (truth 3.0); the 95% CI contained the truth in 96% of answered trials.

## 3. Privacy invariant: what left the labs

- 6,000 results crossed a lab boundary (5,801 answered, 199 refused).
- 29,602 stratum cells released; smallest 5 (policy minimum 5); **0 below policy**.
- 401 model estimates released; fewest exposed cases 20 (policy minimum 20); **0 below policy**.
- Fields that ever left a lab: `cases_exposed, checks, ci_high, ci_low, compute_trace, estimate, lab_id, log_estimate, n_bucket, operation, query_id, refusal, result_hash, se, status, strata, strata_suppressed`. No row-level field exists in the schema.

## 4. Planner: routing without a language model

A TF-IDF + logistic-regression classifier picks the analysis; the catalog fills every other slot (groups, local vs remote labs); fixed rules screen the purpose. 60 labelled questions, leave-one-out accuracy **78%**; held-out paraphrases **7 / 8**. Below confidence 0.5 it abstains (recipe 'unsupported') instead of guessing. Same question, same plan, every time.

## Limits

- Twins are simple: one confounder (age), no survey weights, no missingness. Real data is harder.
- The planner's training set is small and written by us; accuracy on real researchers' phrasing is untested.
- The transport test has power only when the minority is large enough for the gate to release cells.
- This measures the deterministic layer. The language-model layer (deep mode) is blinded to lab names but not yet measured here: a name-swap test on it is the next step.
