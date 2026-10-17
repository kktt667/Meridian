# CRITERIA — the biomed pivot (pitch + demo plan)

**AI reads the chart. Maths decides eligibility. Every patient gets a record, not a story.**

Same engine as RUBRIC (SPEC.md, MATHS.md). New document: a patient's clinical notes.
New rubric: a clinical trial's eligibility criteria. New decision: *eligible / ineligible /
refer to coordinator*. Nothing in the maths changes. The market, the evidence and the
regulatory timing are all stronger than hiring. Read this once, then build.

("Unicorn TVL" read as: a wedge with a credible path to a $1B+ company. Section 6.)

---

## 1. The research that makes this a business (verified Oct 2026)

| Finding | Source | Use on the slide |
|---|---|---|
| Nine LLMs screened physician-validated vignettes against real Phase II–III protocols in a control version and **33 identity variants** (same clinical facts, different labels). Eligibility was mostly stable, but the **"homeless" label cut the eligibility score (−0.12, p<.001)** and tore through adherence/resource judgments (−0.60). The bias appears exactly where the model must *infer* rather than *read*. | Soffer, Omar, Nadkarni, Klang et al., medRxiv Nov 2025 → **JAMIA 2026;33(8):1504** | "The label did the deciding." Our demo reproduces this test live. |
| Adding race/sex to a note that is **irrelevant to matching** changed the trial recommendations LLMs generated (NDCG@10 varied by group; e.g. Llama-3 8B 67.7% Native American vs 72.6% Asian). | Ji et al., **npj Digital Medicine 2025;8:246** | "Irrelevant information moved the answer." |
| Nine LLMs, 1,000 ED cases, labels varied, clinical facts fixed: Black, unhoused and LGBTQIA+ labels pushed toward more urgent care, invasive procedures and psych evaluations; high-income → more advanced testing. Labels shifted recommendations more than guidelines did. | Omar et al., **Nature Medicine, 7 Apr 2025** | "This is the whole field, not one model." |
| Reasoning models (o3-mini, DeepSeek-R1) still misrepresent race for 78–89% of conditions tested. **Newer ≠ fairer.** | JMIR 2026;e82256 | Kills "just wait for GPT-6". |
| The industry is nonetheless shipping LLM screeners: TrialGPT (NIH; 87.3% criterion accuracy vs experts 88.7–90%, 42.6% screening-time cut). Real-world adaptation at UT Health San Antonio with a local Mistral-7B: sensitivity 81.8%, specificity 97.8%, PPV 75%; **no subgroup or fairness analysis at all**; manual screening took ~20 min/patient. | Jin et al. Nat Commun 2024; **JAMIA 2026;33(4):909** | "Deployed, useful, and nobody checked for whom it fails." |
| Benchmark criteria already contain proxies. The n2c2 2018 cohort-selection task (288 patients, 13 criteria) includes **"patient must speak English"** and **"makes own medical decisions"** as eligibility criteria. | n2c2 2018 Track 1 | Live proxy-audit example. |

**The one-line thesis:** the papers show LLMs are safe when they *read* a fixed criterion and
unsafe when they *infer* about a person. So make the LLM read only, and let a transparent gate
infer nothing. That is exactly what RUBRIC does.

## 2. Why the economics are better than hiring

- **Recruitment is the bottleneck of drug development.** In a study of 114 UK trials, only 31%
  met enrollment goals; a third of publicly-funded trials needed a time extension; ~20% of
  cancer trials fail to complete (Fogel 2018 review, Contemp Clin Trials Commun, citing Bower,
  Campbell, Stensland). Phase III trials burn tens of thousands of dollars per day of delay
  (vendor-cited Tufts figure, ~$56k/day; treat as indicative).
- **The AI-recruitment market is small but compounding:** ~$0.67B (2026) → $2.1B (2031), 25%
  CAGR (Mordor). Deals in the last 18 months: Tempus acquired Deep 6 AI (Mar 2025); Biorce
  $52.5M Series A (Feb 2026); Triomics $22M Series B (May 2026); Iterative Health $77M Series C
  (May 2026); IQVIA launched IQVIA.ai (Mar 2026) claiming 150 agents to halve Phase III
  enrollment timelines. **Every one of them puts an LLM in the deciding seat.**
- **Equity is a sponsor cost line, not charity.** FDORA 2022 made Diversity Action Plans
  statutory for Phase III; the FDA's June 2024 draft guidance was pulled (Jan 2025), restored by
  court order (Feb 2025), and is still not final, but the statute stands and sponsors are
  already writing DAPs. NIHR INCLUDE in the UK requires trials to show how they reach under-
  served groups. A screener that *proves* it did not exclude on proxies is a DAP deliverable.

## 3. Why now (UK first, dated)

- **6 Oct 2026** — the UK government accepted all 44 recommendations of the National Commission
  into the Regulation of AI in Healthcare (report 10 Sep 2026): lifecycle regulation,
  post-market monitoring, performance across settings and populations, transparency.
- **31 Oct 2026** — MHRA **AI Airlock Phase 3** application window for the first wave; theme is
  **post-market surveillance**. Our equity ledger *is* post-market surveillance. Apply.
- **5 Feb 2026** — UK GDPR Art. 22A–D in force: solely-automated significant decisions must
  give the outcome, allow representations, offer human intervention and a contest route. The
  decision record (SPEC §S5) produces all four.
- **2 Dec 2027** — EU AI Act Annex III high-risk obligations (delayed by the Digital Omnibus).
  Annex III point 5 covers AI assessing eligibility for healthcare services. Trial eligibility is
  the adjacent case; waiting-list prioritisation and prior authorisation are squarely in.

## 4. The product (what changes vs RUBRIC)

```
S0 CRITERIA   coordinator pastes NCT protocol → structured criteria vector + PROXY AUDIT
              flags: "speaks English", "reliable transport", "owns smartphone", "no substance
              use history", "lives within 50 miles", "stable housing", BMI caps
              → keep with justification / soften to a support flag / remove. Hashed + versioned.
S1 PRIVACY    notes redacted (name, DOB, address, race, insurance, employment, housing).
              Readers never see identity. Identity exists only in the audit set.
S2 FACT JURY  3 model families transcribe the SAME schema: HbA1c, creatinine, MI < 6 mo,
              on insulin, BMI, age… each with a verbatim evidence quote. Vote → fact / unresolved.
S3 GATE       inclusion/exclusion = deterministic rules on facts. Conformal band per group on
              the calibration set → "refer to coordinator" when uncertain. No LLM verdict ever.
S4 STRESS     10,000 perturbations per patient + the 33-identity swap from Soffer et al.
              Identity-flip rate is 0 by construction (identity never reaches the gate).
              Proxy-criterion sensitivity: which criterion excludes whom.
S5 LEDGER     per-trial equity ledger: screened / eligible / referred / excluded-by-criterion,
              by group, with the justification text for each proxy criterion. Export → DAP /
              NIHR INCLUDE / MHRA post-market report. Patient-facing record + contest route.
```

Things that stay exactly the same: features registry pattern (`pipeline/features.py`),
readers (`pipeline/readers.py`), jury voting, conformal envelope, einsum stress test,
closed-form counterfactual, fairness refit.

## 5. The demo (what the judges see, 4 minutes)

1. **Paste a protocol.** Real NCT criteria. The proxy audit lights up "must speak English" and
   "reliable transportation". Coordinator types a justification for one, softens the other to a
   "support needed" flag. Rubric hash appears.
2. **One patient, thirty-three identities.** Same synthetic note; a dropdown swaps the
   demographic sentence (race × sex × housing × insurance × employment × LGBT+ × disability).
   Two panels run side by side:
   - **LLM-as-judge** ("Is this patient eligible?") → eligibility flips across variants.
     Print the flip rate and the variant that caused it (expect housing/SES, per Soffer).
   - **CRITERIA** → the fact vector is identical in all 33 runs, gate output identical, flip
     rate 0.00. Evidence quotes shown next to each fact.
3. **Press "Run trial".** 600 synthetic patients × 3 readers stream through the dashboard:
   jury agreement, eligible / refer / excluded, cost so far (~0.2p/patient), coordinator queue
   with the three disagreeing quotes for each unresolved fact.
4. **The ledger.** Excluded-by-criterion by group. The "English" criterion accounts for X% of
   exclusions in group Y. Click → the justification the coordinator wrote in step 1. Export.
5. **Failure demo (required by any serious judge).** Feed 100 notes from a different
   condition: conformal coverage alarm, refer-rate spike, the engine refuses rather than guesses.

Runnable scaffold for steps 1–2 is in `trial/` (offline, no key needed; add a key for the
LLM-judge panel). See the bottom of this file.

## 6. Business and the unicorn path

**Wedge (year 1).** Per-trial SaaS to sites and CROs: £2–5k per site per trial for pre-screening
plus the equity ledger. 20-minute manual screens → seconds; coordinator only sees the refer
queue. Sell the ledger to the sponsor as the DAP/INCLUDE evidence pack.

**Expansion (year 2–3).** The same engine is the compliance rail for every regulated
"document → facts → gate" decision: prior authorisation (the nH Predict litigation in the US
is the cautionary tale), transplant listing, waiting-list prioritisation (EU Annex III), NHS
referral triage. Readers are commodities; the audited gate and the ledger are the moat.

**Why $1B is plausible.** Tempus paid for Deep 6 to own trial matching; IQVIA is building the
same thing in-house. Neither can show a regulator an identity-flip rate of zero with a proof,
because an LLM made the call. We can. The company that owns the *audit rail* for clinical AI
decisions becomes mandatory infrastructure once lifecycle regulation lands (UK 2027+, EU Dec
2027). Comparable: Holistic AI (SotonShip's technical sponsor) does exactly this for
enterprise AI governance; we do it inside the clinical decision, with maths instead of
checklists.

**Honest caveats to say out loud if asked.** FDA DAP guidance is not final. UK data on ethnic
representation in trials is patchy (NIHR INCLUDE's own finding: trials often don't collect it,
which is an argument *for* the ledger). The "80% of trials miss timelines" statistic circulates
widely but we could not trace it to a primary source; use Bower's 31%.

## 7. Three-minute pitch script

**0:00** "Same patient. Same labs. Same scan. We changed one word in the note, *homeless*, and
the AI screener said she was no longer eligible for the trial. That's not our experiment. That
is Mount Sinai, nine models, JAMIA, this year."

**0:25** "Trial recruitment is the slowest, most expensive step in medicine. Only a third of UK
trials hit enrollment. So the industry is doing the obvious thing: Tempus, IQVIA, Triomics,
Biorce, half a billion in the last eighteen months, all putting an LLM in the chair that decides
who gets screened in. None of them published subgroup performance."

**0:50** "The research says something precise: LLMs are reliable when they *read* a fixed
criterion and unreliable when they *infer* about a person. So we split the job. Three different
models read the chart and transcribe facts, with a quote for each. They vote. A transparent gate
applies the criteria. The gate never sees who the patient is. Identity cannot move the decision,
and we can prove it, not just test it." *(data-processing inequality, one slide)*

**1:30** *(demo)* protocol → proxy audit → 33-identity swap: judge flips, CRITERIA doesn't →
ledger.

**2:30** "Why now: the UK accepted all 44 AI-in-healthcare recommendations eleven days ago.
MHRA's Airlock Phase 3 is post-market surveillance; applications close in two weeks; our
ledger *is* post-market surveillance. Art. 22 rights have been live since February. The EU
deadline is December 2027. Everyone who put an LLM in the chair will need what we built."

**2:50** "We sell the ledger to the sponsor and the screen to the site. Then the same rail goes
under prior auth, transplant listing and waiting lists. Thank you."

## 8. Judge Q&A (add to the MATHS.md flashcards)

- **"Isn't excluding on English a legitimate criterion?"** Sometimes (consent, PROs). That is
  why it's a *justified, logged, measured* criterion, not a hidden one. The ledger shows what
  it cost in each group; the sponsor decides whether an interpreter line is cheaper.
- **"The LLM reader could still misread one group's notes."** Yes. Three families vote, every
  fact has a quote, disagreement goes to a human, and we report extraction accuracy *per group*
  on the calibration set. The error becomes a measured number, not a vibe.
- **"Why not fine-tune a fair model?"** EquityGuard (Ji et al.) narrows the gap; it doesn't
  prove anything, and it needs weight access. Our guarantee lives at the interface, so it holds
  for closed models too.
- **"Clinical notes are messier than CVs."** Correct, and that is why the refer rate matters.
  UT Health's TrialGPT false negatives sat just under the threshold: that is what a conformal
  band exists for.
- **"Is this a medical device?"** Decision support for research screening, human in the loop:
  likely not a device, but the Airlock exists precisely to find out, which is why we apply.

## 9. Build plan for the weekend (delta from the hiring build)

| When | What | Reuses |
|---|---|---|
| Fri night | `trial/criteria.py` registry (done), synthetic patient generator (Opus, 600 notes with hidden truth + identity labels), 33-variant renderer | SPEC §5 generator |
| Sat am | Proxy audit over pasted NCT text (regex + one Haiku call), rubric hash | S0 |
| Sat pm | Readers on notes (schema swap only), jury, gate, conformal, stress + identity swap | S2–S4 unchanged |
| Sat night | Ledger + decision record + dashboard panels (judge vs CRITERIA) | S5 |
| Sun am | Failure demo, PREREG results, slides | §5.5 |

Data: synthetic first (label everything SYNTHETIC). Public patient descriptions + real
ClinicalTrials.gov criteria are in the TrialGPT repo (SIGIR 2016 shipped; TREC 2021/22 via
wget). n2c2 2018 needs a DUA; don't wait for it.

## 10. Sources

- Soffer et al., *Sociodemographic bias in LLM clinical trial screening*, JAMIA 2026;33(8):1504 — https://pubmed.ncbi.nlm.nih.gov/42118957/
- Ji et al., *Mitigating the risk of health inequity exacerbated by LLMs*, npj Digit Med 2025;8:246 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12049425
- Omar et al., *Sociodemographic biases in medical decision making by LLMs*, Nat Med 2025 — https://www.nature.com/articles/s41591-025-03626-6
- JMIR 2026 reasoning-model stereotypes — https://www.jmir.org/2026/1/e82256
- TrialGPT, Nat Commun 2024 / PMC10418514; real-world adaptation JAMIA 2026;33(4):909 — https://academic.oup.com/jamia/article/33/4/909/8460629
- TrialGPT code + data — https://github.com/ncbi-nlp/TrialGPT
- n2c2 2018 cohort selection — https://n2c2.dbmi.hms.harvard.edu/data-sets
- Fogel 2018 review (enrollment statistics) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6092479
- Market: Mordor Intelligence, AI in clinical trial patient recruitment — https://www.mordorintelligence.com/industry-reports/ai-in-clinical-trial-patient-recruitment-market
- FDA DAP status — https://www.ajmc.com/view/fda-quietly-removes-draft-guidance-on-diversity-in-clinical-trials-following-executive-order-on-dei
- UK National Commission (44 recs) — https://www.gov.uk/government/publications/national-commission-into-the-regulation-of-ai-in-healthcare-recommendations-for-a-future-regulatory-framework
- MHRA AI Airlock Phase 3 — https://www.fieldfisher.com/en/insights/mhra-ai-airlock-from-pilot-sandbox-to-scaling-regulatory-pathway-for-ai-medical-devices
- EU AI Act delay to Dec 2027 — https://www.traverssmith.com/knowledge/knowledge-container/eu-agrees-to-delay-key-ai-act-compliance-deadlines/
- NIHR INCLUDE — https://evidence.nihr.ac.uk/alert/include-roadmap-can-help-researchers-make-trials-more-inclusive/
- SotonShip 26 (tracks, VC-for-a-day, live demo fair; Holistic AI technical sponsor) — https://sotonship.space
