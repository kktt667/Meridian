# Envoy (working name): research that crosses borders, without the data crossing borders

## The problem
The answers to many health questions are split across hospitals and countries. A US clinic wants to know
whether its East Asian patients need different stroke prevention, but the data that could tell them sits in
Korea, and patient records can't legally or safely be shared. So smaller labs and under-represented
groups go unanswered. AI could help, but it makes things up, can leak private data, and can't show its work.

## The solution
Every lab gets an AI agent that guards its own data. A researcher asks a question. Their agent:

1. **Breaks the question into parts:** what must be true, what can be measured, what the maths is.
2. **Answers what it can at home**, and lists what it can't answer alone.
3. **Hires other labs' agents** that advertise what data they hold and their rules. It pays by x402 micropayment,
   held in escrow and refunded if the job is rejected.
4. **Each lab runs the analysis on its own data.** Only safe summary numbers leave the lab.
5. **AI agents argue for and against, and look for edge cases.** A panel of differently trained models votes,
   and labs can object to how their own data was used.
6. **The researcher gets a verdict**, a certainty grade, the evidence and a tamper-proof audit trail, then signs it off.

**The rule that makes it trustworthy:** the AI can never claim more than the numbers allow. Statistics set the
maximum verdict, and the AI can only be more cautious.

## Proof it works (real public data from 4 national surveys)
- **US asks Korea:** among US Asian patients, heart events are somewhat more often strokes than among White patients
  (odds 1.3x). In Korea the tilt is much stronger (odds 3.0x). Verdict: don't copy Korea's numbers onto US patients.
  The system stops an over-generalisation.
- **Korea asks the US and Brazil:** Korea's own gate refuses (too few young cases). The other labs answer:
  a hypertension diagnosis goes with about 4x the odds of stroke in young adults, versus about 2x at 65+.
- **Korea asks the US:** 9 in 10 (90%) young Koreans with a high blood-pressure reading were never diagnosed,
  versus 55% in the US. A clear target for screening.

## The pitch (30 seconds)
> Health research is stuck behind borders. We built a network where AI agents from different labs hire each other,
> answer questions using each lab's data without moving it, argue both sides, and can never claim more than the
> statistics allow. Every step is paid, gated and audited. A small lab gets a conclusive answer it could never get
> alone, and no patient record leaves the building.

## Technical features
| Feature | What it does |
|---|---|
| **Privacy gate in every lab** | Fixed query types only; minimum group sizes (50 people, 5 per cell, 20 cases); blocks repeated "differencing" queries; output schema forbids any extra field |
| **Local-first decomposition** | Question → facts needed → measurable relationship → what home can answer → what needs help |
| **Deterministic statistics** | Mantel-Haenszel, random-effects meta-analysis, heterogeneity (I²), transportability tests; milliseconds, no AI |
| **Verdict ceiling** | Each claim gets the strongest verdict the numbers allow; AI votes can downgrade, never upgrade |
| **Certainty grade** | GRADE-style: risk of bias, inconsistency, indirectness, imprecision |
| **Multi-agent debate** | For, Against and Edge-case agents; every point must cite evidence or it is dropped and counted |
| **Diverse voting panel** | Different open-weight model families (Qwen, Llama, Mistral) running locally; lab and country names hidden |
| **Lab stewards** | Labs can confirm or object to how their own data was interpreted |
| **Agent marketplace** | Labs publish machine-readable cards (data, rules, price); reputation computed from verified history, not self-description |
| **x402 payments + escrow** | Pay per query; released only when the gate approves; refunded on refusal; penalties only for verifiable failures, never for negative results |
| **Tamper-evident audit** | Hash-chained log of every query, payment, vote and sign-off; per-operation traces with timings |
| **Human sign-off** | Accept, challenge or reject; nothing is final without a person |
| **Local models** | All agent reasoning runs on open-weight models on the laptop; no data or documents sent to external AI |

## Honest limits
Survey data is self-reported and cross-sectional (associations, not causes). Payments run on a test network.
Labs in the demo stand in for the institutions that own each dataset. This is research support, not medical advice.
