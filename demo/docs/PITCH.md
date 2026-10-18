# Pitch script (medium technical, ~2:45 spoken; cut the bracketed lines for 1:30)

## 1. The problem (0:00-0:35)

Eighty percent of the world's clinical evidence comes from about twenty percent of its people. When a doctor in
Southampton treats a patient of Korean descent, the guidelines on the screen were built mostly on white, Western
cohorts. The data that would actually describe that patient exists: Korea runs one of the best national health
examinations in the world. But it cannot leave Korea. Not because anyone is hiding it; because the law, the licence
and the ethics board all say the same thing: patient data does not travel.

So the medicine we practise on minorities is borrowed from majorities, and nobody checks whether the loan is good.
That is a bias problem at the scale of every clinic on earth, and it will not be fixed by collecting more data into one
place, because that place is exactly what the law forbids.

## 2. The idea (0:35-0:55)

Meridian moves the question instead of the data.

Every lab, hospital or national survey keeps its data where it is and runs a small node that can answer a fixed set of
statistical questions through its own privacy gate. A researcher's agent plans the analysis, hires the labs that can
help on an open market, pays each one per answer, combines only the summary numbers that come back, and, before
anyone acts on the result, tests whether the evidence actually transfers to the patient in front of them.

[It is a marketplace for answers, where the raw data never changes hands.]

## 3. How it works (0:55-1:35)

Five mechanisms make it trustworthy.

- **The gate.** Each lab's node only answers three pre-approved statistical operations and releases only counts and
  coefficients: never a row, never a group smaller than five people, never an estimate built on fewer than twenty cases.
  It also refuses overlapping queries that could be subtracted to isolate individuals. The lab enforces this in its own
  process; the network cannot override it.
- **The market.** Labs do not describe themselves. Each one proves what it can answer by dry-running its own gate, then
  bids with its published price and its track record from the platform ledger. Selection is capability first, then
  reliability, then price, with a written reason for every outcome. Payment is per answer over x402, a pay-per-request
  web standard settled in USDC on chain; a refusal costs nothing.
- **Pre-registration.** The analysis plan and decision rules are fingerprinted before any data is touched, so the
  conclusion cannot be steered after the fact.
- **Bias-aware review.** Fixed statistics set the strongest verdict the numbers allow, including an explicit
  transportability test: does the effect measured abroad match the effect measured locally? A review panel of three
  lenses, methodologist, clinician and ethics, scores eight weaknesses of every claim and can only make a verdict more
  cautious, never more confident. Every vote comes with the numbers that produced it. The language model is allowed
  to do one thing: reword a summary that code already wrote, with lab names hidden, under a guard that rejects any new
  number, changed verdict or loaded word.
- **The audit chain.** Every step, from the question to the payment to the human's signature, is written to a
  hash-chained log. Change one value and every entry after it breaks. Nobody has to trust us; they can check.

## 4. The video: record presentation mode, read the banners

Open `http://localhost:8000/?present=1&replay=<run_id>` (a clean Q6 run, e.g. `20261018-100448-926112`). Controls and
detail text are hidden; each stage shows one large banner in the top-left. **The voiceover is the banner text.** Read it
as it appears; the map and the single panel on the right illustrate it. Pacing is ~45 s to the verdicts, then you click
the four tabs.

| Stage | Banner (say this) | What moves |
|---|---|---|
| 00 | Four health databases in three countries. None may share a single patient record. Each sits behind its own locked gate. | Korea left, three labs right, dotted gate rings |
| 01 | A researcher in Korea asks: is high blood pressure a bigger stroke risk in young adults? Korea alone has too few young stroke cases. The system works out what Korea can answer itself, and what needs other countries. | Bars pick the analysis; blue pulses to the labs it needs |
| 02 | Labs that could help bid for the job. Each proves it can answer by test-running its own gate, then bids with a price and a track record the platform computed. Two are hired. No sales pitch counts. | Two lines go solid, one lab fades; bid cards |
| 03 | The plan is locked before any data is touched. Fingerprinted. Nobody can move the goalposts. | Padlock on Korea; the fingerprint |
| 04 | Each lab answers from behind its gate. Only summary counts leave, never a person. Korea's own gate refuses its own researcher: too few people to protect. The others answer and are paid per answer, automatically. | Korea's ring flashes red; packets out, green rings, Σ back, gold coins; feed: refused / answered / paid |
| 05 | The summaries are combined with fixed statistics. Young adults: four times the odds of stroke. Over 65: the labs disagree, so the system refuses to average them into a false consensus. | Three intervals, "no effect" line; verdict ceilings |
| 06 | Three reviewers look for weaknesses: imprecise, too close to no effect, labs disagreeing, small groups. They can make a verdict more cautious, never more confident. | Reviewer nodes light up; red cell = the 65+ disagreement; three dots per claim |
| 07 | The write-up is produced by code. The language model may only reword it, with countries hidden, and a guard checks every number and verdict. | "model rewrite rejected, code version shown" |
| 08 | Verdicts. What a normal analysis would have said is shown struck through. | C3: inconsistent, "a normal analysis: ~~supported~~" |
| 09 (tab) | Did the bias checks work? We plant bias in synthetic labs where we know the truth and watch the same system catch it. Click **Plant bias now**. | Bars: 72 / 93 / 98% caught, false alarms 2-3%; small group 0/20 alone vs 20/20; 0 privacy breaks |
| 10 (tab) | The audit trail: every step, each line fingerprinting the one before. Click **Change one value in a copy**. | Chain turns red from that line |
| 11 (tab) | A human signs. Nothing is a conclusion until a qualified person accepts it. Click **Accept and sign**. | Signature joins the chain; counter: valid · 27 |

Bottom strip throughout: rows of data moved **0** · paid (with real settlements) · audit chain.

For a 1:30 cut: stages 00-05, then 08, then the tamper click, then sign. For 2:45: everything.

## 5. Scale and strategy (2:25-2:45)

Every national health survey, biobank and hospital registry on earth is a potential node, because joining means
running a small server, not handing over data. Labs earn from assets they already hold and cannot otherwise use.
Researchers and public health bodies pay per answer; pharma pays for evidence on the populations their trials never
reached. A share of every payment funds new labs in under-represented populations, so the network becomes less
biased the more it is used.

The moat is the ledger: every answered question adds verified track record and audit history that a competitor
cannot copy. We start with heart disease and stroke, the world's largest killers and the conditions where population
differences matter most.

What runs today: four national datasets from three countries, real per-query payments on a test network, a gate that
has never leaked, a bias detector we have measured, and an audit chain anyone can verify. What is simulated: deposits,
penalties and subscriptions. Everything else you saw is live.

## One-liners to keep in your pocket
- "We move the question, not the data."
- "Labs are judged on what their gate can do and what they have delivered, never on what they say."
- "The AI can make a conclusion more cautious. It can never make it more confident."
- "Same numbers, two answers. A normal analysis says yes. Ours tests it and says no."
- "Change one value and the chain breaks. Nobody has to trust us."
