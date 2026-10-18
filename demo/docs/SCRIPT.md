# Recording script: technical demo

Record `http://localhost:8000/?present=1&replay=20261018-100448-926112` at 1600x1000 or wider, dark theme.
Each row = one banner on screen. **Screen** is what the viewer sees. **Happening** is the technical truth. **Say** is the
voiceover, medium-technical. Total about 3:00; the 1:30 cut is marked with *.

---

### Cold open (before pressing play, 0:00-0:12) *
**Screen:** the map: Korea on the left, three labs on the right, each in a dotted ring.
**Say:** "Most clinical evidence comes from majority populations, because the data that would fix that cannot legally
move. This is a network where the data never moves. Each lab keeps its records behind its own gate, and only the
question travels."

### 00 Four health databases in three countries *
**Screen:** banner; four nodes; the ring around each lab.
**Happening:** four real national surveys (Korea KNHANES, US BRFSS, US NHANES, Brazil PNS), each loaded only inside its
own process. The ring is the gate: code that decides what may leave.
**Say:** "Four national health surveys from three countries. None may share a single patient record. The dotted ring is
each lab's gate: its own rules, enforced in its own process."

### 01 A researcher in Korea asks a question *
**Screen:** right panel: four bars, one lit at 0.81; chips with the words that decided it; "answerable at home" vs
"needs other labs".
**Happening:** a TF-IDF + logistic-regression classifier picks one of three approved analyses with a calibrated
probability (abstains below 0.5). The catalogue, not a model, decides local vs remote: which labs publish the
variables this analysis needs. Purpose is screened by fixed rules.
**Say:** "Is high blood pressure a bigger stroke risk in young adults than in older adults? Korea alone has too few
young stroke cases. A small classifier picks the analysis in milliseconds and shows the words that decided it. The
catalogue works out what Korea can answer itself and what needs other countries. No chatbot anywhere in this step."

### 02 Labs that could help bid for the job *
**Screen:** lines to the labs; two go solid, one fades; bid cards: capability, track record, price, hired / not hired.
**Happening:** each lab dry-runs its own gate on the job (pre-flight: yes/no only, no numbers). Accepted labs bid with
facts: published price x queries, pre-flight capability, reputation computed from the platform ledger. Selection is
capability, then answer rate, then SLA failures, then price, with a written reason per lab. Deposits are refunded to
the labs not hired.
**Say:** "Labs do not describe themselves. Each one proves it can answer by test-running its own gate, then bids with a
price and a track record the platform computed from past jobs. Two are hired on record and price. The third can only
partly answer and is not hired. No sales pitch counts."

### 03 The plan is locked before any data is touched *
**Screen:** padlock on Korea; a SHA-256 fingerprint; four facts to establish.
**Happening:** the protocol (population, exposure, comparison, outcome; the facts; the estimand; the decision rules)
is hashed into the audit log before the first query.
**Say:** "Before any data is touched, the whole plan is fingerprinted: the facts to establish and the rules that will
decide them. Nobody can move the goalposts after seeing results."

### 04 Each lab answers from behind its gate *
**Screen:** Korea's ring flashes red; then packets fly to the US and Brazil, rings flash green, a sigma flies back to
the centre, gold coins fly out. Feed: refused / answered · counts only / paid · receipt.
**Happening:** inside each lab, in order: schema check, query budget, differencing check, group and variable
availability, minimum population, then the operation (here logistic regression with age and sex covariates). Korea
has fewer than 20 young stroke cases, so its own gate refuses its own researcher, free. The US and Brazil release only
a coefficient, a standard error and a size bucket. Each answered query is paid in USDC over x402 on a public test
chain; a refusal returns 409 before payment.
**Say:** "Watch Korea first: its own gate refuses its own researcher. Fewer than twenty young stroke cases, too few to
release safely. Nothing leaves, nothing is charged. That refusal is why the network exists. Now the US and Brazil: the
packet is the question, the green flash is the gate approving, the sigma coming back is a summary, a coefficient and
an error bar, never a person. The coin is the payment, settled on a public chain, one receipt per answer."

### 05 The summaries are combined with fixed statistics *
**Screen:** three intervals on a chart with a dotted "no effect" line; three claims with their ceilings.
**Happening:** DerSimonian-Laird random-effects pooling of the released log odds ratios, Cochran's Q and I-squared for
disagreement, a ratio of odds ratios for young vs old. Fixed rules set a ceiling per claim: supported if the interval
excludes no-effect, inconsistent if the labs disagree beyond chance (Q p < 0.10 and I-squared >= 50%).
**Say:** "Fixed statistics, no model. Young adults: hypertension multiplies the odds of stroke four-fold, consistent
across labs: supported. The effect is nearly twice as strong in the young as in the old: supported. But at 65 and
over the labs disagree: I-squared 71 percent. A normal analysis would average them. This one refuses: inconsistent."

### 06 Three independent reviewers vote on every claim
**Screen:** three reviewer nodes under the evidence; a grid of weaknesses (red = worse); three dots per claim.
**Happening:** eight bias features per claim computed from the aggregates (imprecision, closeness to no-effect,
inconsistency, refused share, single source, cross-population, small cells, weak effect). Three platform lenses with
published weights: methodologist, clinician, ethics/regulatory. Score = features x weights; above 1.6 the lens votes
to downgrade. Final verdict = statistics ceiling unless the majority votes more cautious. Every vote carries its
contribution vector. (Deep mode swaps this for three open-weight language models arguing, blinded to lab names.)
**Say:** "Three independent reviewers, built into the platform, not the labs: a methodologist, a clinician and an ethics
reviewer. Each votes on every claim: keep the verdict the statistics allow, or make it more cautious. The votes are
scored from eight weaknesses in the numbers, not from words, and each comes with its reasons. The red cell is the
disagreement at 65 plus. A reviewer can never make a claim more confident."

### 07 The write-up is produced by code
**Screen:** spinner, then "model rewrite rejected, code version shown".
**Happening:** a template narrative is generated from the audit objects, so every number is traceable. A local
open-weight model (qwen2.5) receives it with countries replaced by Lab A / Lab B and may only reword. A guard rejects
any new number, dropped claim ID, changed verdict word, or loaded word (risk, cause, significant). Here it was
rejected; the template stands.
**Say:** "The write-up is written by code from the numbers. The language model is allowed to do one thing: reword it,
with the countries hidden. A guard checks every number and every verdict word. Here it rejected the rewrite for
changing the wording, so the code's version is shown. The model never decides anything."

### 08 Verdicts *
**Screen:** three verdict cards. C3: inconsistent, with "a normal analysis: ~~supported~~" and the amber reason.
**Happening:** per claim, the pipeline also computes what a naive analysis of the same numbers concludes (assume
transfer, pool despite disagreement, no review). A name-swap test renames every lab and reruns the panel: identical
votes.
**Say:** "Same numbers, two answers. A normal analysis says supported. This one says the labs disagree, and shows you
both. Rename every lab and rerun: the votes do not change. Identity is not an input."

### 09 Did the bias checks work? (tab; click Plant bias now)
**Screen:** bars fill: 72 / 93 / 98 percent caught; false alarms 2-3 percent; small group 0/20 alone vs 20/20 via the
network; 0 privacy breaks.
**Happening:** synthetic twin labs are generated with a planted truth (a minority here, its home population abroad,
effect sizes differing 2x), registered in the catalogue, and pushed through the same gate and statistics live. Six
scenarios, sixty trials each, about a second.
**Say:** "How do we know the bias detection works? We plant bias in synthetic labs where we know the truth and run the
same system, live, right now. Evidence that does not transfer is caught 93 percent of the time at a minority of a
thousand people, with 3 percent false alarms. A small group that its own lab must refuse gets an answer through the
network every time. And of over a thousand released answers, zero broke the privacy rules."

### 10 Audit trail (tab; click Verify) *
**Screen:** 27 plain lines: question received, analysis chosen, labs hired, plan locked, asked / answered / paid,
panel voted, name-swap test, summary written, verdicts, human signed. Green: valid · 27.
**Happening:** every entry includes the SHA-256 of the previous one; Verify recomputes the whole chain. (Do not click
"change one value" in the video; keep it for Q&A.)
**Say:** "Every step is written down in order, each line fingerprinting the one before: what was asked, what left each
lab, what was paid, how the reviewers voted. Verify: nothing has changed since it happened."

### 11 Sign (tab; click Accept and sign) *
**Screen:** status "accepted · chain valid · 28 entries"; the human node turns green.
**Happening:** the sign-off is appended to the chain as its own entry.
**Say:** "And only now, when a qualified person signs, is it a conclusion. The signature becomes the last line of the
chain."

### Close (0:10) *
**Screen:** hold on the map.
**Say:** "Four countries, zero patient records moved, every answer paid for and receipted, every step verifiable.
The question travels. The data stays home."

---

## If asked afterwards
- **Fast vs deep:** everything you saw was fast mode: code and machine learning, one optional language-model call to
  reword. Deep mode runs the same pipeline with three open-weight models arguing and voting, blinded.
- **Why only three analyses?** Because the gate can be proven over three. New analyses are added by writing the
  operation and its gate checks, never by loosening the gate.
- **What is simulated?** Deposits, stakes, subscriptions. Payments are real on a test network; the gate, statistics,
  reviewers, twins and audit are real.

---

## Voiceover, read straight through (about 3:00)

[Map, before pressing play]
Most of the world's clinical evidence comes from a minority of its people. The data that could fix that exists, in
national health surveys and hospital records on every continent, but it cannot legally move. So the medicine we
practise on minorities is borrowed from majorities, and nobody checks whether the loan is good.

This is a network where the data never moves. Four national health surveys, from Korea, the United States and Brazil,
each sitting behind its own gate: code that decides what may leave, enforced inside the lab's own process. Only the
question travels.

[01]
A researcher in Korea wants to know whether high blood pressure is a bigger stroke risk in young adults than in
older ones. Korea alone has too few young stroke cases to answer. So the first thing the system does is read the
question. Not with a chatbot: with a small classifier that picks one of the approved analyses and shows the words that
decided it. Then the catalogue, not a model, works out what Korea can establish itself and what needs other countries.

[02]
Next, finding help. Labs on this network do not describe themselves. Each one proves it can answer by test-running
its own gate on the job, then bids with a published price and a track record the platform computed from past work.
Two are hired, on record and price. The third can only partly answer, so it is not hired and its deposit goes back.

[03]
Before a single number is released, the whole plan is fingerprinted: the facts to establish and the rules that will
decide them. Nobody can move the goalposts after seeing the results.

[04]
Now the labs answer, and watch Korea first. Its own gate refuses its own researcher: fewer than twenty young stroke
cases, too few to release safely. Nothing leaves, nothing is charged. That refusal is the reason this network exists.
The question goes to the United States and Brazil. The green flash is each gate approving. What comes back is a
summary: a coefficient and an error bar, adjusted for age and sex, never a person. And the coin is the payment, settled
on a public chain, one receipt per answer.

[05]
The summaries are combined with fixed statistics, no model involved. In young adults, hypertension multiplies the odds
of stroke four-fold, consistently across labs: supported. The effect is nearly twice as strong in the young as in the
old: supported. But at sixty-five and over, the labs disagree with each other, I-squared seventy-one percent. A normal
analysis would average them into one number. This one refuses, and calls it inconsistent.

[06]
Then three independent reviewers vote on every claim. They are built into the platform, not the labs: a methodologist,
a clinician and an ethics reviewer. Each one scores the claim on eight weaknesses in the numbers, imprecision, closeness
to no effect, disagreement between labs, small groups, and votes to keep the verdict the statistics allow, or to make
it more cautious. Never more confident. Every vote comes with its reasons.

[07]
The write-up is produced by code, straight from the numbers. The language model is allowed to do one thing: reword it,
with the countries hidden. A guard then checks every number and every verdict word. Here it rejected the rewrite, so the
code's version stands. The model never decides anything.

[08]
The verdicts. Where a normal analysis would have concluded something different, that conclusion is shown struck
through. Same numbers, two answers. Rename every lab and run the review again: the votes do not change.

[09, click Plant bias now]
How do we know the bias detection works? We plant bias in synthetic labs where we know the truth, and run the same
system live, now. Evidence that does not transfer between populations is caught ninety-three percent of the time, with
three percent false alarms. A small group that its own lab must refuse gets an answer through the network every time.
And of more than a thousand released answers, none broke the privacy rules.

[10, click Verify]
Everything you just watched was written down as it happened: what was asked, what left each lab, what was paid, how
each reviewer voted. Each line carries the fingerprint of the line before it. Verified: nothing has changed.

[11, click Accept and sign]
And only now, when a qualified person signs, does it become a conclusion. The signature is the last line of the chain.

[Hold on the map]
Four countries. Zero patient records moved. Every answer paid for and receipted, every step verifiable. The question
travels. The data stays home.

## Voiceover, 1:30, spoken (about 230 words)

[map] Almost everything we know in medicine comes from a handful of populations, and the data that could change that
is locked away by law. So we built a network where the data never has to move. Four national health surveys, three
countries, each one behind its own gate. The only thing that travels is the question.

[01] Here, a researcher in Korea wants to know whether high blood pressure is a bigger stroke risk in young people.
Korea alone doesn't have enough cases. A small classifier reads the question, picks the right analysis, and the
catalogue figures out which other labs can help. No chatbot involved.

[02] Those labs test the job against their own rules and bid: a price, and a track record the platform has computed
for them. Two get hired.

[03] Then the plan is locked, hashed, before anyone touches a number.

[04] Now watch Korea. Its own gate turns its own researcher down: fewer than twenty cases, too few to be safe. Nothing
leaves, nothing's charged. The US and Brazil answer with a coefficient and an error bar, never a person, and each
answer is paid for on chain, with a receipt.

[05] Fixed statistics pull it together. In young adults, four times the odds of stroke. Over sixty-five, the labs
disagree, so rather than average them, the system says so.

[06] Three independent reviewers, a methodologist, a clinician and an ethics lens, score each claim and vote. They can
make it more cautious, never more confident.

[07] The write-up is written by code. The language model only rewords it, blinded, with a guard checking every number.

[08] And here, struck through, is what a normal analysis would have told you.

[10, Verify] Every step went into a hash chain. Verified.

[11, Sign] A person signs, and only now is it a conclusion. Zero records moved. The question travels. The data stays home.
