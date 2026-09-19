# Research questions, hypotheses and experiment map

A guide that links every research question to its hypotheses, the experiments already run (with the notebook that analyses them), the current evidence, and what is still missing. All numbers come from the executed notebooks in this folder.

**Countries.**

| country | why it is in the thesis | language(s) | party system |
|---|---|---|---|
| **USA 2024** | benchmark with a known result; the model's "home" language | English | two parties, Electoral College |
| **El Salvador 2024** | first LLM-persona simulation of a Salvadoran election; a hegemonic candidate (Bukele 84.6%) | Spanish / English | fragmented, three candidates simulated |
| **Brazil 2026** | real programmes available; the election is still pending, so the focus is on *how* the model decides | Portuguese / English | multiparty, polarised runoff (Lula vs Flávio Bolsonaro) |

**Common setup.** Nemotron personas; gpt-oss-20b; reasoning low; temperature 0. Three persona arms:

- **demographic**;
- **cultural background**;
- **persona**.

USA adds a fourth arm, **career + Big Five**.

---

## Core research questions

### RQ1. How accurately can persona-driven LLMs reproduce real vote choice across countries that differ in language and party-system complexity (USA, Brazil, El Salvador)?

**Hypotheses.**
- **H1₀:** Accuracy does not depend on the country: the error against the real result is the same in the USA and El Salvador.
- **H1₁:** Accuracy is higher in the USA (English, two-party system, abundant training data) than in El Salvador and Brazil (non-English, more fragmented or less represented).
- **H1₂ (sub-hypothesis):** At aggregate level the model reproduces the *structure* of an electorate (ranking of regions, direction of cleavages) better than its *level* (national shares, size of gaps).

**Metrics.**

| metric | used for |
|---|---|
| national vote-share error | all countries |
| state/department winner accuracy | all countries |
| correlation of regional shares with the real result | all countries |
| Electoral College votes | USA |
| total variation distance (TVD) | all countries |
| group-gap error vs exit polls | where exit polls exist |

**Evidence so far.**

| country | best arm | national error | regional structure |
|---|---|---|---|
| USA | Cultural background: Trump 298 EV, 92.2% of states correct (real 312 EV) | Trump popular vote −2.7 pp (47.1 vs 49.8) | r = 0.89–0.90 with real state shares |
| El Salvador | Demographic · EN: Bukele 82.8% (real 84.6%) | −1.8 pp (Spanish 94.2%) | Bukele r = 0.39 (demographic · EN, 29.8% Morazán to 98.9% San Salvador); Flores r = 0.40–0.63 (persona arms) |
| Brazil | — (no result yet; election in October 2026) | — | — |

- **H1₂ is supported.**
  - Ranking and direction are right: state correlation 0.9 in the USA; the FMLN geography points the right way in El Salvador (Flores r = 0.40–0.63).
  - Levels and gaps are wrong: Black voters get 1% Trump against ~13% real; Bukele falls to 29–43% in the cultural and persona arms (Flores wins the cultural arms).
- **H1₁ is partly supported.** The USA is the most accurate case; in El Salvador the English demographic arm is close nationally (82.8% vs 84.6%) with a moderate regional correlation (r = 0.39), but Spanish overshoots (94.2%) and persona text collapses Bukele's lead.

**Missing.**
- The Brazil comparison after October 2026: first round and a possible runoff.
- Optionally, 2026 polls as an interim benchmark, clearly labelled as polls.

**Notebooks:** 02b, 02c, 04a, 04c, 06.

---

### RQ2. Which persona attributes and prompting strategies most improve national alignment with real election outcomes?

**Hypotheses.**
- **H2₀:** Adding persona text or changing the prompt design does not change alignment with the real result.
- **H2₁:** Richer persona text improves alignment, because the model can reason about the individual voter.
- **H2₂:** Prompt features that should be irrelevant also move the result, and must be controlled: prompt language, candidate order and context length.

**Evidence so far.**

| factor | USA | El Salvador | Brazil |
|---|---|---|---|
| + cultural background | **improves** (fixes LA, MI; 92.2% of states) | lowers Bukele to 36.6–43.2% (Flores wins) | Lula 80–84% (short), 87% (full) |
| + persona | **worsens** (flips 7 states to Harris) | lowers Bukele to 28.8–32.9% | Lula 80% (short), 81–90% (full) |
| + career + Big Five | no gain (269–269 tie) | — (no Big Five data) | — |
| language (local vs English) | — | 18% (demographic), 24% (cultural), 36% (persona) of votes change; direction flips with persona text | 21–22% of votes change |
| candidate order | — | **primacy**: Bukele 93.5% first vs 54.5% second | not tested (Lula always first) |
| context length (full vs short) | — | — | Lula 75–90% full vs 69–84% short; 16–21% of votes change, never the winner |

- H2₁ holds only for **cultural background in the USA**. The persona text pushes every country toward the candidate the model favours by default (Harris, Flores/Sánchez, Lula).
- **H2₂ is strongly supported**: language and order effects are as large as, or larger than, the persona effects.

**Limitation, not re-run.** The candidate order is fixed in every main arm (Bukele first in El Salvador, Lula first in Brazil). The primacy effect measured in El Salvador is reported as a limitation of the levels; comparisons between arms and languages stay valid because the order is the same in all of them.

**Done (19 Sep 2026).** All full-context Brazil runs, 3 arms × PT/EN, 50,000 each.

**Notebooks:** 02b, 02c, 04a, 04b, 05.

---

## Additional research questions (suggested)

### RQ3. Does the LLM vote on the **candidate label** or on the **programme**? (Brazil 2026, H0/H1 of notebook 05)

- **H3₀ (human-like):** vote choice follows the party label and the model's prior about each candidate; the written proposals weigh little.
- **H3₁ (expected):** vote choice follows the proposals, whatever name they carry.

**Test: candidate-label swap.**
- The same 2,000 voters (seed 2026) answer twice, with the original prompt and with the candidate names and stance labels swapped; every proposal stays in place. Repeated on 50,000 voters with the full programmes (demographic arm, PT).
- H3₁ predicts that more than 50% of voters follow the proposals; exact binomial test.
- The breakdown by original vote shows asymmetric loyalty.

**Result.** 72.9% (PT) and 71.7% (EN) of the 2,000 voters keep the name; 76.0% of the 50,000 with full programmes. Exact binomial test of 'follows > 50%': p = 1 in every case. **H3₀ is supported, H3₁ rejected.** Explanations cite proposal content in ≥ 98.6% of cases: the model argues from the programmes but decides by the name. The first recorded swap was invalid (both programmes under 'Lula') and was redone.

### RQ4. Is the simulated vote **stable**, i.e. does it depend on the persona and not on sampling noise?

- **H4₀:** Repeated identical runs give different votes.
- **H4₁:** At temperature 0 the vote is reproducible.
- **Evidence:**
  - Temperature 0: 4% of 100 voters change vote across 3 runs, against 19–22% at 0.6–1.0.
  - State shares repeat within 0.13 pp.
  - **H4₁ supported** (02a).
- **Reasoning effort:** `low` is justified by **cost and time**: fewer output tokens, lower latency and fewer GPU hours for 200,000+ requests per arm. Medium/high levels were not run.

### RQ5. Does the **prompt language** change the vote of the same person?

- **H5₀:** Same persona, same content, different language → same vote.
- **H5₁:** Language changes individual votes and national shares.
- **Evidence:** El Salvador, 18% of voters change in the demographic arm, 24% in the cultural and 36% in the persona arm; Brazil, 21–22% in every short arm. **H5₁ supported** (04a, 04b, 05).
- **Open sub-question:** is it the language of the *instructions* or of the *persona text*? A 2 × 2 design (instructions EN/local × persona text EN/local) would separate the two.

### RQ6. Do synthetic voters form a **realistic electorate** or a set of **stereotypes**?

- **H6₀:** Simulated group differences match real ones (exit polls, regional results).
- **H6₁:** The model exaggerates cleavages: group gaps are too large and each group is pushed to its "typical" side.
- **Evidence:**
  - USA: gaps 2–13× too large.
  - El Salvador: the department range reaches 69 pp (demographic · EN, 29.8–98.9%), against 15 pp in reality; with cultural text Flores reaches 89–95% in Morazán against 14% real.
  - **H6₁ supported** (02c, 04c).
- **Possible follow-up:** calibration (post-stratification of the simulated probabilities) to test whether the structure can be rescued once the level is corrected.

### RQ7. Is the **Kish multipurpose allocation** better than naive designs for LLM simulation?

- **H7₁:** For the same budget, Kish (competitive EV + state + popular vote) gives more effective information than 1,000 per state, with the precision where races are close.
- **Evidence:**
  - Design effect: 1.17 (Kish) vs 2.26 (1,000 per state).
  - The margin standard error is below the real 2024 margin in every swing state except Wisconsin.
  - Supported by design (01).
- **Missing:** an empirical check comparing Kish-weighted vs unweighted national estimates.

---

## Experiment map

| # | experiment | status | RQ | notebook |
|---|---|---|---|---|
| E1 | Kish allocation of 200,000 USA calls | done | RQ7 | 01 |
| E2 | Temperature 0 / 0.6 / 1 stability (100 × 3) + AZ/AL/VA reruns | done | RQ4 | 02a |
| E3 | USA 4 arms × 200,023 personas | done | RQ1, RQ2, RQ6 | 02b, 02c |
| E4 | TranslateGemma boundary translation, SV + Brazil | done | RQ5 (enabler) | 03 |
| E5 | El Salvador EN/ES pilot (500) + full (50,000) demographic, corrected sample (19 Sep 2026) | done | RQ1, RQ5 | 04a |
| E6 | El Salvador candidate order (1,000 × 4; first sample, demographic-only prompts, within-persona comparison) | done (reported as a limitation) | RQ2 | 04a |
| E7 | El Salvador cultural + persona × EN/ES (50,000), corrected sample (19 Sep 2026) | done | RQ1, RQ2, RQ5, RQ6 | 04b, 04c |
| E8 | Brazil short context, 3 arms × PT/EN (50,000) | done | RQ2, RQ5 | 05 |
| E9 | Brazil full context: demographic PT (clean re-run), cultural (completion), persona (new) | done (3 arms × PT/EN, 50,000) | RQ2 | 05 |
| E10 | Brazil candidate-label swap, corrected (names + stance labels; PT + EN 2,000; full PT 50,000) | done: 72–76% keep the name, H3₀ supported | RQ3 | 05 |
| E11 | Brazil vs the real 2026 result (after October 2026) | pending the election | RQ1 | — |

## Suggested priority

1. **E11**, as soon as the Brazilian result is published.

Out of scope by decision: comparing model sizes, re-running with a randomised candidate order, and testing medium/high reasoning (reasoning `low` is justified by cost and time).
