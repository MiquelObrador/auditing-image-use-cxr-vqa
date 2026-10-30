# Evaluation settings, answer extraction, checkpoints and the inverted label for `swap`

> **Public version.** Local paths and verbatim dataset text (MMXU options) have been replaced by descriptions; everything else is unchanged.

**Date: 2026-09-28.** Written in response to an external review. Checked against the code (file:line), not
against memory. The last section fixes the readings for the new swap/duplicate runs (fine-tuned models, clean
set) and for the constant-token control on the zero-shot models, **before** launching them.

## 1. Decoding (all models)

All models use **greedy** decoding (`do_sample=False`, `num_beams=1` by default) with `max_new_tokens=20`. The
temperature/top_p of each checkpoint's `generation_config.json` has no effect because sampling is off.

| model | class (decoding) | repetition_penalty | visual resolution | system prompt |
|---|---|---|---|---|
| Qwen2-VL-7B | `mmxu_eval/models/qwen2vl.py:77` | not passed (checkpoint default) | `MIN/MAX_PIXELS = 256·28², 1280·28²` (`oracle_evaluate.py:56-57`) | chat-template default ("You are a helpful assistant.") |
| InternVL2-8B | `mmxu_eval/models/internvl2.py:183` | not passed | `max_num_tiles=4` (`:139`), 448px tiles + thumbnail; ignores min/max_pixels | `internlm2-chat` template default; prompt prefixed with `Image-1: <image>\nImage-2: <image>` |
| HuatuoGPT-Vision-7B | `mmxu_eval/models/huatuo.py:84-85` | **1.0** (the checkpoint ships 1.05) | min/max_pixels as above | chat-template default |
| Qwen2.5-VL-7B | `mmxu_eval/models/qwen25vl.py:85-86` | **1.0** (the checkpoint ships 1.05) | min/max_pixels as above | chat-template default |
| A, B, B2 (TAC) | `tac_mcq_train/model_tac.py:176` | not passed | RAD-DINO fixed 518×518; min/max_pixels ignored | **"You are a helpful assistant. Answer multiple-choice questions with a single letter."** (`:36-37`) |
| C (generic) and text-only | `qwen25vl_baseline_train/model_qwen25vl.py:81` | **1.0** | evaluation with the protocol's min/max_pixels (training used 268,324 px/image) | chat-template default |

**System prompt: the TAC models carry a task system prompt and the generic model does not.**
- **TAC models (A, B, B2):** *"You are a helpful assistant. Answer multiple-choice questions with a single
  letter."* (`tac_mcq_train/model_tac.py:36-37`, set on Libra's `libra_llama_3` template, `:55-56`). The same
  prompt is used in training (`train_lora.py`, through `build_conv_prompt`) and in evaluation.
- **Generic C, the text-only model and the four zero-shot VLMs:** no task system prompt. The Qwen2/Qwen2.5-VL
  chat template adds its default ("You are a helpful assistant."). InternVL2 uses its template's default system
  message.

It is the one systematic difference in the input between the TAC models and the others. The user prompt is
identical for all: `mmxu_eval/prompts.py`, ending in "Answer with the option's letter from the given choices
directly."

## 2. Answer extraction and failure rate

`mmxu_eval/evaluate.py:22,27-33`: `parse_letter` takes the **first A–D appearing as a whole word**
(regex `\b([A-Da-d])\b`, case-insensitive) and upper-cases it. There is no fallback from option text to letter:
with no match the row is `unparseable` and counts as wrong. An exception during loading or generation gives
`error`, also counted as wrong.

Known risk: the lowercase English article "a" would match as A. **Checked across all CSVs: 0 rows in any model
or condition take the letter from a lowercase character.**

| model | unparseable | error | output format |
|---|---|---|---|
| Qwen2-VL, HuatuoGPT | 0 in every condition | 0 | bare letter (100%) |
| Qwen2.5-VL-7B | **27/2992 in `blind`**, 0 in the rest | 0 | bare letter (99–100%) |
| InternVL2-8B | 0 in every condition | 0 | **"A: <option text>" in 66–70% of rows with images; bare letter in 100% of `blind`**. In all 2,108 (full), 1,968 (hole) and 1,966 (hole_random) "letter: text" answers, the text is that letter's option — no letter/text contradictions |
| A, B, C, text-only | 0 in every condition | 0 | bare letter (100%) |
| B2 | 5/3096 (clean) and 4/2992 (MMXU-test) in `blind`, 0 in the rest | 0 | bare letter |

For InternVL2, `blind` changes the output format as well as the input. The parser absorbs it without error, but
this is one of the reasons for the constant-token control (§6).

## 3. Exact checkpoints

No `revision=` is pinned anywhere in the repo. The cached snapshot (= `refs/main` at download) is the one used:

| model | HF id | snapshot |
|---|---|---|
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| InternVL2-8B | `OpenGVLab/InternVL2-8B` | `6fb9ad6924f69424e57fab2ab061d707688f0296` |
| HuatuoGPT-Vision-7B | `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` | `451ac32400e36cfd07b41b62cbe63e6894895b38` |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| TAC models: LLM | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| TAC models: TAC (A) | `X-iZhang/libra-v1.0-3b` (`mm_tac_projector.bin`) | `a9a3f66296682a0fe7eb9cc67a564774ed195eb8` |
| TAC models: encoder | `microsoft/rad-dino` | `110cbc18d5133582e320b43d53bf5c44e410c936` |
| C and text-only | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |

## 4. Backbone of the text-only model (defines the hard subset)

**Qwen2.5-VL-3B-Instruct**, the same checkpoint as the generic model C (`train_lora_textonly.py:42`). It is
fine-tuned with LoRA on the clean split (`qwen25vl_baseline_train/runs/textonly_clean`, 93,693 rows,
`splits_clean_2026-09-24.json`). It never receives images, in training or evaluation. The vision tower is present
but unused, with no LoRA. Same data, formats, permutation and steps as C.

## 5. The inverted label for `swap`

**How it is built** (`mmxu_eval/oracle_evaluate.py:74-120`):
- `label_option` assigns each option a class with three regexes (improved / worsen / no change). Exactly one
  match gives that class, none gives NONE, more than one gives AMBIG.
- The class of the correct option is inverted: worsen ↔ improved; no change stays no change.
- `answer_inverted` is the letter of the single option with the inverted class. If there are zero or several,
  it is left empty.
- Consequence: in `no change` the "inverted label" is the correct answer itself, so it is not informative.

**Problem with the current rule, found in the audit:** it inverts the *lexical* class of the correct option, not
the question's `content_type`. When the two disagree, it builds a wrong label.
- Example: an `improved` question whose correct option states that the finding is stable (lexically
  no change) gets that same answer as its "inverted" label.
- The class of the correct option matches `content_type` in only 1,101/1,188 worsen and 839/1,007 improved
  questions (clean set).

**"Unambiguous" rule (new).** All four conditions:
1. `content_type` ∈ {worsen, improved};
2. the lexical class of the correct option = `content_type`;
3. exactly one option has the opposite class;
4. no negation (no / not / without / none / neither / nor) in the correct option or the opposite one.

| population | worsen: unambiguous / total | improved: unambiguous / total | current rule (worsen / improved) |
|---|---|---|---|
| clean set | 706 / 1,188 | 285 / 1,007 | 745 / 330 |
| full MMXU-test | 551 / 997 | 234 / 996 | 589 / 319 |

Main exclusions:
- **2 or more opposite-class options:** improved 502 and worsen 311 on the clean set. Typical case: two distractors
  one generic worsening option and one new-finding option.
- **Lexical class of the correct option ≠ `content_type`:** improved 168, worsen 87.

**Manual audit.**
- **Unambiguous rule:** 30 random pairs from the clean set (seed 1), correct option → inverted option.
  - **29/30 correct** (opposite direction).
  - In 5, the direction is opposite but the finding is worded differently (a new finding in one region
    against an improvement of a different finding).
  - **1 lexicon error:** an option describing a loss of lung volume as a "reduction" is classed as improved by
    `reduc`, but volume loss is worsening.
  - Estimated precision ≈ 97%.
- **Earlier strict rule** (which also required the correct option to be the only one of its class): 30/30
  correct, but it discards valid cases.

## 6. Readings fixed before launching

**Swap and duplicate in the fine-tuned models** (A, B, C, clean set; runner `run_eval_clean.py`):
- **duplicate:** P(pred = option of class "no change") under duplicate, against the same fraction under full.
  - A model that compares the two images should move towards "no change" when shown the same image twice.
  - Reported overall and by `content_type`.
- **swap:** accuracy against the original label and against the inverted label, by `content_type`, with the
  current rule and **restricted to the unambiguous questions**.
  - Reasoning about direction predicts, in worsen/improved: acc(vs inverted) > acc(vs original).
  - The reverse, as the old TAC model showed on MMXU-test, means swapping the roles does not reverse the answer.
  - `no change` is reported only against the original label.

**Constant-token control on the zero-shot VLMs** (Qwen2-VL, InternVL2, HuatuoGPT, Qwen2.5-VL-7B; full MMXU-test;
runner `clean_eval/run_both_other_zeroshot.py`):
- **Condition:** both images from another patient, with the same procedure as on the clean set (per-patient
  derangement, seed 0), applied to the 2,992 rows.
- **Reported:** accuracy, and Δpred(both_other) next to Δpred(blind) and the null Δpred(hole_random).
- **Reading:**
  - Δpred(both_other) clearly above the null and of the order of Δpred(blind): the model reads image content.
    Changing the content with the same format and length changes its answers.
  - Δpred(both_other) ≈ null while Δpred(blind) is high: blind's effect came from the change in input format or
    length, not from content. The claim that the model "reads the image" would not hold.

**Noted as optional, not launched:** the control of trained A without a real prior.

## 7. Recalculated swap for the ISBI zero-shot VLMs (shuffled letters, predictions already saved)

Script: `clean_eval/swap_inverted_recompute.py`; report: `clean_eval/swap_inverted_recompute_report.md`.
The per-row permutation is rebuilt as in `mmxu_eval/swap_permuted_check.py` and matches the saved CSV in all
2,992 rows. The paper's numbers are reproduced exactly with the old rule.

| model | vs original − full (all rows) | vs inverted − full, paper (old rule, n=1,723 incl. no change) | vs inverted − full, unambiguous rule (worsen+improved, n=785) | same subset: acc(vs inverted) − acc(vs original) |
|---|---|---|---|---|
| HuatuoGPT-7B | −8.56 [−10.32, −6.88] — **unchanged** | −4.64 [−7.53, −1.72] | **−7.52 [−13.09, −2.00]** | +1.0 [−5.2, +7.2] |
| Qwen2.5-VL-7B | −5.72 [−7.62, −3.85] — **unchanged** | −8.30 [−11.12, −5.53] | **−17.20 [−22.12, −12.25]** | −11.9 [−17.7, −5.9] |

By class (unambiguous rule, vs inverted − full):

| model | worsen | improved |
|---|---|---|
| HuatuoGPT | +5.3 [−1.3, +12.0] (n=551) | −37.6 [−46.9, −28.4] (n=234) |
| Qwen2.5-VL | −16.0 (n=551) | −20.1 (n=234) |

**Readings:**
- **The "vs original" column does not depend on the rule and does not move** (−8.56 and −5.72, identical to the
  paper).
- **The paper's "vs inverted" numbers were diluted.** They averaged in the 815 `no change` questions, whose
  "inverted label" is their own answer, and used the defective rule. With the unambiguous rule: HuatuoGPT goes
  from −4.6 to −7.5pp, Qwen2.5-VL from −8.3 to −17.2pp.
- **HuatuoGPT's verdict changes.** The paper read "the pattern reverses: distinguishes direction" by comparing
  original (−8.56, over 2,992 rows) with inverted (−4.64, over 1,723 rows): two different subsets. On the same
  questions (unambiguous, n=785) the difference is +1.0pp [−5.2, +7.2], with no evidence of direction reasoning.
  The by-class asymmetry (worsen ≈ 0, improved −37.6) suggests a response bias, not directional reasoning.
- **Qwen2.5-VL is reinforced.** On the same questions it gets the inverted label 11.9pp less often than the
  original [−17.7, −5.9]: swapping the roles does not reverse its answer.
