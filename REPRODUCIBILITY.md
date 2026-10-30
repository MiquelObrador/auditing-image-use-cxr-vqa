# Reproducibility supplement

Companion to *Visual counterfactuals to evaluate image use in longitudinal chest X-ray VQA*
(ISBI 2027 submission). Every value below was read from the code or the authors' run artifacts, not from the paper.
The datasets are not distributed here: see `README.md` for access and `config.py` for paths.

Where a detail is not listed here, the scripts named in each section are the reference.

---

## 1. Data

| item | value |
|---|---|
| Benchmark | MMXU-test (`MMXU-test.jsonl`, 3000 rows) and MMXU-dev (`MMXU-dev.jsonl`, obtained from the MMXU authors), from the MMXU release (arXiv:2502.11651) |
| Rows used from MMXU-test | 2992 of 3000: those with a valid Chest ImaGenome bounding box on both images (`bbox_analysis/mmxu_bbox_table.csv`, built by `scripts/00_prepare.sh`; list in `data_ids/mmxu_test_eval_qids.csv`) |
| Patients in those 2992 rows | 1198 (the patient is the `pNNNNNNNN` folder of `cur_image_path`) |
| Images | MIMIC-CXR-JPG v2.0.0 JPEGs at their original resolution (`config.MIMIC_JPG_ROOT`) |
| Bounding boxes | Recovered from Chest ImaGenome scene graphs. MMXU-test ships none |
| Official MIMIC-CXR split | `mimic-cxr-2.0.0-split.csv.gz` (`config.MIMIC_SPLIT_CSV`), 377,110 rows, SHA-256 `515997bd6649045d7443d60c59a4ce9f6cca6c478871b8f2fb13454462bedb2f` |
| Decontaminated set | `data_ids/clean_set_ids_2026-09-24.csv` (MMXU `source`, `qid` and `usable` only): rows whose patient is in the official *validate* or *test* split, `usable` rows only. That is 3096 rows (3022 from MMXU-dev, 74 from MMXU-test) from 331 patients. The set is built from all 3000 MMXU-test rows and needs no box, so 2 of the 74 (qids 1894 and 2199) lie outside the 2992 boxed rows; hence 2992 − 2920 = 72, not 74. Rebuild with `clean_eval/build_clean_split.py` (writes to `data_ids/regenerated/`); see `docs/CLEAN_SET.md` |
| Fine-tuning split | `data_ids/splits_clean_2026-09-24.json`: MMXU-dev with every decontaminated-set patient removed, leaving 93,693 training rows from 13,037 patients |

MMXU-test qids (1–3000) also occur in MMXU-dev. In the decontaminated set, MMXU-dev qids are
offset by +1,000,000.

---

## 2. Prompts

### 2.1 Zero-shot VLMs and evaluation of every model

This is MMXU's own zero-shot template (paper Fig. 14), filled from `question` and `options`
(`mmxu_eval/prompts.py`). The same text is used in every condition. No report or history is
ever included.

```
Here are two chest X-RAY images of the same patient. Previous chest X-RAY is Image-1. And current chest X-RAY is Image-2. Your task is to evaluate the differences between the two images based on the provided question.

Question:
{question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

Answer with the option's letter from the given choices directly.
```

- **Image order.** Previous image first, then current. For the TAC model the tensor axis is
  (current, previous), as Libra expects. `tac_mcq_train/model_tac.py` does the reordering.
- **System prompt, zero-shot VLMs.** Each model's default chat template.
- **System prompt, TAC model.** A neutral one replaces Libra's report-generation prompt:
  `You are a helpful assistant. Answer multiple-choice questions with a single letter.`
- **System prompt, fine-tuned Qwen2.5-VL-3B.** Its default chat template.

### 2.2 Fine-tuning targets (`tac_mcq_train/format_targets.py`)

Each training example keeps the evaluation prompt and changes only its last line. The format
is drawn per (sample, epoch), with seed 0:

| format | share | last line of the prompt | target |
|---|---|---|---|
| A | 65% | the evaluation instruction, unchanged | the letter, e.g. `A` |
| B | 20% | `Answer with the text of the correct option.` | the option text |
| C | 15% | `Briefly justify your answer, then give the option's letter.` | `<justification> The answer is A.` |

- **Where format C's justification comes from.** Up to two sentences (at most 40 words) from
  the row's current regional report in MMXU-dev, filtered to change-describing sentences.
  Format C is used only where such a justification exists, which is 68.9% of rows. Its
  probability is rescaled so the overall share stays at 15%.
- **Option order.** Permuted per (sample, epoch). MMXU's letters are already balanced, so this
  corrects no dataset bias; it removes a positional shortcut.
- **Images.** Every training example has both images.

---

## 3. Conditions

| condition | implementation |
|---|---|
| *full* | Both images, unmodified |
| *hole* | Annotated box filled with mid-grey RGB (128, 128, 128); rest of the image unchanged (`mmxu_eval/oracle_transforms.py`) |
| *hole_random* | Same-size box, same grey, at the horizontally mirrored position, or shifted vertically for midline regions (`hole_random_transforms.py`) |
| *blind* | No image at all. **Zero-shot VLMs:** the image entries are left out of the chat message and the text is unchanged. **TAC model:** the `<image>` token is not inserted. **Fine-tuned Qwen2.5-VL-3B:** no image entries |
| *no_prior* | Current image only. The prompt still mentions Image-1 |
| *mismatch*, MMXU-test | A random derangement over patients (seed 0), as on the decontaminated set: each row receives the prior of a row of another patient, and no row keeps a prior of its own patient (`clean_eval/run_mismatch_zeroshot.py`). This is the *mismatch* of the paper. MMXU's original protocol shifted priors by one row (`np.roll(arange(n), 1)`, `oracle_evaluate.py --condition mismatch`), which lands on the same patient in 2 of 2992 rows; it is kept only for comparison |
| *mismatch*, decontaminated set | A random derangement over patients (seed 0): each patient receives another patient's priors, and an assertion rules out same-patient priors. The one-row shift is not used here because MMXU-dev rows are grouped by patient |
| *swap* | Previous and current images exchange positions; the prompt is unchanged. Scored against the original label and against the inverted one. The inversion uses a regex lexicon on the option texts (`oracle_evaluate.py`: `IMPROVED_RE`, `WORSENED_RE`, `NOCHANGE_RE`). Rows whose options cannot be mapped unambiguously are excluded from the inverted score, which leaves n = 1723 |
| *swap* with letter permutation (zero-shot VLMs) | As *swap*, with answer letters permuted per row (`SeedSequence(0).spawn(n)`), compared with `full` under the **same** permutation (`oracle_full_permuted.csv`; the two permutations match row by row, asserted in `clean_eval/paper_parallel_table.py`). Comparing it instead with `full` at the original letters mixes in the cost of the permutation itself (3.4–5.3pp) |
| *swap* reversal rate (paper) | Share of the worsened or improved rows whose predicted direction under *swap* is the opposite of the direction the same model predicted under *full*; a row whose *full* answer is not a direction counts as not reversed. Null: the same rate under *mismatch*, which breaks the pair but gives no reason to reverse (`clean_eval/swap_direction_audit.py`, columns "flip") |
| *swap* turn metric (supplementary) | acc(inverted) − acc(original) under *swap*, per class, over worsen/improved rows with exactly one opposite-class option and no negation (`clean_eval/swap_inverted_recompute.py`, `clean_eval/swap_turn_zeroshot.py`, `clean_eval/swapdup_report.py`) |
| *duplicate* | The current image in both positions. The metric is P(prediction = "no change") |
| *current_other* (A only) | The previous image stays and the current image comes from the source row of the *mismatch* derangement (`tac_mcq_train/run_eval_clean.py --condition current_other`, `clean_eval/current_other_report.py`) |
| *both_other* | Both images from another patient: same number of visual tokens as *full*, wrong content. Zero-shot VLMs on MMXU-test: random derangement over patients, seed 0 (`clean_eval/run_both_other_zeroshot.py`). Fine-tuned models on the decontaminated set: the source row of the *mismatch* derangement |

---

## 4. Decoding and answer extraction

- **Decoding.** Greedy for every model (`do_sample=False`), `max_new_tokens=20`.
- **Answer extraction** (`mmxu_eval/evaluate.py::parse_letter`). The first standalone letter
  A–D, case-insensitive, matched by `\b([A-Da-d])\b`. No answer is ever forced.
- **Unparseable outputs** count as incorrect.

Unparseable rate, over all conditions evaluated for each model:

| model | unparseable / rows |
|---|---|
| Qwen2-VL-7B-Instruct | 0 / 17,952 |
| InternVL2-8B | 0 / 20,944 |
| HuatuoGPT-Vision-7B-Qwen2.5VL | 0 / 41,888 |
| Qwen2.5-VL-7B-Instruct | 53 / 41,888 (0.13%) |
| TAC model, pretrained TAC (A) | 0 / 24,352 |
| TAC model, TAC from scratch (B) | 0 / 21,256 |
| Qwen2.5-VL-3B fine-tuned (C) | 0 / 24,352 |
| Qwen2.5-VL-3B text-only | 0 / 6,088 |

- **Where Qwen2.5-VL-7B's 53 failures are.** All are under *blind*: 27/2992 in the standard run
  and 26/2992 in the letter-permuted run. They are refusals such as "To provide an accurate
  answer, I would need to see the image…".
- **Effect on Qwen2.5-VL-7B's gate.** Counting these as incorrect makes the gate at most 0.9pp
  larger than scoring them at chance would. It does not change the verdict.
- **Absolute zero-shot accuracies vs. MMXU's Table 3.** Ours do not match MMXU's Table 3:
  Qwen2-VL-7B is +3.2pp above it and InternVL2-8B −5.9pp below (0.408 vs. 0.467). Prompt,
  template, generation settings and resolution were ruled out, so the gap is most likely
  version drift (dataset, checkpoint or libraries), which we cannot verify. The paper's
  conclusions rest on paired differences between conditions, not on absolute accuracy.

---

## 5. Checkpoints

All checkpoints are from the Hugging Face Hub, at the snapshot revisions cached locally when
the experiments ran.

| model | repository | revision |
|---|---|---|
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| InternVL2-8B | `OpenGVLab/InternVL2-8B` | `6fb9ad6924f69424e57fab2ab061d707688f0296` |
| HuatuoGPT-Vision-7B | `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` | `451ac32400e36cfd07b41b62cbe63e6894895b38` |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Qwen2.5-VL-3B-Instruct (C and text-only) | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| RAD-DINO | `microsoft/rad-dino` | `110cbc18d5133582e320b43d53bf5c44e410c936` |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Libra 3B (source of the TAC) | `X-iZhang/libra-v1.0-3b` | `a9a3f66296682a0fe7eb9cc67a564774ed195eb8` |

How the TAC model is assembled (`tac_mcq_train/libra-3b-baseassembly/`):

- Libra 3B's `config.json` (TAC projector, `mm_vision_tower = microsoft/rad-dino`,
  `mm_vision_select_layer = all`, `image_aspect_ratio = pad`).
- Libra's stage-1 TAC weights (`mm_tac_projector.bin`, renamed to `mm_projector.bin`). Stage 1
  is Libra's visual-feature alignment, trained with the language model frozen.
- The stock Llama-3.2-3B-Instruct as language model. None of Libra's stage-2 LoRA is used.

---

## 6. Image preprocessing and resolution

| model | preprocessing |
|---|---|
| Qwen2-VL-7B, Qwen2.5-VL-7B, HuatuoGPT-Vision-7B | Native dynamic resolution, `min_pixels = 256·28² = 200,704`, `max_pixels = 1280·28² = 1,003,520` per image (`oracle_evaluate.py`) |
| InternVL2-8B | Dynamic tiling at 448×448, at most 4 tiles per image plus a thumbnail. This is below the official maximum of 12, because of GPU memory |
| TAC model (A, B) | RAD-DINO processor: pad to square (`image_aspect_ratio = pad`), resize shortest edge to 518 (bicubic), centre-crop 518×518, normalise with mean 0.5307 and std 0.2583 |
| Qwen2.5-VL-3B (C), **training** | Fixed at 518×518 = 268,324 pixels per image (`min_pixels = max_pixels`), to match RAD-DINO's input |
| Qwen2.5-VL-3B (C), **evaluation** | Also 518×518 = 268,324 pixels (`--pixels 268324`; runs in `qwen25vl_baseline_train/runs/qwen_clean/px518/`, script `scripts/02_finetune.sh`). An earlier evaluation at the protocol's standard range (200,704–1,003,520 pixels) is kept in `qwen_clean/` for comparison: C's accuracy differs by less than 1pp in every condition |

---

## 7. Fine-tuning hyperparameters

The recipe is the same for A, B, B2, C and the two text-only models unless the table says otherwise
(`tac_mcq_train/train_lora.py`, `tac_mcq_train/train_lora_textonly_llama.py`,
`qwen25vl_baseline_train/train_lora_qwen25vl.py`, `train_lora_textonly.py`). B2 trains the TAC from scratch
alone first (`--stage1-tac-only`, language model frozen, pilot gate `pilot_gate_stage1.py`), then the LoRA with
that TAC frozen (`--tac-state-init`). Each model is a **single training run**.

| hyperparameter | value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05, no bias |
| LoRA targets | Every `nn.Linear` of the language model. The vision tower, the TAC (`mm_projector`) and `lm_head` are excluded by module name. For Qwen2.5-VL they are excluded with `exclude_modules = model.visual.*`, because its vision tower reuses the language model's module names. An assertion checks that no vision-tower parameter is trainable |
| Optimiser, learning rate | AdamW (HF `Trainer` default), 5e-5 |
| Schedule | cosine, 3% warm-up |
| Batch | 2 per step × 8 accumulation = 16 effective |
| Training length | 1 epoch = 5856 optimiser steps over 93,693 rows |
| Precision | bf16, gradient checkpointing |
| Seed | 0, for model init, data order, option permutation and format choice |
| A (pretrained TAC) | TAC frozen at Libra's stage-1 weights |
| B (TAC from scratch) | TAC re-initialised with PyTorch's default init (Libra's stage-1 init, no pretrained weights) and trained jointly with LoRA at a TAC learning rate of 2e-5. An assertion checks that no tensor still equals Libra's |
| Text-only | Same Qwen2.5-VL-3B checkpoint and recipe as C; never sees an image in training or evaluation |

The *hard subset* is the rows this text-only model answers wrongly: 675 on the decontaminated
set and 582 on MMXU-test.

---

## 8. Statistics

- **Bootstrap.** Paired and clustered by patient, N = 5000 resamples, seed 0, 95% percentile
  CIs.
- **TOST.** A ±5pp margin, tested as a 90% CI entirely inside it.
- **Where to find everything.** `scripts/03_analysis.sh` runs every analysis; `README.md` maps
  each paper result to its script. The paper's main table (both panels) is
  produced by `clean_eval/paper_parallel_table.py`; the hard subset by
  `clean_eval/paper_tables.py --c-dir qwen25vl_baseline_train/runs/qwen_clean/px518 `; the A−C, B−C and difference-in-differences contrasts by
  `C_DIR=<.../qwen_clean/px518> python3 clean_eval/compare_models.py` →
  `compare_models_report_c518.md`.

---

## 9. Hardware and software

| item | value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 4090 (24 GB) for every run |
| Training wall-clock | A 11.9 h, B 12.2 h, B2 9.0 h (stage 1) + 11.9 h (stage 2), C 16.3 h, text-only Qwen 4.0 h, text-only Llama 3.1 h |
| Peak VRAM | A 11.9 GiB, B 14.2 GiB, B2 13.9 GiB (stage 1) / 11.9 GiB (stage 2), C 11.4 GiB, text-only Qwen 8.5 GiB, text-only Llama 7.4 GiB |
| Zero-shot Qwen / HuatuoGPT environment | Python 3.12.13, torch 2.13.0+cu130, transformers 5.14.1 |
| InternVL2-8B environment | Python 3.10.20, torch 2.13.0+cu130, transformers 4.44.2 (its remote code needs ~4.37–4.44) |
| TAC model environment | Python 3.10.21, torch 2.4.0+cu121, transformers 4.51.3, peft 0.21.0, Libra v1.0.3 (`0d6c56a`), cloned by `scripts/00_prepare.sh` |
| Qwen2.5-VL-3B fine-tuning environment | Python 3.12.13, torch 2.13.0+cu130, transformers 5.14.1, peft 0.21.0 |
