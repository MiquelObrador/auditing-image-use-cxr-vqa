# Visual counterfactuals to evaluate image use in longitudinal chest X-ray VQA

Code to reproduce every result of the paper (ISBI 2027 submission): the visual counterfactuals
(the images removed, replaced by another patient's, swapped or occluded, with the prompt fixed)
applied to four zero-shot VLMs on MMXU-test and to four fine-tuned models on a decontaminated
set of held-out patients, the two text-only controls, and the statistics, tables and figure.

**This repository contains code and documentation only, never data.**
- No images, reports, scene graphs or MMXU files are included.
- No file derived from them containing text is included.
- `data_ids/` holds MMXU row ids only (with the file they come from and a `usable` flag), no
  MIMIC-CXR identifiers. The MIMIC-CXR patient, study and image of each row are recovered by
  joining on `qid` with MMXU (`cur_image_path`, `prior_image_path`).
- No trained weights are distributed.

## 1. Data access

All three datasets require credentialed access. Request it yourself and keep to their terms of
use.

| dataset | where | what is needed |
|---|---|---|
| **MIMIC-CXR-JPG 2.0.0** | [physionet.org/content/mimic-cxr-jpg/2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/): credentialed PhysioNet account, CITI "Data or Specimens Only Research" training, signed data use agreement | the `files/` images, `mimic-cxr-2.0.0-split.csv.gz`, `mimic-cxr-2.0.0-metadata.csv.gz` |
| **Chest ImaGenome 1.0.0** | [physionet.org/content/chest-imagenome/1.0.0](https://physionet.org/content/chest-imagenome/1.0.0/): same PhysioNet credentialing | `silver_dataset/scene_graph.zip`, `silver_dataset/splits/images_to_avoid.csv` (bounding boxes) |
| **MMXU** (Mu et al., ACL Findings 2025, [arXiv:2502.11651](https://arxiv.org/abs/2502.11651)) | from its authors (see the paper; code at `github.com/linjiemu/MMXU`) | `MMXU-test.jsonl`, `MMXU-dev.jsonl` |

The model checkpoints are public on Hugging Face. `meta-llama/Llama-3.2-3B-Instruct` requires
accepting Meta's licence and a `huggingface-cli login`. The exact revisions are listed in
`docs/EVAL_SETTINGS.md` §3.

### Where to put the data

Paths are read by `config.py`. By default it looks inside `data/` (git-ignored), so the
simplest setup is a set of symlinks:

```bash
mkdir -p data
ln -s /path/to/MMXU-test.jsonl                     data/MMXU-test.jsonl
ln -s /path/to/MMXU-dev.jsonl                      data/MMXU-dev.jsonl
ln -s /path/to/mimic-cxr-jpg/2.0.0                 data/mimic-cxr-jpg     # contains files/p10 ... files/p19
ln -s /path/to/mimic-cxr-2.0.0-split.csv.gz        data/mimic-cxr-2.0.0-split.csv.gz
ln -s /path/to/mimic-cxr-2.0.0-metadata.csv.gz     data/mimic-cxr-2.0.0-metadata.csv.gz
ln -s /path/to/chest-imagenome/1.0.0               data/chest-imagenome   # contains silver_dataset/
```

Each path can instead be set with an environment variable:
- data: `MMXU_TEST_JSONL`, `MMXU_DEV_JSONL`, `MIMIC_JPG_ROOT`, `MIMIC_SPLIT_CSV`,
  `MIMIC_METADATA_CSV`, `IMAGENOME_DIR`;
- code: `LIBRA_DIR`;
- all of the above at once: `DATA_DIR`.

## 2. Environments

| file | used for |
|---|---|
| `envs/mmxu_eval.yml` | zero-shot Qwen2-VL-7B, HuatuoGPT-Vision-7B, Qwen2.5-VL-7B; Chest ImaGenome boxes |
| `envs/internvl_eval.yml` | zero-shot InternVL2-8B (its remote code needs transformers ≤ 4.44) |
| `envs/tacmcq_train.yml` | models A, B and B2 (RAD-DINO + TAC + Llama-3.2-3B, through Libra v1.0.3) and the text-only Llama |
| `envs/qwen25vl_baseline_train.yml` | model C and the text-only Qwen (Qwen2.5-VL-3B) |
| `envs/analysis.yml` | statistics, tables, linear shortcut model (CPU) |

Create them with `conda env create -f envs/<file>.yml`. The scripts call them by these names;
override with `ENV_MMXU`, `ENV_INTERNVL`, `ENV_TAC`, `ENV_QWEN`. All runs used one 24 GB GPU
(RTX 4090).

## 3. Run order

| step | script | GPU | what it does |
|---|---|---|---|
| 0 | `scripts/00_prepare.sh` | — | clones Libra at the pinned commit; builds the TAC assembly; extracts the Chest ImaGenome boxes and checks they give the 2,992 evaluated rows |
| 0b | `python3 clean_eval/build_clean_split.py --base-splits tac_mcq_train/splits.json` (optional) | — | rebuilds the decontaminated set from the official split into `data_ids/regenerated/`, to compare with the shipped lists (`docs/CLEAN_SET.md`). `tac_mcq_train/build_split.py` makes `splits.json` |
| 1 | `scripts/01_zeroshot.sh` | ~25 h | 4 zero-shot VLMs × counterfactuals on MMXU-test |
| 2 | `scripts/02_finetune.sh` | ~90 h | trains A, B, B2, C and the two text-only models on `data_ids/splits_clean_2026-09-24.json`; evaluates them on the decontaminated set and on MMXU-test |
| 3 | `scripts/03_analysis.sh` | — | every statistic, table and number of the paper (minutes) |
| 4 | `scripts/04_figure.sh` | — | Fig. 2, from the numbers of Table 1 (Fig. 1 is drawn in LaTeX) |

Training uses the shipped split lists, not a rebuilt one: our copy of MIMIC-CXR-JPG lacked
about 8% of the images, so rebuilding from a complete copy would select more rows (see
`docs/CLEAN_SET.md`).

## 4. What reproduces what

| paper result | produced by |
|---|---|
| Table 1a (zero-shot VLMs, MMXU-test) and Table 1b (fine-tuned models, decontaminated set): *blind*, *both_other*, *mismatch*, *swap* | `clean_eval/paper_parallel_table.py` |
| Table 1, *hole* column (*hole* minus a same-size grey patch elsewhere) | `clean_eval/hole_zeroshot_report.py` (1a), `clean_eval/hole_finetuned_report.py` (1b, 3,064 rows with the region's box, `clean_eval/build_clean_bbox_table.py`) |
| Table 1, text-only rows | `clean_eval/textonly_llama_report.py`, `scripts/paper_numbers.py` |
| Table 2 (MMXU-test vs. decontaminated set) | `clean_eval/contamination_table.py` |
| Fig. 2 (benefit from content vs. from visual tokens) | `figures/fig_audit_ladder.py`, from `clean_eval/paper_parallel_table.json` |
| Visual counterfactuals (*full, blind, both_other, mismatch, current_other, swap, duplicate, hole, hole_random*) | `mmxu_eval/oracle_evaluate.py` (zero-shot), `clean_eval/run_mismatch_zeroshot.py` and `clean_eval/run_both_other_zeroshot.py` (zero-shot *mismatch* and *both_other*), `tac_mcq_train/run_eval_clean.py` and `run_eval.py` (fine-tuned) |
| Answer letters permuted per question (zero-shot *swap* and its *full* baseline) | `mmxu_eval/swap_permuted_check.py`, `mmxu_eval/permuted_check.py` |
| *swap*: reversal rate against *mismatch* (A 25% vs. 14%, B, B2, C at most 4%) | `clean_eval/swap_direction_audit.py` (columns "flip", worsened and improved questions) |
| A with the current scan replaced (11.9pp) | `clean_eval/current_other_report.py` |
| Contamination: 2,920 of 2,992 rows (97.6%) from official-train patients; C 0.809 with images, 0.812 without | `scripts/paper_numbers.py` |
| Decontaminated set: 3,096 rows, 331 patients | `clean_eval/build_clean_split.py`, shipped as `data_ids/clean_set_ids_2026-09-24.csv` |
| Per-class analyses (e.g. *blind* pushes A and B2 towards "worsened") | `clean_eval/per_class_report.py` |
| Linear model on the options alone (0.67) and its heaviest features | `shortcut_analysis/options_only_linear.py` |
| Chest ImaGenome progression check (Limitations) | `docs/results/imagenome_progression.md` (result table; the code of this check is not part of this release) |
| Pre-registered criteria per model | `clean_eval/criteria_report_clean.py` |

Patient-clustered bootstrap (N=5000, seed 0, 95% percentile CIs) and TOST (±5pp, 90% CI)
are implemented in each analysis script (`clean_eval/paper_tables.Boot`). Supplementary
reports run by `scripts/03_analysis.sh` (A−C contrasts, constant-token control, swap turn per
class, hard subset, C at two resolutions) are not in the paper.

## 5. Training code and configurations

| model | code | key settings |
|---|---|---|
| A — pretrained TAC (Libra), frozen | `tac_mcq_train/train_lora.py` | LoRA r=16, α=32, dropout 0.05 on the LLM linears; lr 5e-5, cosine, 1 epoch, effective batch 16 |
| B — TAC from scratch | `train_lora.py --unfreeze-tac --tac-init scratch --tac-lr 2e-5` | as A; TAC trained jointly; pilot and gate (`pilot_gate.py`) |
| B2 — TAC from scratch, two stages | `train_lora.py --stage1-tac-only --unfreeze-tac --tac-init scratch --tac-lr 2e-5`, then `train_lora.py --tac-state-init <stage-1 TAC>` | stage 1 trains the TAC alone (language model frozen; pilot and gate `pilot_gate_stage1.py`), stage 2 the LoRA with that TAC frozen |
| C — generic Qwen2.5-VL-3B | `qwen25vl_baseline_train/train_lora_qwen25vl.py` | as A; images at 518² (268,324 px, as RAD-DINO) |
| text-only Qwen | `qwen25vl_baseline_train/train_lora_textonly.py` | same checkpoint and recipe as C, never shown an image |
| text-only Llama | `tac_mcq_train/train_lora_textonly_llama.py` | same assembly and recipe as A, B, B2, never shown an image |

Answer formats, letter permutation and loss masking are in `tac_mcq_train/format_targets.py`.
Decoding, answer extraction and checkpoint revisions are in `docs/EVAL_SETTINGS.md`, and every
other detail is in `REPRODUCIBILITY.md`. The pre-registrations, written before each training
run, are in `docs/preregistrations/`. What was fixed before the experiments and what was added
during the study is listed in `docs/preregistrations/README.md`.

## 6. Repository layout

```
config.py                   data / code paths (environment variables, default data/)
data_ids/                   identifier lists: decontaminated set, fine-tuning split, evaluated rows
mmxu_eval/                  evaluation pipeline, counterfactuals, prompt, answer extraction, model wrappers
bbox_analysis/              Chest ImaGenome boxes for MMXU-test (hole / hole_random)
tac_mcq_train/              models A, B, B2 and the text-only Llama: Libra assembly, formats, training, evaluation
qwen25vl_baseline_train/    model C and the text-only Qwen
clean_eval/                 decontaminated set, statistics and tables
shortcut_analysis/          linear model on the options; hedged wording
figures/                    Fig. 2
scripts/                    run order (00–04) and the remaining paper numbers
envs/                       conda environments
docs/                       evaluation settings, decontaminated set, pre-registrations, Chest ImaGenome result
```

## Licence

Code: MIT (see `LICENSE`). The datasets and model checkpoints keep their own licences and
terms of use.

