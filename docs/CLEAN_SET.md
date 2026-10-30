# The decontaminated evaluation set (frozen 2026-09-24)

## Why

MMXU builds its dev and test sets from the whole of Chest ImaGenome and only guarantees that no
patient is in both. It does not follow MIMIC-CXR's official split:

- 2,920 of the 2,992 evaluated MMXU-test rows (97.6%) come from patients in the official
  **train** split (`scripts/paper_numbers.py`).
- Libra pretrained its temporal module (TAC) on that split (Libra README, "we strictly follow the
  official split").
- RAD-DINO did too (see §3).

So a model built from those components may have seen the evaluation patients.

## Definition

A patient is **clean** if it is in the official MIMIC-CXR **validate** or **test** split. A row
of MMXU-dev or MMXU-test is in the clean set if its patient is. The current and previous image of
a row always belong to the same patient; `build_clean_split.py` asserts it.

## Shipped files (`data_ids/`, identifiers only)

| file | content |
|---|---|
| `clean_set_ids_2026-09-24.csv` | `source` (dev/test), `qid` (MMXU row id), `usable` — 3,350 rows |
| `splits_clean_2026-09-24.json` | train / val / val_fast row ids of MMXU-dev used for **all** fine-tuning, with no clean patient |
| `mmxu_test_eval_qids.csv` | the 2,992 MMXU-test rows with a usable Chest ImaGenome box on both images |

Notes on the columns:
- `source` is needed because the MMXU-test qids (1–3000) also exist in MMXU-dev.
- The MIMIC-CXR identifiers are not shipped. They are recovered by joining on `source` and `qid` with
  MMXU: `cur_image_path` and `prior_image_path` (`files/pXX/p<subject_id>/s<study_id>/<dicom_id>.jpg`)
  give the patient, study and image, and the patient's official split comes from
  `mimic-cxr-2.0.0-split.csv.gz`. This reproduces all 3,350 rows of the list used in the paper.
- `usable` means: options A–D, and both images present in the authors' copy of MIMIC-CXR-JPG
  2.0.0, which lacked about 8% of the images. The same filter is applied in
  `tac_mcq_train/build_split.py`.
  - A complete copy marks more rows usable. Use the shipped lists to reproduce the paper.
  - Rebuilding with `clean_eval/build_clean_split.py` writes to `data_ids/regenerated/`, so it can
    be compared with the shipped lists without overwriting them.

## 1. Verification against the official split

`build_clean_split.py` reads `mimic-cxr-2.0.0-split.csv.gz` from MIMIC-CXR-JPG 2.0.0, at the path
in `config.MIMIC_SPLIT_CSV`. It applies it to the 377,110 images of
`mimic-cxr-2.0.0-metadata.csv.gz`, and stops unless it reproduces, exactly, Table 3 of Johnson et
al. 2019 (MIMIC-CXR-JPG, arXiv:1901.07042):

| | train | validate | test |
|---|---|---|---|
| images | 368,960 | 2,991 | 5,159 |
| studies | 222,758 | 1,808 | 3,269 |
| patients in validate ("a random set of 500 patients") | — | 500 | — |

The authors' copy of the split file has SHA-256
`515997bd6649045d7443d60c59a4ce9f6cca6c478871b8f2fb13454462bedb2f` (377,110 rows). No patient has
more than one split.

An earlier reconstruction from a derived source agreed with the official file on all 65,379
patients, with 0 discrepancies. That source was the per-image `split` column of Medical-Diff-VQA's
`mimic_all.csv`. The clean set regenerated from the official file is identical to the one used.

## 2. Size

| | rows | usable | patients with usable rows |
|---|---|---|---|
| MMXU-dev, clean | 3,276 | 3,022 | 304 |
| MMXU-test, clean | 74 | 74 | 27 |
| **total** | 3,350 | **3,096** | **331** |

Effect on the fine-tuning split (`build_split.py` output → `splits_clean_2026-09-24.json`):

| split | before | after | removed | patients |
|---|---|---|---|---|
| train | 96,426 | 93,693 | −2,733 | 13,037 |
| val | 10,503 | 10,214 | −289 | 1,455 |
| val_fast | 2,000 | 1,938 | −62 | — |

`val_fast` is the earlier fixed subsample minus the clean rows.

Expected power: the 95% CI of mismatch−full on 331 patients is about 5.0pp wide (10th–90th
percentile 4.5–5.4pp). This was estimated by subsampling patients from a TAC model evaluated on
MMXU-test.

## 3. RAD-DINO and MAIRA's split

**RAD-DINO's pretraining used the official MIMIC-CXR train split.**
- RAD-DINO paper (Pérez-García et al., arXiv:2401.10815v3), appendix D.1: *"All images are used
  for pre-training from the given datasets, except MIMIC-CXR where their recommended training
  split is used."*
- The card of the public `microsoft/rad-dino` checkpoint lists 368,960 MIMIC-CXR images. That is
  exactly the size of the official train split. The card does not say "official split" literally.

The card also says *"Images in the validation and test sets used to train MAIRA were excluded"*.
MAIRA-1 (Hyland et al., arXiv:2311.13668, Table 1) uses:
- its own validation set: 2,709 subjects, not the official validate split of 500;
- the standard MIMIC-CXR test split, filtered to frontal views with Findings.

So the MAIRA exclusion falls inside the official train split and does not affect the clean set.
One minor inconsistency cannot be resolved from what is published: if MAIRA's validation images
were excluded, the checkpoint should have fewer than 368,960 images. It only concerns
official-train images.

**Consequence:**
- The clean set is outside the pretraining of both Libra's TAC and RAD-DINO.
- On the full MMXU-test, RAD-DINO saw most patients, not only the TAC. Every model with RAD-DINO
  evaluated on the full MMXU-test carries that exposure, including the one with a TAC trained
  from scratch.
