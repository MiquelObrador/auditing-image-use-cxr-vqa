# What was fixed before the experiments and what was added during the study

Section 2.4 of the paper lists what was fixed before the experiments: the image-benefit
criterion, the 5pp threshold (for the region and, in fine-tuned models, *mismatch*), *blind*,
*mismatch*, *swap* and *hole* on MMXU-test, and *blind* and *mismatch* on the decontaminated set.
The zero-shot counterfactuals on MMXU-test were the study's starting protocol. The two files in
this folder (in Spanish, with local paths and literal dataset text removed) were written before
the fine-tuning runs they cover:

- `PREREGISTRO_2026-09-21.md`, before any fine-tuning: the image gate (*blind* − *full*
  negative, 95% CI excluding zero), use of the previous scan (*mismatch* − *full* ≤ −5pp, 95% CI
  excluding zero), and how *blind* and *mismatch* are built.
- `PREREGISTRO_SPLIT_LIMPIO_2026-09-24.md`, before training on the decontaminated split:
  the decontaminated set and *blind* and *mismatch* on it.

Everything else in the paper was added during the study:

| addition | where |
|---|---|
| Model B2 (TAC from scratch, two stages), 26 Sep 2026 | `tac_mcq_train/train_lora.py --stage1-tac-only`, `pilot_gate_stage1.py`, `scripts/02_finetune.sh` |
| TOST equivalence test (±5pp, 90% CI) and the *Eq.* verdict | `clean_eval/paper_parallel_table.py`, `clean_eval/hole_finetuned_report.py` |
| *both_other*, *duplicate* and *current_other* | `mmxu_eval/oracle_evaluate.py`, `clean_eval/run_both_other_zeroshot.py`, `tac_mcq_train/run_eval_clean.py` |
| *hole* compared with a same-size grey patch elsewhere (*hole_random*), and *hole* on the fine-tuned models (15 Oct 2026) | `mmxu_eval/hole_random_transforms.py`, `clean_eval/build_clean_bbox_table.py`, `clean_eval/hole_finetuned_report.py` |
| *mismatch* on MMXU-test with a patient derangement (16 Oct 2026), as on the decontaminated set. The pre-registered *mismatch* shifted priors by one row, which gives 2 of 2,992 rows a prior of their own patient; it is kept for comparison as `oracle_mismatch.csv` | `clean_eval/run_mismatch_zeroshot.py` |
| Per-class analyses (16 Oct 2026) | `clean_eval/per_class_report.py` |
| Text-only Llama-3.2-3B control (19 Oct 2026) | `tac_mcq_train/train_lora_textonly_llama.py`, `clean_eval/textonly_llama_report.py` |
| *swap* reversal rate against *mismatch* (19 Oct 2026) | `clean_eval/swap_direction_audit.py` |
| Table 2 (MMXU-test vs. decontaminated set) | `clean_eval/contamination_table.py` |
