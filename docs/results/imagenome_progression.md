# Model A's recipe on Chest ImaGenome progression: the visual counterfactuals on a second benchmark

Supports the third limitation of the paper ("A's recipe gives the same picture on Chest ImaGenome's progression task"). The training and evaluation code of this check is not part of this release; the table below is the full result (aggregates only). `mismatch` = prior image of another patient, `no_prior` = prior replaced by the current image, flip = the model's predicted direction (improved/worsened) reverses against `full`.

Recipe of model A (Libra's frozen TAC + Llama-3.2-3B-Instruct, LoRA) fine-tuned on the Chest ImaGenome comparison relations of the official MIMIC-CXR train split. Evaluation: all relations of the official validate + test splits with both images, **14,952 relations, 411 patients** (0 without both images left out), unseen by Libra's TAC and by the fine-tuning. Three classes, letter answer, greedy. mismatch = prior of another patient (per-patient derangement, seed 0, as in the paper).

Paired by relation; patient bootstrap (5,000 replicates, seed 0), 95% CI, pp. Verdicts as in Table 1: Sig. = |Δ| ≥ 5pp with the 95% CI excluding zero; Eq. = 90% CI inside ±5pp; otherwise Inconcl. Balanced accuracy = mean of the per-class accuracies (classes: improved 14.6%, worsened 25.4%, no change 60.0% of the relations), same bootstrap.

## 1. Accuracy and balanced accuracy

| condition | accuracy | − full | verdict | balanced acc. | − full (balanced) | Δpred | unparseable |
|---|---|---|---|---|---|---|---|
| full | 0.693 [0.675, 0.710] | — | — | 0.564 [0.543, 0.585] | — | — | 0 |
| blind | 0.464 [0.452, 0.478] | -22.8 [-24.4, -21.2] | Sig. | 0.382 [0.375, 0.390] | -18.2 [-20.1, -16.1] | 49.1% [47.9, 50.4] | 0 |
| no_prior | 0.571 [0.546, 0.596] | -12.2 [-14.4, -10.0] | Sig. | 0.439 [0.422, 0.455] | -12.5 [-14.5, -10.6] | 31.3% [29.0, 33.5] | 0 |
| mismatch | 0.511 [0.488, 0.536] | -18.1 [-20.6, -15.7] | Sig. | 0.450 [0.429, 0.472] | -11.4 [-13.7, -9.0] | 42.6% [40.1, 45.2] | 0 |
| both_other | 0.531 [0.511, 0.552] | -16.1 [-18.5, -13.9] | Sig. | 0.351 [0.340, 0.361] | -21.3 [-23.8, -18.9] | 38.4% [35.8, 41.0] | 0 |
| swap | 0.528 [0.503, 0.553] | -16.5 [-18.7, -14.2] | Sig. | 0.316 [0.305, 0.327] | -24.8 [-27.4, -22.1] | 37.2% [34.5, 39.9] | 0 |

## 2. Swap direction (corrected metric)

turn(X) = P(pred = inverted class) − P(pred = original class) on the improved and worsened relations; Δturn = turn(X) − turn(full). mismatch is the null (another patient's prior: no reason to invert the direction); the swap effect controlled by it is Δturn(swap) − Δturn(mismatch) = turn(swap) − turn(mismatch). Flip = the predicted class under X is the opposite (improved <-> worsened) of the class predicted under full, over all relations of the class.

| class | n | turn(full) | turn(swap) | Δturn(swap) | Δturn(mismatch) | Δturn(swap) − Δturn(mismatch) | flip swap | flip mismatch |
|---|---|---|---|---|---|---|---|---|
| improved | 2,179 | -23.3 [-28.7, -18.0] | +43.6 [+38.4, +49.0] | **+66.9 [+58.0, +76.4]** | +52.5 [+41.7, +63.0] | **+14.4 [+3.2, +25.2]** | 28.3% [24.1, 32.8] | 11.0% [8.3, 14.0] |
| worsened | 3,797 | -52.2 [-56.2, -48.2] | +22.2 [+17.3, +27.1] | **+74.4 [+66.4, +82.5]** | -2.7 [-9.6, +4.8] | **+77.1 [+66.4, +87.7]** | 27.3% [23.6, 31.2] | 3.7% [2.5, 5.1] |
| improved+worsened | 5,976 | -41.7 [-45.6, -37.8] | +30.0 [+26.2, +33.9] | **+71.7 [+64.6, +79.0]** | +17.5 [+12.9, +22.2] | **+54.2 [+46.7, +61.7]** | 27.7% [24.4, 31.1] | 6.3% [5.1, 7.7] |
