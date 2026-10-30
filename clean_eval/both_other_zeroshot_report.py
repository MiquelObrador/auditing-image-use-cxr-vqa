#!/usr/bin/env python3
"""Control de tokens constantes en los 4 VLM zero-shot, MMXU-test completo (EVAL_SETTINGS §6).
accuracy de full/blind/both_other/hole_random; Δpred frente a full; diferencias pareadas.
Bootstrap por paciente, N=5000, semilla 0, IC95%."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL  # noqa: E402
sys.path.insert(0, str(HERE)); from swap_inverted_recompute import boot
MODELS = {"Qwen2-VL-7B": "mmxu_eval", "InternVL2-8B": "internvl_eval", "HuatuoGPT-Vision-7B": "huatuo_eval", "Qwen2.5-VL-7B": "qwen25vl_eval"}
P = {d["id"]: d["cur_image_path"].split("/")[2] for d in map(json.loads, open(MMXU_TEST_JSONL))}
load = lambda p: {int(r["qid"]): r for r in csv.DictReader(open(p))}
L = ["# Control de tokens constantes — VLM zero-shot, MMXU-test completo\n",
     "Ambas imágenes de otro paciente (derangement por paciente, semilla 0). Δpred = % de filas cuya letra cambia frente a full. "
     "Bootstrap por paciente, N=5000, semilla 0, IC95%.\n",
     "| modelo | acc. full | acc. blind | acc. otro paciente | otro − full (pp) | otro − blind (pp) | Δpred blind (%) | Δpred otro paciente (%) | Δpred nulo hole_random (%) | Δpred otro / nulo |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for m, d in MODELS.items():
    R = {c: load(REPO / d / "oracle_results" / f"oracle_{c}.csv") for c in ("full", "blind", "both_other", "hole_random")}
    qs = sorted(set.intersection(*[set(v) for v in R.values()])); assert len(qs) == 2992
    pats = np.array([P[q] for q in qs])
    ok = {c: np.array([R[c][q]["correct"] == "True" for q in qs], float) for c in R}
    ch = {c: np.array([R[c][q]["pred_letter"] != R["full"][q]["pred_letter"] for q in qs], float) for c in ("blind", "both_other", "hole_random")}
    a = lambda x: boot(x, pats)[0].replace("+", "").replace(" [", " [")
    accs = [f"{ok[c].mean():.3f}" for c in ("full", "blind", "both_other")]
    L.append(f"| {m} | " + " | ".join(accs) + f" | {boot(ok['both_other'] - ok['full'], pats)[0]} | {boot(ok['both_other'] - ok['blind'], pats)[0]} | "
             f"{a(ch['blind'])} | {a(ch['both_other'])} | {a(ch['hole_random'])} | {ch['both_other'].mean() / ch['hole_random'].mean():.1f}× |")
text = "\n".join(L) + "\n"; (HERE / "both_other_zeroshot_report.md").write_text(text); print(text)
