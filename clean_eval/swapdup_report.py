#!/usr/bin/env python3
"""swap y duplicate en A, B, C sobre el conjunto limpio (EVAL_SETTINGS_2026-09-28.md §6).

duplicate: P(opcion predicha de clase "no change") bajo duplicate frente a full.
swap: acc vs. original y vs. invertida (regla antigua = columna answer_inverted; regla inequivoca
= swap_inverted_recompute.unambiguous_inverted sobre las opciones sin permutar), por clase, y
acc(vs invertida) - acc(vs original) pareado en el mismo subconjunto.
Bootstrap por paciente, N=5000, semilla 0, IC95%, en pp."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(HERE))
from swap_inverted_recompute import unambiguous_inverted, label_option, boot
OFF = 1_000_000
MODELS = {"A (TAC preentrenado)": REPO / "tac_mcq_train/runs/tac_pre_clean",
          "B (TAC desde cero)": REPO / "tac_mcq_train/runs/tac_scratch_clean",
          "C (genérico)": REPO / "qwen25vl_baseline_train/runs/qwen_clean"}
CT = ("worsen", "improved", "no change")
ent = {}
for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
    for d in map(json.loads, open(fn)):
        ent[d["id"] + (OFF if src == "dev" else 0)] = d
pat = lambda qs: np.array([ent[q]["cur_image_path"].split("/")[2] for q in qs])
load = lambda p: {int(r["qid"]): r for r in csv.DictReader(open(p))}
pct = lambda arr, qs: boot(arr, pat(qs))[0].replace("+", "") if False else boot(arr, pat(qs))[0]
L = ["# swap y duplicate — conjunto limpio (A, B, C)\n", "Bootstrap por paciente, N=5000, semilla 0, IC95%. Valores en pp.\n"]

L += ["## duplicate: P(respuesta de clase 'no change')\n",
      "| modelo | clase | n | full | duplicate | duplicate − full |", "|---|---|---|---|---|---|"]
for m, d in MODELS.items():
    F, D = load(d / "clean_full.csv"), load(d / "clean_duplicate.csv")
    qs_all = sorted(set(F) & set(D))
    nc = lambda R, q: float(label_option(ent[q]["options"].get(R[q]["pred_letter"], "")) == "no change") if R[q]["pred_letter"] else 0.0
    for ct in ("todas",) + CT:
        qs = [q for q in qs_all if ct == "todas" or ent[q]["content_type"] == ct]
        f = np.array([nc(F, q) for q in qs]); dd = np.array([nc(D, q) for q in qs])
        L.append(f"| {m} | {ct} | {len(qs)} | {boot(f, pat(qs))[0].lstrip('+')} | {boot(dd, pat(qs))[0].lstrip('+')} | {boot(dd - f, pat(qs))[0]} |")
L.append("")

L += ["## swap\n", "| modelo | clase | regla | n | vs. original − full | vs. invertida − full | acc(invertida) − acc(original) |",
      "|---|---|---|---|---|---|---|"]
for m, d in MODELS.items():
    F, S = load(d / "clean_full.csv"), load(d / "clean_swap.csv")
    qs_all = sorted(set(F) & set(S))
    fc = lambda qs: np.array([F[q]["correct"] == "True" for q in qs], float)
    orig = lambda qs: np.array([S[q]["correct"] == "True" for q in qs], float)
    s, _ = boot(orig(qs_all) - fc(qs_all), pat(qs_all))
    L.append(f"| {m} | todas | — | {len(qs_all)} | {s} | — | — |")
    for ct in CT:
        qc = [q for q in qs_all if ent[q]["content_type"] == ct]
        if ct == "no change":
            s, _ = boot(orig(qc) - fc(qc), pat(qc)); L.append(f"| {m} | no change | — | {len(qc)} | {s} | (autoinversión) | — |"); continue
        for rule in ("antigua", "inequívoca"):
            if rule == "antigua":
                qs = [q for q in qc if S[q]["answer_inverted"]]; tgt = {q: S[q]["answer_inverted"] for q in qs}
            else:
                tgt = {q: unambiguous_inverted(ent[q]["options"], ent[q]["answer"], ct) for q in qc}
                qs = [q for q in qc if tgt[q]]
            inv = np.array([S[q]["pred_letter"] == tgt[q] for q in qs], float)
            a, _ = boot(orig(qs) - fc(qs), pat(qs)); b, _ = boot(inv - fc(qs), pat(qs)); c, _ = boot(inv - orig(qs), pat(qs))
            L.append(f"| {m} | {ct} | {rule} | {len(qs)} | {a} | {b} | {c} |")
    qwi = [q for q in qs_all if ent[q]["content_type"] != "no change" and unambiguous_inverted(ent[q]["options"], ent[q]["answer"], ent[q]["content_type"])]
    inv = np.array([S[q]["pred_letter"] == unambiguous_inverted(ent[q]["options"], ent[q]["answer"], ent[q]["content_type"]) for q in qwi], float)
    a, _ = boot(orig(qwi) - fc(qwi), pat(qwi)); b, _ = boot(inv - fc(qwi), pat(qwi)); c, _ = boot(inv - orig(qwi), pat(qwi))
    L.append(f"| {m} | worsen+improved | inequívoca | {len(qwi)} | {a} | {b} | **{c}** |")
text = "\n".join(L) + "\n"; (HERE / "swapdup_report.md").write_text(text); print(text)
