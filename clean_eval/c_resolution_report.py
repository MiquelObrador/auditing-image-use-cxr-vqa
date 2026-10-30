#!/usr/bin/env python3
"""Genérico C: evaluación a la resolución del protocolo (~1.248 tokens/imagen) frente a su
resolución de entrenamiento (268.324 px = 518², ~320 tokens/imagen). Conjunto limpio: full,
blind, both_other, mismatch, swap; MMXU-test completo: full, blind.
Bootstrap por paciente, N=5000, semilla 0, IC95%, en pp."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(HERE)); from swap_inverted_recompute import boot, unambiguous_inverted
OFF = 1_000_000
C = REPO / "qwen25vl_baseline_train/runs/qwen_clean"
ent = {}
for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
    for d in map(json.loads, open(fn)):
        ent[(src == "dev") * OFF + d["id"]] = d
load = lambda p: {int(r["qid"]): r for r in csv.DictReader(open(p))}
pat = lambda qs: np.array([ent[q]["cur_image_path"].split("/")[2] for q in qs])
ok = lambda R, qs: np.array([R[q]["correct"] == "True" for q in qs], float)
L = ["# Genérico C: resolución de evaluación del protocolo vs. resolución de entrenamiento (518²)\n",
     "Bootstrap por paciente, N=5000, semilla 0, IC95%, en pp.\n"]
for pop, pref, conds in (("conjunto limpio", "clean", ("full", "blind", "both_other", "mismatch", "swap")),
                         ("MMXU-test completo", "oracle", ("full", "blind"))):
    std = {c: load(C / f"{pref}_{c}.csv") for c in conds}
    px = {c: load(C / "px518" / f"{pref}_{c}.csv") for c in conds}
    qs = sorted(set.intersection(*[set(v) for v in list(std.values()) + list(px.values())]))
    assert all(len(v) == len(qs) for v in list(std.values()) + list(px.values())), pop
    P = pat(qs)
    L += [f"## {pop} (n={len(qs)} filas, {len(set(P))} pacientes)\n",
          "| condición | accuracy protocolo (~1.248 tok) | accuracy 518² (~320 tok) | 518² − protocolo | predicciones que cambian |",
          "|---|---|---|---|---|"]
    for c in conds:
        a, b = ok(std[c], qs), ok(px[c], qs)
        ch = np.array([std[c][q]["pred_letter"] != px[c][q]["pred_letter"] for q in qs], float)
        L.append(f"| {c} | {a.mean():.4f} | {b.mean():.4f} | {boot(b - a, P)[0]} | {ch.mean()*100:.1f}% |")
    L += ["", "| contraste | protocolo | 518² |", "|---|---|---|"]
    for c in conds[1:]:
        if c == "swap":
            continue
        L.append(f"| {c} − full | {boot(ok(std[c], qs) - ok(std['full'], qs), P)[0]} | {boot(ok(px[c], qs) - ok(px['full'], qs), P)[0]} |")
    if "swap" in conds:
        wi = [q for q in qs if ent[q]["content_type"] != "no change"
              and unambiguous_inverted(ent[q]["options"], ent[q]["answer"], ent[q]["content_type"])]
        for name, R in (("protocolo", std), ("518²", px)):
            inv = np.array([R["swap"][q]["pred_letter"] == unambiguous_inverted(ent[q]["options"], ent[q]["answer"], ent[q]["content_type"]) for q in wi], float)
            o = ok(R["swap"], wi)
            L.append(f"| swap ({name}): vs. original − full, todas | {boot(ok(R['swap'], qs) - ok(R['full'], qs), P)[0]} | |" if name == "protocolo" else
                     f"| swap ({name}): vs. original − full, todas | | {boot(ok(R['swap'], qs) - ok(R['full'], qs), P)[0]} |")
            L.append(f"| swap ({name}): acc(invertida) − acc(original), inequívocas n={len(wi)} | "
                     + (f"{boot(inv - o, pat(wi))[0]} | |" if name == "protocolo" else f"| {boot(inv - o, pat(wi))[0]} |"))
    L.append("")
text = "\n".join(L) + "\n"; (HERE / "c_resolution_report.md").write_text(text); print(text)
