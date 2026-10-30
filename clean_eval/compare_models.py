#!/usr/bin/env python3
"""Comparaciones pre-registradas (PREREGISTRO_SPLIT_LIMPIO_2026-09-24, §4b), sobre CSV ya guardados.
  1. X − C (C = genérico), pareado por fila: accuracy(full), blind−full, mismatch−full.
  2. DiD: (X_limpio − X_completo) − (C_limpio − C_completo).
Bootstrap por paciente N=5000 semilla 0, sobre la UNIÓN de pacientes de las dos poblaciones
(comparten 27), mismos índices para todo."""
import csv, json, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
N_BOOT, SEED, OFF = 5000, 0, 1_000_000
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
MODELS = {"A (TAC preentrenado)": T / "tac_pre_clean", "B (TAC desde cero)": T / "tac_scratch_clean",
          "B2 (TAC desde cero, 2 etapas)": T / "tac_b2_stage2", "C (genérico)": Q / "qwen_clean"}
# C_DIR=<dir> sustituye el directorio de C (p.ej. qwen_clean/px518, C a 518²) y escribe el informe
# con sufijo _c518; sin C_DIR el comportamiento es el de siempre.
if os.environ.get("C_DIR"):
    MODELS["C (genérico)"] = Path(os.environ["C_DIR"])
REF = "C (genérico)"
POPS = {"limpio": "clean_{}.csv", "completo": "oracle_{}.csv"}
QTYS = ("accuracy", "blind−full", "mismatch−full")

pat = {"limpio": {}, "completo": {}}
for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
    for line in open(fn):
        d = json.loads(line); p = d["cur_image_path"].split("/")[2]
        pat["limpio"][d["id"] + (OFF if src == "dev" else 0)] = p
        if src == "test": pat["completo"][d["id"]] = p
corr = lambda p: {int(r["qid"]): r["correct"] == "True" for r in csv.DictReader(open(p))}

rows, vals = {}, {}
for pop, tmpl in POPS.items():
    per = {m: {c: corr(d / tmpl.format(c)) for c in ("full", "blind", "mismatch")} for m, d in MODELS.items()}
    qs = sorted(set.intersection(*[set(v) for m in per.values() for v in m.values()]))
    assert all(len(v) == len(qs) for m in per.values() for v in m.values()), pop
    rows[pop] = (qs, np.array([pat[pop][q] for q in qs]))
    for m, c in per.items():
        f, b, mm = (np.array([c[k][q] for q in qs], float) for k in ("full", "blind", "mismatch"))
        vals[(m, pop, "accuracy")], vals[(m, pop, "blind−full")], vals[(m, pop, "mismatch−full")] = f, b - f, mm - f

union = np.unique(np.concatenate([p for _, p in rows.values()]))
pidx = {p: i for i, p in enumerate(union)}
boot = np.random.default_rng(SEED).integers(0, len(union), (N_BOOT, len(union)))
W = np.zeros((N_BOOT, len(union))); np.add.at(W, (np.arange(N_BOOT)[:, None], boot), 1.0)

def rep(pop, arr):
    ids = np.array([pidx[p] for p in rows[pop][1]])
    s, c = np.zeros(len(union)), np.zeros(len(union)); np.add.at(s, ids, arr); np.add.at(c, ids, 1.0)
    return arr.mean(), (W @ s) / (W @ c)
def ci(pt, r):
    lo, hi = np.percentile(r, [2.5, 97.5]); return f"{pt*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]"
def gap(who, qty):
    (pc, rc), (pf, rf) = rep("limpio", vals[(who, "limpio", qty)]), rep("completo", vals[(who, "completo", qty)])
    return pc - pf, rc - rf

L = ["# Comparaciones entre modelos — conjunto limpio y MMXU-test completo\n",
     f"Bootstrap por paciente sobre la unión de pacientes ({len(union)}), N={N_BOOT}, semilla {SEED}, IC95%. pp.\n",
     "## Valores por modelo\n", "| modelo | población | accuracy | blind−full | mismatch−full |", "|---|---|---|---|---|"]
for m in MODELS:
    for pop in POPS:
        L.append(f"| {m} | {pop} (n={len(rows[pop][0])}) | " + " | ".join(ci(*rep(pop, vals[(m, pop, q)])) for q in QTYS) + " |")
L += ["", "## 1. Contraste X − C, pareado por fila\n", "| contraste | población | Δ accuracy | Δ blind−full | Δ mismatch−full |", "|---|---|---|---|---|"]
for m in MODELS:
    if m == REF: continue
    for pop in POPS:
        L.append(f"| {m.split()[0]} − C | {pop} | " + " | ".join(ci(*rep(pop, vals[(m, pop, q)] - vals[(REF, pop, q)])) for q in QTYS) + " |")
L += ["", "## 2. Diferencia en diferencias: (X_limpio − X_completo) − (C_limpio − C_completo)\n",
      "| modelo | X: limpio − completo (acc.) | C: limpio − completo (acc.) | DiD accuracy | DiD blind−full | DiD mismatch−full |", "|---|---|---|---|---|---|"]
for m in MODELS:
    if m == REF: continue
    cells = []
    for q in QTYS:
        (px, rx), (pc, rc) = gap(m, q), gap(REF, q); cells.append(ci(px - pc, rx - rc))
    L.append(f"| {m} | {ci(*gap(m, 'accuracy'))} | {ci(*gap(REF, 'accuracy'))} | " + " | ".join(cells) + " |")
text = "\n".join(L) + "\n"
out = "compare_models_report_c518.md" if os.environ.get("C_DIR") else "compare_models_report.md"
Path(__file__).with_name(out).write_text(text); print(text)
