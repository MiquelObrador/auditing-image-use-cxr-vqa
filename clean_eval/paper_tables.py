#!/usr/bin/env python3
"""Tablas finales para el paper (LaTeX, booktabs) + subconjunto difícil.

Tabla 1 — conjunto limpio, A / B / C: accuracy(full), accuracy(blind), blind−full,
  mismatch−full, Δpred(blind), Δpred(mismatch), nulo de Δpred (hole_random, medido en MMXU-test
  completo porque el conjunto limpio no tiene bbox de la previa) y veredictos de los 3 criterios.
Tabla 2 — control de tokens constantes, conjunto limpio, A / B2 / C.
Tabla 3 — subconjunto difícil (preguntas que falla el solo-texto reentrenado con el split limpio),
  en el conjunto limpio y en MMXU-test completo: accuracy de A y C, con y sin imagen, y A − C.

Bootstrap por paciente, N=5000, semilla 0, IC95%. Δpred = fracción de filas cuya letra predicha
cambia respecto a full (una respuesta no parseable cuenta como una letra más).

Uso: python3 clean_eval/paper_tables.py [--textonly DIR] [--out-dir DIR]
Salida: paper_tables.tex y paper_tables.md
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
N_BOOT, SEED, OFF = 5000, 0, 1_000_000
DELTA, FLOOR, CHANCE = 0.05, 0.395, 0.25
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
MAIN = {"A": ("TAC (Libra-pretrained)", T / "tac_pre_clean"),
        "B": ("TAC (from scratch, joint)", T / "tac_scratch_clean"),
        "C": ("Generic (Qwen2.5-VL-3B)", Q / "qwen_clean")}
CONTROL = {"A": MAIN["A"], "B2": ("TAC (from scratch, two-stage)", T / "tac_b2_stage2"), "C": MAIN["C"]}
PREFIX = {"clean": "clean_", "full": "oracle_"}


def patient_maps():
    pat = {"clean": {}, "full": {}}
    for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        for line in open(fn):
            d = json.loads(line)
            p = d["cur_image_path"].split("/")[2]
            pat["clean"][d["id"] + (OFF if src == "dev" else 0)] = p
            if src == "test":
                pat["full"][d["id"]] = p
    return pat


def load(path):
    return {int(r["qid"]): r for r in csv.DictReader(open(path))}


class Boot:
    def __init__(self, qs, pat):
        u, self.inv = np.unique([pat[q] for q in qs], return_inverse=True)
        self.n_pat = len(u)
        idx = np.random.default_rng(SEED).integers(0, len(u), (N_BOOT, len(u)))
        self.W = np.zeros((N_BOOT, len(u)))
        np.add.at(self.W, (np.arange(N_BOOT)[:, None], idx), 1.0)
        self.cnt = np.zeros(len(u))
        np.add.at(self.cnt, self.inv, 1.0)

    def ci(self, arr):
        arr = np.asarray(arr, float)
        s = np.zeros(len(self.cnt))
        np.add.at(s, self.inv, arr)
        r = (self.W @ s) / (self.W @ self.cnt)
        lo, hi = np.percentile(r, [2.5, 97.5])
        lo90, hi90 = np.percentile(r, [5, 95])
        return arr.mean(), lo, hi, lo90, hi90


acc_s = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
pp_s = lambda t: f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"
pct_s = lambda t: f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"
tex = lambda s: re.sub(r"-(?=\d)", "$-$", s.replace("−", "$-$").replace("+", "$+$").replace("%", r"\%"))
correct = lambda rows, qs: np.array([rows[q]["correct"] == "True" for q in qs], float)
changed = lambda a, b, qs: np.array([a[q]["pred_letter"] != b[q]["pred_letter"] for q in qs], float)


def table_main(pat, md, tx):
    md += ["## Tabla 1 — conjunto limpio (A, B, C)\n",
           "| modelo | accuracy | acc. sin imagen | blind−full (pp) | mismatch−full (pp) | "
           "Δpred blind (%) | Δpred mismatch (%) | Δpred nulo, MMXU-test (%) | criterios 1/2/3 |",
           "|---|---|---|---|---|---|---|---|---|"]
    body = []
    for key, (name, d) in MAIN.items():
        c = {k: load(d / f"clean_{k}.csv") for k in ("full", "blind", "mismatch")}
        qs = sorted(set.intersection(*[set(v) for v in c.values()]))
        assert all(len(v) == len(qs) for v in c.values()), key
        bt = Boot(qs, pat["clean"])
        f, b, m = (correct(c[k], qs) for k in ("full", "blind", "mismatch"))
        acc, accb, gate, mis = bt.ci(f), bt.ci(b), bt.ci(b - f), bt.ci(m - f)
        dpb = bt.ci(changed(c["blind"], c["full"], qs))
        dpm = bt.ci(changed(c["mismatch"], c["full"], qs))
        null_path = d / "oracle_hole_random.csv"
        if null_path.exists():
            ff, hr = load(d / "oracle_full.csv"), load(null_path)
            qn = sorted(set(ff) & set(hr))
            dpn = pct_s(Boot(qn, pat["full"]).ci(changed(hr, ff, qn)))
        else:
            dpn = "pendiente"
        c1, c2, c3 = acc[1] > FLOOR, gate[2] < 0, mis[2] < 0 and mis[0] <= -DELTA
        tost = mis[3] > -DELTA and mis[4] < DELTA
        v = "/".join("✓" if x else "✗" for x in (c1, c2, c3)) + (" (TOST ≈0)" if tost and not c3 else "")
        md.append(f"| {key}: {name} | {acc_s(acc)} | {acc_s(accb)} | {pp_s(gate)} | {pp_s(mis)} | "
                  f"{pct_s(dpb)} | {pct_s(dpm)} | {dpn} | {v} |")
        vt = " ".join(r"\cmark" if x else r"\xmark" for x in (c1, c2, c3)) + (r"$^{\dagger}$" if tost and not c3 else "")
        body.append(f"{key} & {acc_s(acc)} & {acc_s(accb)} & {pp_s(gate)} & {pp_s(mis)} & "
                    f"{pct_s(dpb)} & {pct_s(dpm)} & {dpn} & {vt} \\\\")
    n_line = f"n = {len(qs)} rows, {bt.n_pat} patients"
    md += ["", f"{n_line}. ✓/✗: criterios (1) accuracy > {FLOOR}, (2) blind−full < 0, "
           "(3) mismatch−full ≤ −5pp, todos con IC95%. TOST ≈0: IC90% de mismatch−full dentro de "
           "±5pp. El nulo de Δpred (hole_random) está medido en MMXU-test completo.", ""]
    tx += [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Clean evaluation set (" + n_line + r"; patients in the official MIMIC-CXR "
           r"validate/test split, unseen by Libra's TAC, RAD-DINO and our fine-tuning). Accuracy "
           r"with and without the images, paired differences (pp) and prediction flips "
           r"($\Delta$pred, \%). 95\% patient-clustered bootstrap CIs. The $\Delta$pred null "
           r"(\textit{hole\_random}) is measured on the full MMXU-test, since the clean set lacks "
           r"prior-image boxes. Criteria: (1) accuracy $>0.395$, (2) blind$-$full $<0$, "
           r"(3) mismatch$-$full $\le -5$pp. $^{\dagger}$Equivalent to zero (TOST, $\pm5$pp).}",
           r"\label{tab:clean_main}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}", r"\toprule",
           r"Model & Acc. & Acc. (no image) & Blind$-$Full & Mismatch$-$Full & "
           r"$\Delta$pred blind & $\Delta$pred mismatch & $\Delta$pred null & Criteria \\",
           r"\midrule"] + [tex(x) for x in body] + [
           r"\bottomrule", r"\end{tabular}}",
           r"\\[2pt]\footnotesize A: " + MAIN["A"][0] + "; B: " + MAIN["B"][0] + "; C: " + MAIN["C"][0] + ".",
           r"\end{table*}", ""]


def table_control(pat, md, tx):
    md += ["## Tabla 2 — control de tokens constantes, conjunto limpio (A, B2, C)\n",
           "| modelo | acc. full | acc. blind | acc. otro paciente | otro−full (pp) | otro−blind (pp) |",
           "|---|---|---|---|---|---|"]
    body = []
    for key, (name, d) in CONTROL.items():
        c = {k: load(d / f"clean_{k}.csv") for k in ("full", "blind", "both_other")}
        qs = sorted(set.intersection(*[set(v) for v in c.values()]))
        bt = Boot(qs, pat["clean"])
        f, b, o = (correct(c[k], qs) for k in ("full", "blind", "both_other"))
        cells = [acc_s(bt.ci(f)), acc_s(bt.ci(b)), acc_s(bt.ci(o)), pp_s(bt.ci(o - f)), pp_s(bt.ci(o - b))]
        md.append(f"| {key}: {name} | " + " | ".join(cells) + " |")
        body.append(f"{key} & " + " & ".join(cells) + r" \\")
    md.append("")
    tx += [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Constant-token control on the clean set: both images replaced by another "
           r"patient's pair (same number of visual tokens, correct information removed). If the "
           r"image gate reflects use of image content, accuracy with another patient's images "
           r"falls towards blind; if it only reflects the tokens being absent, it stays at full. "
           r"95\% patient-clustered bootstrap CIs; differences in pp.}",
           r"\label{tab:clean_control}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}", r"\toprule",
           r"Model & Full & Blind & Other patient & Other$-$Full & Other$-$Blind \\",
           r"\midrule"] + [tex(x) for x in body] + [
           r"\bottomrule", r"\end{tabular}}",
           r"\\[2pt]\footnotesize A: " + CONTROL["A"][0] + "; B2: " + CONTROL["B2"][0] + "; C: " + CONTROL["C"][0] + ".",
           r"\end{table*}", ""]


def table_hard(pat, md, tx, textonly):
    md += ["## Tabla 3 — subconjunto difícil (preguntas que falla el solo-texto)\n",
           "| población | n filas / pacientes (% del total) | solo-texto | A | A sin imagen | C | C sin imagen | A − C (pp) | A full − A sin imagen (pp) |",
           "|---|---|---|---|---|---|---|---|---|"]
    body = []
    for pop, label in (("clean", "Clean set"), ("full", "Full MMXU-test")):
        tpath = textonly / f"{PREFIX[pop]}blind.csv"
        if not tpath.exists():
            md.append(f"| {pop} | pendiente ({tpath.name} no existe) | | | | | | | |")
            continue
        to = load(tpath)
        A = {k: load(MAIN["A"][1] / f"{PREFIX[pop]}{k}.csv") for k in ("full", "blind")}
        C = {k: load(MAIN["C"][1] / f"{PREFIX[pop]}{k}.csv") for k in ("full", "blind")}
        allq = sorted(set(to) & set(A["full"]) & set(A["blind"]) & set(C["full"]) & set(C["blind"]))
        hard = [q for q in allq if to[q]["correct"] != "True"]
        bt = Boot(hard, pat[pop])
        a, ab = correct(A["full"], hard), correct(A["blind"], hard)
        c, cb = correct(C["full"], hard), correct(C["blind"], hard)
        share = f"{len(hard)} / {bt.n_pat} ({100*len(hard)/len(allq):.1f}%)"
        cells = ["0.000 (by def.)", acc_s(bt.ci(a)), acc_s(bt.ci(ab)), acc_s(bt.ci(c)),
                 acc_s(bt.ci(cb)), pp_s(bt.ci(a - c)), pp_s(bt.ci(a - ab))]
        md.append(f"| {pop} | {share} | " + " | ".join(cells) + " |")
        body.append(f"{label} & {share} & " + " & ".join(cells) + r" \\")
    md += ["", f"Azar: {CHANCE}. El solo-texto acierta 0 por construcción del subconjunto. "
           "Salvedad: el subconjunto se selecciona por los errores de un modelo de texto, así que "
           "concentra preguntas cuyo texto engaña (el modelo eligió un distractor verosímil); el "
           "azar (0,25) no es un suelo neutral ahí. La referencia de uso de imagen es la diferencia "
           "con y sin imagen. A − C mezcla uso de imagen con que A sin imagen se aparta del atajo del "
           "solo-texto (C se parece al solo-texto y cae cerca de cero por selección); la parte "
           "atribuible a la imagen es A full − A sin imagen.", ""]
    tx += [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Hard subset: questions the text-only model (Qwen2.5-VL-3B fine-tuned without "
           r"images, clean split) answers incorrectly. Accuracy of the TAC model (A) and the generic "
           r"model (C) with and without images; chance is 0.25. The subset is selected by a text model's "
           r"errors, so C (which relies on the same textual cues) falls below chance by construction; "
           r"the image-attributable effect is A$-$A (no image). 95\% patient-clustered bootstrap CIs.}",
           r"\label{tab:hard_subset}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}", r"\toprule",
           r"Population & $n$ rows / patients & Text-only & A & A (no image) & C & C (no image) & A$-$C (pp) & A$-$A (no image) (pp) \\",
           r"\midrule"] + [tex(x) for x in body] + [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textonly", type=Path, default=Q / "textonly_clean")
    ap.add_argument("--c-dir", type=Path, default=None,
                    help="directorio de C alternativo (p.ej. qwen_clean/px518); por defecto qwen_clean")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    pat = patient_maps()
    md = ["# Tablas finales — conjunto limpio\n",
          f"Bootstrap por paciente, N={N_BOOT}, semilla {SEED}, IC95%. Solo-texto: {args.textonly}\n"]
    tx = [r"% Requiere \usepackage{booktabs,amssymb,pifont,graphicx};",
          r"% \newcommand{\cmark}{\ding{51}} \newcommand{\xmark}{\ding{55}}", ""]
    if args.c_dir is not None:
        # C alternativo: solo se recalcula el subconjunto difícil (las otras dos tablas usan
        # condiciones, como hole_random, que no existen para todas las resoluciones de C).
        MAIN["C"] = (MAIN["C"][0], args.c_dir)
        md.append(f"C: {args.c_dir}\n")
    else:
        table_main(pat, md, tx)
        table_control(pat, md, tx)
    table_hard(pat, md, tx, args.textonly)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paper_tables.md").write_text("\n".join(md) + "\n")
    (args.out_dir / "paper_tables.tex").write_text("\n".join(tx) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
