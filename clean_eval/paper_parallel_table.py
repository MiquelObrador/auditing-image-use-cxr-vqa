#!/usr/bin/env python3
"""Tabla principal del paper (2026-09-29): dos paneles con columnas paralelas.

(a) VLM zero-shot, MMXU-test completo (n=2992): Qwen2-VL-7B, InternVL2-8B, HuatuoGPT-7B,
    Qwen2.5-VL-7B.
(b) Modelos afinados, conjunto limpio (n=3096): A (TAC preentrenado), B (TAC desde cero), C
    (genérico Qwen2.5-VL-3B).

Columnas: accuracy(full); blind−full; both_other−full; mismatch−full; swap−full (vs. etiqueta
original); Δpred bajo blind; Δpred bajo both_other.

Dos detalles que no son obvios:
  * swap zero-shot se compara contra full CON LA MISMA permutación de letras
    (oracle_full_permuted.csv), no contra full con letras originales. La permutación sola
    cuesta 3-5pp y contaminaba la cifra antigua. Se comprueba que las dos permutaciones
    coinciden fila a fila.
  * mismatch y swap no existen para Qwen2-VL-7B ni InternVL2-8B: el protocolo solo evalúa las
    condiciones temporales en los modelos que pasan el gate (celda "---").

Veredicto por celda de diferencia: CI95 que excluye cero; TOST (IC90 dentro de ±5pp) = Eq.;
Sig. = |estimación| ≥ 5pp con CI95 que excluye cero. Bootstrap por paciente, N=5000, semilla 0.

Uso: python3 clean_eval/paper_parallel_table.py
Salida: clean_eval/paper_parallel_table.md
"""
import csv
import json
from pathlib import Path

import numpy as np

from paper_tables import Boot, load, patient_maps, OFF  # noqa: F401  (mismo bootstrap)

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DELTA = 0.05
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"

ZS = [("Qwen2-VL-7B", REPO / "mmxu_eval/oracle_results"),
      ("InternVL2-8B", REPO / "internvl_eval/oracle_results"),
      ("HuatuoGPT-7B", REPO / "huatuo_eval/oracle_results"),
      ("Qwen2.5-VL-7B", REPO / "qwen25vl_eval/oracle_results")]
FT = [("A: pretrained TAC", T / "tac_pre_clean"),
      ("B: TAC from scratch", T / "tac_scratch_clean"),
      # B2 (2026-09-26, a posteriori): TAC desde cero entrenado solo primero, luego la LoRA
      ("B2: TAC from scratch, 2 stages", T / "tac_b2_stage2"),
      # C evaluado a su resolución de entrenamiento (518², 268.324 px), 2026-09-29
      ("C: generic", Q / "qwen_clean/px518")]


def opt(path):
    return load(path) if path.exists() else None


def verdict(t):
    m, lo, hi, lo90, hi90 = t
    excl = lo > 0 or hi < 0
    if abs(m) >= DELTA and excl:
        return "Sig."
    if -DELTA < lo90 and hi90 < DELTA:
        return "Eq."
    return "Inconcl."


def diff_cell(bt, a, b, dpred):
    t = bt.ci(a - b)
    hw = (t[2] - t[1]) / 2
    excl = t[1] > 0 or t[2] < 0
    d = bt.ci(dpred)
    cell = f"{t[0]*100:+.1f}±{hw*100:.1f} {verdict(t)}{' *' if excl else ''}; Δpred {d[0]*100:.1f}%"
    return cell, dict(est=t[0], lo=t[1], hi=t[2], lo90=t[3], hi90=t[4], verdict=verdict(t),
                      dpred=d[0], dpred_lo=d[1], dpred_hi=d[2])


def row(name, full, cond, pat, swap=None, swap_base=None):
    """Cada celda de condición: condición−full ± semiancho del IC95%, veredicto, `*` si el IC95%
    excluye cero, y Δpred (filas cuya letra cambia frente a full; para swap zero-shot, frente a
    full con la MISMA permutación de letras). Gate: blind−full con IC95% < 0; las filas que no lo
    pasan son descriptivas."""
    qs = sorted(full)
    # una condición cuenta solo si cubre todas las filas (p.ej. una corrida a medias no entra)
    cond = {k: (R if R is not None and set(full) <= set(R) else None) for k, R in cond.items()}
    bt = Boot(qs, pat)
    corr = lambda R, col="correct": np.array([R[q][col] == "True" for q in qs], float)
    chg = lambda R, B: np.array([R[q]["pred_letter"] != B[q]["pred_letter"] for q in qs], float)
    f = corr(full)
    stats = {"acc_full": bt.ci(f)[:3]}
    cells = [f"{f.mean():.3f}"]
    for k in ("blind", "both_other", "mismatch"):
        R = cond.get(k)
        if R:
            c, stats[k] = diff_cell(bt, corr(R), f, chg(R, full))
            cells.append(c)
        else:
            cells.append("---")
    if swap is not None and set(full) <= set(swap):
        base_rows = swap_base if swap_base is not None else full
        # zero-shot (swap_permuted): columna correct_vs_original; afinados: `correct` ya es vs.
        # la etiqueta original (answer_inverted va aparte).
        col = "correct_vs_original" if "correct_vs_original" in next(iter(swap.values())) else "correct"
        c, stats["swap"] = diff_cell(bt, corr(swap, col), corr(base_rows), chg(swap, base_rows))
        cells.append(c)
    else:
        cells.append("---")
    gate = "blind" in stats and stats["blind"]["hi"] < 0
    stats["gate"] = bool(gate)
    label = name if gate else f"{name} (descriptive)"
    return f"| {label} | " + " | ".join(cells) + f" | {'✓' if gate else '✗'} |", stats


def main():
    pat = patient_maps()
    head = "| model | acc. full | blind−full | both_other−full | mismatch−full | swap−full | gate |"
    out = {}
    md = ["# Tabla paralela del paper — zero-shot (MMXU-test) y afinados (conjunto limpio)\n",
          "Cada celda: condición−full en pp ± semiancho del IC95%, veredicto (Sig. = |Δ| ≥ 5pp con "
          "IC95% que excluye cero; Eq. = IC90% dentro de ±5pp, TOST; si no, Inconcl.), `*` = IC95% "
          "excluye cero, y Δpred = % de filas cuya letra cambia frente a full. Bootstrap por "
          "paciente, N=5000, semilla 0. Gate: blind−full con IC95% < 0; las filas sin gate son "
          "descriptivas.\n",
          "## (a) VLM zero-shot, MMXU-test (n=2992)\n", head, "|" + "---|" * 7]
    for name, d in ZS:
        full = load(d / "oracle_full.csv")
        # mismatch: derangement por paciente (clean_eval/run_mismatch_zeroshot.py, 2026-10-16), no
        # la previa de la fila anterior del protocolo original (oracle_mismatch.csv)
        cond = {k: opt(d / f"oracle_{k}.csv") for k in ("blind", "both_other")}
        cond["mismatch"] = opt(d / "oracle_mismatch_patient.csv")
        sp, fp = opt(d / "oracle_swap_permuted.csv"), opt(d / "oracle_full_permuted.csv")
        if sp is not None:
            assert all(sp[q]["answer_permuted"] == fp[q]["answer_permuted"] for q in sp), \
                f"{name}: la permutación de swap y de full_permuted no coincide"
        line, out[name] = row(name, full, cond, pat["full"], swap=sp, swap_base=fp)
        md.append(line)
    md += ["", "mismatch zero-shot: previa de otro paciente (derangement por paciente, semilla 0, "
           "`clean_eval/run_mismatch_zeroshot.py`; 0 filas con previa del mismo paciente), como en el "
           "conjunto limpio. swap zero-shot: swap con letras permutadas vs. etiqueta original, menos "
           "full con la MISMA permutación (oracle_full_permuted.csv).", "",
           "## (b) Modelos afinados, conjunto limpio (n=3096)\n", head, "|" + "---|" * 7]
    for name, d in FT:
        full = load(d / "clean_full.csv")
        cond = {k: opt(d / f"clean_{k}.csv") for k in ("blind", "both_other", "mismatch")}
        line, out[name] = row(name, full, cond, pat["clean"], swap=opt(d / "clean_swap.csv"))
        md.append(line)
    md.append("")
    (HERE / "paper_parallel_table.json").write_text(json.dumps(
        out, indent=1, default=lambda x: float(x) if isinstance(x, np.floating) else list(map(float, x))))
    text = "\n".join(md) + "\n"
    (HERE / "paper_parallel_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
