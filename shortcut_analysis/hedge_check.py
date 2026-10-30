#!/usr/bin/env python3
"""Comprobacion directa del mecanismo sugerido por options_only_linear.py: la opcion correcta
usa lenguaje cauteloso de informe y los distractores lenguaje absoluto. Las listas salen de los
rasgos de mas peso del lineal entrenado en MMXU-dev (no se eligieron mirando MMXU-test).

Uso: python3 shortcut_analysis/hedge_check.py
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

from textual_shortcut import boot_ci, fmt, MMXU_TEST_JSONL, EVAL_QIDS

HERE = Path(__file__).resolve().parent
HEDGE = r"\b(small|possibly|possible|almost|minimal|minimally|nearly|likely|suggesting|slight|slightly|mild|mildly)\b"
EXTREME = r"\b(complete|completely|significant|significantly|disappeared|new mass|new abnormality|entirely|total|totally)\b"


def main():
    qids = {int(r["qid"]) for r in csv.DictReader(open(EVAL_QIDS))}
    rows = [d for d in (json.loads(l) for l in open(MMXU_TEST_JSONL)) if d["id"] in qids]
    g_h = g_e = d_h = d_e = n_d = 0
    ok, pats = [], []
    for d in rows:
        for L, t in d["options"].items():
            h, e = bool(re.search(HEDGE, t, re.I)), bool(re.search(EXTREME, t, re.I))
            if L == d["answer"]:
                g_h += h; g_e += e
            else:
                d_h += h; d_e += e; n_d += 1
        s = {L: len(re.findall(HEDGE, t, re.I)) - len(re.findall(EXTREME, t, re.I))
             for L, t in d["options"].items()}
        top = [L for L in s if s[L] == max(s.values())]
        ok.append((d["answer"] in top) / len(top))
        pats.append(d["cur_image_path"].split("/")[2])
    n = len(rows)
    text = "\n".join([
        f"# Lenguaje cauteloso frente a absoluto en las opciones — MMXU-test ({n} filas)\n",
        "| | opción correcta | distractores |", "|---|---|---|",
        f"| contiene palabra cautelosa | {g_h/n:.1%} | {d_h/n_d:.1%} |",
        f"| contiene palabra absoluta | {g_e/n:.1%} | {d_e/n_d:.1%} |", "",
        f"Heurístico 'la más cautelosa y menos absoluta' (desempate al azar): "
        f"{fmt(boot_ci(np.array(ok), np.array(pats)))}", ""])
    (HERE / "hedge_check_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
