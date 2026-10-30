#!/usr/bin/env python3
"""Giro de respuesta bajo swap en los VLM zero-shot (HuatuoGPT-7B, Qwen2.5-VL-7B), MMXU-test.

Giro = acc(etiqueta invertida) − acc(etiqueta original), ambas bajo swap_permuted y con la regla
inequívoca de swap_inverted_recompute.py. Es la misma métrica que la columna
acc(invertida) − acc(original) de swapdup_report.md para los modelos afinados. Por clase.
Bootstrap por paciente, N=5000, semilla 0.

Uso: python3 clean_eval/swap_turn_zeroshot.py   ->  clean_eval/swap_turn_zeroshot_report.md
"""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swap_inverted_recompute as S  # noqa: E402  (misma regla, misma reconstrucción de la permutación)


def main():
    raw = {d["id"]: d for d in map(json.loads, open(S.MMXU_TEST_JSONL))}
    L = ["# Giro de respuesta bajo swap — VLM zero-shot, MMXU-test\n",
         "acc(invertida) − acc(original) bajo swap_permuted, regla inequívoca, en pp. Bootstrap por "
         "paciente, N=5000, semilla 0, IC95%.\n",
         "| modelo | clase | n | acc(original) | acc(invertida) | giro |", "|---|---|---|---|---|---|"]
    for name, d in S.MODELS.items():
        sp = {int(r["qid"]): r for r in csv.DictReader(open(d / "oracle_swap_permuted.csv"))}
        qids = sorted(sp)
        seeds = np.random.SeedSequence(0).spawn(len(qids))
        inv_letter = {}
        for i, q in enumerate(qids):
            p, e = list(np.random.default_rng(seeds[i]).permutation(S.LETTERS)), raw[q]
            opts = {S.LETTERS[k]: e["options"][p[k]] for k in range(4)}
            ans = next(l for k, l in enumerate(S.LETTERS) if p[k] == e["answer"])
            assert ans == sp[q]["answer_permuted"]
            inv_letter[q] = S.unambiguous_inverted(opts, ans, e["content_type"])
        for ct in ("worsen", "improved", None):
            qs = [q for q in qids if inv_letter[q] and (ct is None or raw[q]["content_type"] == ct)]
            inv = np.array([sp[q]["pred_letter"] == inv_letter[q] for q in qs], float)
            orig = np.array([sp[q]["correct_vs_original"] == "True" for q in qs], float)
            pats = np.array([raw[q]["cur_image_path"].split("/")[2] for q in qs])
            L.append(f"| {name} | {ct or 'worsen+improved'} | {len(qs)} | {orig.mean():.3f} | "
                     f"{inv.mean():.3f} | {S.boot(inv - orig, pats)[0]} |")
    text = "\n".join(L) + "\n"
    (Path(__file__).resolve().parent / "swap_turn_zeroshot_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
