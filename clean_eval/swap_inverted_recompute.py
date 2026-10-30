#!/usr/bin/env python3
"""Recalcula el swap con letras barajadas de los VLM zero-shot del ISBI (HuatuoGPT-7B,
Qwen2.5-VL-7B) sobre las predicciones ya guardadas, con la regla antigua y con la regla
inequívoca de EVAL_SETTINGS_2026-09-28.md §5, lado a lado y por clase.

La permutación por fila se reconstruye igual que mmxu_eval/swap_permuted_check.py
(SeedSequence(0).spawn(n) en el orden de load_oracle_samples) y se valida contra las columnas
answer_permuted / answer_inverted_permuted guardadas. Diferencias pareadas contra full (letras
originales), igual que mmxu_eval/analyze_swap_permuted.py. Bootstrap por paciente, N=5000, semilla 0.

Uso: python3 clean_eval/swap_inverted_recompute.py
"""
import csv, json, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL  # noqa: E402
HERE = Path(__file__).resolve().parent
N_BOOT, SEED = 5000, 0
LETTERS = ["A", "B", "C", "D"]
MODELS = {"HuatuoGPT-7B": REPO / "huatuo_eval/oracle_results", "Qwen2.5-VL-7B": REPO / "qwen25vl_eval/oracle_results"}
PAPER = {"HuatuoGPT-7B": (-0.0856, -0.0464), "Qwen2.5-VL-7B": (-0.0572, -0.0830)}

src = (REPO / "mmxu_eval/oracle_evaluate.py").read_text()
ns = {}
exec(compile("import re\n" + src[src.index("IMPROVED_RE ="):src.index("def build_qid_lexicon")], "lex", "exec"), ns)
label_option, INVERT = ns["label_option"], ns["INVERT_CLASS"]
NEG = re.compile(r"\b(no|not|without|none|neither|nor)\b", re.I)


def unambiguous_inverted(opts, answer, ct):
    if ct not in ("worsen", "improved"):
        return None
    labs = {L: label_option(t) for L, t in opts.items()}
    if labs[answer] != ct:
        return None
    m = [L for L, l in labs.items() if l == INVERT[ct]]
    if len(m) != 1 or NEG.search(opts[answer]) or NEG.search(opts[m[0]]):
        return None
    return m[0]


def boot(arr, pats):
    arr = np.asarray(arr, float)
    u, inv = np.unique(pats, return_inverse=True)
    s, c = np.zeros(len(u)), np.zeros(len(u))
    np.add.at(s, inv, arr); np.add.at(c, inv, 1.0)
    idx = np.random.default_rng(SEED).integers(0, len(u), (N_BOOT, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return f"{arr.mean()*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]", arr.mean()


def main():
    raw = {d["id"]: d for d in map(json.loads, open(MMXU_TEST_JSONL))}
    L = ["# Swap con letras barajadas (ISBI): regla antigua vs. regla inequívoca\n",
         "Predicciones ya guardadas (`oracle_swap_permuted.csv`), sin GPU. Diferencias pareadas "
         "contra `full` (letras originales), como en `mmxu_eval/analyze_swap_permuted.py`. "
         "Bootstrap por paciente, N=5000, semilla 0, IC95%, en pp.\n"]
    for name, d in MODELS.items():
        full = {int(r["qid"]): r for r in csv.DictReader(open(d / "oracle_full.csv"))}
        sp = {int(r["qid"]): r for r in csv.DictReader(open(d / "oracle_swap_permuted.csv"))}
        qids = sorted(sp)
        seeds = np.random.SeedSequence(0).spawn(len(qids))
        perms = {q: list(np.random.default_rng(seeds[i]).permutation(LETTERS)) for i, q in enumerate(qids)}
        new_inv, mism = {}, 0
        for q in qids:
            e, p = raw[q], perms[q]
            opts = {LETTERS[i]: e["options"][p[i]] for i in range(4)}
            ans = next(l for i, l in enumerate(LETTERS) if p[i] == e["answer"])
            mism += ans != sp[q]["answer_permuted"]
            new_inv[q] = unambiguous_inverted(opts, ans, e["content_type"])
        assert mism == 0, f"{name}: la permutación reconstruida no coincide en {mism} filas"
        pat = lambda qs: np.array([raw[q]["cur_image_path"].split("/")[2] for q in qs])
        fc = lambda qs: np.array([full[q]["correct"] == "True" for q in qs], float)
        # vs original: todas las filas
        s_o, v_o = boot(np.array([sp[q]["correct_vs_original"] == "True" for q in qids], float) - fc(qids), pat(qids))
        L += [f"## {name}\n", f"Permutación reconstruida: coincide en las {len(qids)} filas.\n",
              f"**vs. original − full** (n={len(qids)}, no depende de la regla): {s_o}  "
              f"(paper: {PAPER[name][0]*100:+.2f}; {'igual' if abs(v_o - PAPER[name][0]) < 5e-5 else 'DISTINTO'})\n",
              "| subconjunto | regla | n | vs. original − full | vs. invertida − full |", "|---|---|---|---|---|"]
        old_q = [q for q in qids if sp[q]["answer_inverted_permuted"]]
        rows = [("todas (como en el paper)", "antigua", old_q)]
        for ct in ("worsen", "improved", "no change"):
            rows.append((ct, "antigua", [q for q in old_q if raw[q]["content_type"] == ct]))
        rows.append(("worsen+improved", "antigua", [q for q in old_q if raw[q]["content_type"] != "no change"]))
        new_q = [q for q in qids if new_inv[q]]
        rows.append(("worsen+improved", "inequívoca", new_q))
        for ct in ("worsen", "improved"):
            rows.append((ct, "inequívoca", [q for q in new_q if raw[q]["content_type"] == ct]))
        for lab, rule, qs in rows:
            if rule == "antigua":
                inv = np.array([sp[q]["correct_vs_inverted"] == "True" for q in qs], float)
            else:
                inv = np.array([sp[q]["pred_letter"] == new_inv[q] for q in qs], float)
            orig = np.array([sp[q]["correct_vs_original"] == "True" for q in qs], float)
            s_i, v_i = boot(inv - fc(qs), pat(qs))
            s_oo, _ = boot(orig - fc(qs), pat(qs))
            tag = f" (paper: {PAPER[name][1]*100:+.2f}; {'igual' if abs(v_i - PAPER[name][1]) < 5e-5 else 'DISTINTO'})" if lab.startswith("todas") else ""
            L.append(f"| {lab} | {rule} | {len(qs)} | {s_oo} | {s_i}{tag} |")
        L.append("")
    text = "\n".join(L) + "\n"
    (HERE / "swap_inverted_recompute_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
