#!/usr/bin/env python3
"""¿Cuanto añade la pregunta sobre las opciones solas? Mismo lineal por opcion que
options_only_linear.py, variante (b), frente a (c) = (b) + rasgos que cruzan pregunta y opcion.

Los rasgos de la pregunta sola son identicos en las cuatro opciones de una fila y se cancelan
en el argmax, asi que la pregunta solo puede aportar a traves de interacciones:
  * pares (palabra de la pregunta, palabra de la opcion), sin stopwords, con hashing (2^20);
  * pares (palabra de la pregunta, clase lexica de la opcion);
  * solape de palabras pregunta-opcion (one-hot 0..5+).
Hiperparametros fijos (C=1), entrenado en el train limpio, evaluado en las 2.992 filas de
MMXU-test. Diferencia (c)-(b) pareada por fila, bootstrap por paciente N=5000 semilla 0.

Uso: python3 shortcut_analysis/question_options_linear.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from textual_shortcut import label_option, words, boot_ci, fmt, fmt_pp, CLASSES, REPO, MMXU_TEST_JSONL, MMXU_DEV_JSONL, EVAL_QIDS
from options_only_linear import load, expand, SPLITS_FILE, LETTERS

HERE = Path(__file__).resolve().parent


def cross_tokens(data):
    toks, overlap = [], []
    for d in data:
        qw = sorted(words(d["question"]))
        for L in LETTERS:
            ow = words(d["options"][L])
            cls = label_option(d["options"][L])
            toks.append(" ".join([f"{q}|{o}" for q in qw for o in sorted(ow)] +
                                 [f"{q}|CLS={cls}" for q in qw]))
            f = np.zeros(6)
            f[min(len(ow & set(qw)), 5)] = 1
            overlap.append(f)
    return toks, np.array(overlap)


def main():
    train_ids = set(json.loads(SPLITS_FILE.read_text())["train"])
    train = load(MMXU_DEV_JSONL, train_ids)
    test_qids = {int(r["qid"]) for r in csv.DictReader(
        open(EVAL_QIDS))}
    test = sorted(load(MMXU_TEST_JSONL, test_qids), key=lambda d: d["id"])
    pats = np.array([d["cur_image_path"].split("/")[2] for d in test])
    ct = np.array([d["content_type"] for d in test])
    gold = np.array([LETTERS.index(d["answer"]) for d in test])

    tr_t, tr_s, tr_y = expand(train)
    te_t, te_s, _ = expand(test)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xb_tr = hstack([vec.fit_transform(tr_t), csr_matrix(tr_s)]).tocsr()
    Xb_te = hstack([vec.transform(te_t), csr_matrix(te_s)]).tocsr()

    hv = HashingVectorizer(n_features=2**20, token_pattern=r"\S+", lowercase=False,
                           alternate_sign=False, norm="l2")
    tr_x, tr_o = cross_tokens(train)
    te_x, te_o = cross_tokens(test)
    Xc_tr = hstack([Xb_tr, hv.transform(tr_x), csr_matrix(tr_o)]).tocsr()
    Xc_te = hstack([Xb_te, hv.transform(te_x), csr_matrix(te_o)]).tocsr()

    ok = {}
    for name, Xtr, Xte in [("(b) solo opciones (+ clase, n con su clase, rango de longitud)", Xb_tr, Xb_te),
                           ("(c) (b) + pregunta×opción", Xc_tr, Xc_te)]:
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, tr_y)
        ok[name] = (clf.decision_function(Xte).reshape(-1, 4).argmax(1) == gold).astype(float)

    (nb, b), (nc, c) = ok.items()
    lines = [f"# ¿Cuánto añade la pregunta? Lineal por opción — MMXU-test ({len(test)} filas, "
             f"{len(set(pats))} pacientes)\n",
             f"Entrenado en {len(train)} preguntas del train limpio; C=1 fijo. "
             "Bootstrap por paciente, N=5000, semilla 0.\n",
             "| variante | accuracy | " + " | ".join(CLASSES) + " |", "|---|---|---|---|---|"]
    for name, x in ok.items():
        lines.append(f"| {name} | {fmt(boot_ci(x, pats))} | " +
                     " | ".join(fmt(boot_ci(x[ct == k], pats[ct == k])) for k in CLASSES) + " |")
    lines += ["", f"**Aporte de la pregunta, (c) − (b), pareado:** {fmt_pp(boot_ci(c - b, pats))} pp",
              "Por clase: " + ", ".join(f"{k} {fmt_pp(boot_ci((c-b)[ct == k], pats[ct == k]))}"
                                        for k in CLASSES),
              f"Filas que (c) acierta y (b) no: {int(((c == 1) & (b == 0)).sum())}; "
              f"al revés: {int(((c == 0) & (b == 1)).sum())}.", ""]
    text = "\n".join(lines) + "\n"
    (HERE / "question_options_linear_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
