#!/usr/bin/env python3
"""¿Se deduce la respuesta de las opciones solas? Modelo lineal (CPU) que ve SOLO las cuatro
opciones, sin pregunta ni imagenes. Paso previo, barato, al modelo de lenguaje solo-opciones.

Cada opcion se puntua por separado con una regresion logistica binaria (es la correcta / no lo
es); se predice la opcion de mayor puntuacion. Dos variantes, hiperparametros fijos (C=1, sin
ajuste: no se mira el test para elegir nada):
  (a) texto de la opcion: TF-IDF de palabras, 1-2 gramas;
  (b) (a) + rasgos de conjunto: clase lexica de la opcion, cuantas opciones comparten su clase,
      y rango de longitud entre las cuatro. Un lineal por opcion no ve "la distinta" sin esto.
Entrenamiento: train del split limpio (format_targets.SPLITS_FILE). Evaluacion: MMXU-test,
las mismas 2.992 filas del resto de analisis; bootstrap por paciente, N=5000, semilla 0.

Uso: python3 shortcut_analysis/options_only_linear.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from textual_shortcut import label_option, boot_ci, fmt, CLASSES, REPO, MMXU_TEST_JSONL, MMXU_DEV_JSONL, EVAL_QIDS

HERE = Path(__file__).resolve().parent
from config import SPLITS_FILE  # noqa: E402
LETTERS = ("A", "B", "C", "D")
LABELS = ("worsen", "improved", "no change", "NONE", "AMBIG")


def load(path, ids=None):
    out = []
    for line in open(path):
        d = json.loads(line)
        if (ids is None or d["id"] in ids) and tuple(sorted(d["options"])) == LETTERS:
            out.append(d)
    return out


def set_features(d):
    """Por opcion: one-hot de clase (5), one-hot de n opciones con su clase (4), rango de
    longitud en caracteres (4)."""
    cls = {L: label_option(d["options"][L]) for L in LETTERS}
    cnt = Counter(cls.values())
    ranks = {L: r for r, L in enumerate(sorted(LETTERS, key=lambda L: len(d["options"][L])))}
    rows = []
    for L in LETTERS:
        f = np.zeros(13)
        f[LABELS.index(cls[L])] = 1
        f[5 + cnt[cls[L]] - 1] = 1
        f[9 + ranks[L]] = 1
        rows.append(f)
    return rows


def expand(data):
    texts, setf, y = [], [], []
    for d in data:
        texts += [d["options"][L] for L in LETTERS]
        setf += set_features(d)
        y += [L == d["answer"] for L in LETTERS]
    return texts, np.array(setf), np.array(y)


def main():
    train_ids = set(json.loads(SPLITS_FILE.read_text())["train"])
    train = load(MMXU_DEV_JSONL, train_ids)
    test_qids = {int(r["qid"]) for r in csv.DictReader(
        open(EVAL_QIDS))}
    test = sorted(load(MMXU_TEST_JSONL, test_qids), key=lambda d: d["id"])
    assert len(test) == len(test_qids)
    pats = np.array([d["cur_image_path"].split("/")[2] for d in test])
    ct = np.array([d["content_type"] for d in test])
    print(f"train {len(train)} preguntas | test {len(test)} preguntas")

    tr_t, tr_s, tr_y = expand(train)
    te_t, te_s, _ = expand(test)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr_a, Xte_a = vec.fit_transform(tr_t), vec.transform(te_t)

    lines = [f"# Modelo lineal que ve solo las opciones — MMXU-test ({len(test)} filas, "
             f"{len(set(pats))} pacientes)\n",
             f"Entrenado en {len(train)} preguntas del train limpio; C=1 fijo, sin ajuste. "
             f"Bootstrap por paciente, N=5000, semilla 0.\n",
             "| variante | accuracy | " + " | ".join(CLASSES) + " |", "|---|---|---|---|---|"]
    gold = np.array([LETTERS.index(d["answer"]) for d in test])
    for name, Xtr, Xte in [
            ("(a) texto de la opción (TF-IDF 1-2 gramas)", Xtr_a, Xte_a),
            ("(b) (a) + clase léxica, n con su clase, rango de longitud",
             hstack([Xtr_a, csr_matrix(tr_s)]).tocsr(), hstack([Xte_a, csr_matrix(te_s)]).tocsr())]:
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, tr_y)
        scores = clf.decision_function(Xte).reshape(-1, 4)
        ok = (scores.argmax(1) == gold).astype(float)
        lines.append(f"| {name} | {fmt(boot_ci(ok, pats))} | " +
                     " | ".join(fmt(boot_ci(ok[ct == k], pats[ct == k])) for k in CLASSES) + " |")
        if name.startswith("(a)"):
            top = np.argsort(clf.coef_[0])
            feats = np.array(vec.get_feature_names_out())
            lines += ["", f"(a) rasgos con más peso hacia 'correcta': "
                      f"{', '.join(feats[top[-15:]][::-1])}",
                      f"(a) rasgos con más peso hacia 'incorrecta': {', '.join(feats[top[:15]])}", ""]
    text = "\n".join(lines) + "\n"
    (HERE / "options_only_linear_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
