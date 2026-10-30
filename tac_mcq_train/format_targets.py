#!/usr/bin/env python3
"""Mezcla de formatos de entrenamiento y limpieza de justificaciones.

Anticolapso en dos direcciones opuestas: que el modelo solo sepa emitir "A", y que recaiga en
el modo informe que bloqueaba a Libra. Ver PREREGISTRO_2026-09-21.md secciones 7 y 7.1.

  A (65%)  letra sola            objetivo: "A"
  B (20%)  texto de la opcion    objetivo: "Improved"
  C (15%)  justificacion + letra objetivo: "<just. <=40 palabras> The answer is A."

Las tres llevan SIEMPRE las dos imagenes: ningun ejemplo de entrenamiento carece de imagen.
Las tres difieren de la plantilla de evaluacion UNICAMENTE en la ultima linea (la instruccion),
para que el formato A sea byte a byte el prompt del protocolo del ISBI.

Ejecutado como script, valida la limpieza imprimiendo ejemplos y las proporciones reales.

Uso: python3 format_targets.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import MMXU_DEV_JSONL, SPLITS_FILE as _SPLITS_FILE  # noqa: E402
# Split unico para todo entrenamiento desde 2026-09-24: splits.json sin los pacientes del split
# oficial validate/test de MIMIC-CXR (clean_eval/). Las carreras anteriores usaron splits.json.
SPLITS_FILE = _SPLITS_FILE
sys.path.insert(0, str(REPO / "mmxu_eval"))
from prompts import build_prompt, TEMPLATE  # noqa: E402

SEED = 0
# La ultima linea de mmxu_eval/prompts.py::TEMPLATE. Se localiza, no se copia a mano, para que
# si la plantilla del ISBI cambiara esto reventase en vez de divergir en silencio.
INSTR_A = TEMPLATE.rsplit("\n", 1)[-1]
assert INSTR_A == "Answer with the option's letter from the given choices directly.", INSTR_A
INSTR_B = "Answer with the text of the correct option."
INSTR_C = "Briefly justify your answer, then give the option's letter."

FORMAT_P = {"A": 0.65, "B": 0.20, "C": 0.15}
# Solo el 68.9% de las filas de train tienen un `cur_region_report` que sobreviva a la
# limpieza (medido; _validate lo recomprueba y falla si se mueve). Si sorteasemos C al 15%
# plano, las que no tienen justificacion caerian a A y el C REALIZADO seria 0.689*0.15 = 10.3%,
# por debajo de lo pre-registrado. Se condiciona el sorteo para que el marginal sea el 15%.
JUST_COVERAGE = 0.689
P_C_GIVEN_JUST = min(1.0, FORMAT_P["C"] / JUST_COVERAGE)

MAX_JUST_WORDS = 40
MIN_JUST_WORDS = 4
MAX_JUST_SENTENCES = 2
# Frases que delatan que el texto viene de un informe que el modelo NO puede ver. Entrenar
# sobre ellas ensenaria a citar una fuente inexistente.
META_RE = re.compile(
    r"related explanatory information|key comparison sentence|\breports?\b|according to|"
    r"\bmention|provided information|\bthe text\b|explanatory information|\bsentence\b",
    re.I)
SECTION_RE = re.compile(r"\b(FINDINGS|IMPRESSION|COMPARISON|INDICATION|TECHNIQUE|EXAMINATION)\s*:\s*",
                        re.I)
# Numeracion residual de la seccion IMPRESSION ("2. Mild pulmonary edema...").
LISTNUM_RE = re.compile(r"^\s*\d+\s*[.)]\s*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Una justificacion para una pregunta TEMPORAL que no menciona ningun cambio no sirve.
CHANGE_RE = re.compile(
    r"worse|improv|unchanged|no change|stable|\bnew\b|resolv|increas|decreas|persist|"
    r"redemonstrat|similar|interval|prior|previous|clear|recurr|develop|progress|reduc",
    re.I)


def clean_justification(raw):
    """cur_region_report -> justificacion utilizable, o None si no lo es.

    Corta SIEMPRE por frase completa: `cur_region_report` concatena trozos de FINDINGS y de
    IMPRESSION, asi que truncar por numero de palabras producia fragmentos agramaticales
    ("Increased opacity at.") y repeticiones ("X remains enlarged. X remains enlarged, ...").
    """
    if not isinstance(raw, str):
        return None
    t = SECTION_RE.sub("", raw).strip()
    if not t or "___" in t:          # marcador de anonimizacion de MIMIC
        return None
    if META_RE.search(t):
        return None

    kept, seen, n_words = [], set(), 0
    for sent in SENT_SPLIT_RE.split(re.sub(r"\s+", " ", t)):
        sent = LISTNUM_RE.sub("", sent).strip()
        if not sent.endswith((".", "!", "?")):
            continue                 # frase incompleta (cola truncada del campo): se descarta
        if len(sent.split()) < MIN_JUST_WORDS:
            continue
        key = re.sub(r"[^a-z0-9 ]", "", sent.lower()).strip()
        # dedup por prefijo: "X remains enlarged." vs "X remains enlarged, but is stable."
        if any(key.startswith(k[:40]) or k.startswith(key[:40]) for k in seen):
            continue
        if n_words + len(sent.split()) > MAX_JUST_WORDS:
            break
        kept.append(sent)
        seen.add(key)
        n_words += len(sent.split())
        if len(kept) >= MAX_JUST_SENTENCES:
            break

    if not kept:
        return None
    out = " ".join(kept)
    return out if CHANGE_RE.search(out) else None


def rng_for(qid, epoch):
    """Determinista por (muestra, epoca): reproducible, pero la permutacion y el formato
    cambian entre epocas -- aumentacion real sin inflar el dataset."""
    return np.random.default_rng(np.random.SeedSequence([SEED, int(qid), int(epoch)]))


def permute_options(entry, rng):
    """Devuelve (options_permutadas, letra_correcta_nueva). El dataset YA esta equilibrado en
    letras (25/25/25/25, tambien dentro de cada content_type), asi que esto no corrige ningun
    sesgo del dataset: impide que el modelo aprenda un atajo posicional."""
    letters = ["A", "B", "C", "D"]
    texts = [entry["options"][L] for L in letters]
    order = rng.permutation(4)
    new_opts = {letters[i]: texts[order[i]] for i in range(4)}
    old_idx = letters.index(entry["answer"])
    new_answer = letters[int(np.where(order == old_idx)[0][0])]
    assert new_opts[new_answer] == entry["options"][entry["answer"]]
    return new_opts, new_answer


def swap_instruction(prompt, new_instr):
    """Sustituye SOLO la ultima linea del prompt (la instruccion)."""
    head, _, tail = prompt.rpartition("\n")
    assert tail == INSTR_A, f"la ultima linea no es la instruccion esperada: {tail!r}"
    return head + "\n" + new_instr


def make_example(entry, epoch):
    """entry: fila cruda de MMXU-dev.jsonl. Devuelve (prompt_text, target, formato)."""
    rng = rng_for(entry["id"], epoch)
    opts, answer = permute_options(entry, rng)
    permuted = {**entry, "options": opts, "answer": answer}
    prompt_a = build_prompt(permuted)

    just = clean_justification(entry.get("cur_region_report"))
    p_c = P_C_GIVEN_JUST if just is not None else 0.0
    p_b = FORMAT_P["B"]
    fmt = rng.choice(["A", "B", "C"], p=[1.0 - p_b - p_c, p_b, p_c])

    if fmt == "C":
        return swap_instruction(prompt_a, INSTR_C), f"{just} The answer is {answer}.", "C"
    if fmt == "B":
        return swap_instruction(prompt_a, INSTR_B), opts[answer], "B"
    return prompt_a, answer, "A"


def _validate():
    splits = json.loads(SPLITS_FILE.read_text())
    train = set(splits["train"])
    rows = []
    with open(MMXU_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in train:
                rows.append(d)
    print(f"filas de train: {len(rows)}\n")

    ok = sum(1 for d in rows if clean_justification(d.get("cur_region_report")) is not None)
    cov = ok / len(rows)
    print(f"=== LIMPIEZA DE JUSTIFICACIONES ===")
    print(f"  cur_region_report utilizable: {ok}/{len(rows)} ({100*cov:.1f}%)")
    assert abs(cov - JUST_COVERAGE) < 0.01, (
        f"la cobertura real ({cov:.3f}) se ha movido respecto a JUST_COVERAGE "
        f"({JUST_COVERAGE}); actualizala o el formato C se desviara de lo pre-registrado")

    from collections import Counter
    cnt, tlen = Counter(), Counter()
    ex = {"A": [], "B": [], "C": []}
    for d in rows:
        p, t, f_ = make_example(d, epoch=0)
        cnt[f_] += 1
        tlen[f_] += len(t.split())
        if len(ex[f_]) < 3:
            ex[f_].append((p, t))
    print(f"\n=== PROPORCIONES REALES (epoca 0) ===")
    for f_ in "ABC":
        print(f"  {f_}: {cnt[f_]:7d} ({100*cnt[f_]/len(rows):5.1f}%)  "
              f"objetivo medio {tlen[f_]/max(1,cnt[f_]):5.1f} palabras")

    print(f"\n=== 30 JUSTIFICACIONES LIMPIAS (validacion a mano, exigida por el preregistro) ===")
    shown = 0
    for d in rows:
        j = clean_justification(d.get("cur_region_report"))
        if j:
            print(f"  [{d['content_type']:9s}] {j}")
            shown += 1
            if shown >= 30:
                break

    print(f"\n=== UN EJEMPLO COMPLETO DE CADA FORMATO ===")
    for f_ in "ABC":
        p, t = ex[f_][0]
        print(f"\n----- FORMATO {f_} -----")
        print(p)
        print(f"  >>> OBJETIVO: {t!r}")

    print(f"\n=== la permutacion varia entre epocas (misma muestra, 3 epocas) ===")
    d = rows[0]
    for e in range(3):
        p, t, f_ = make_example(d, epoch=e)
        opts = re.search(r"Options:\n(.+?)\n\n", p, re.S).group(1).replace("\n", " | ")
        print(f"  epoca {e} [{f_}] {opts}  -> {t[:60]!r}")


if __name__ == "__main__":
    _validate()
