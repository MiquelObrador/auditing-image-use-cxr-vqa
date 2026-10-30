#!/usr/bin/env python3
"""Construye los splits de MMXU-dev, disjuntos por paciente.

Filtra a las muestras usables (esquema ABCD + ambas imagenes presentes en disco), separa el
10% de los PACIENTES como validacion, y fija una submuestra de 2.000 filas (`val_fast`) para
evaluar durante el entrenamiento y para medir la linea base sin entrenar.

La disjuncion con MMXU-test ya esta verificada (0 pacientes, 0 imagenes, 0 pares en comun);
aqui se vuelve a comprobar como asercion, porque es barato y es el supuesto que sostiene todo.

Salida: splits.json  {train: [id...], val: [id...], val_fast: [id...]}

Uso: python3 build_split.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MIMIC_JPG_ROOT, MMXU_DEV_JSONL, MMXU_TEST_JSONL  # noqa: E402
IMG_ROOT = MIMIC_JPG_ROOT
SEED = 0
VAL_PATIENT_FRAC = 0.10
N_VAL_FAST = 2000


def patient_of(path: str) -> str:
    """files/pXX/pNNNNNNNN/sNNNNNNNN/<dicom>.jpg -> pNNNNNNNN. El segmento 2, no el 1: "pXX" es el
    cubo de 2 digitos de MIMIC-CXR, no el paciente. (Convencion ya verificada en el repo.)"""
    return path.split("/")[2]


def load_usable():
    usable, seen_ids = [], set()
    n_total = n_schema = n_missing = 0
    exists_cache = {}

    def exists(rel):
        if rel not in exists_cache:
            exists_cache[rel] = (IMG_ROOT / rel).exists()
        return exists_cache[rel]

    with open(MMXU_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            n_total += 1
            if tuple(sorted(d["options"].keys())) != ("A", "B", "C", "D"):
                n_schema += 1
                continue
            if not (exists(d["cur_image_path"]) and exists(d["prior_image_path"])):
                n_missing += 1
                continue
            assert d["id"] not in seen_ids, f"id duplicado en MMXU-dev: {d['id']}"
            seen_ids.add(d["id"])
            pc, pp = patient_of(d["cur_image_path"]), patient_of(d["prior_image_path"])
            assert pc == pp, f"paciente distinto entre cur/prior en id={d['id']}: {pc} vs {pp}"
            usable.append((d["id"], pc))
    return usable, dict(total=n_total, schema=n_schema, missing=n_missing)


def test_patients():
    pats = set()
    with open(MMXU_TEST_JSONL) as f:
        for line in f:
            pats.add(patient_of(json.loads(line)["cur_image_path"]))
    return pats


def main():
    usable, stats = load_usable()
    patients = sorted({p for _, p in usable})
    print(f"MMXU-dev: {stats['total']} filas")
    print(f"  descartadas por esquema de opciones : {stats['schema']}")
    print(f"  descartadas por imagen ausente      : {stats['missing']}")
    print(f"  USABLES: {len(usable)} ({100*len(usable)/stats['total']:.1f}%) "
          f"en {len(patients)} pacientes")

    overlap = set(patients) & test_patients()
    assert not overlap, f"SOLAPAMIENTO con MMXU-test en {len(overlap)} pacientes: {sorted(overlap)[:5]}"
    print(f"  disjuncion con MMXU-test: OK (0 pacientes en comun)")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(patients))
    n_val_pat = int(round(VAL_PATIENT_FRAC * len(patients)))
    val_pats = {patients[i] for i in perm[:n_val_pat]}

    train_ids = [i for i, p in usable if p not in val_pats]
    val_ids = [i for i, p in usable if p in val_pats]
    assert not (set(train_ids) & set(val_ids))

    rng_fast = np.random.default_rng(SEED)
    idx = rng_fast.choice(len(val_ids), size=min(N_VAL_FAST, len(val_ids)), replace=False)
    val_fast_ids = sorted(val_ids[i] for i in idx)

    print(f"\nsplits (disjuntos por paciente, semilla {SEED}):")
    print(f"  train    : {len(train_ids):7d} filas | {len(patients)-n_val_pat:6d} pacientes")
    print(f"  val      : {len(val_ids):7d} filas | {n_val_pat:6d} pacientes")
    print(f"  val_fast : {len(val_fast_ids):7d} filas (submuestra fija de val)")

    out = {"seed": SEED, "val_patient_frac": VAL_PATIENT_FRAC,
           "train": sorted(train_ids), "val": sorted(val_ids), "val_fast": val_fast_ids}
    (HERE / "splits.json").write_text(json.dumps(out))
    print(f"\nescrito: {HERE/'splits.json'}")


if __name__ == "__main__":
    main()
