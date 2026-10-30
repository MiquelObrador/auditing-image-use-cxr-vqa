#!/usr/bin/env python3
"""Los tres criterios pre-registrados (PREREGISTRO_2026-09-21.md #2), sobre MMXU-test.

  1. Accuracy (full)          claramente por encima de 0,395 (suelo de nombre de region)
  2. Gate de imagen           blind - full negativo, IC95% que excluye cero
  3. Uso de la previa         mismatch - full <= -5pp, IC95% que excluye cero

Bootstrap agrupado por paciente, N=5000, semilla 0 -- duplicado (no importado) desde
paper/material/tables/make_tables.py, por convencion explicita del proyecto.

Uso: python3 criteria_report.py --dir runs/full
  (espera runs/full/oracle_full.csv, oracle_blind.csv, oracle_mismatch.csv)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL  # noqa: E402
N_BOOT = 5000
SEED = 0
DELTA = 0.05  # margen TOST, 5pp
REGION_FLOOR = 0.395
CHANCE = 0.250


def load_patient_map():
    patient_of = {}
    with open(MMXU_TEST_JSONL) as f:
        for line in f:
            d = json.loads(line)
            patient_of[d["id"]] = d["cur_image_path"].split("/")[2]
    return patient_of


def boot_ci(arr, cluster_ids, seed=SEED):
    arr = np.asarray(arr, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(uniq)
    sums = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    np.add.at(sums, inv, arr)
    np.add.at(counts, inv, 1.0)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    means = sums[boot_idx].sum(axis=1) / counts[boot_idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    lo90, hi90 = np.percentile(means, [5, 95])
    return arr.mean(), lo, hi, lo90, hi90


def paired_diff_ci(a, b, cluster_ids, seed=SEED):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return boot_ci(diff, cluster_ids, seed)


def load_csv(path):
    return {int(r["qid"]): r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    patient_of = load_patient_map()
    full = load_csv(d / "oracle_full.csv")
    blind = load_csv(d / "oracle_blind.csv")
    mismatch = load_csv(d / "oracle_mismatch.csv")

    common = sorted(set(full) & set(blind) & set(mismatch))
    assert len(common) == len(full), (
        f"filas no coincidentes entre condiciones: full={len(full)} comun={len(common)}")
    patients = np.array([patient_of[q] for q in common])
    n_patients = len(set(patients))
    print(f"MMXU-test: {len(common)} filas, {n_patients} pacientes\n")

    correct_full = np.array([full[q]["correct"] == "True" for q in common])
    correct_blind = np.array([blind[q]["correct"] == "True" for q in common])
    correct_mismatch = np.array([mismatch[q]["correct"] == "True" for q in common])

    lines = ["# Criterios pre-registrados sobre MMXU-test\n",
             f"n={len(common)} filas, {n_patients} pacientes. Bootstrap por paciente, "
             f"N={N_BOOT}, semilla {SEED}. IC95% (percentil); TOST con margen ±5pp, IC90%.\n"]

    # --- Criterio 1: accuracy ---
    acc, lo, hi, _, _ = boot_ci(correct_full, patients)
    c1 = acc > REGION_FLOOR and lo > REGION_FLOOR
    lines.append(f"## Criterio 1 -- Accuracy vs. suelo de nombre de region ({REGION_FLOOR})\n")
    lines.append(f"accuracy(full) = {acc:.4f} [{lo:.4f}, {hi:.4f}]  "
                 f"(azar={CHANCE}, suelo={REGION_FLOOR})")
    lines.append(f"**{'CUMPLE' if c1 else 'NO CUMPLE'}** "
                 f"(criterio: IC95% enteramente por encima de {REGION_FLOOR})\n")

    # --- Criterio 2: gate de imagen ---
    d_blind, lo_b, hi_b, _, _ = paired_diff_ci(correct_blind, correct_full, patients)
    c2 = hi_b < 0
    lines.append(f"## Criterio 2 -- Gate de imagen (blind - full)\n")
    lines.append(f"blind - full = {d_blind:+.4f} [{lo_b:+.4f}, {hi_b:+.4f}]")
    lines.append(f"accuracy(blind) = {correct_blind.mean():.4f}")
    lines.append(f"**{'CUMPLE' if c2 else 'NO CUMPLE'}** "
                 f"(criterio: negativo, IC95% excluye cero)\n")

    # --- Criterio 3: uso de la previa ---
    d_mis, lo_m, hi_m, lo90_m, hi90_m = paired_diff_ci(correct_mismatch, correct_full, patients)
    c3 = hi_m < 0 and d_mis <= -DELTA
    tost_equiv = lo90_m > -DELTA and hi90_m < DELTA
    lines.append(f"## Criterio 3 -- Uso de la previa (mismatch - full)\n")
    lines.append(f"mismatch - full = {d_mis:+.4f} [{lo_m:+.4f}, {hi_m:+.4f}] (IC95%), "
                 f"[{lo90_m:+.4f}, {hi90_m:+.4f}] (IC90%, TOST)")
    lines.append(f"accuracy(mismatch) = {correct_mismatch.mean():.4f}")
    lines.append(f"**{'CUMPLE' if c3 else 'NO CUMPLE'}** "
                 f"(criterio: <= -{DELTA*100:.0f}pp, IC95% excluye cero)")
    if tost_equiv:
        lines.append(f"(nota: IC90% enteramente dentro de ±{DELTA*100:.0f}pp -- "
                     f"equivalencia TOST con full, en vez de diferencia)")
    lines.append("")

    lines.append("## Veredicto\n")
    n_ok = sum([c1, c2, c3])
    lines.append(f"{n_ok}/3 criterios cumplidos.")
    lines.append(f"  1. Accuracy > suelo de region : {'SI' if c1 else 'NO'}")
    lines.append(f"  2. Gate de imagen             : {'SI' if c2 else 'NO'}")
    lines.append(f"  3. Uso de la previa (mismatch): {'SI' if c3 else 'NO'}")

    text = "\n".join(lines) + "\n"
    (d / "criteria_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
