#!/usr/bin/env python3
"""Los tres criterios pre-registrados sobre el CONJUNTO LIMPIO (PREREGISTRO_SPLIT_LIMPIO_2026-09-24).

Mismos criterios y mismo bootstrap que tac_mcq_train/criteria_report.py (duplicado, no
importado, por convencion del proyecto); solo cambia la poblacion y el mapa qid -> paciente.
Lectura principal: conjunto limpio completo. Desgloses (secundarios): por origen dev/test y por
content_type.

Uso: python3 clean_eval/criteria_report_clean.py --dir tac_mcq_train/runs/X
  (espera clean_full.csv, clean_blind.csv, clean_mismatch.csv en --dir)
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
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
N_BOOT = 5000
SEED = 0
DELTA = 0.05
REGION_FLOOR = 0.395
DEV_QID_OFFSET = 1_000_000


def patient_map():
    pat, src = {}, {}
    for source, fname in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                q = d["id"] + (DEV_QID_OFFSET if source == "dev" else 0)
                pat[q] = d["cur_image_path"].split("/")[2]
                src[q] = source
    return pat, src


def boot_ci(arr, cluster_ids, seed=SEED):
    arr = np.asarray(arr, dtype=float)
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    sums = np.zeros(len(uniq))
    counts = np.zeros(len(uniq))
    np.add.at(sums, inv, arr)
    np.add.at(counts, inv, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    lo90, hi90 = np.percentile(means, [5, 95])
    return arr.mean(), lo, hi, lo90, hi90


def load_csv(path):
    return {int(r["qid"]): r for r in csv.DictReader(open(path))}


def criteria(qs, full, blind, mis, pat):
    pats = np.array([pat[q] for q in qs])
    cf = np.array([full[q]["correct"] == "True" for q in qs], float)
    cb = np.array([blind[q]["correct"] == "True" for q in qs], float)
    cm = np.array([mis[q]["correct"] == "True" for q in qs], float)
    acc = boot_ci(cf, pats)
    gate = boot_ci(cb - cf, pats)
    mm = boot_ci(cm - cf, pats)
    return {
        "n": len(qs), "n_pat": len(set(pats)), "acc": acc, "acc_blind": cb.mean(),
        "gate": gate, "mismatch": mm,
        "c1": acc[1] > REGION_FLOOR, "c2": gate[2] < 0, "c3": mm[2] < 0 and mm[0] <= -DELTA,
        "tost": mm[3] > -DELTA and mm[4] < DELTA,
    }


def fmt(c):
    a, g, m = c["acc"], c["gate"], c["mismatch"]
    return (f"| {c['n']} / {c['n_pat']} | {a[0]:.4f} [{a[1]:.4f},{a[2]:.4f}] | "
            f"{c['acc_blind']:.4f} | {g[0]*100:+.2f} [{g[1]*100:+.2f},{g[2]*100:+.2f}] | "
            f"{m[0]*100:+.2f} [{m[1]*100:+.2f},{m[2]*100:+.2f}] "
            f"(IC90 [{m[3]*100:+.2f},{m[4]*100:+.2f}]) | "
            f"{'SI' if c['c1'] else 'NO'}/{'SI' if c['c2'] else 'NO'}/{'SI' if c['c3'] else 'NO'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    pat, src = patient_map()
    full = load_csv(d / "clean_full.csv")
    blind = load_csv(d / "clean_blind.csv")
    mis = load_csv(d / "clean_mismatch.csv")
    qs = sorted(set(full) & set(blind) & set(mis))
    assert len(qs) == len(full) == len(blind) == len(mis), "condiciones con filas distintas"

    head = ("| n filas / pacientes | accuracy(full) | accuracy(blind) | blind−full (pp) | "
            "mismatch−full (pp) | criterios 1/2/3 |\n|---|---|---|---|---|---|")
    main_c = criteria(qs, full, blind, mis, pat)
    lines = [f"# Criterios pre-registrados — conjunto limpio — {d}\n",
             f"Bootstrap por paciente, N={N_BOOT}, semilla {SEED}, IC95%; TOST ±5pp con IC90%. "
             f"Criterios: (1) IC95% de accuracy > {REGION_FLOOR}; (2) blind−full con IC95% < 0; "
             f"(3) mismatch−full ≤ −5pp con IC95% < 0.\n",
             "## Lectura principal: conjunto limpio completo\n", head, fmt(main_c), ""]
    n_ok = sum([main_c["c1"], main_c["c2"], main_c["c3"]])
    lines.append(f"**{n_ok}/3 criterios cumplidos.**")
    if main_c["tost"]:
        lines.append("(mismatch−full: IC90% dentro de ±5pp — equivalencia TOST con full)")
    lines += ["", "## Desgloses (secundarios, no pre-registrados como criterio)\n",
              "| subconjunto " + head[1:].split("\n")[0], "|---" + head.split("\n")[1]]
    for name, sel in [("origen = dev", lambda q: src[q] == "dev"),
                      ("origen = test", lambda q: src[q] == "test"),
                      ("worsen", lambda q: full[q]["content_type"] == "worsen"),
                      ("improved", lambda q: full[q]["content_type"] == "improved"),
                      ("no change", lambda q: full[q]["content_type"] == "no change")]:
        sub = [q for q in qs if sel(q)]
        lines.append(f"| {name} " + fmt(criteria(sub, full, blind, mis, pat)))

    text = "\n".join(lines) + "\n"
    (d / "criteria_report_clean.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
