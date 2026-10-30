"""Cruza MMXU-test.jsonl (vía data.py) con bbox_analysis/mmxu_bbox_table.csv.

Produce, por muestra, todo lo que ya da data.load_samples() más region, cells_max,
bbox_prior y bbox_cur (en píxeles, ya recortados a los límites de la imagen). Solo
las muestras presentes en mmxu_bbox_table.csv (2992/3000 -- 8 sin bbox utilizable,
ver bbox_analysis/bbox_size_report.md).
"""
import csv
from pathlib import Path
from typing import Iterator

from data import load_samples

_REPO_ROOT = Path(__file__).resolve().parents[1]
BBOX_TABLE_PATH = _REPO_ROOT / "bbox_analysis" / "mmxu_bbox_table.csv"


def _load_bbox_rows(path: Path = BBOX_TABLE_PATH) -> dict[int, dict]:
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            rows[qid] = {
                "region": row["region"],
                "cells_max": int(row["cells_max"]),
                "bbox_cur": (
                    int(row["bbox_cur_x1"]), int(row["bbox_cur_y1"]),
                    int(row["bbox_cur_x2"]), int(row["bbox_cur_y2"]),
                ),
                "bbox_prior": (
                    int(row["bbox_prior_x1"]), int(row["bbox_prior_y1"]),
                    int(row["bbox_prior_x2"]), int(row["bbox_prior_y2"]),
                ),
            }
    return rows


def load_oracle_samples() -> Iterator[dict]:
    bbox_rows = _load_bbox_rows()
    for sample in load_samples():
        bbox = bbox_rows.get(sample["qid"])
        if bbox is None:
            continue
        yield {**sample, **bbox}


if __name__ == "__main__":
    # Dry-run sin modelo: confirma el cruce y las bboxes de 1-2 muestras reales.
    n = 0
    for i, sample in enumerate(load_oracle_samples()):
        n += 1
        if i >= 2:
            continue
        print(f"--- muestra (qid={sample['qid']}, region={sample['region']}, "
              f"cells_max={sample['cells_max']}) ---")
        print(f"bbox_prior={sample['bbox_prior']}  bbox_cur={sample['bbox_cur']}")
    print(f"\ntotal muestras con bbox utilizable: {n}")
