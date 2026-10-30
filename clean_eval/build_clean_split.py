#!/usr/bin/env python3
"""Build the decontaminated evaluation set and the fine-tuning split without it.

Clean set = every MMXU-dev or MMXU-test row whose patient is in the OFFICIAL MIMIC-CXR
validate or test split. Neither Libra's temporal module nor RAD-DINO was pretrained on those
patients: both state that they use the official training split (see docs/CLEAN_SET.md).

Inputs (paths in config.py, overridable with the flags below):
  * mimic-cxr-2.0.0-split.csv.gz     official split (PhysioNet, MIMIC-CXR-JPG 2.0.0)
  * mimic-cxr-2.0.0-metadata.csv.gz  used to check that the split reproduces, image by image
                                     and study by study, Table 3 of Johnson et al. 2019
  * MMXU-dev.jsonl, MMXU-test.jsonl
  * --base-splits (optional): the train/val/val_fast split of MMXU-dev produced by
    tac_mcq_train/build_split.py; the clean patients are removed from it.

Outputs (identifiers only, never text), in --out-dir:
  clean_set_ids_<date>.csv   source, qid, usable (as the shipped list; the MIMIC-CXR
                             identifiers are recovered from MMXU by qid)
  splits_clean_<date>.json   only with --base-splits

`usable` = ABCD options and both images present under MIMIC_JPG_ROOT. Our local copy of
MIMIC-CXR-JPG lacked about 8% of the images, so a complete copy marks more rows usable: the
lists in data_ids/ are the ones used in the paper. The default --out-dir is
data_ids/regenerated/, so the shipped files are never overwritten.

Usage:
  python3 clean_eval/build_clean_split.py [--official-split PATH] [--base-splits tac_mcq_train/splits.json]
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import (MMXU_DEV_JSONL, MMXU_TEST_JSONL, MIMIC_JPG_ROOT, MIMIC_SPLIT_CSV,  # noqa: E402
                    MIMIC_METADATA_CSV)

DATE = "2026-09-24"
CLEAN_SPLITS = ("validate", "test")
# Johnson et al. 2019, MIMIC-CXR-JPG, Table 3; validate is "a random set of 500 patients" (sec. 4).
TABLE3_IMAGES = {"train": 368960, "validate": 2991, "test": 5159}
TABLE3_STUDIES = {"train": 222758, "validate": 1808, "test": 3269}
VALIDATE_PATIENTS = 500
FIELDS = ["source", "qid", "usable"]   # written columns; the rows keep the MIMIC-CXR ids in memory


def official_split(path):
    split_of = defaultdict(set)
    for r in csv.DictReader(gzip.open(path, "rt")):
        split_of[r["subject_id"]].add(r["split"])
    multi = [p for p, s in split_of.items() if len(s) > 1]
    assert not multi, f"{len(multi)} patients with more than one split in the official file"
    return {p: next(iter(s)) for p, s in split_of.items()}


def check_against_table3(split_of, metadata):
    meta = list(csv.DictReader(gzip.open(metadata, "rt")))
    missing = {m["subject_id"] for m in meta} - set(split_of)
    assert not missing, f"{len(missing)} metadata patients without a split"
    images = Counter(split_of[m["subject_id"]] for m in meta)
    studies = {s: len({m["study_id"] for m in meta if split_of[m["subject_id"]] == s})
               for s in TABLE3_IMAGES}
    val_pats = sum(1 for s in split_of.values() if s == "validate")
    assert dict(images) == TABLE3_IMAGES, f"images {dict(images)} != Table 3 {TABLE3_IMAGES}"
    assert studies == TABLE3_STUDIES, f"studies {studies} != Table 3 {TABLE3_STUDIES}"
    assert val_pats == VALIDATE_PATIENTS, f"validate has {val_pats} patients, not {VALIDATE_PATIENTS}"
    print(f"  reproduces Table 3: images {dict(images)}, studies {studies}, "
          f"{val_pats} patients in validate")


def ids_of(path):
    """files/pXX/pNNNNNNNN/sMMMMMMMM/<dicom>.jpg -> (NNNNNNNN, MMMMMMMM, <dicom>)"""
    parts = path.split("/")
    return parts[2][1:], parts[3][1:], Path(parts[4]).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-split", type=Path, default=MIMIC_SPLIT_CSV)
    ap.add_argument("--metadata", type=Path, default=MIMIC_METADATA_CSV)
    ap.add_argument("--base-splits", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data_ids" / "regenerated")
    args = ap.parse_args()

    split_of = official_split(args.official_split)
    print(f"official split: {len(split_of)} patients")
    check_against_table3(split_of, args.metadata)

    exists_cache = {}

    def exists(rel):
        if rel not in exists_cache:
            exists_cache[rel] = (MIMIC_JPG_ROOT / rel).exists()
        return exists_cache[rel]

    rows, dev_patient = [], {}
    for source, path in (("dev", MMXU_DEV_JSONL), ("test", MMXU_TEST_JSONL)):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                subj, cur_study, cur_dicom = ids_of(d["cur_image_path"])
                subj_p, prior_study, prior_dicom = ids_of(d["prior_image_path"])
                assert subj == subj_p, f"{source} qid={d['id']}: different patient in cur/prior"
                if source == "dev":
                    dev_patient[d["id"]] = subj
                s = split_of.get(subj)
                assert s is not None, f"{source} qid={d['id']}: patient {subj} has no official split"
                if s not in CLEAN_SPLITS:
                    continue
                usable = (tuple(sorted(d["options"])) == ("A", "B", "C", "D")
                          and exists(d["cur_image_path"]) and exists(d["prior_image_path"]))
                rows.append({"source": source, "qid": d["id"], "subject_id": subj,
                             "cur_study_id": cur_study, "cur_dicom_id": cur_dicom,
                             "prior_study_id": prior_study, "prior_dicom_id": prior_dicom,
                             "mimic_split": s, "usable": usable})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"clean_set_ids_{DATE}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nclean set -> {out_csv}")
    for source in ("dev", "test"):
        rs = [r for r in rows if r["source"] == source]
        us = [r for r in rs if r["usable"]]
        print(f"  {source}: {len(rs)} rows ({len(us)} usable) | "
              f"{len({r['subject_id'] for r in us})} patients with usable rows")

    if args.base_splits is not None:
        clean_pats = {r["subject_id"] for r in rows}
        old = json.loads(args.base_splits.read_text())
        new = {"seed": old["seed"], "val_patient_frac": old["val_patient_frac"],
               "derived_from": "tac_mcq_train/splits.json",
               "excluded": f"clean_set_ids_{DATE}.csv (patients in the official validate/test split)"}
        for k in ("train", "val", "val_fast"):
            kept = sorted(i for i in old[k] if dev_patient[i] not in clean_pats)
            new[k] = kept
            print(f"  {k:8s}: {len(old[k])} -> {len(kept)} rows")
        out_json = args.out_dir / f"splits_clean_{DATE}.json"
        out_json.write_text(json.dumps(new))
        print(f"split without the clean set -> {out_json}")


if __name__ == "__main__":
    main()
