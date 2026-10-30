#!/usr/bin/env python3
"""Region boxes for the clean set (data_ids/clean_set_ids_2026-09-24.csv, 3,096 usable rows: 3,022 MMXU-dev + 74
MMXU-test), for the `hole` and `hole_random` conditions of the fine-tuned models.

Same source and rules as bbox_analysis/analyze_bbox_size.py, which built bbox_analysis/mmxu_bbox_table.csv for
MMXU-test (the table the zero-shot hole runs used):
  - region: the first region that bbox_analysis/regions.extract_regions finds in the question;
  - box: the `original_x1..y2` coordinates of that region in the Chest ImaGenome silver scene graph of EACH image
    (silver_dataset/scene_graph.zip); the row is dropped if the region is missing in either scene graph;
  - image size: Rows/Columns of mimic-cxr-2.0.0-metadata.csv.gz;
  - each box is clipped to its image; a row is dropped if a box does not overlap its image at all.

Checks (aggregates only, written to clean_eval/clean_bbox_table.md):
  - the 74 clean MMXU-test rows must reproduce mmxu_bbox_table.csv exactly (region and the eight coordinates);
  - for MMXU-dev rows, agreement of the scene-graph box with the dataset's own `cur_image_bboxs` /
    `prior_image_bboxs` for the same region, when present.

Output: clean_eval/clean_bbox_table.csv (per row; git-ignored) and clean_eval/clean_bbox_table.md (aggregates).
Usage: python3 clean_eval/build_clean_bbox_table.py
"""
import collections
import csv
import gzip
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL, MIMIC_METADATA_CSV, IMAGENOME_DIR  # noqa: E402
sys.path.insert(0, str(REPO / "bbox_analysis"))
sys.path.insert(0, str(REPO / "tac_mcq_train"))
sys.path.insert(0, str(REPO / "mmxu_eval"))
from regions import extract_regions  # noqa: E402
from run_eval_clean import clean_samples, DEV_QID_OFFSET  # noqa: E402

SCENE_GRAPH_ZIP = IMAGENOME_DIR / "silver_dataset" / "scene_graph.zip"
METADATA_CSV_GZ = MIMIC_METADATA_CSV
TEST_TABLE = REPO / "bbox_analysis/mmxu_bbox_table.csv"
OUT_CSV = REPO / "clean_eval/clean_bbox_table.csv"
OUT_MD = REPO / "clean_eval/clean_bbox_table.md"
COLS = ["qid", "region", "bbox_cur_x1", "bbox_cur_y1", "bbox_cur_x2", "bbox_cur_y2", "W_cur", "H_cur",
        "bbox_prior_x1", "bbox_prior_y1", "bbox_prior_x2", "bbox_prior_y2", "W_prior", "H_prior"]


def main():
    raw = {}
    for source, fname in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        for line in open(fname):
            e = json.loads(line)
            raw[e["id"] + (DEV_QID_OFFSET if source == "dev" else 0)] = e
    samples = clean_samples()
    zf = zipfile.ZipFile(SCENE_GRAPH_ZIP)
    cache = {}

    def sg_regions(dicom):
        if dicom not in cache:
            try:
                with zf.open(f"scene_graph/{dicom}_SceneGraph.json") as f:
                    sg = json.load(f)
                cache[dicom] = {o["bbox_name"]: (o["original_x1"], o["original_y1"], o["original_x2"],
                                                 o["original_y2"]) for o in sg["objects"]}
            except KeyError:
                cache[dicom] = {}
        return cache[dicom]

    dims = {}
    with gzip.open(METADATA_CSV_GZ, "rt") as f:
        for r in csv.DictReader(f):
            dims[r["dicom_id"]] = (int(r["Columns"]), int(r["Rows"]))

    rows, reasons, pats_out = [], collections.Counter(), collections.defaultdict(set)
    for s in samples:
        e = raw[s["qid"]]
        regs = extract_regions(e["question"])
        region = regs[0] if regs else None
        if region is None:
            reasons["no region recognised in the question"] += 1
            pats_out["no region"].add(s["patient"])
            continue
        dc, dp = Path(e["cur_image_path"]).stem, Path(e["prior_image_path"]).stem
        sc, sp = sg_regions(dc), sg_regions(dp)
        if region not in sc or region not in sp:
            reasons["region missing in the scene graph of an image"] += 1
            pats_out["region missing"].add(s["patient"])
            continue
        if dc not in dims or dp not in dims:
            reasons["image size not in metadata.csv"] += 1
            continue
        row = {"qid": s["qid"], "region": region, "patient": s["patient"], "source": "dev" if s["qid"] >= DEV_QID_OFFSET else "test"}
        bad = False
        for pfx, box, (W, H) in (("cur", sc[region], dims[dc]), ("prior", sp[region], dims[dp])):
            x1, y1, x2, y2 = box
            if not (x2 > x1 and y2 > y1):
                bad = True
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(x2, W), min(y2, H)
            if cx2 <= cx1 or cy2 <= cy1:
                bad = True
            row.update({f"bbox_{pfx}_x1": cx1, f"bbox_{pfx}_y1": cy1, f"bbox_{pfx}_x2": cx2, f"bbox_{pfx}_y2": cy2,
                        f"W_{pfx}": W, f"H_{pfx}": H, f"raw_{pfx}": box})
        if bad:
            reasons["box degenerate or entirely outside its image"] += 1
            pats_out["off frame"].add(s["patient"])
            continue
        rows.append(row)

    # check 1: the clean MMXU-test rows reproduce mmxu_bbox_table.csv
    ref = {int(r["id"]): r for r in csv.DictReader(open(TEST_TABLE))}
    test_rows = [r for r in rows if r["source"] == "test"]
    coord = [c for c in COLS if c.startswith("bbox_")]
    same = sum(r["region"] == ref[r["qid"]]["region"] and all(str(r[c]) == ref[r["qid"]][c] for c in coord)
               for r in test_rows if r["qid"] in ref)
    in_ref = sum(r["qid"] in ref for r in test_rows)
    clean_test_in_ref = sum(s["qid"] in ref for s in samples if s["qid"] < DEV_QID_OFFSET)

    # check 2: dev rows, scene-graph box vs the dataset's own box for the same region (before clipping).
    # `cur_image_bboxs` holds one box per region; `prior_image_bboxs` a list of boxes per region. When the prior box
    # differs, look for it in the scene graphs of the other images of the same prior study.
    study_of, by_study = {}, collections.defaultdict(list)
    with gzip.open(METADATA_CSV_GZ, "rt") as f:
        for r in csv.DictReader(f):
            study_of[r["dicom_id"]] = r["study_id"]
            by_study[r["study_id"]].append(r["dicom_id"])
    agree = {"cur": collections.Counter(), "prior": collections.Counter()}
    for r in rows:
        if r["source"] != "dev":
            continue
        e = raw[r["qid"]]
        own = (e.get("cur_image_bboxs") or {}).get(r["region"])
        agree["cur"]["absent" if own is None else
                     "identical" if tuple(own) == tuple(r["raw_cur"]) else "different"] += 1
        own = (e.get("prior_image_bboxs") or {}).get(r["region"])
        if own is None:
            agree["prior"]["absent"] += 1
        elif any(tuple(b) == tuple(r["raw_prior"]) for b in own):
            agree["prior"]["identical"] += 1
        else:
            agree["prior"]["different"] += 1
            dp = Path(e["prior_image_path"]).stem
            others = [d for d in by_study[study_of[dp]] if d != dp]
            if any(tuple(b) == tuple(sg_regions(d).get(r["region"], ())) for d in others for b in own):
                agree["prior"]["different: from another image of the same prior study"] += 1

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_all, p_all = len(samples), len({s["patient"] for s in samples})
    n_in, p_in = len(rows), len({r["patient"] for r in rows})
    by_src = collections.Counter(r["source"] for r in rows)
    md = ["# Region boxes for the clean set (hole / hole_random)\n",
          "Generated by `clean_eval/build_clean_bbox_table.py`. Aggregates only; the per-row table "
          "(`clean_bbox_table.csv`) is git-ignored. Same source and rules as `bbox_analysis/analyze_bbox_size.py` "
          "(the MMXU-test table of the zero-shot hole runs): question region via `extract_regions`, its box in the "
          "ImaGenome silver scene graph of each image (`original_*` coordinates), image size from the DICOM "
          "metadata, boxes clipped to the image.\n",
          "| | rows | patients |", "|---|---|---|",
          f"| clean set | {n_all:,} | {p_all} |",
          f"| with the region's box in both images | {n_in:,} (dev {by_src['dev']:,}, test {by_src['test']}) | {p_in} |",
          f"| left out | {n_all - n_in:,} | {p_all - p_in} patients with no row left |", "",
          "Rows left out, by reason:", ""]
    md += [f"- {k}: {v:,}" for k, v in reasons.most_common()] or ["- none"]
    md += ["", "Checks:", "",
           f"- The clean MMXU-test rows with a box ({len(test_rows)}) reproduce `mmxu_bbox_table.csv`: {same}/{in_ref} "
           f"identical in region and the eight coordinates ({clean_test_in_ref} of the 74 clean test rows are in that "
           "table).",
           "- MMXU-dev rows: the scene-graph box of the image shown against the dataset's own box for the same "
           "region (before clipping):"]
    for pfx in ("cur", "prior"):
        a = agree[pfx]
        extra = (f" (of which {a['different: from another image of the same prior study']:,} match another image "
                 "of the same prior study)") if pfx == "prior" else ""
        md.append(f"  - {pfx}: identical {a['identical']:,}, different {a['different']:,}{extra}, region absent from "
                  f"the dataset's boxes {a['absent']:,}.")
    md += ["  The boxes used are always those of the image actually shown, from its own scene graph, as for "
           "MMXU-test.", ""]
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error", type(e).__name__)
        sys.exit(1)
