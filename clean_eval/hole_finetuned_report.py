#!/usr/bin/env python3
"""`hole` and `hole_random` on the fine-tuned models (A, B, B2, C), clean set, rows with the region's box in both
images (clean_eval/clean_bbox_table.csv). Transforms of the paper: mmxu_eval/oracle_transforms.gray_inside_box
(grey 128, the box itself) and mmxu_eval/hole_random_transforms.hole_random_bbox (same size, mirrored side or shifted
vertically), applied to both images by mmxu_eval/oracle_evaluate.run, through tac_mcq_train/run_eval_clean.py.

Modes:
  smoke   box check on the first 20 rows of the box subset (the rows the 20-row smoke runs use) and on all of them,
          plus, per model, the 20-row `full` against the model's original clean_full.csv on the same rows.
  report  per model: hole−full and hole_random−full with 95% CI (patient bootstrap on the box subset, 5000, seed 0),
          Sig./Eq. (TOST ±5pp with the 90% CI), Δpred of each condition against full. `full` is the model's original
          clean_full.csv restricted to the box subset. -> clean_eval/hole_finetuned_report.md

Aggregates only. Errors print only their type.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "mmxu_eval"))
from paper_tables import Boot, load, patient_maps  # noqa: E402  (same bootstrap as the paper's tables)
from hole_random_transforms import hole_random_bbox  # noqa: E402

DELTA = 0.05
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
MODELS = [("A: pretrained TAC", T / "tac_pre_clean"),
          ("B: TAC from scratch", T / "tac_scratch_clean"),
          ("B2: TAC from scratch, 2 stages", T / "tac_b2_stage2"),
          ("C: generic (518²)", Q / "qwen_clean/px518")]
BOX_CSV = HERE / "clean_bbox_table.csv"
SMOKE = REPO / "tac_mcq_train/runs/hole_smoke"
OUT = HERE / "hole_finetuned_report.md"


def boxes():
    rows = list(csv.DictReader(open(BOX_CSV)))
    return rows


def box_stats(rows):
    """Covered area (share of the image), valid boxes, laterality (radiological convention: the patient's left is
    on the image's right), for hole and hole_random, over both images of each row."""
    area = {"hole": [], "hole_random": []}
    valid = {"hole": [], "hole_random": []}
    side = {"hole": [], "hole_random": []}
    for r in rows:
        for p in ("cur", "prior"):
            W, H = int(r[f"W_{p}"]), int(r[f"H_{p}"])
            b = tuple(int(r[f"bbox_{p}_{k}"]) for k in ("x1", "y1", "x2", "y2"))
            for cond, (x1, y1, x2, y2) in (("hole", b), ("hole_random", hole_random_bbox(r["region"], b, (W, H)))):
                a = max(0, x2 - x1) * max(0, y2 - y1)
                area[cond].append(a / (W * H))
                valid[cond].append(a > 0)
                if r["region"].startswith(("left ", "right ")):
                    cx = (x1 + x2) / 2
                    patient_left = r["region"].startswith("left ")
                    side[cond].append((cx > W / 2) == patient_left)   # box on the region's own side
    return {c: dict(area_mean_pct=100 * float(np.mean(area[c])), area_median_pct=100 * float(np.median(area[c])),
                    valid_pct=100 * float(np.mean(valid[c])),
                    own_side_pct=100 * float(np.mean(side[c])) if side[c] else None, n_images=len(area[c]))
            for c in area}


def smoke():
    rows = boxes()
    out = {"box_check_first20": box_stats(rows[:20]), "box_check_all": box_stats(rows)}
    rep = {}
    for name, d in MODELS:
        tag = d.parent.name if d.name == "px518" else d.name
        f20 = SMOKE / f"{tag}_full.csv"
        if not f20.exists():
            rep[name] = "missing"
            continue
        s, o = load(f20), load(d / "clean_full.csv")
        qs = sorted(s)
        rep[name] = dict(n=len(qs), same_letter=sum(s[q]["pred_letter"] == o[q]["pred_letter"] for q in qs),
                         acc_smoke=float(np.mean([s[q]["correct"] == "True" for q in qs])),
                         acc_original_same_rows=float(np.mean([o[q]["correct"] == "True" for q in qs])),
                         hole_rows=len(load(SMOKE / f"{tag}_hole.csv")) if (SMOKE / f"{tag}_hole.csv").exists() else 0,
                         hole_random_rows=len(load(SMOKE / f"{tag}_hole_random.csv"))
                         if (SMOKE / f"{tag}_hole_random.csv").exists() else 0)
    out["full_reproduction"] = rep
    (HERE / "hole_finetuned_smoke.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    ok = all(isinstance(v, dict) and v["same_letter"] == v["n"] and v["hole_rows"] == v["n"]
             and v["hole_random_rows"] == v["n"] for v in rep.values())
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return ok


def verdict(t):
    m, lo, hi, lo90, hi90 = t
    if abs(m) >= DELTA and (lo > 0 or hi < 0):
        return "Sig."
    if -DELTA < lo90 and hi90 < DELTA:
        return "Eq."
    return "Inconcl."


def cell(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}] {verdict(t)}"


def report():
    pat = patient_maps()["clean"]
    rows = boxes()
    sub = {int(r["qid"]) for r in rows}
    st = box_stats(rows)
    smk = json.loads((HERE / "hole_finetuned_smoke.json").read_text())
    md = ["# `hole` and `hole_random` on the fine-tuned models (clean set)\n",
          "Rows of the clean set with the question region's box in both images (`clean_eval/clean_bbox_table.md`): "
          f"**{len(sub):,} rows, {len({pat[q] for q in sub})} patients** (of 3,096 and 331). Left out: 22 rows whose "
          "region is missing from the scene graph of one image, and 10 whose box lies entirely outside its image. "
          "Boxes from the Chest ImaGenome silver scene graphs of the images shown, with the rules of "
          "`bbox_analysis/analyze_bbox_size.py` (the 72 clean MMXU-test rows reproduce `mmxu_bbox_table.csv` "
          "exactly).\n",
          "Transforms of the paper, on both images: `hole` = `oracle_transforms.gray_inside_box` (grey 128 inside the "
          "box); `hole_random` = the same grey patch, same size, at `hole_random_transforms.hole_random_bbox` (mirrored "
          "to the other side for paired regions, shifted vertically for midline ones). `full` = each model's original "
          "`clean_full.csv` on the same rows. Same models, checkpoints and settings as the main table (C at 518²).\n",
          "Statistics: paired by row, patient bootstrap on this subset (5,000 replicates, seed 0, `paper_tables.Boot`), "
          "95% CI. Sig. = |Δ| ≥ 5pp with the 95% CI excluding zero; Eq. = 90% CI inside ±5pp (TOST); otherwise "
          "Inconcl. Δpred = share of rows whose predicted letter changes against full.\n",
          "## Box check\n",
          "| | covered area, mean (median) | valid boxes | box on the region's own side (left/right regions) |",
          "|---|---|---|---|"]
    for c in ("hole", "hole_random"):
        s = st[c]
        md.append(f"| {c} ({s['n_images']:,} images) | {s['area_mean_pct']:.1f}% ({s['area_median_pct']:.1f}%) | "
                  f"{s['valid_pct']:.1f}% | {s['own_side_pct']:.1f}% |")
    md += ["", "Radiological convention: the patient's left is on the image's right. `hole_random` moves paired "
           "regions to the other side by design, so its own-side share should be near 0%.", "",
           "Smoke test (first 20 rows of the subset): `full` reproduces each model's original run "
           + "; ".join(f"{n.split(':')[0]} {v['same_letter']}/{v['n']} same letter, accuracy {v['acc_smoke']:.2f} vs "
                       f"{v['acc_original_same_rows']:.2f}" for n, v in smk["full_reproduction"].items()) + ".", "",
           "## Results\n",
           "| model | acc. full (subset) | hole−full | hole_random−full | hole−hole_random | Δpred hole | "
           "Δpred hole_random |",
           "|---|---|---|---|---|---|---|"]
    res = {}
    for name, d in MODELS:
        full = load(d / "clean_full.csv")
        H, R = load(d / "clean_hole.csv"), load(d / "clean_hole_random.csv")
        qs = sorted(sub)
        assert set(H) == sub and set(R) == sub and sub <= set(full), name
        bt = Boot(qs, pat)
        corr = lambda X: np.array([X[q]["correct"] == "True" for q in qs], float)
        chg = lambda X: np.array([X[q]["pred_letter"] != full[q]["pred_letter"] for q in qs], float)
        f = corr(full)
        tf, th, tr = bt.ci(f), bt.ci(corr(H) - f), bt.ci(corr(R) - f)
        thr = bt.ci(corr(H) - corr(R))
        dh, dr = bt.ci(chg(H)), bt.ci(chg(R))
        status = {k: sum(X[q]["status"] != "ok" for q in qs) for k, X in (("hole", H), ("hole_random", R))}
        res[name] = dict(full=tf[:3], hole=th, hole_random=tr, hole_minus_hole_random=thr, dpred_hole=dh[:3],
                         dpred_hole_random=dr[:3],
                         not_ok=status)
        md.append(f"| {name} | {tf[0]:.3f} [{tf[1]:.3f}, {tf[2]:.3f}] | {cell(th)} | {cell(tr)} | {cell(thr)} | "
                  f"{dh[0]*100:.1f}% [{dh[1]*100:.1f}, {dh[2]*100:.1f}] | {dr[0]*100:.1f}% [{dr[1]*100:.1f}, "
                  f"{dr[2]*100:.1f}] |")
    bad = {n: v["not_ok"] for n, v in res.items() if any(v["not_ok"].values())}
    md += ["", "pp ×100; [95% CI]. hole−hole_random is the cost specific to covering the asked region rather than an "
           "equal grey patch elsewhere. Rows with an unparseable answer or an error: "
           + (", ".join(f"{n.split(':')[0]} {v}" for n, v in bad.items()) if bad else "none") + ".", ""]
    OUT.write_text("\n".join(md))
    (HERE / "hole_finetuned_stats.json").write_text(json.dumps(
        {n: {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in r.items()} for n, r in res.items()},
        indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        ok = {"smoke": smoke, "report": report}[sys.argv[1]]()
        sys.exit(0 if ok is not False else 2)
    except Exception as e:
        print("error", type(e).__name__)
        sys.exit(1)
