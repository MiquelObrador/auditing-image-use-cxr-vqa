#!/usr/bin/env python3
"""Per-class analysis of the fine-tuned models on the clean set (saved predictions only, no GPU).

Class = content_type of the row (worsen / improved / no change), as in criteria_report_clean.md. Per model and class:
  - n rows / patients, accuracy(full);
  - blind−full, both_other−full, mismatch−full (all rows of the class) and hole−hole_random (rows with the region's
    box in both images, clean_eval/clean_bbox_table.csv);
  - swap turn = acc(vs. inverted) − acc(vs. original) on the swap run, with the unambiguous inverted label of
    swap_inverted_recompute.unambiguous_inverted (as in swapdup_report.py); defined for worsen and improved only;
  - distribution of the predicted class (label_option of the predicted option's text) under full and blind, overall
    and by correct class.
And the question of the request: does removing the image push every model towards "worsened"? Paired by patient,
(blind−full)_A − (blind−full)_X on the worsen rows, for X = C (both resolutions) and B2.

Bootstrap by patient on each subset (paper_tables.Boot: 5000 replicates, seed 0), 95% CI, pp.
Models: A, B, B2 (TAC) and C twice: at 518² (the main table's C) and at the protocol's default resolution
(qwen_clean, the run behind criteria_report_clean.md's +4.12 in worsen). Aggregates only.

Usage: python3 clean_eval/per_class_report.py   -> clean_eval/per_class_report.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(HERE))
from paper_tables import Boot, load  # noqa: E402
from swap_inverted_recompute import unambiguous_inverted, label_option  # noqa: E402

OFF = 1_000_000
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
MODELS = [("A", "A (pretrained TAC)", T / "tac_pre_clean"),
          ("B", "B (TAC from scratch)", T / "tac_scratch_clean"),
          ("B2", "B2 (TAC from scratch, 2 stages)", T / "tac_b2_stage2"),
          ("C518", "C (generic, 518², main table)", Q / "qwen_clean/px518"),
          ("Cdef", "C (generic, default resolution)", Q / "qwen_clean")]
CT = ("worsen", "improved", "no change")
PRED = ("worsen", "improved", "no change", "other")


def entries():
    ent = {}
    for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        for d in map(json.loads, open(fn)):
            ent[d["id"] + (OFF if src == "dev" else 0)] = d
    return ent


ENT = entries()
PAT = {q: e["cur_image_path"].split("/")[2] for q, e in ENT.items()}


def pp(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"


def pclass(R, q):
    L = R[q]["pred_letter"]
    if not L:
        return "other"
    c = label_option(ENT[q]["options"][L])
    return c if c in CT else "other"


def opt(d, c):
    p = d / f"clean_{c}.csv"
    return load(p) if p.exists() else None


def main():
    box = {int(r["qid"]) for r in __import__("csv").DictReader(open(HERE / "clean_bbox_table.csv"))}
    runs = {k: {c: opt(d, c) for c in ("full", "blind", "both_other", "mismatch", "swap", "hole", "hole_random")}
            for k, _, d in MODELS}
    qs_all = sorted(runs["A"]["full"])
    for k in runs:
        assert sorted(runs[k]["full"]) == qs_all, k
        # a condition counts only if it covers all its rows (a run still in progress does not enter)
        for c, X in runs[k].items():
            need = box if c in ("hole", "hole_random") else set(qs_all)
            if X is not None and not need <= set(X):
                runs[k][c] = None
    cls = {q: ENT[q]["content_type"] for q in qs_all}
    md = ["# Per-class analysis — fine-tuned models, clean set\n",
          "Saved predictions, no GPU. Class = `content_type` of the row, as in `criteria_report_clean.md`. Paired "
          "differences by row; patient bootstrap on each subset (5,000 replicates, seed 0), 95% CI, in pp. "
          "`hole−hole_random` on the rows with the region's box in both images (`clean_bbox_table.md`). Swap turn = "
          "acc(vs. inverted) − acc(vs. original) on the swap run, unambiguous inverted label "
          "(`swap_inverted_recompute.unambiguous_inverted`), worsen and improved only. Predicted class = "
          "`label_option` of the predicted option's text (`other` = no class, ambiguous or unparseable).\n",
          "**Two versions of C.** The main table's C is evaluated at 518² (`qwen_clean/px518`); the +4.12 in worsen "
          "quoted in the request comes from C at the protocol's default resolution (`qwen_clean`). Their `blind` "
          "runs give the same answers (blind has no image), so they differ only through `full`. Both are shown.\n",
          "## 1. Per class\n",
          "| model | class | n rows / patients | acc. full | blind−full | both_other−full | mismatch−full | "
          "hole−hole_random (n) | swap turn (n) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, name, _ in MODELS:
        R = runs[k]
        for c in ("all",) + CT:
            qs = [q for q in qs_all if c == "all" or cls[q] == c]
            bt = Boot(qs, PAT)
            corr = lambda X, col="correct": np.array([X[q][col] == "True" for q in qs], float)
            f = corr(R["full"])
            cells = [f"{len(qs):,} / {len({PAT[q] for q in qs})}", f"{f.mean():.3f}"]
            for cond in ("blind", "both_other", "mismatch"):
                cells.append(pp(bt.ci(corr(R[cond]) - f)) if R[cond] else "—")
            if R["hole"] and R["hole_random"]:
                qb = [q for q in qs if q in box]
                hb = np.array([(R["hole"][q]["correct"] == "True") - (R["hole_random"][q]["correct"] == "True")
                               for q in qb], float)
                cells.append(f"{pp(Boot(qb, PAT).ci(hb))} ({len(qb):,})")
            else:
                cells.append("—")
            if R["swap"] and c in ("worsen", "improved", "all"):
                tgt = {q: (unambiguous_inverted(ENT[q]["options"], ENT[q]["answer"], cls[q])
                           if cls[q] != "no change" else None) for q in qs}
                qw = [q for q in qs if tgt[q]]
                turn = np.array([(R["swap"][q]["pred_letter"] == tgt[q]) - (R["swap"][q]["correct"] == "True")
                                 for q in qw], float)
                cells.append(f"{pp(Boot(qw, PAT).ci(turn))} ({len(qw):,})" + (" worsen+improved" if c == "all" else ""))
            else:
                cells.append("—" if c != "no change" else "— (no inverse)")
            md.append(f"| {name} | {c} | " + " | ".join(cells) + " |")
    md += ["", "## 2. Predicted class under full and blind (% of rows)\n",
           "| model | correct class | full: worsen / improved / no change / other | blind: worsen / improved / "
           "no change / other |", "|---|---|---|---|"]
    dist = {}
    for k, name, _ in MODELS:
        R = runs[k]
        for c in ("all",) + CT:
            qs = [q for q in qs_all if c == "all" or cls[q] == c]
            row = []
            for cond in ("full", "blind"):
                cnt = Counter(pclass(R[cond], q) for q in qs)
                dist[(k, c, cond)] = {p: 100 * cnt[p] / len(qs) for p in PRED}
                row.append(" / ".join(f"{dist[(k, c, cond)][p]:.1f}" for p in PRED))
            md.append(f"| {name} | {c} | " + " | ".join(row) + " |")
    # the question: is the blind gain in worsen a shift towards "worsened" shared by all models?
    qw = [q for q in qs_all if cls[q] == "worsen"]
    bt = Boot(qw, PAT)
    bf = lambda k: (np.array([runs[k]["blind"][q]["correct"] == "True" for q in qw], float)
                    - np.array([runs[k]["full"][q]["correct"] == "True" for q in qw], float))
    pw = lambda k, cond, qs: np.array([pclass(runs[k][cond], q) == "worsen" for q in qs], float)
    md += ["", "## 3. Does removing the image push the models towards \"worsened\"?\n",
           "Share of ALL rows (any correct class) answered with a worsen option, full → blind, with the paired "
           "change (patient bootstrap on the 3,096 rows):", "",
           "| model | full | blind | blind − full |", "|---|---|---|---|"]
    bta = Boot(qs_all, PAT)
    shift = {}
    for k, name, _ in MODELS:
        a, b = pw(k, "full", qs_all), pw(k, "blind", qs_all)
        shift[k] = bta.ci(b - a)
        md.append(f"| {name} | {a.mean()*100:.1f}% | {b.mean()*100:.1f}% | {pp(bta.ci(b - a))} |")
    md += ["", f"Worsen rows ({len(qw):,} rows, {len({PAT[q] for q in qw})} patients): blind−full per model and the "
           "paired difference against A, (blind−full)_A − (blind−full)_X:", "",
           "| model X | blind−full in worsen | (blind−full)_A − (blind−full)_X |", "|---|---|---|"]
    res = {}
    for k, name, _ in MODELS:
        t = bt.ci(bf(k))
        d = bt.ci(bf("A") - bf(k)) if k != "A" else None
        res[k] = dict(blind_minus_full_worsen=list(map(float, t[:3])),
                      A_minus_X=list(map(float, d[:3])) if d else None)
        md.append(f"| {name} | {pp(t)} | {pp(d) if d else '—'} |")
    # swap−full (vs. the original label, original letters), verdict and Δpred, as in the main table
    md += ["", "## 4. swap − full (all rows, vs. the original label)\n",
           "Verdict as in `paper_parallel_table.py`: Sig. = |Δ| ≥ 5pp with the 95% CI excluding zero; Eq. = 90% CI "
           "inside ±5pp (TOST); otherwise Inconcl. Δpred = share of rows whose predicted letter changes against full.",
           "", "| model | swap − full | verdict | Δpred |", "|---|---|---|---|"]
    swap_res = {}
    for k, name, _ in MODELS:
        R = runs[k]
        if not R["swap"]:
            md.append(f"| {name} | — | — | — |")
            continue
        corr = lambda X: np.array([X[q]["correct"] == "True" for q in qs_all], float)
        t = bta.ci(corr(R["swap"]) - corr(R["full"]))
        ch = bta.ci(np.array([R["swap"][q]["pred_letter"] != R["full"][q]["pred_letter"] for q in qs_all], float))
        m, lo, hi, lo90, hi90 = t
        v = "Sig." if abs(m) >= 0.05 and (lo > 0 or hi < 0) else "Eq." if -0.05 < lo90 and hi90 < 0.05 else "Inconcl."
        swap_res[k] = dict(swap_minus_full=list(map(float, t)), verdict=v, dpred=list(map(float, ch[:3])))
        md.append(f"| {name} | {pp(t)} | {v} | {ch[0]*100:.1f}% [{ch[1]*100:.1f}, {ch[2]*100:.1f}] |")
    md.append("")
    res["swap"] = swap_res
    f = lambda t: f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"
    g = lambda k: f(res[k]["A_minus_X"])
    md += ["## 5. Reading\n",
           "- **Removing the image does not push every model towards \"worsened\"; it pushes the two models whose "
           "answers depend on whether an image is present** (A, and B2, which reacts to the image's absence but not to "
           f"its content: both_other and mismatch ≈ 0). Share of all rows answered with a worsen option, blind − full: A {f(shift['A'])}, B2 "
           f"{f(shift['B2'])}; B {f(shift['B'])} and C at 518² {f(shift['C518'])}, which barely react to removing "
           "the image at all (blind−full overall ≈ 0). Without the image, A and B2 fall back to \"worsened\" (about "
           "60% of answers), which raises accuracy on worsen rows and costs it on improved and no change.",
           "- **So A's gain on worsen is that shift, and it is not specific to A: B2 shows it too, larger** "
           f"((blind−full)_A − (blind−full)_B2 = {g('B2')} on worsen rows).",
           f"- **Against C, the answer depends on which C.** With the main table's C (518²): A − C = {g('C518')}, "
           "because C does not change without the image. With C at the default resolution: A − C = "
           f"{g('Cdef')}, not distinguishable. But that C's +4.1 on worsen is not a shift of its blind answers (its "
           "blind run gives the same answers as the 518² one); it comes from a lower full accuracy on worsen at the "
           "default resolution (0.773 vs 0.812), i.e. from the resolution mismatch with training.",
           ""]
    (HERE / "per_class_report.md").write_text("\n".join(md))
    (HERE / "per_class_stats.json").write_text(json.dumps(
        {"worsen_question": res, "pred_class_pct": {f"{k}|{c}|{cond}": v for (k, c, cond), v in dist.items()}},
        indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error", type(e).__name__)
        sys.exit(1)
