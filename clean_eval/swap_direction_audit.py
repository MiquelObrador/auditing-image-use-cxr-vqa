#!/usr/bin/env python3
"""Audit of the swap direction metric (review, Flaw #1), on saved predictions only (no GPU).

The swap turn, acc(vs. inverted) − acc(vs. original) under swap, has a strongly negative null: a model blind to the
order of the images keeps answering the original label, so its turn is ≈ −acc. This script measures the turn against
its own baseline and the class flips directly.

  turn(X)  = P(pred = inverted letter) − P(pred = original letter) under condition X, on the rows with an
             unambiguous inverted label (swap_inverted_recompute.unambiguous_inverted), worsen and improved only;
  Δturn    = turn(swap) − turn(full), paired by row;
  flip     = share of rows whose predicted CLASS under swap is the opposite of the class the same model predicted
             under full (worsen <-> improved; class = label_option of the predicted option's text), by correct class.
             Two denominators: all rows of the class, and the rows where full predicted worsen or improved.
             Null reference: the same flip rate between full and mismatch (another patient's prior; no reason to
             invert the direction).

Fine-tuned A, B, B2, C (518², main table) on the clean set (3,096 rows); zero-shot VLMs on MMXU-test (2,992 rows),
where swap is run with permuted letters, so the baseline is full with the SAME permutation (oracle_full_permuted.csv);
the permutation is rebuilt as in swap_inverted_recompute.py and checked against the saved columns.
Patient bootstrap (paper_tables.Boot: 5,000 replicates, seed 0), 95% CI, pp.

Usage: python3 clean_eval/swap_direction_audit.py   -> clean_eval/swap_direction_audit.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(HERE))
from paper_tables import Boot, load  # noqa: E402
from swap_inverted_recompute import unambiguous_inverted, label_option, INVERT  # noqa: E402

OFF = 1_000_000
LETTERS = ["A", "B", "C", "D"]
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
FT = [("A (pretrained TAC)", T / "tac_pre_clean"), ("B (TAC from scratch)", T / "tac_scratch_clean"),
      ("B2 (TAC from scratch, 2 stages)", T / "tac_b2_stage2"), ("C (generic, 518²)", Q / "qwen_clean/px518")]
ZS = [("Qwen2-VL-7B", REPO / "mmxu_eval/oracle_results"), ("InternVL2-8B", REPO / "internvl_eval/oracle_results"),
      ("HuatuoGPT-7B", REPO / "huatuo_eval/oracle_results"), ("Qwen2.5-VL-7B", REPO / "qwen25vl_eval/oracle_results")]
DIR = ("worsen", "improved")


def entries():
    ent = {}
    for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        for d in map(json.loads, open(fn)):
            ent[(src, d["id"])] = d
    return ent


ENT = entries()


def pp(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"


def pct(t):
    return f"{t[0]*100:.1f}% [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def analyse(name, qs, opts, ans, ct, pat, full, swap, mis, mis_base=None, mis_opts=None):
    """opts[q]: {letter: text} as shown to the model; ans[q]: correct letter in that order; ct[q]: class."""
    cls = lambda R, q, o=opts: (label_option(o[q][R[q]["pred_letter"]]) if R[q]["pred_letter"] else "none")
    # null: full vs. mismatch with the SAME letter order (zero-shot: both with the original letters)
    same_order = mis_base is None      # fine-tuned: full, swap and mismatch share the letters, so turn(mismatch) exists
    mis_base, mis_opts = (mis_base or full), (mis_opts or opts)
    inv = {q: unambiguous_inverted(opts[q], ans[q], ct[q]) for q in qs}
    out, rows = {}, []
    for c in DIR + ("worsen+improved",):
        sub = [q for q in qs if (ct[q] == c if c in DIR else ct[q] in DIR)]
        qu = [q for q in sub if inv[q]]
        bt = Boot(qu, pat)
        turn = lambda R: np.array([(R[q]["pred_letter"] == inv[q]) - (R[q]["pred_letter"] == ans[q]) for q in qu],
                                  float)
        tf, ts = turn(full), turn(swap)
        res = dict(n_unamb=len(qu), turn_full=bt.ci(tf)[:3], turn_swap=bt.ci(ts)[:3], d_turn=bt.ci(ts - tf)[:3])
        if same_order:
            tm = turn(mis)
            # mismatch as the null of the direction: turn(mismatch) − turn(full), and the swap effect beyond it,
            # turn(swap) − turn(mismatch) = Δturn(swap) − Δturn(mismatch); paired by row
            res.update(turn_mismatch=bt.ci(tm)[:3], d_turn_mismatch=bt.ci(tm - tf)[:3],
                       d_turn_swap_minus_mismatch=bt.ci(ts - tm)[:3])
        # flips of the predicted class, all rows of the class
        bts = Boot(sub, pat)
        fc = {q: cls(full, q) for q in sub}
        for tag, R, B, o in (("swap", swap, full, opts), ("mismatch", mis, mis_base, mis_opts)):
            bc = {q: cls(B, q, o) for q in sub}
            fl = np.array([bc[q] in DIR and cls(R, q, o) == INVERT[bc[q]] for q in sub], float)
            res[f"flip_{tag}_all"] = bts.ci(fl)[:3]
            qd = [q for q in sub if bc[q] in DIR]
            fd = np.array([cls(R, q, o) == INVERT[bc[q]] for q in qd], float)
            res[f"flip_{tag}_given_dir"] = Boot(qd, pat).ci(fd)[:3]
        res["n_rows"] = len(sub)
        res["n_full_dir"] = len([q for q in sub if fc[q] in DIR])
        out[c] = res
        dm = (f"{pp(res['d_turn_mismatch'])} | **{pp(res['d_turn_swap_minus_mismatch'])}**" if same_order
              else "— | —")
        rows.append(f"| {name} | {c} | {len(qu):,} | {pp(res['turn_full'])} | {pp(res['turn_swap'])} | "
                    f"**{pp(res['d_turn'])}** | {dm} | {pct(res['flip_swap_all'])} | {pct(res['flip_mismatch_all'])} | "
                    f"{pct(res['flip_swap_given_dir'])} ({res['n_full_dir']:,}) | {pct(res['flip_mismatch_given_dir'])} |")
    return out, rows


def main():
    md = ["# Swap direction audit (review Flaw #1)\n",
          "Saved predictions, no GPU. turn(X) = P(pred = inverted letter) − P(pred = original letter) under X, on rows "
          "with an unambiguous inverted label (worsen, improved). Δturn = turn(swap) − turn(full), paired. Flip = the "
          "predicted class under swap (or mismatch, as a null) is the opposite of the class the same model predicted "
          "under full; over all rows of the class, and over the rows where full predicted worsen/improved (n in "
          "brackets, for swap). Zero-shot: swap with permuted letters against full with the same permutation; the "
          "mismatch null (patient derangement) against full, both with the original letters. For the fine-tuned models "
          "(same letters in full, swap and mismatch) the turn under mismatch is also given: Δturn mismatch = "
          "turn(mismatch) − turn(full), and the swap effect beyond it, turn(swap) − turn(mismatch). Patient bootstrap, "
          "5,000 replicates, seed 0, 95% CI, pp.\n",
          "| model | class | n (unamb.) | turn(full) | turn(swap) | Δturn | Δturn mismatch = turn(mis) − turn(full) | "
          "swap beyond mismatch = turn(swap) − turn(mis) | flip swap (all rows) | flip mismatch "
          "(all rows) | flip swap (full predicted a direction) | flip mismatch (idem) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    # fine-tuned, clean set
    md_ft = []
    for name, d in FT:
        full, swap, mis = load(d / "clean_full.csv"), load(d / "clean_swap.csv"), load(d / "clean_mismatch.csv")
        qs = sorted(full)
        assert set(qs) == set(swap) == set(mis), name
        e = {q: ENT[("dev", q - OFF)] if q >= OFF else ENT[("test", q)] for q in qs}
        pat = {q: e[q]["cur_image_path"].split("/")[2] for q in qs}
        res[name], rows = analyse(name, qs, {q: e[q]["options"] for q in qs}, {q: e[q]["answer"] for q in qs},
                                  {q: e[q]["content_type"] for q in qs}, pat, full, swap, mis)
        md_ft += rows
    md += ["| **fine-tuned, clean set** | | | | | | | | | | | |"] + md_ft
    # zero-shot, MMXU-test, permuted letters
    md_zs = []
    for name, d in ZS:
        fp, sp, mis = load(d / "oracle_full_permuted.csv"), load(d / "oracle_swap_permuted.csv"), \
            load(d / "oracle_mismatch_patient.csv")
        qs = sorted(sp)
        assert sorted(fp) == qs and set(mis) == set(qs), name
        seeds = np.random.SeedSequence(0).spawn(len(qs))
        opts, ans, bad = {}, {}, 0
        for i, q in enumerate(qs):
            e, p = ENT[("test", q)], list(np.random.default_rng(seeds[i]).permutation(LETTERS))
            opts[q] = {LETTERS[k]: e["options"][p[k]] for k in range(4)}
            ans[q] = next(L for k, L in enumerate(LETTERS) if p[k] == e["answer"])
            bad += ans[q] != sp[q]["answer_permuted"] or ans[q] != fp[q]["answer_permuted"]
        assert bad == 0, f"{name}: permutation mismatch in {bad} rows"
        # mismatch was run with the ORIGINAL letters: its null is against full with the original letters
        orig = {q: ENT[("test", q)]["options"] for q in qs}
        full_o = load(d / "oracle_full.csv")
        assert set(full_o) == set(qs), name
        pat = {q: ENT[("test", q)]["cur_image_path"].split("/")[2] for q in qs}
        res[name], rows = analyse(name, qs, opts, ans, {q: ENT[("test", q)]["content_type"] for q in qs}, pat,
                                  fp, sp, mis, mis_base=full_o, mis_opts=orig)
        md_zs += rows
    md += ["| **zero-shot, MMXU-test** | | | | | | | | | | | |"] + md_zs + [
        "", "Δturn mismatch and swap-beyond-mismatch are given for the fine-tuned models only: the zero-shot mismatch "
        "runs use the original letters and the zero-shot swap runs permuted ones, so their turns are not on the same "
        "scale.", ""]
    (HERE / "swap_direction_audit.md").write_text("\n".join(md))
    (HERE / "swap_direction_audit.json").write_text(json.dumps(
        res, indent=1, default=lambda x: float(x) if isinstance(x, np.floating) else list(map(float, x))))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("error", type(ex).__name__)
        sys.exit(1)
