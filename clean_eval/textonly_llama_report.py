#!/usr/bin/env python3
"""Text-only control paired with the TAC backbone (Llama-3.2-3B-Instruct, review Flaw #6).

Accuracy of the text-only Llama (tac_mcq_train/runs/textonly_llama, trained and evaluated without images, `blind`
path of the TAC wrapper) on the clean set and MMXU-test, and paired differences against A, B, B2 (full and blind), and
the matching pair for the generic backbone (C full − Qwen2.5-VL-3B text-only). Patient bootstrap (paper_tables.Boot:
5,000 replicates, seed 0), 95% CI, pp. Aggregates only.

Usage: python3 clean_eval/textonly_llama_report.py   -> clean_eval/textonly_llama_report.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from paper_tables import Boot, load, patient_maps  # noqa: E402

T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
LL = T / "textonly_llama"
SETS = {"clean": ("clean set (3,096 rows, 331 patients)", "clean_{}.csv"),
        "full": ("MMXU-test (2,992 rows)", "oracle_{}.csv")}
TAC = [("A", T / "tac_pre_clean"), ("B", T / "tac_scratch_clean"), ("B2", T / "tac_b2_stage2")]
C_DIR, QT = Q / "qwen_clean/px518", Q / "textonly_clean"


def pp(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"


def acc(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    pats = patient_maps()
    st = json.loads((LL / "train_stats.json").read_text())
    md = ["# Text-only control on the TAC backbone (Llama-3.2-3B-Instruct)\n",
          "LoRA on Libra's Llama-3.2-3B-Instruct trained WITHOUT images (`tac_mcq_train/train_lora_textonly_llama.py`): "
          "same assembly, LoRA (r 16, alpha 32, all LLM linear layers), data (clean split, "
          f"{st['n_train']:,} rows), formats and optimisation as A, B, B2; prompt of the TAC wrapper's `blind` path. "
          f"1 epoch in {st['train_runtime_s']/3600:.2f} h, final loss {st['train_loss']:.4f}. Evaluated with the "
          "`blind` path (no image ever). Paired differences by row; patient bootstrap, 5,000 replicates, seed 0, 95% "
          "CI, pp.\n"]
    out = {}
    for key, (title, tmpl) in SETS.items():
        L = load(LL / tmpl.format("blind"))
        qs = sorted(L)
        bt = Boot(qs, pats[key])
        c = lambda R: np.array([R[q]["correct"] == "True" for q in qs], float)
        unp = sum(L[q]["status"] != "ok" for q in qs)
        l = c(L)
        res = {"textonly_llama": bt.ci(l)[:3], "unparseable_or_error": unp}
        md += [f"## {title}\n", f"Text-only Llama: accuracy **{acc(bt.ci(l))}** ({unp} rows without a parsed letter).\n",
               "| model X | acc. X full | X full − text-only Llama | X blind − text-only Llama |", "|---|---|---|---|"]
        for name, d in TAC:
            F, B = load(d / tmpl.format("full")), load(d / tmpl.format("blind"))
            assert set(qs) <= set(F) and set(qs) <= set(B), name
            f, b = c(F), c(B)
            res[name] = dict(full=bt.ci(f)[:3], full_minus_textonly=bt.ci(f - l)[:3],
                             blind_minus_textonly=bt.ci(b - l)[:3])
            md.append(f"| {name} | {acc(bt.ci(f))} | {pp(bt.ci(f - l))} | {pp(bt.ci(b - l))} |")
        # generic backbone pair, for reference
        Qt, Cf = load(QT / tmpl.format("blind")), load(C_DIR / tmpl.format("full"))
        if set(qs) <= set(Qt) and set(qs) <= set(Cf):
            qt, cf = c(Qt), c(Cf)
            res["Qwen_textonly"] = bt.ci(qt)[:3]
            res["C_full_minus_Qwen_textonly"] = bt.ci(cf - qt)[:3]
            res["textonly_Llama_minus_Qwen"] = bt.ci(l - qt)[:3]
            md += ["", f"Generic backbone, for reference: Qwen2.5-VL-3B text-only {acc(bt.ci(qt))}; C (518²) full − "
                   f"Qwen text-only {pp(bt.ci(cf - qt))}. Text-only Llama − text-only Qwen {pp(bt.ci(l - qt))}."]
        md.append("")
        out[key] = res
    (HERE / "textonly_llama_report.md").write_text("\n".join(md))
    (HERE / "textonly_llama_report.json").write_text(json.dumps(
        out, indent=1, default=lambda x: float(x) if isinstance(x, np.floating) else list(map(float, x))))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error", type(e).__name__)
        sys.exit(1)
