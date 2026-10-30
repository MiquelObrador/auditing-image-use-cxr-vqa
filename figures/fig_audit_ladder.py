#!/usr/bin/env python3
"""Fig. 2 of the paper: what each model loses without images (x) against what it loses with another patient's images (y).

One point per model: the four zero-shot VLMs (MMXU-test) and the fine-tuned A, B, B2, C (clean set).
  x = full − blind (pp): lost without images;
  y = full − both_other (pp): lost with both images from another patient (same number of visual tokens);
  95% patient-bootstrap CIs on both axes (5,000 replicates, seed 0).
Reference lines: y = x (all the image benefit comes from content), y = 0 (benefit from token presence only), x = 0.
Markers: circle = zero-shot (blue), square = fine-tuned (vermillion); filled if mismatch−full is Sig. (the model
reads the prior image), hollow otherwise. Colours are Okabe-Ito (colour-blind safe); shape and fill still carry the
encoding in black and white. Labels of the points crowded near the origin are placed in free space with thin leader
lines. All text is 9pt (ISBI: no type smaller than 9pt); the marker key and the axis definitions are in the caption,
not in the figure, so short axis and line labels keep the figure at one column without growing it.

Source: clean_eval/paper_parallel_table.json, written by clean_eval/paper_parallel_table.py (the numbers of Table 1),
so the figure and the table cannot disagree.
Output: figures/fig_audit_ladder.pdf (vector, one column, 3.4 x 2.3 in) and fig_audit_ladder.csv (the values).

Usage: python3 figures/fig_audit_ladder.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "clean_eval" / "paper_parallel_table.json"
# table name -> (short label, zero-shot?, label position). A position in points is an offset from the marker;
# a position tagged "data" is where the label goes in data coordinates, joined to the marker by a leader line
# (used for the points crowded around the origin).
POINTS = {
    "Qwen2-VL-7B": ("Qwen2-VL", True, ("data", (3.4, -1.55))),
    "InternVL2-8B": ("InternVL2", True, ("data", (-5.1, 3.1))),
    "HuatuoGPT-7B": ("Huatuo", True, ("data", (9.4, 4.0))),
    "Qwen2.5-VL-7B": ("Qwen2.5-VL", True, ("data", (4.4, 6.3))),
    "A: pretrained TAC": ("A", False, ("pts", (-9, 2))),
    "B: TAC from scratch": ("B", False, ("data", (-3.4, -1.3))),
    "B2: TAC from scratch, 2 stages": ("B2", False, ("pts", (3, 3))),
    "C: generic": ("C", False, ("data", (-1.8, 5.4))),
}
INK, GREY = "#000000", "#8c8c8c"
ZS_COL, FT_COL = "#0072B2", "#D55E00"  # Okabe-Ito blue (zero-shot) and vermillion (fine-tuned)


def main():
    S = json.loads(SRC.read_text())
    rows = []
    for name, (lab, zs, off) in POINTS.items():
        s = S[name]
        b, o = s["blind"], s["both_other"]
        mis = s.get("mismatch")
        rows.append(dict(
            model=lab, setting="zero-shot, MMXU-test" if zs else "fine-tuned, clean set",
            x_full_minus_blind_pp=-100 * b["est"], x_lo=-100 * b["hi"], x_hi=-100 * b["lo"],
            y_full_minus_both_other_pp=-100 * o["est"], y_lo=-100 * o["hi"], y_hi=-100 * o["lo"],
            mismatch_minus_full_pp=100 * mis["est"] if mis else None,
            mismatch_verdict=mis["verdict"] if mis else "missing",
            gate=s["gate"], _zs=zs, _off=off))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs = [v for r in rows for v in (r["x_lo"], r["x_hi"])]
    ys = [v for r in rows for v in (r["y_lo"], r["y_hi"])]
    x0, x1 = min(-7, min(xs) - 1), max(xs) + 1.5
    y0, y1 = min(-2, min(ys) - 1), max(ys) + 1.5
    # reference lines
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
    d = min(x1, y1)
    ax.plot([max(x0, y0), d], [max(x0, y0), d], color=GREY, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.text(d - 0.3, d - 0.3, "content only", ha="right", va="top", color="#555555")
    ax.text(x1 - 0.3, -0.35, "token presence only", ha="right", va="top", color="#555555")
    for r in rows:
        filled = r["mismatch_verdict"] == "Sig."
        col = ZS_COL if r["_zs"] else FT_COL
        xy = (r["x_full_minus_blind_pp"], r["y_full_minus_both_other_pp"])
        ax.errorbar(*xy,
                    xerr=[[r["x_full_minus_blind_pp"] - r["x_lo"]], [r["x_hi"] - r["x_full_minus_blind_pp"]]],
                    yerr=[[r["y_full_minus_both_other_pp"] - r["y_lo"]], [r["y_hi"] - r["y_full_minus_both_other_pp"]]],
                    fmt="o" if r["_zs"] else "s", ms=4.2, mew=0.8, mec=col, mfc=col if filled else "white",
                    ecolor=col, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)
        mode, pos = r["_off"]
        if mode == "data":
            ax.annotate(r["model"], xy, xytext=pos, textcoords="data", color=col, zorder=4,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY, shrinkA=1.5, shrinkB=2.5))
        else:
            ax.annotate(r["model"], xy, xytext=pos, textcoords="offset points", color=col, zorder=4)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Lost without images (pp)")
    ax.set_ylabel("Lost with another\npatient's images (pp)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(HERE / "fig_audit_ladder.pdf")
    fig.savefig(HERE / "fig_audit_ladder.png", dpi=300)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(HERE / "fig_audit_ladder.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    for r in rows:
        print(f"{r['model']:11s} x {r['x_full_minus_blind_pp']:+5.1f} [{r['x_lo']:+5.1f},{r['x_hi']:+5.1f}]  "
              f"y {r['y_full_minus_both_other_pp']:+5.1f} [{r['y_lo']:+5.1f},{r['y_hi']:+5.1f}]  "
              f"mismatch {r['mismatch_verdict']}")


if __name__ == "__main__":
    main()
