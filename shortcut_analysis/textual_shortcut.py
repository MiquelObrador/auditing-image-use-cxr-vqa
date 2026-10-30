#!/usr/bin/env python3
"""Shared helpers for the textual-shortcut analysis: the option->class lexicon (same as
mmxu_eval/oracle_evaluate.py), word tokenizer and patient-clustered bootstrap (N=5000, seed 0).
Imported by options_only_linear.py, question_options_linear.py and hedge_check.py."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL, EVAL_QIDS  # noqa: E402
N_BOOT = 5000
SEED = 0
CLASSES = ("worsen", "improved", "no change")

IMPROVED_RE = re.compile(
    r"improv|resolv|better|decreas|reduc|clear(ed|ing)?\b|less |diminish|smaller|regress", re.I)
WORSENED_RE = re.compile(
    r"worse|worsen|increas|progress|larger|develop|new |appear|greater|more |enlarg|deteriorat", re.I)
NOCHANGE_RE = re.compile(
    r"no change|unchanged|no significant|stable|same|remain|not chang|no observ|persist|similar|no differ", re.I)

STOPWORDS = set("""a an the of in on at to for from by with and or is are was were be been being
has have had do does did what which how when where who whom this that these those it its as
there their between across two both image images chest x ray xray cxr""".split())


def label_option(text):
    i = bool(IMPROVED_RE.search(text))
    w = bool(WORSENED_RE.search(text))
    nc = bool(NOCHANGE_RE.search(text))
    hits = [lab for lab, flag in (("no change", nc), ("improved", i), ("worsen", w)) if flag]
    return hits[0] if len(hits) == 1 else ("NONE" if not hits else "AMBIG")


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def boot_ci(arr, cl, seed=SEED):
    arr = np.asarray(arr, float)
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    s = np.zeros(len(u)); c = np.zeros(len(u))
    np.add.at(s, inv, arr); np.add.at(c, inv, 1.0)
    idx = np.random.default_rng(seed).integers(0, len(u), (N_BOOT, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return arr.mean(), lo, hi


def diff_ci(a, mask_a, b, mask_b, cl, seed=SEED):
    """media(a | mask_a) - media(b | mask_b), remuestreando pacientes conjuntamente."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    def sums(x, m):
        s = np.zeros(len(u)); c = np.zeros(len(u))
        np.add.at(s, inv[m], x[m]); np.add.at(c, inv[m], 1.0)
        return s, c
    sa, ca = sums(a, mask_a); sb, cb = sums(b, mask_b)
    idx = np.random.default_rng(seed).integers(0, len(u), (N_BOOT, len(u)))
    d = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return a[mask_a].mean() - b[mask_b].mean(), lo, hi


def expected_pick(scores, gold):
    """1/k si la correcta esta entre las k opciones de maxima puntuacion."""
    best = max(scores.values())
    top = [L for L, v in scores.items() if v == best]
    return (gold in top) / len(top), len(top)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


def fmt_pp(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f},{t[2]*100:+.1f}]"
