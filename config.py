"""Data and model locations. No dataset is distributed with this repository (see README).

Each path can be set with an environment variable; by default it points inside ``data/``,
where the README asks you to place (or symlink) the credentialed files.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO / "data"))


def _path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, DATA_DIR / default))


# MMXU (Mu et al., 2025): obtained from its authors.
MMXU_TEST_JSONL = _path("MMXU_TEST_JSONL", "MMXU-test.jsonl")
MMXU_DEV_JSONL = _path("MMXU_DEV_JSONL", "MMXU-dev.jsonl")

# MIMIC-CXR-JPG 2.0.0 (PhysioNet, credentialed): directory that contains files/p10 ... files/p19.
MIMIC_JPG_ROOT = _path("MIMIC_JPG_ROOT", "mimic-cxr-jpg")
MIMIC_SPLIT_CSV = _path("MIMIC_SPLIT_CSV", "mimic-cxr-2.0.0-split.csv.gz")
MIMIC_METADATA_CSV = _path("MIMIC_METADATA_CSV", "mimic-cxr-2.0.0-metadata.csv.gz")

# Chest ImaGenome 1.0.0 (PhysioNet, credentialed): directory that contains silver_dataset/.
IMAGENOME_DIR = _path("IMAGENOME_DIR", "chest-imagenome")

# Libra (Zhang et al.), code only, pinned to v1.0.3 = 0d6c56a44b0f74259bdd8f05929465790b49f6e5.
LIBRA_DIR = Path(os.environ.get("LIBRA_DIR", REPO / "third_party" / "Libra"))

# Identifier lists shipped with this repository (identifiers only, no text).
CLEAN_SET_CSV = REPO / "data_ids" / "clean_set_ids_2026-09-24.csv"
SPLITS_FILE = REPO / "data_ids" / "splits_clean_2026-09-24.json"
# The 2,992 MMXU-test rows with a usable Chest ImaGenome box on both images (column: qid).
EVAL_QIDS = REPO / "data_ids" / "mmxu_test_eval_qids.csv"
