"""Carga MMXU-test.jsonl y resuelve las rutas de imagen contra MIMIC-CXR-JPG-512.

Aviso de nombres de campo: el enunciado de la tarea (y la Fig. 9 del paper)
asumen "current_image_path" / "history_image_path" / "qid". El JSONL real
trae "cur_image_path" / "prior_image_path" / "id" -- verificado leyendo el
propio fichero. El mapeo semántico es el que ya se documentó en context.md:
prior_image_path = imagen previa = Image-1, cur_image_path = imagen actual =
Image-2. La traducción de nombres ocurre aquí, una sola vez.
"""
import json
import os
from pathlib import Path
from typing import Iterator

from prompts import build_prompt

_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(_REPO_ROOT))
from config import MMXU_TEST_JSONL, MIMIC_JPG_ROOT  # noqa: E402

MMXU_JSONL_PATH = Path(os.environ.get("MMXU_JSONL_PATH", MMXU_TEST_JSONL))
MMXU_IMAGE_ROOT = Path(os.environ.get("MMXU_IMAGE_ROOT", MIMIC_JPG_ROOT))


def load_samples(
    jsonl_path: Path = MMXU_JSONL_PATH,
    image_root: Path = MMXU_IMAGE_ROOT,
) -> Iterator[dict]:
    """Itera MMXU-test.jsonl y produce, por muestra:

        qid, prompt_text, image_prior_path, image_current_path, answer, content_type

    No comprueba que los ficheros de imagen existan -- eso es responsabilidad
    de quien consuma la ruta (evaluate.py decide qué hacer si falta una).
    """
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            yield {
                "qid": entry["id"],
                "prompt_text": build_prompt(entry),
                "image_prior_path": image_root / entry["prior_image_path"],
                "image_current_path": image_root / entry["cur_image_path"],
                "answer": entry["answer"],
                "content_type": entry["content_type"],
            }


if __name__ == "__main__":
    # Dry-run sin modelo: confirma que el JSONL real se parsea bien y que las
    # rutas de imagen resueltas existen en disco.
    for i, sample in enumerate(load_samples()):
        if i >= 2:
            break
        print(f"--- muestra {i} (qid={sample['qid']}) ---")
        for key in ("image_prior_path", "image_current_path"):
            path = sample[key]
            print(f"{key}: {path}  (exists={path.exists()})")
        print(f"answer={sample['answer']}  content_type={sample['content_type']}")
        print("prompt_text:")
        print(sample["prompt_text"])
        print()
