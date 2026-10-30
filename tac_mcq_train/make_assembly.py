#!/usr/bin/env python3
"""Build the RAD-DINO + TAC + Llama-3.2-3B-Instruct assembly that models A and B start from.

The assembly is a directory with two files from Libra's public 3B checkpoint, byte for byte:
  config.json          (X-iZhang/libra-v1.0-3b)
  mm_projector.bin     (= mm_tac_projector.bin of the same checkpoint, renamed)
Libra's loader (libra/model/builder.py::load_pretrained_model, branch without "lora" in the
name) then takes the language model from the stock meta-llama/Llama-3.2-3B-Instruct
(model_base) and injects only the TAC weights; RAD-DINO comes from microsoft/rad-dino,
named in config.json. No Libra fine-tuning of the language model is used.

The directory name must contain "libra" and not "lora": the loader dispatches on the name.

Usage: python3 tac_mcq_train/make_assembly.py   (needs `huggingface-cli login` with access to
       meta-llama/Llama-3.2-3B-Instruct for the later training/evaluation steps)
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO_ID = "X-iZhang/libra-v1.0-3b"
REVISION = "a9a3f66296682a0fe7eb9cc67a564774ed195eb8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "libra-3b-baseassembly")
    args = ap.parse_args()
    assert "libra" in args.out.name.lower() and "lora" not in args.out.name.lower(), \
        "the directory name must contain 'libra' and not 'lora' (Libra's loader dispatches on it)"
    args.out.mkdir(parents=True, exist_ok=True)
    for src, dst in (("config.json", "config.json"), ("mm_tac_projector.bin", "mm_projector.bin")):
        path = hf_hub_download(REPO_ID, src, revision=REVISION)
        shutil.copyfile(path, args.out / dst)
        print(f"{REPO_ID}@{REVISION[:7]}/{src} -> {args.out / dst}")


if __name__ == "__main__":
    main()
