#!/usr/bin/env python3
"""Qwen25VLTrainBaseline: envuelve Qwen2.5-VL-3B-Instruct en la interfaz `VLMBaseline` de
mmxu_eval/models/base.py, igual que model_tac.py hace para el ensamblaje Libra.

A diferencia de Libra, `blind`/`mismatch`/`swap` no necesitan ningun truco: la arquitectura
acepta cualquier numero de imagenes en la lista de contenido (incluida ninguna), asi que el
predict() es casi identico al ya usado en produccion en mmxu_eval/models/qwen25vl.py -- la
unica adicion es la carga opcional de un adaptador LoRA.
"""
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "mmxu_eval"))

from models.base import VLMBaseline  # noqa: E402

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class Qwen25VLTrainBaseline(VLMBaseline):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 20,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype, device_map=self.device_map)
        if self.adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

        processor_kwargs = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)

    def predict(self, image_prior, image_current, prompt_text: str) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if image_prior is not None:
            content.append({"type": "image", "image": image_prior})
        if image_current is not None:
            content.append({"type": "image", "image": image_current})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[chat_text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.0)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
