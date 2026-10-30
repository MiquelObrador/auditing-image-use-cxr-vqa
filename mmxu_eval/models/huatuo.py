from PIL import Image

from .base import VLMBaseline

DEFAULT_MODEL_NAME = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"


class HuatuoVisionBaseline(VLMBaseline):
    """HuatuoGPT-Vision-7B, variante entrenada sobre la arquitectura Qwen2.5-VL (PubMedVision
    encima de Qwen/Qwen2.5-VL-7B-Instruct, sin trust_remote_code). Idéntica en mecanismo a
    Qwen2VLBaseline -- mismo Qwen2VLImageProcessor, mismo min_pixels/max_pixels, mismo
    process_vision_info -- así que la comparación con Qwen2-VL es de modelo, no de resolución.

    Única diferencia real: repetition_penalty. El generation_config.json publicado con este
    checkpoint trae 1.05 (el de Qwen2-VL-7B-Instruct trae 1.0); se fija a 1.0 explícitamente
    para que no sea una variable de decodificación distinta entre los dos modelos."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 20,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self) -> None:
        # Imports diferidos: que importar mmxu_eval.models no falle si
        # transformers/torch todavía no están instalados o están rotos.
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        processor_kwargs = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)

    def predict(
        self,
        image_prior: Image.Image | None,
        image_current: Image.Image | None,
        prompt_text: str,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if image_prior is not None:
            content.append({"type": "image", "image": image_prior})    # Image-1: previa
        if image_current is not None:
            content.append({"type": "image", "image": image_current})  # Image-2: actual
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
