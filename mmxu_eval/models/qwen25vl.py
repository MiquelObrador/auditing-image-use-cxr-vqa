from PIL import Image

from .base import VLMBaseline

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen25VLBaseline(VLMBaseline):
    """Qwen2.5-VL-7B-Instruct genérico (sin PubMedVision) -- la base exacta de
    HuatuoGPT-Vision-7B-Qwen2.5VL. Ablación: mismo checkpoint base, misma arquitectura, única
    diferencia el entrenamiento médico -- aísla ese factor de la generación de arquitectura
    (Qwen2-VL vs. Qwen2.5-VL) que la comparación con Qwen2-VL-7B-Instruct no aislaba (ver
    paper/limitations.md).

    Idéntico en mecanismo a HuatuoVisionBaseline/Qwen2VLBaseline -- mismo
    Qwen2VLImageProcessor, mismo min_pixels/max_pixels, mismo process_vision_info.
    `repetition_penalty` fijado a 1.0 (el `generation_config.json` publicado trae 1.05, igual
    que el checkpoint de HuatuoGPT) para que la única variable entre este modelo y HuatuoGPT
    sea el entrenamiento, no la configuración de decodificación."""

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
            content.append({"type": "image", "image": image_prior})
        if image_current is not None:
            content.append({"type": "image", "image": image_current})
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
