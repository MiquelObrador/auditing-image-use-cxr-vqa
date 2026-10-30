#!/usr/bin/env python3
"""TacMcqBaseline: el ensamblaje RAD-DINO + TAC(etapa 1) + Llama-3.2-3B-Instruct, envuelto en
la interfaz `VLMBaseline` de mmxu_eval/models/base.py.

Cumplir esa interfaz permite evaluar con el binario existente `mmxu_eval/oracle_evaluate.py`
SIN modificar ni un fichero de mmxu_eval (ver run_eval.py): el mismo prompt, las mismas
condiciones, el mismo CSV.

Tres detalles del montaje que no son obvios y de los que depende que esto sea correcto:

  * `images` tiene forma [2, B, C, H, W] y el eje 0 es (ACTUAL, PREVIA) -- no es el batch.
    Verificado en run_libra.py:121-126, train.py:1460-1466 y dino_encoder.py:88-98.
    La interfaz de mmxu_eval recibe (image_prior, image_current) en ese orden: se invierte.
  * El builder despacha por el NOMBRE del modelo (`if 'libra' in model_name.lower()`), no por
    el contenido del directorio, y fuerza fp16. Se pasa el nombre explicito y se recarga el
    proyector en su bf16 original.
  * El turno del asistente sale en MAYUSCULAS por un bug de Libra (conversation.py:124); el
    chat_template real de Llama-3.2 usa minusculas. El arreglo debe ser identico aqui y en
    entrenamiento o los prompts divergen.
"""
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import LIBRA_DIR  # noqa: E402
sys.path.insert(0, str(LIBRA_DIR))
sys.path.insert(0, str(REPO / "mmxu_eval"))

from models.base import VLMBaseline  # noqa: E402

DEFAULT_ASSEMBLY = str(HERE / "libra-3b-baseassembly")
MODEL_NAME = "libra-3b-baseassembly"   # debe contener "libra" y NO "lora"
MODEL_BASE = "meta-llama/Llama-3.2-3B-Instruct"
NEUTRAL_SYSTEM = ("You are a helpful assistant. Answer multiple-choice questions with a "
                  "single letter.")


def fix_casing(prompt: str) -> str:
    return prompt.replace("<|start_header_id|>ASSISTANT<|end_header_id|>",
                          "<|start_header_id|>assistant<|end_header_id|>")


def build_conv_prompt(query: str, with_image: bool, system: str = NEUTRAL_SYSTEM) -> str:
    """Envuelve el texto de mmxu_eval en la plantilla de conversacion de Libra.

    with_image=False omite el token <image>: es la condicion `blind`. Sin eso, el -200 de
    IMAGE_TOKEN_INDEX llegaria a la tabla de embeddings con images=None.
    """
    from libra.conversation import conv_templates
    from libra.constants import DEFAULT_IMAGE_TOKEN

    qs = (DEFAULT_IMAGE_TOKEN + "\n" + query) if with_image else query
    conv = conv_templates["libra_llama_3"].copy()
    conv.system = system
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return fix_casing(conv.get_prompt())


class TacMcqBaseline(VLMBaseline):
    def __init__(
        self,
        model_name: str = DEFAULT_ASSEMBLY,
        adapter_path: str | None = None,
        tac_state_path: str | None = None,
        device_map=None,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 20,
        system_prompt: str = NEUTRAL_SYSTEM,
        # oracle_evaluate.py los inyecta incondicionalmente; este modelo no los usa (RAD-DINO
        # tiene resolucion fija 518x518), pero la firma debe aceptarlos.
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        self.assembly = model_name
        self.adapter_path = adapter_path
        self.tac_state_path = tac_state_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load(self) -> None:
        import torch
        from libra.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.assembly, MODEL_BASE, MODEL_NAME,
            device_map=self.device_map, device="cuda",
        )
        # El builder fuerza fp16 (builder.py:43). bf16 tiene 8 bits de exponente y fp16 solo 5,
        # asi que el viaje de ida y vuelta puede desbordar: se recarga el proyector del fichero.
        model = model.to(torch.bfloat16)
        raw = torch.load(Path(self.assembly) / "mm_projector.bin", map_location="cpu")
        model.load_state_dict({k: v.to(torch.bfloat16) for k, v in raw.items()}, strict=False)

        if self.tac_state_path is not None:
            # TAC entrenado (ronda de mejora acotada, ejecucion con --unfreeze-tac): no es un
            # modulo PEFT, se guarda y se carga aparte del adaptador LoRA.
            tac_raw = torch.load(self.tac_state_path, map_location="cpu")
            model.get_model().mm_projector.load_state_dict(
                {k: v.to(torch.bfloat16) for k, v in tac_raw.items()})

        if self.adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()

        model = model.cuda().eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model, self.tokenizer, self.image_processor = model, tokenizer, image_processor

    def _tensor(self, image_current, image_prior):
        """-> [2, 1, C, H, W] con eje 0 = (actual, previa)."""
        import torch
        from libra.mm_utils import process_images

        def proc(img):
            return process_images([img], self.image_processor, self.model.config)[0]

        return torch.stack([torch.stack([proc(image_current)]),
                            torch.stack([proc(image_prior)])]).to("cuda", torch.bfloat16)

    def predict(
        self,
        image_prior: Image.Image | None,
        image_current: Image.Image | None,
        prompt_text: str,
        system: str | None = None,
    ) -> str:
        """`system` solo lo usa el chequeo de retencion, para los prompts fuera de tarea: el
        system prompt de la tarea ("answer with a single letter") CONTRADICE una instruccion
        como "reply with the word BANANA", y entonces mediriamos el system prompt en vez del
        seguimiento de instrucciones. oracle_evaluate.py llama con 3 argumentos posicionales,
        asi que el contrato de VLMBaseline se mantiene."""
        import torch
        from libra.mm_utils import tokenizer_image_token
        from libra.constants import IMAGE_TOKEN_INDEX

        system = system if system is not None else self.system_prompt

        if image_current is None and image_prior is None:
            # `blind`: images=None corta toda la rama visual (libra_arch.py:149) y el LLM hace
            # un forward de solo texto. Nada de imagenes grises: seria fuera de distribucion.
            prompt = build_conv_prompt(prompt_text, with_image=False, system=system)
            input_ids = torch.tensor(
                self.tokenizer(prompt, add_special_tokens=False).input_ids
            ).unsqueeze(0).cuda()
            images = None
        else:
            if image_current is None:
                raise ValueError("no hay condicion del protocolo que pase la actual a None "
                                 "con la previa presente")
            if image_prior is None:
                # `no_prior`: modo NATIVO de Libra -- duplica la actual como previa y lo detecta
                # con un sesgo entrenado (cos ~ 1 -> LFE_prior_bias al maximo). Es comportamiento
                # dentro de distribucion, y por eso `no_prior` y `duplicate` son indistinguibles
                # en esta arquitectura (ver PREREGISTRO seccion 4.2).
                image_prior = image_current
            prompt = build_conv_prompt(prompt_text, with_image=True, system=system)
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            images = self._tensor(image_current, image_prior)

        attn = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids, images=images, attention_mask=attn,
                do_sample=False, max_new_tokens=self.max_new_tokens,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
