"""InternVL2-8B (OpenGVLab), segundo VLM genérico para el protocolo de 5 condiciones.

Verificado contra el código real del repo del modelo (modeling_internvl_chat.py,
model card), no por analogía con Qwen2-VL -- la entrada multi-imagen de InternVL2 no funciona
igual:

- InternVL2 no acepta `PIL.Image` directamente: cada imagen se trocea en tiles 448x448 según
  su relación de aspecto (`dynamic_preprocess`, tantos como quepan hasta `max_num`), se
  normaliza (media/std de ImageNet) y se apilan en un tensor [n_tiles, 3, 448, 448]. Con dos
  imágenes, se concatenan los tiles de ambas y se pasa `num_patches_list = [n_tiles_img1,
  n_tiles_img2]` para que el modelo sepa dónde empieza cada imagen.
- El texto de la pregunta debe llevar un placeholder `<image>` por imagen (no tokens sueltos):
  `chat()` los sustituye internamente por la secuencia completa `<img><IMG_CONTEXT>*n</img>`
  usando `num_patches_list` -- confirmado leyendo el cuerpo de `chat()`, no asumido. La
  convención oficial del model card es anteponer `"Image-1: <image>\\nImage-2: <image>\\n"`;
  es la única desviación textual respecto al prompt de Qwen2-VL, forzada por la arquitectura.
- Condición "blind" (image_prior=None, image_current=None): se llama a
  `model.chat(tokenizer, None, prompt_text, ...)` -- vía nativa de InternVL2 para texto puro,
  sin ningún placeholder `<image>` en el prompt. Confirmado en el cuerpo de `chat()`: con
  `pixel_values=None` se salta todo el procesado visual y `num_patches_list` es irrelevante.
  Mismo criterio que Qwen2-VL: el prompt no cambia, solo se retiran las imágenes.
- `chat(..., return_history=False)` devuelve un string plano (confirmado en el código, no
  asumido) -- no hace falta desempaquetar una tupla.
- `min_pixels`/`max_pixels` (aceptados por compatibilidad con oracle_evaluate.py, que los
  inyecta incondicionalmente) no tienen equivalente en InternVL2 y se ignoran: el control de
  resolución aquí es `max_num` (tope de tiles de 448x448 por imagen), fijado igual en las
  cinco condiciones para que la única variable sea el contenido de la imagen. Como
  contrapartida, la condición "box" (recortes pequeños) generará menos tiles que "full" --
  eso es resolución efectiva, no solo información, y se mide (ver `tile_log`) y se reporta,
  no se corrige.
- `max_num_tiles=4` (no el `max_num=12` oficial): restricción de recursos, RTX 4090 de 24GB.
  Investigado a fondo si esto explicaba el hueco frente al 0.467 de la Tabla 3 del paper de
  MMXU (nuestra corrida con `max_num=4` da 0.408): se compiló `flash-attn` (2.8.3.post1, contra
  un `nvcc` 13.0 vía conda-forge que coincidiera con el `torch` del entorno -- el `nvcc` del
  sistema era 12.0 y `torch.utils.cpp_extension` rechaza compilar si no coinciden; `MAX_JOBS=4`
  porque el build por defecto agota la RAM compilando varios kernels grandes en paralelo) y se
  probó `max_num=12` con `flash_attention_2`. Resultado: 40 y 300 muestras reales sin errores,
  pero el accuracy se estabiliza en ~0.41-0.42, no escala hacia 0.467 -- la resolución explica
  como mucho ~1pp del hueco de 5.9pp, no la mayoría. Se comprobó también el prompt exacto de
  evaluación zero-shot del paper (Fig. 14 del PDF, apéndice D): idéntico carácter a carácter al
  que construye este módulo, incluido el prefijo `Image-1: <image>\\nImage-2: <image>\\n`
  derivado independientemente del código de `chat()` -- descartado como explicación. El hueco
  restante (y el de Qwen2-VL, que va en dirección CONTRARIA: +3.2pp sobre el paper, no −5.9pp)
  se documenta como probable deriva de versión de dataset/checkpoint/librerías, no verificable
  desde aquí -- ver `paper/limitations.md`. Con esto, se revierte a `max_num_tiles=4` y
  `use_flash_attn=False` (más abajo): es la configuración que produjo todos los resultados ya
  reportados, y flash-attn no cambió la conclusión lo suficiente como para justificar el riesgo
  de memoria de `max_num=12` (23.85/24.56GB de pico, validado pero sin margen) en una corrida
  de horas sin vigilancia.
"""
import os

from PIL import Image

from .base import VLMBaseline

DEFAULT_MODEL_NAME = "OpenGVLab/InternVL2-8B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Verbatim del model card de OpenGVLab/InternVL2-8B. Import de torchvision diferido
    (mismo motivo que qwen2vl.py: que importar mmxu_eval.models no falle sin torch/torchvision
    instalados)."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Verbatim del model card de OpenGVLab/InternVL2-8B."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Verbatim del model card de OpenGVLab/InternVL2-8B."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_pixel_values(image: Image.Image, transform, max_num: int, input_size: int = 448):
    """image ya cargada (PIL) -> tensor [n_tiles, 3, input_size, input_size]."""
    import torch

    tiles = dynamic_preprocess(
        image.convert("RGB"), image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(t) for t in tiles])


class InternVL2Baseline(VLMBaseline):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 20,
        max_num_tiles: int = 4,
        min_pixels: int | None = None,  # no aplica a InternVL2 (usa max_num_tiles); aceptado
        max_pixels: int | None = None,  # e ignorado porque oracle_evaluate.py lo inyecta
    ):                                   # incondicionalmente -- ver docstring del módulo.
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_num_tiles = max_num_tiles
        self.model = None
        self.tokenizer = None
        self.transform = None
        # Registro opcional de nº de tiles por llamada a predict(), en orden de fila (mismo
        # orden que load_oracle_samples()) -- para el hallazgo "nº medio de tiles por
        # condición" que pide la tarea. Activado con la variable de entorno INTERNVL_TILE_LOG
        # (ruta de fichero); si no está definida, no se registra nada. Cero cambios en
        # oracle_evaluate.py -- ver internvl_eval/ para cómo se usa.
        self._tile_log_path = os.environ.get("INTERNVL_TILE_LOG")
        self._tile_log_file = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,  # instalado y verificado (2.8.3.post1, ver docstring), pero
            trust_remote_code=True,  # se deja en False para que esta configuración sea
        ).eval().cuda()               # exactamente la que produjo oracle_results/oracle_*.csv.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        self.transform = build_transform(input_size=448)
        if self._tile_log_path:
            self._tile_log_file = open(self._tile_log_path, "a")

    def predict(
        self,
        image_prior: Image.Image | None,
        image_current: Image.Image | None,
        prompt_text: str,
    ) -> str:
        import torch

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)

        if image_prior is None and image_current is None:
            # blind: vía nativa de InternVL2 para texto puro (pixel_values=None), prompt
            # intacto, sin ningún placeholder <image>. Confirmado en el código de chat():
            # con pixel_values=None se salta el procesado visual entero.
            response = self.model.chat(
                self.tokenizer, None, prompt_text, generation_config,
                history=None, return_history=False,
            )
            num_tiles = 0
        elif image_prior is not None and image_current is not None:
            pv_prior = image_to_pixel_values(image_prior, self.transform, self.max_num_tiles)
            pv_current = image_to_pixel_values(image_current, self.transform, self.max_num_tiles)
            pixel_values = torch.cat((pv_prior, pv_current), dim=0).to(torch.bfloat16).cuda()
            num_patches_list = [pv_prior.size(0), pv_current.size(0)]
            num_tiles = sum(num_patches_list)
            # Convención oficial del model card. Única desviación textual forzada por la
            # arquitectura respecto al prompt de Qwen2-VL -- el TEMPLATE de prompts.py no
            # cambia, esto se antepone.
            question = "Image-1: <image>\nImage-2: <image>\n" + prompt_text
            response = self.model.chat(
                self.tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list, history=None, return_history=False,
            )
        else:
            raise ValueError(
                "InternVL2Baseline.predict recibió una sola imagen (la otra None) -- "
                "el protocolo del oráculo siempre pasa las dos o ninguna (blind)."
            )

        if self._tile_log_file is not None:
            self._tile_log_file.write(f"{num_tiles}\n")
            self._tile_log_file.flush()

        return response
