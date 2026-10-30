from .base import VLMBaseline
from .qwen2vl import Qwen2VLBaseline
from .internvl2 import InternVL2Baseline
from .huatuo import HuatuoVisionBaseline
from .qwen25vl import Qwen25VLBaseline

MODEL_REGISTRY = {
    "qwen2vl": Qwen2VLBaseline,
    "internvl2": InternVL2Baseline,
    "huatuo": HuatuoVisionBaseline,
    "qwen25vl": Qwen25VLBaseline,
}


def get_model(name: str, **kwargs) -> VLMBaseline:
    """Instancia (sin cargar) el modelo registrado bajo `name`.

    Para añadir un modelo nuevo: crear models/<nombre>.py con una clase que
    implemente VLMBaseline, importarla arriba y añadir una entrada aquí.
    data.py y evaluate.py no cambian.
    """
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Modelo desconocido: '{name}'. Disponibles: {sorted(MODEL_REGISTRY)}"
        ) from None
    return cls(**kwargs)
