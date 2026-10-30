from abc import ABC, abstractmethod

from PIL import Image


class VLMBaseline(ABC):
    """Interfaz mínima que debe implementar cualquier VLM evaluado.

    Añadir un modelo nuevo = un fichero nuevo en models/ que implemente esta
    clase + una entrada en models/__init__.py. data.py y evaluate.py no
    cambian.
    """

    @abstractmethod
    def load(self) -> None:
        """Carga el modelo/procesador en memoria. Se llama una vez antes de predict()."""

    @abstractmethod
    def predict(
        self,
        image_prior: Image.Image | None,
        image_current: Image.Image | None,
        prompt_text: str,
    ) -> str:
        """Devuelve el texto crudo generado por el modelo, sin parsear la letra.

        image_prior es Image-1 (previa) en el prompt; image_current es Image-2 (actual).
        None en cualquiera de las dos significa "sin imagen" (entrada visual a cero,
        condición "blind" del experimento oráculo) -- prompt_text no cambia.
        """
