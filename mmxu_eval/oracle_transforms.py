"""Transformaciones de imagen para el experimento oráculo (crop-to-box / gray-outside-box /
gray-inside-box).

Historial: `gray_outside_box` se llamó `mask_outside_box` hasta que se descubrió que la
documentación del experimento (condición "mask" tapando la región) describía justo lo
contrario de lo que hace el código (deja la región visible, tapa el resto). El código y su
comportamiento nunca cambiaron -- solo el nombre, para que no vuelva a pasar. Ver
../context.md, sección de hallazgos consolidados.

`gray_inside_box` es el complemento exacto de `gray_outside_box` (mismo relleno, mismo tamaño
de imagen): tapa la región y deja el resto visible -- la condición que faltaba para poder
afirmar "el modelo necesita la región", en vez de solo "la región sola no basta".

Todas operan sobre PIL.Image ya cargadas, con una bbox en píxeles (x1, y1, x2, y2) ya
recortada a los límites de la imagen (ver bbox_analysis/mmxu_bbox_table.csv, columnas
bbox_cur_*/bbox_prior_*).
"""
from PIL import Image


def crop_to_box(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Recorte exacto a la bbox, sin margen."""
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def gray_outside_box(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Condición `keep`: copia la imagen con todo excepto la bbox relleno de `fill` (gris medio
    por defecto). La región queda visible, en su posición y escala originales."""
    x1, y1, x2, y2 = bbox
    region = image.crop((x1, y1, x2, y2))
    out = Image.new("RGB", image.size, fill)
    out.paste(region, (x1, y1))
    return out


def gray_inside_box(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Condición `hole`: complemento exacto de `gray_outside_box`. Copia la imagen con la bbox
    rellena de `fill`; el resto queda visible."""
    x1, y1, x2, y2 = bbox
    out = image.copy()
    out.paste(Image.new(image.mode, (x2 - x1, y2 - y1), fill), (x1, y1))
    return out
