"""Control `hole_random` para MMXU-test -- copia verbatim de
`biovilt_eval/hole_random_transforms.py` (no se importa entre fases del proyecto por
convención; ver ese fichero para el razonamiento completo). Tapa un parche del MISMO tamaño
que la bbox real, pero en la posición anatómica homóloga del lado contrario (reflejo
especular horizontal) para regiones con contraparte izquierda/derecha, o desplazado
verticalmente para regiones de la línea media -- en vez de la región preguntada. Si tapar el
lado contrario cuesta lo mismo que tapar la región real, la posición dentro de la imagen no
importa (choque de distribución del encoder al ver un parche gris), no la información de esa
región concreta.

PAIRED_REGIONS / UNPAIRED_REGIONS: las 24 regiones que aparecen en
`bbox_analysis/mmxu_bbox_table.csv` (subconjunto de las 29 de Chest ImaGenome; verificado por
conteo) son todas subconjunto de las 29 ya clasificadas para ImaGenome/BioViL-T -- mismas
listas, sin cambios.
"""

PAIRED_REGIONS = {
    "left apical zone", "right apical zone",
    "left clavicle", "right clavicle",
    "left costophrenic angle", "right costophrenic angle",
    "left hemidiaphragm", "right hemidiaphragm",
    "left hilar structures", "right hilar structures",
    "left lower lung zone", "right lower lung zone",
    "left lung", "right lung",
    "left mid lung zone", "right mid lung zone",
    "left upper lung zone", "right upper lung zone",
}

UNPAIRED_REGIONS = {
    "abdomen", "aortic arch", "cardiac silhouette", "carina", "cavoatrial junction",
    "mediastinum", "right atrium", "spine", "svc", "trachea", "upper mediastinum",
}


def mirror_bbox(bbox: tuple[float, float, float, float], width: float) -> tuple[float, float, float, float]:
    """Reflejo horizontal de la bbox respecto al eje vertical de la imagen -- mismo tamaño,
    lado contrario, misma altura (y1,y2 sin cambiar)."""
    x1, y1, x2, y2 = bbox
    return (width - x2, y1, width - x1, y2)


def vertical_shift_bbox(bbox: tuple[float, float, float, float], height: float) -> tuple[float, float, float, float]:
    """Desplaza la bbox verticalmente su propia altura -- hacia abajo si cabe en la imagen,
    si no hacia arriba. Fallback para regiones sin contraparte izquierda/derecha."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    if y2 + h <= height:
        return (x1, y1 + h, x2, y2 + h)
    return (x1, max(0.0, y1 - h), x2, max(h, y2 - h))


def hole_random_bbox(
    region: str,
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Bbox de control para `hole_random`: mismo tamaño que `bbox`, en la posición homóloga
    del lado contrario (regiones pareadas) o desplazada verticalmente (regiones sin pareja).
    Coordenadas devueltas como enteros (píxeles), ya recortadas a los límites de la imagen."""
    width, height = image_size
    if region in PAIRED_REGIONS:
        x1, y1, x2, y2 = mirror_bbox(bbox, width)
    elif region in UNPAIRED_REGIONS:
        x1, y1, x2, y2 = vertical_shift_bbox(bbox, height)
    else:
        raise ValueError(f"región no clasificada como pareada/no-pareada: {region!r}")
    x1 = max(0, min(int(round(x1)), width))
    x2 = max(0, min(int(round(x2)), width))
    y1 = max(0, min(int(round(y1)), height))
    y2 = max(0, min(int(round(y2)), height))
    return (x1, y1, x2, y2)
