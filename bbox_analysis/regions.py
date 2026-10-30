"""Las 29 regiones anatómicas de Chest ImaGenome y su extracción desde texto libre.

Compartido entre analyze_region_prior.py y analyze_bbox_size.py para que ambos
análisis usen exactamente la misma definición de región.
"""

REGIONS = [
    "right apical zone", "right upper lung zone", "right mid lung zone",
    "right lower lung zone", "right hilar structures", "right costophrenic angle",
    "right hemidiaphragm", "right clavicle", "right atrium", "right lung",
    "left apical zone", "left upper lung zone", "left mid lung zone",
    "left lower lung zone", "left hilar structures", "left costophrenic angle",
    "left hemidiaphragm", "left clavicle", "left lung",
    "upper mediastinum", "mediastinum", "cardiac silhouette",
    "cavoatrial junction", "aortic arch", "carina", "trachea", "spine",
    "svc", "abdomen",
]


def extract_regions(question: str):
    """Regiones encontradas en el texto, con preferencia por la coincidencia
    más larga (evita que 'right lung' capture parte de 'right lower lung zone')."""
    q = question.lower()
    found = [r for r in REGIONS if r in q]
    return [r for r in found if not any(r != o and r in o for o in found)]
