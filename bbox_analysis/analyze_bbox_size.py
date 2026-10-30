#!/usr/bin/env python3
"""
Caracteriza el tamaño de las regiones anotadas en MMXU-test.

Pregunta: ¿son las cajas lo bastante pequeñas como para que la poda de tokens
visuales pueda marcar la diferencia? Se mide proyectando cada bbox sobre la
rejilla 12x12 de la etapa 4 de Swin-base.

MMXU-test.jsonl no trae bboxes; se recuperan de Chest ImaGenome
(ImaGenome/silver_dataset/scene_graph.zip), que sí cubre las 4563 imágenes
únicas de MMXU-test. Las dimensiones reales de imagen (necesarias para pasar
de píxeles a fracción de imagen) se leen de mimic-cxr-2.0.0-metadata.csv.gz
(columnas Rows/Columns de la cabecera DICOM original), que cubre 4563/4563
dicoms -- no hace falta el píxel de la imagen, solo sus dimensiones.

Se descartó una vía anterior (leer W,H con PIL de los JPG locales en
la copia local de MIMIC-CXR-JPG): cubría solo 4058/4563 dicoms, y una inspección
encontró que 37.6% de esas imágenes son landscape (W>H), inusual para un CXR
frontal -- se sospechó un problema de rotación de archivo. Verificado y
descartado: PIL coincide con Rows/Columns oficiales en el 100% de las 4058
imágenes comprobadas. El landscape es real (portátiles AP de UCI), no un bug
de almacenamiento. Se usa igualmente el metadata.csv porque da cobertura
completa sin depender de los JPG en disco.

Trampa de coordenadas evitada: cada objeto del scene graph trae la caja en dos
espacios, uno "224" y otro "original". El espacio 224 NO llena el lienzo 224x224
(es un resize que preserva aspecto). Se usan siempre las coordenadas
original_x1..y2, normalizadas por el W,H real (Rows/Columns de la cabecera DICOM).
"""
import json
import os
import csv
import gzip
import zipfile
import collections
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regions import extract_regions

# Ancla el cwd al directorio del script: los literales de salida de abajo
# (mmxu_bbox_table.csv, fig_*.png, bbox_size_report.md) siguen siendo rutas
# relativas simples, pero ahora aterrizan aquí sin importar desde dónde se
# invoque el script.
os.chdir(Path(__file__).resolve().parent)

# Datos de entrada compartidos, viven en la raíz del repo (un nivel arriba).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MMXU_TEST_JSONL, IMAGENOME_DIR, MIMIC_METADATA_CSV  # noqa: E402
DATA_PATH = str(MMXU_TEST_JSONL)
SCENE_GRAPH_ZIP = str(IMAGENOME_DIR / "silver_dataset" / "scene_graph.zip")
IMAGES_TO_AVOID_CSV = str(IMAGENOME_DIR / "silver_dataset" / "splits" / "images_to_avoid.csv")
METADATA_CSV_GZ = str(MIMIC_METADATA_CSV)
GRID = 12
N_CELLS = GRID * GRID
BUDGETS = {"keep-50%": 72, "keep-25%": 36, "keep-10%": 14}

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

records = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
n = len(records)
for r in records:
    regs = extract_regions(r["question"])
    r["_region"] = regs[0] if regs else None

avoid_dicoms = set()
with open(IMAGES_TO_AVOID_CSV) as f:
    for row in csv.DictReader(f):
        avoid_dicoms.add(row["dicom_id"])

zf = zipfile.ZipFile(SCENE_GRAPH_ZIP)


def dicom_of(path):
    return path.split("/")[-1].replace(".jpg", "")


scene_graph_cache = {}


def get_scene_graph_regions(dicom):
    """dict bbox_name -> (x1,y1,x2,y2) en píxeles originales, o {} si no existe."""
    if dicom in scene_graph_cache:
        return scene_graph_cache[dicom]
    try:
        with zf.open(f"scene_graph/{dicom}_SceneGraph.json") as f:
            sg = json.load(f)
        out = {
            o["bbox_name"]: (o["original_x1"], o["original_y1"], o["original_x2"], o["original_y2"])
            for o in sg["objects"]
        }
    except KeyError:
        out = {}
    scene_graph_cache[dicom] = out
    return out


dims_by_dicom = {}
with gzip.open(METADATA_CSV_GZ, "rt") as f:
    for row in csv.DictReader(f):
        dims_by_dicom[row["dicom_id"]] = (int(row["Columns"]), int(row["Rows"]))  # (W, H)


def get_dims(dicom):
    return dims_by_dicom.get(dicom)  # (W, H) o None


# ---------------------------------------------------------------------------
# 1. Construcción de la tabla por muestra + verificación de estructura
# ---------------------------------------------------------------------------

rows = []
reasons = collections.Counter()
for r in records:
    region = r["_region"]
    if region is None:
        reasons["sin región reconocida"] += 1
        continue

    cur_path, prior_path = r["cur_image_path"], r["prior_image_path"]
    cur_dicom, prior_dicom = dicom_of(cur_path), dicom_of(prior_path)

    sg_cur = get_scene_graph_regions(cur_dicom)
    sg_prior = get_scene_graph_regions(prior_dicom)

    if region not in sg_cur or region not in sg_prior:
        reasons["región ausente en el scene graph de alguna imagen"] += 1
        continue

    dims_cur, dims_prior = get_dims(cur_dicom), get_dims(prior_dicom)
    if dims_cur is None or dims_prior is None:
        reasons["dimensiones no encontradas en metadata.csv"] += 1
        continue

    bbox_cur, bbox_prior = sg_cur[region], sg_prior[region]
    W_cur, H_cur = dims_cur
    W_prior, H_prior = dims_prior

    # sanidad: la caja no puede salirse de la imagen
    x1, y1, x2, y2 = bbox_cur
    assert x2 > x1 and y2 > y1, f"caja degenerada en {cur_dicom}"
    x1, y1, x2, y2 = bbox_prior
    assert x2 > x1 and y2 > y1, f"caja degenerada en {prior_dicom}"

    rows.append(dict(
        id=r["id"], region=region, content_type=r["content_type"],
        W_cur=W_cur, H_cur=H_cur, bbox_cur=bbox_cur,
        W_prior=W_prior, H_prior=H_prior, bbox_prior=bbox_prior,
        avoid=(cur_dicom in avoid_dicoms or prior_dicom in avoid_dicoms),
    ))

print("=== 1. Verificación de estructura ===")
print(f"n MMXU-test = {n}")
print(f"muestras analizables (región + ambas cajas + ambas imágenes locales): {len(rows)}")
for k, v in reasons.most_common():
    print(f"  excluidas por: {k}: {v}")
n_avoid = sum(1 for row in rows if row["avoid"])
print(f"de las analizables, en images_to_avoid.csv de ImaGenome: {n_avoid} (se marcan, no se descartan)")

# Overflow de la caja fuera del marco de la imagen: no es redondeo de píxel.
# Se ha visto overflow de hasta ~240px (~8-9% de W/H) en regiones cercanas al
# borde (spine, costophrenic angle, lower lung zone) -- probablemente el
# pipeline automático de bboxes de ImaGenome corrió sobre una versión de
# imagen ligeramente distinta a los JPG locales. Se cuantifica y se recorta
# la caja a los límites de la imagen antes de área/rejilla (no se descarta la
# muestra: el recorte es la interpretación conservadora -- "lo que cabe en la
# imagen real").
oob = 0
oob_px = []
offframe = []  # (id, pfx, region, bbox, W, H) -- caja que tras recortar no deja área
kept_rows = []
for row in rows:
    degenerate = False
    for pfx in ("cur", "prior"):
        x1, y1, x2, y2 = row[f"bbox_{pfx}"]
        W, H = row[f"W_{pfx}"], row[f"H_{pfx}"]
        overflow = max(0, x2 - W, y2 - H)
        if overflow > 0:
            oob += 1
            oob_px.append(overflow)
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(x2, W), min(y2, H)
        if cx2 <= cx1 or cy2 <= cy1:
            # la caja original no solapa la imagen en absoluto -- no es
            # redondeo, es un fallo del detector automático (visto en
            # 'right costophrenic angle' con x1,x2 ambos negativos: la caja
            # cayó fuera del lado equivocado del cuerpo).
            offframe.append((row["id"], pfx, row["region"], (x1, y1, x2, y2), W, H))
            degenerate = True
        row[f"bbox_{pfx}"] = (cx1, cy1, cx2, cy2)
    if not degenerate:
        kept_rows.append(row)

print(f"cajas que exceden los límites de la imagen: {oob}/{2*len(rows)} "
      f"(overflow máx={max(oob_px) if oob_px else 0}px, mediana={np.median(oob_px) if oob_px else 0:.1f}px) "
      f"-- recortadas a los límites de la imagen antes de calcular área/rejilla")
print(f"cajas completamente fuera del marco (fallo del detector automático, excluidas): {len(offframe)}")
for oid, pfx, region, bbox, W, H in offframe:
    print(f"   id={oid} ({pfx}) region={region} bbox={bbox} W,H=({W},{H})")

rows = kept_rows
assert len(rows) >= 2500, "el conjunto analizable es mucho menor de lo esperado (~2537)"

# ---------------------------------------------------------------------------
# 2-3. Área relativa y cobertura en rejilla 12x12
# ---------------------------------------------------------------------------

def cells_overlap(x1n, y1n, x2n, y2n, grid=GRID):
    """nº de celdas que la caja toca aunque sea parcialmente (rectángulo contiguo)."""
    c0 = max(0, min(grid - 1, math.floor(x1n * grid)))
    c1 = max(0, min(grid - 1, math.ceil(x2n * grid) - 1))
    r0 = max(0, min(grid - 1, math.floor(y1n * grid)))
    r1 = max(0, min(grid - 1, math.ceil(y2n * grid) - 1))
    return (c1 - c0 + 1) * (r1 - r0 + 1)


def cells_center(x1n, y1n, x2n, y2n, grid=GRID):
    """nº de celdas cuyo CENTRO cae dentro de la caja (criterio estricto)."""
    def count_1d(a, b):
        i_min = math.ceil(a * grid - 0.5)
        i_max = math.floor(b * grid - 0.5)
        i_min = max(0, i_min)
        i_max = min(grid - 1, i_max)
        return max(0, i_max - i_min + 1)
    return count_1d(x1n, x2n) * count_1d(y1n, y2n)


for row in rows:
    x1, y1, x2, y2 = row["bbox_cur"]
    W, H = row["W_cur"], row["H_cur"]
    row["area_rel_cur"] = ((x2 - x1) * (y2 - y1)) / (W * H)
    x1n, y1n, x2n, y2n = x1 / W, y1 / H, x2 / W, y2 / H
    row["cells_overlap_cur"] = cells_overlap(x1n, y1n, x2n, y2n)
    row["cells_center_cur"] = cells_center(x1n, y1n, x2n, y2n)

    x1, y1, x2, y2 = row["bbox_prior"]
    W, H = row["W_prior"], row["H_prior"]
    row["area_rel_prior"] = ((x2 - x1) * (y2 - y1)) / (W * H)
    x1n, y1n, x2n, y2n = x1 / W, y1 / H, x2 / W, y2 / H
    row["cells_overlap_prior"] = cells_overlap(x1n, y1n, x2n, y2n)
    row["cells_center_prior"] = cells_center(x1n, y1n, x2n, y2n)

    row["cells_max"] = max(row["cells_overlap_cur"], row["cells_overlap_prior"])

    for k in ("area_rel_cur", "area_rel_prior"):
        assert 0 < row[k] <= 1, f"área relativa fuera de (0,1]: {row[k]}"
    for k in ("cells_overlap_cur", "cells_overlap_prior"):
        assert 1 <= row[k] <= N_CELLS, f"celdas fuera de [1,144]: {row[k]}"

area_all = np.array([row["area_rel_cur"] for row in rows] + [row["area_rel_prior"] for row in rows])
cells_overlap_all = np.array([row["cells_overlap_cur"] for row in rows] + [row["cells_overlap_prior"] for row in rows])
cells_center_all = np.array([row["cells_center_cur"] for row in rows] + [row["cells_center_prior"] for row in rows])
cells_max = np.array([row["cells_max"] for row in rows])

print("\n=== 2. Área relativa de la bbox (combinando cur+prior, n={}) ===".format(len(area_all)))
print(f"mediana={np.median(area_all):.4f}  Q1={np.percentile(area_all,25):.4f}  Q3={np.percentile(area_all,75):.4f}")

print("\n=== 3. Cobertura en rejilla 12x12 (144 celdas) ===")
print(f"[solapamiento, primario] mediana={np.median(cells_overlap_all):.1f}  "
      f"Q1={np.percentile(cells_overlap_all,25):.1f}  Q3={np.percentile(cells_overlap_all,75):.1f}")
print(f"[centro, estricto]       mediana={np.median(cells_center_all):.1f}  "
      f"Q1={np.percentile(cells_center_all,25):.1f}  Q3={np.percentile(cells_center_all,75):.1f}  "
      f"(fracción con 0 celdas-centro: {(cells_center_all==0).mean():.3f})")
print(f"[max(cur,prior), por muestra, n={len(cells_max)}] mediana={np.median(cells_max):.1f}")

budget_fracs = {}
for name, k in BUDGETS.items():
    frac_per_image = (cells_overlap_all <= k).mean()
    frac_per_sample = (cells_max <= k).mean()
    budget_fracs[name] = (k, frac_per_image, frac_per_sample)
    print(f"  {name} ({k} celdas): {frac_per_image:.3f} de las imágenes caben; "
          f"{frac_per_sample:.3f} de las muestras (peor de las dos imágenes) caben")

# ---------------------------------------------------------------------------
# Sanidad cruzada: regiones grandes deben salir grandes, pequeñas pequeñas
# ---------------------------------------------------------------------------

by_region_cells = collections.defaultdict(list)
for row in rows:
    by_region_cells[row["region"]].append(row["cells_overlap_cur"])
    by_region_cells[row["region"]].append(row["cells_overlap_prior"])

med_lung = np.median(by_region_cells.get("right lung", [0])) if "right lung" in by_region_cells else None
med_carina = np.median(by_region_cells["carina"]) if "carina" in by_region_cells else None
print(f"\nsanidad cruzada: mediana celdas right lung={med_lung}  "
      f"carina={med_carina}  right clavicle={np.median(by_region_cells.get('right clavicle',[float('nan')]))}")
if med_lung is not None and med_carina is not None:
    assert med_lung > med_carina, "BUG: right lung debería tocar más celdas que carina"

# ---------------------------------------------------------------------------
# 4. Desglose por región
# ---------------------------------------------------------------------------

region_rows = []
for region, cells in by_region_cells.items():
    n_img = len(cells)
    region_rows.append((region, n_img // 2, np.median(cells), np.percentile(cells, 25), np.percentile(cells, 75)))
region_rows.sort(key=lambda x: -x[2])

print("\n=== 4. Desglose por región (ordenado por mediana de celdas, desc.) ===")
print(f"{'región':26s} {'n':>5s} {'mediana':>8s} {'Q1':>6s} {'Q3':>6s}")
for region, n_muestras, med, q1, q3 in region_rows:
    print(f"{region:26s} {n_muestras:5d} {med:8.1f} {q1:6.1f} {q3:6.1f}")

# ---------------------------------------------------------------------------
# 5. Cruce con content_type
# ---------------------------------------------------------------------------

by_ct = collections.defaultdict(list)
for row in rows:
    by_ct[row["content_type"]].append(row["cells_max"])

print("\n=== 5. Cruce con content_type (mediana de cells_max por clase) ===")
for ct, vals in sorted(by_ct.items()):
    print(f"  {ct:12s} n={len(vals):4d}  mediana={np.median(vals):.1f}")

# regiones "pequeñas" (mediana cells_max <= 36) — reparto por clase
small_regions = {r for r, _, med, _, _ in region_rows if med <= 36}
ct_in_small = collections.Counter(row["content_type"] for row in rows if row["region"] in small_regions)
ct_all = collections.Counter(row["content_type"] for row in rows)
print(f"\nregiones con mediana <=36 celdas: {sorted(small_regions)}")
print(f"reparto por clase EN esas regiones: {dict(ct_in_small)}  (total: {dict(ct_all)})")

# ---------------------------------------------------------------------------
# 6. Propuesta de subconjunto
# ---------------------------------------------------------------------------

THRESHOLD = 36  # keep-25%, ver justificación en el informe
subset = [row for row in rows if row["cells_max"] <= THRESHOLD]
subset_ct = collections.Counter(row["content_type"] for row in subset)
subset_region = collections.Counter(row["region"] for row in subset)

print(f"\n=== 6. Propuesta de subconjunto: cells_max <= {THRESHOLD} ===")
print(f"muestras: {len(subset)}/{len(rows)} ({len(subset)/len(rows):.1%})")
print(f"reparto por clase: {dict(subset_ct)}")
print("reparto por región (top 10):")
for region, c in subset_region.most_common(10):
    print(f"   {c:5d}  {region}")

# ---------------------------------------------------------------------------
# CSV de la tabla por muestra
# ---------------------------------------------------------------------------

with open("mmxu_bbox_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "id", "region", "content_type",
        "area_rel_cur", "area_rel_prior",
        "cells_overlap_cur", "cells_overlap_prior", "cells_max",
        "cells_center_cur", "cells_center_prior",
        "in_images_to_avoid",
        "bbox_cur_x1", "bbox_cur_y1", "bbox_cur_x2", "bbox_cur_y2", "W_cur", "H_cur",
        "bbox_prior_x1", "bbox_prior_y1", "bbox_prior_x2", "bbox_prior_y2", "W_prior", "H_prior",
    ])
    for row in rows:
        bx1c, by1c, bx2c, by2c = row["bbox_cur"]
        bx1p, by1p, bx2p, by2p = row["bbox_prior"]
        w.writerow([
            row["id"], row["region"], row["content_type"],
            f"{row['area_rel_cur']:.5f}", f"{row['area_rel_prior']:.5f}",
            row["cells_overlap_cur"], row["cells_overlap_prior"], row["cells_max"],
            row["cells_center_cur"], row["cells_center_prior"],
            row["avoid"],
            bx1c, by1c, bx2c, by2c, row["W_cur"], row["H_cur"],
            bx1p, by1p, bx2p, by2p, row["W_prior"], row["H_prior"],
        ])
print("\nTabla por muestra guardada en mmxu_bbox_table.csv")

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(area_all, bins=30, edgecolor="white", color="#4C72B0")
axes[0].set_xlabel("área relativa de la bbox (área caja / área imagen)")
axes[0].set_ylabel("nº de imágenes (cur+prior)")
axes[0].set_title("Área relativa")

axes[1].hist(cells_overlap_all, bins=np.arange(0, N_CELLS + 4, 4), edgecolor="white", color="#4C72B0")
for name, (k, _, _) in budget_fracs.items():
    axes[1].axvline(k, linestyle="--", color="#C44E52", alpha=0.7)
    axes[1].text(k, axes[1].get_ylim()[1] * 0.9, name, rotation=90, fontsize=8, va="top")
axes[1].set_xlabel("celdas tocadas (rejilla 12x12 = 144)")
axes[1].set_title("Cobertura en rejilla (solapamiento)")
fig.tight_layout()
fig.savefig("fig_bbox_distributions.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 8))
labels = [r for r, *_ in region_rows]
meds = [m for _, _, m, _, _ in region_rows]
q1s = [q1 for _, _, _, q1, _ in region_rows]
q3s = [q3 for _, _, _, _, q3 in region_rows]
y = np.arange(len(labels))
ax.barh(y, meds, xerr=[np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)], color="#4C72B0", ecolor="gray", capsize=2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("celdas tocadas (mediana, IQR)")
ax.set_title("Cobertura en rejilla por región")
fig.tight_layout()
fig.savefig("fig_region_cells.png", dpi=150)
plt.close(fig)

print("Figuras guardadas: fig_bbox_distributions.png, fig_region_cells.png")

# ---------------------------------------------------------------------------
# Informe markdown
# ---------------------------------------------------------------------------

with open("bbox_size_report.md", "w") as f:
    f.write("# Tamaño de las regiones anotadas en MMXU-test\n\n")
    f.write(
        "¿Son las cajas lo bastante pequeñas como para que la poda de tokens visuales pueda marcar "
        "la diferencia? Se proyecta cada bbox sobre la rejilla 12×12 (144 celdas) de la etapa 4 de "
        "Swin-base. `MMXU-test.jsonl` no trae bboxes; se recuperan de Chest ImaGenome "
        "(`ImaGenome/silver_dataset/scene_graph.zip`).\n\n"
    )

    f.write("## 1. Estructura y cobertura\n\n")
    f.write(f"- n = {n} muestras en MMXU-test.\n")
    f.write(f"- **{len(rows) + len(offframe)}/{n}** analizables tras emparejar región + ambas cajas en los "
            f"scene graphs de ImaGenome (excluidas {reasons['sin región reconocida'] + reasons['región ausente en el scene graph de alguna imagen']}: "
            f"{dict(reasons)}).\n")
    f.write(f"- Dimensiones de imagen (W,H) desde `mimic-cxr-2.0.0-metadata.csv.gz` oficial "
            f"(`Rows`/`Columns` de la cabecera DICOM) — cobertura 4563/4563 dicoms, no depende de tener "
            f"el JPG en disco.\n")
    f.write(
        f"- **{oob}/{2*(len(rows)+len(offframe))}** cajas exceden el marco de la imagen en algún borde "
        f"(overflow máx={max(oob_px) if oob_px else 0}px, mediana={np.median(oob_px) if oob_px else 0:.0f}px) "
        f"— probable artefacto del pipeline automático de detección de ImaGenome, no de las imágenes "
        f"locales (ver caveat de orientación abajo). Se recortan al marco de la imagen antes de calcular "
        f"área/rejilla.\n"
    )
    f.write(
        f"- **{len(offframe)} cajas caen completamente fuera del marco** (no es recorte, es un fallo del "
        f"detector: en 5/6 casos es `right costophrenic angle` con coordenadas x negativas — probable "
        f"confusión de lateralidad del pipeline automático). Excluidas de área/rejilla, id listados en la "
        f"salida del script.\n"
    )
    f.write(f"- **n final analizable: {len(rows)}/{n}**, reparto por clase: "
            f"{dict(collections.Counter(row['content_type'] for row in rows))}.\n")
    f.write(f"- {sum(1 for row in rows if row['avoid'])} de las analizables están en `images_to_avoid.csv` "
            f"(lista de exclusión por calidad de ImaGenome) — marcadas en la tabla CSV, no descartadas.\n\n")

    f.write(
        "**Caveat de orientación (investigado y descartado como problema).** Un chequeo inicial usando "
        "PIL sobre los JPG de la copia local de MIMIC-CXR-JPG encontró que 37.6% de las imágenes eran "
        "landscape (W>H), inusual para un CXR frontal. Se verificó contra `Rows`/`Columns` oficiales: "
        "PIL coincide al 100% (4058/4058 imágenes comprobadas) — el landscape es real (portátiles AP de "
        "UCI, no bug de rotación). Se usa el metadata.csv de todos modos porque da cobertura completa "
        "sin depender de los JPG en disco.\n\n"
    )

    f.write("## 2. Área relativa de la bbox\n\n")
    f.write(f"Combinando imagen actual y previa (n={len(area_all)}): "
            f"mediana={np.median(area_all):.3f}, Q1={np.percentile(area_all,25):.3f}, "
            f"Q3={np.percentile(area_all,75):.3f}.\n\n")
    f.write("![área relativa](fig_bbox_distributions.png)\n\n")

    f.write("## 3. Cobertura en rejilla 12×12 (144 celdas) — el número decisivo\n\n")
    f.write(
        f"- **Solapamiento (primario, celda cuenta si la toca aunque sea parcial)**: mediana="
        f"{np.median(cells_overlap_all):.0f}, Q1={np.percentile(cells_overlap_all,25):.0f}, "
        f"Q3={np.percentile(cells_overlap_all,75):.0f} celdas de 144.\n"
        f"- **Centro (secundario, estricto, celda cuenta si su centro cae dentro)**: mediana="
        f"{np.median(cells_center_all):.0f}, fracción con 0 celdas-centro="
        f"{(cells_center_all==0).mean():.3f}.\n"
        f"- **Por muestra (max de las dos imágenes, n={len(cells_max)})**: mediana={np.median(cells_max):.0f}.\n\n"
    )
    f.write("Presupuestos de poda — fracción de imágenes / muestras que caben dentro del presupuesto:\n\n")
    f.write("| presupuesto | celdas | frac. imágenes (cur+prior) | frac. muestras (peor de las 2) |\n|---|---|---|---|\n")
    for nombre, (k, fi, fs) in budget_fracs.items():
        f.write(f"| {nombre} | {k} | {fi:.3f} | {fs:.3f} |\n")
    f.write("\n![cobertura en rejilla](fig_bbox_distributions.png)\n\n")
    f.write(
        f"**Lectura:** con la mediana en {np.median(cells_overlap_all):.0f}/144 (25% del área), la región "
        f"típica ya ocupa una cuarta parte de la rejilla — el keep-25% (36 celdas) deja fuera a la mitad "
        f"de las muestras ({1-budget_fracs['keep-25%'][2]:.1%} no caben). No es el escenario "
        f"'toda la imagen es la región' de `right lung`/`left lung`, pero tampoco el escenario "
        f"'diminuta como carina' para la mayoría — hay heterogeneidad real que el desglose de abajo "
        f"expone.\n\n"
    )

    f.write("## 4. Desglose por región\n\n")
    f.write("Ordenado por mediana de celdas tocadas (solapamiento), de mayor a menor.\n\n")
    f.write("| región | n | mediana | Q1 | Q3 |\n|---|---|---|---|---|\n")
    for region, n_muestras, med, q1, q3 in region_rows:
        f.write(f"| {region} | {n_muestras} | {med:.0f} | {q1:.0f} | {q3:.0f} |\n")
    f.write("\n![celdas por región](fig_region_cells.png)\n\n")

    f.write("## 5. Cruce con content_type\n\n")
    f.write("| content_type | n | mediana cells_max |\n|---|---|---|\n")
    for ct, vals in sorted(by_ct.items()):
        f.write(f"| {ct} | {len(vals)} | {np.median(vals):.0f} |\n")
    f.write(
        f"\nRegiones con mediana ≤36 celdas (`{', '.join(sorted(small_regions))}`) reparten "
        f"`content_type` como {dict(ct_in_small)} — comparado con el total {dict(ct_all)}. "
        f"El análisis previo (`region_prior_report.md`) ya mostró acoplamiento fuerte región↔clase "
        f"(p. ej. `cardiac silhouette` 78% `no change`); aquí se confirma que ese acoplamiento **no** "
        f"se corrige solo: las regiones pequeñas siguen sobrerrepresentando `no change` "
        f"({ct_in_small['no change']}/{sum(ct_in_small.values())} = "
        f"{ct_in_small['no change']/sum(ct_in_small.values()):.1%} vs "
        f"{ct_all['no change']/sum(ct_all.values()):.1%} en el total). Cualquier subconjunto por tamaño "
        f"hereda parte de ese sesgo; hay que controlarlo en el experimento siguiente, no ignorarlo.\n\n"
    )

    f.write("## 6. Propuesta de subconjunto\n\n")
    f.write(
        f"**Criterio: `cells_max ≤ {THRESHOLD}`** (el presupuesto keep-25%, el más relevante para el "
        f"experimento de tres barras descrito en `context.md`). `cells_max` = la peor de las dos "
        f"imágenes (actual/previa), porque el oráculo necesita cubrir la región en ambas.\n\n"
        f"- **{len(subset)}/{len(rows)} muestras ({len(subset)/len(rows):.1%})**.\n"
        f"- Reparto por clase: {dict(subset_ct)} — "
        f"{'razonablemente equilibrado' if max(subset_ct.values())/min(subset_ct.values()) < 1.5 else 'con sesgo apreciable, ver arriba'}.\n"
        f"- Dominado por `cardiac silhouette` ({subset_region['cardiac silhouette']}), los costophrenic "
        f"angles y las lung zones inferiores — coherente con el desglose de la sección 4.\n\n"
    )
    f.write("| región | n en el subconjunto |\n|---|---|\n")
    for region, c in subset_region.most_common():
        f.write(f"| {region} | {c} |\n")
    f.write(
        f"\n**Nota:** el umbral 36 es el presupuesto keep-25% ya definido en `context.md` para el "
        f"experimento siguiente, no un ajuste post-hoc a la distribución. Con {len(subset)} muestras y "
        f"tres clases razonablemente representadas, es un tamaño viable para el experimento de tres "
        f"barras (Blind / Random-25% / Oracle-25%).\n\n"
    )

    f.write("## Entregables\n\n")
    f.write(
        "- `mmxu_bbox_table.csv`: una fila por muestra (área relativa, celdas por criterio, región, "
        "clase, flag de `images_to_avoid`) — reutilizable directamente para construir el subconjunto y "
        "el experimento de oráculo, sin recomputar.\n"
        "- `fig_bbox_distributions.png`, `fig_region_cells.png`.\n\n"
    )

    f.write("## Fuera de alcance (anotado, no ejecutado)\n\n")
    f.write(
        "- **Desalineación entre imagen previa y actual.** Se podría medir el desplazamiento del "
        "centroide de la misma región entre las dos imágenes, en celdas de rejilla — los \"~3 celdas\" "
        "que menciona `context.md`. Los datos ya están en `mmxu_bbox_table.csv` (bbox_cur/bbox_prior "
        "no se exportan por separado, pero el script los tiene disponibles); no está en el alcance de "
        "esta tarea, queda anotado para decidir después.\n"
        "- Cargar modelos, entrenar, implementar poda, pipeline de evaluación.\n"
    )

print("Informe guardado en bbox_size_report.md")
