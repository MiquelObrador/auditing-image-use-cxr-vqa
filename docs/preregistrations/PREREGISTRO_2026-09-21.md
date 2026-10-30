# Pre-registro — VLM temporal para opción múltiple en MMXU

> **Versión pública.** Se han eliminado rutas locales y fragmentos literales del conjunto de datos; el resto del documento, incluidos los criterios y las lecturas fijadas de antemano, no se ha modificado.

**Fecha: 2026-09-21.** Escrito **antes** de entrenar nada y antes del primer paso de
fwd+bwd. Todo lo que aparece aquí queda fijado; cualquier desviación posterior se anota al
final, en la sección "Desviaciones", con fecha y motivo.

Proyecto independiente del paper de ISBI. No modifica `paper/`, `mmxu_eval/` ni ningún
resultado publicado.

---

## 1. Objetivo

Entrenar un VLM con fusión temporal explícita que responda opción múltiple en MMXU **y que
use de verdad la imagen previa**.

No se persigue batir la Tabla 3 de MMXU. El modelo se declara bueno **solo si cumple los tres
criterios de §2**, ninguno de los cuales es "accuracy alta".

## 2. Criterios de éxito (los tres, no dos de tres)

Medidos sobre **MMXU-test** (2.992 filas con bbox utilizable, 1.201 pacientes), con el mismo
protocolo que el paper: **bootstrap agrupado por paciente, N=5000, semilla 0**, IC95%
percentil. Unidad de remuestreo = paciente (`cur_image_path.split("/")[2]`).

| # | Criterio | Umbral |
|---|---|---|
| 1 | **Accuracy** | claramente por encima del suelo de nombre de región, **0,395** |
| 2 | **Gate de imagen** | `blind − full` **negativo**, IC95% que **excluye cero** |
| 3 | **Uso de la previa** | `mismatch − full` **≤ −5pp**, IC95% que **excluye cero** |

El criterio 3 es el que da sentido al proyecto. Los criterios 1 y 2 son condiciones
necesarias: sin ellos, el 3 no es interpretable (misma lógica de dos pasos que el paper).

**La comparación que importa es `mismatch` frente a `full`.** Ver §4.2 para por qué
`no_prior` y `duplicate` no aportan un contraste independiente en esta arquitectura.

## 3. Referencias fijadas de antemano

- **Suelo de nombre de región: 0,395** (IC95% [0,377, 0,412]), marco B, voto mayoritario por
  región leave-one-out. Fuente: `bbox_analysis/region_prior_report.md`.
- **Azar: 0,250** (4 opciones).
- **Línea base del montaje SIN ENTRENAR**: se mide **antes de entrenar**, sobre las **2.000
  filas de `dev-val-fast`** (§5), y se registra en este fichero antes de lanzar el
  entrenamiento. Es la referencia contra la que se juzga el piloto.
  El "20%" de la exploración previa venía de n=20: es ruido, **no** se usa como línea base.

## 4. Condiciones y cómo se implementan

El protocolo de evaluación es el binario existente `mmxu_eval/oracle_evaluate.py`, invocado
sin modificarlo (registro en caliente en `MODEL_REGISTRY`). El prompt es
`mmxu_eval/prompts.py::build_prompt`, **byte a byte**.

### 4.1 `blind` — sin tokens visuales

`blind` **no** usa imágenes grises, negras ni de ruido: sería una entrada fuera de
distribución y mediríamos eso, no el uso de la imagen.

Se pasa `images=None`, lo que corta limpiamente toda la rama visual
(`libra_arch.py:149`: `if vision_tower is None or images is None or input_ids.shape[1] == 1:
return input_ids, attention_mask, past_key_values, None, labels`), y el LLM hace un forward de
solo texto. El prompt se construye **sin** el token `<image>` (si no, el `-200` de
`IMAGE_TOKEN_INDEX` llegaría a la tabla de embeddings). El texto MMXU queda idéntico.

Es exactamente lo que hace Qwen2-VL en `blind` (no añade la imagen al contenido): **paridad
con el protocolo del ISBI**.

### 4.2 `no_prior` ≡ `duplicate` en esta arquitectura — no son condiciones distintas

Libra sustituye la previa ausente por **una copia de la actual**
(`run_libra.py:111-112`, `train.py:1361-1373`) y detecta ese caso con un **sesgo aprendido**:
en `TAC.TFM`, `prev_weight = cos * self.LFE_prior_bias`, donde `cos` es la similitud coseno
entre actual y previa. Con la previa duplicada, `cos ≈ 1` y el sesgo se aplica al máximo — es
literalmente el detector de "no hay previa" que el modelo entrenó.

Por tanto **`no_prior` y `duplicate` son la misma entrada y el mismo camino de cómputo**. No
se pueden distinguir, y `no_prior` **no es una ablación**: es comportamiento entrenado, dentro
de distribución.

Decisión: se usa **el modo nativo de Libra** para `no_prior` (duplicar la actual), se reporta
**una sola fila**, y se documenta que `duplicate` es redundante con ella. Ninguna conclusión
del proyecto descansa sobre ese contraste.

### 4.3 `mismatch`

Tal cual lo implementa `oracle_evaluate.py`: desplazamiento cíclico sin puntos fijos
(`np.roll(np.arange(n), 1)`) aplicado **solo a la previa**, con la actual intacta, calculado
**sobre la población completa antes de cualquier `--limit`**. Es la única manipulación
temporal genuinamente fuera de distribución disponible.

## 5. Datos

`MMXU-dev.jsonl`: 118.800 muestras. Tras descartar 7 con esquema de opciones no-ABCD y las que
no tienen ambas imágenes en la copia local de MIMIC-CXR-JPG: **106.929 usables, 14.796
pacientes** (90,0%).

Disjunción con MMXU-test **verificada por `subject_id`**: 0 pacientes, 0 imágenes, 0 pares en
común.

- `dev-val`: **10% de los pacientes**, elegidos con `np.random.default_rng(0)` sobre la lista
  ordenada de pacientes. Disjunto por paciente del entrenamiento.
- `dev-val-fast`: submuestra fija de **2.000** filas de `dev-val` (semilla 0), para evaluar
  durante el entrenamiento y para la línea base de §3.
- El resto es el conjunto de entrenamiento.

**Ningún ejemplo de entrenamiento carece de imágenes.** Entrenar sin imagen enseñaría el atajo
del texto y debilitaría precisamente el gate del criterio 2.

## 6. Qué se entrena

| Módulo | Params | Estado |
|---|---|---|
| RAD-DINO | 86M | Congelado (lo está por construcción: `@torch.no_grad()` en `dino_encoder.py:87`) |
| TAC | 39,0M | **Congelado**, y forzado a `.eval()` en cada paso (tiene dropout 0.1; `Trainer.train()` pondría todo en modo train) |
| LLM (Llama-3.2-3B-Instruct) | 3,2B | **LoRA**, único módulo entrenado |

Motivo de congelar el TAC: (a) es la receta de etapa 2 del propio Libra
(`finetune_lora.sh: --freeze_mm_mlp_adapter True`); (b) **razón científica** — si ajustamos la
fusión temporal sobre MMXU y después medimos sensibilidad a `mismatch`, no podríamos
distinguirla de un artefacto de nuestro propio ajuste. Congelado, cualquier uso de la previa
es atribuible a la fusión pre-entrenada.

**Brazo 2, declarado aquí y por adelantado**: si el modelo cumple los criterios 1 y 2 pero
falla el 3, se entrena una variante con el **TAC descongelado** (LR propio 2e-5, el
`--mm_projector_lr` de Libra). Es un **segundo brazo**, no un reemplazo: se reportan los dos.

### 6.1 Hiperparámetros

```
LoRA: r=16, alpha=32, dropout=0.05, todas las lineales (q,k,v,o,gate,up,down)
      ~24M params entrenables (0,75% del LLM)
lr=5e-5, cosine, warmup 3%     # por debajo del 1e-4 típico: prioriza no derivar
epochs=1
batch=2 por dispositivo x grad_accum 8 (efectivo 16)
bf16, gradient_checkpointing, attn_implementation="sdpa"
loss SOLO sobre tokens de respuesta (el resto a IGNORE_INDEX = -100)
```

`batch >= 2` no es opcional: con `batch=1`, `prepare_inputs_labels_for_multimodal` entra por la
rama de reshape de beam-search (`libra_arch.py:158-175`) en vez de por la correcta.

### 6.2 Sin replay de instrucciones generales

Se consideró y **se descarta**. Los datos de instrucciones generales son texto sin imagen, así
que entrenarlos exigiría usar la **misma ruta de forward sin tokens visuales que implementa
`blind`** (§4.1) — entrenaríamos al modelo en la configuración exacta con la que luego lo
medimos, contaminando el instrumento de medida y no solo el modelo. Además choca con la regla
de no entrenar ejemplos sin imagen (§5).

Si el chequeo de retención (§8) mostrase degradación, el replay entra como **brazo declarado**
pero **anclado en imagen**: prompts fuera de tarea sobre el mismo par de CXR, nunca texto
pelado.

## 7. Mezcla de formatos (anticolapso)

El riesgo es doble y en direcciones opuestas: que el modelo **solo** sepa emitir "A", o que
recaiga en el **modo informe** que bloqueaba a Libra. La mezcla se elige contra ambos.

| Formato | Instrucción | Objetivo | Proporción |
|---|---|---|---|
| **A** letra sola | la de `mmxu_eval/prompts.py`, byte a byte | `A` | **65%** |
| **B** texto de la opción | "Answer with the text of the correct option." | `Improved` | **20%** |
| **C** justificación + letra | "Briefly justify your answer, then give the option's letter." | `<justificación ≤40 palabras> The answer is A.` | **15%** |

Razones de estas proporciones:
- **A domina (65%)** porque es el único formato que se usa en evaluación.
- **B (20%) es la palanca anticolapso segura**: obliga a mapear contenido→texto sin ningún
  riesgo de prosa larga.
- **C es la menor (15%)** porque es la única con riesgo de recaída al modo informe. Se acota
  por dos vías: la justificación se **trunca a 40 palabras**, y el objetivo **siempre termina
  en `The answer is X.`**, de modo que incluso en C el modelo debe aterrizar en una letra.

### 7.1 Fuente de la justificación: `cur_region_report`, no `explanation`

`explanation` **no se usa**. El 59,9% contiene *"related explanatory information"*, el 31,6%
cita *"the report"*, el 9,1% *"according to"*; con filtro estricto solo el **1,3%** queda
limpio. Entrenar sobre eso enseñaría a citar un informe que el modelo no puede ver.

Se usa `cur_region_report`: frase real del informe sobre esa región, no vacía en el 99,4%,
**89,6% limpia y con lenguaje de cambio temporal**, mediana 25 palabras.

Limpieza (validada a mano sobre 30 ejemplos antes de entrenar): quitar prefijos `FINDINGS:` /
`IMPRESSION:`, descartar las que contengan el marcador de anonimización `___` de MIMIC,
truncar a 40 palabras. Las muestras que no pasan la limpieza **no van al formato C**: se
asignan a A o B.

## 8. Chequeo de retención (antes y después de entrenar)

16 prompts fijos, ejecutados sobre el montaje **sin entrenar** y sobre el entrenado, con las
salidas **literales** una al lado de la otra. Es un cable-trampa cualitativo, no una
puntuación.

- **6 MMXU formato A (solo la letra)** — el control específico contra la recaída al modo
  informe. Se registra además la **longitud media de la salida**: si crece tras entrenar, hay
  recaída.
- 3 formato B, 3 formato C.
- 2 fuera de tarea pero ancladas en la imagen ("Describe Image-2 in one sentence").
- 2 de seguimiento puro de instrucciones ("Reply with the word BANANA and nothing else").

## 9. Piloto

~4.000 muestras de entrenamiento (disjuntas por paciente de `dev-val`), 1 época. Se reporta:
la loss baja; parseo estricto ≈100% sobre `dev-val-fast`; accuracy sobre `dev-val-fast`
**frente a la línea base sin entrenar de §3**; throughput (muestras/s) y pico de VRAM para
extrapolar; y el chequeo de retención de §8.

Estimación previa: ~1.519 tokens/muestra → del orden de **20–27 h por época** sobre las
106.929. El entrenamiento completo será casi seguro sobre una submuestra estratificada por
paciente, con el n fijado por el throughput **medido**, no inventado.

**Parada obligatoria tras el piloto.** No se evalúa `mismatch` ni `blind` sobre MMXU-test
hasta que el modelo esté entrenado.

---

## Desviaciones

**2026-09-21 — §8, system prompt de los prompts fuera de tarea.** Tal como estaba escrito,
los 4 prompts fuera de tarea heredaban el system prompt de la tarea (*"Answer multiple-choice
questions with a single letter"*), que **contradice** instrucciones como *"Reply with the word
BANANA and nothing else"*. En la primera corrida el modelo sin entrenar devolvió cadena vacía
en 2 de los 4: una línea base rota, sin margen para detectar degradación posterior — el
chequeo habría medido el system prompt, no el seguimiento de instrucciones.

Corrección: los 4 prompts fuera de tarea usan `"You are a helpful assistant."`; los 12 de
formato A/B/C mantienen el system prompt de la tarea, que es el de despliegue. La corrida
defectuosa se conserva como `runs/retention_before_DESCARTADO_v1.json` y **no** se usa como
referencia; ambos lados (antes y después) se recorren con el diseño corregido.

**2026-09-21 — §7.1, corte de las justificaciones.** El pre-registro decía "truncar a 40
palabras". Al validar a mano los 30 ejemplos exigidos, el truncado por número de palabras
producía fragmentos agramaticales (frases cortadas a media oración), frases duplicadas y restos de
numeración de la sección IMPRESSION. Se cambia a **corte por frases completas** (máx. 2
frases y 40 palabras, deduplicadas, exigiendo lenguaje de cambio temporal). Cobertura
resultante: 68,9% de las filas de train (antes 78,9%). Para que el formato C siga siendo el
15% **marginal** pre-registrado, el sorteo se condiciona a que la muestra tenga justificación
utilizable (P(C | hay justificación) = 0,15/0,689). Proporciones realizadas: 65,0 / 20,0 /
15,0.
