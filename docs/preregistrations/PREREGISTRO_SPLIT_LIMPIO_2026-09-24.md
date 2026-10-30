# Pre-registro — reentrenamiento con el split limpio: TAC preentrenado (A) y TAC desde cero (B)

> **Versión pública.** Se han eliminado rutas locales y fragmentos literales del conjunto de datos; el resto del documento, incluidos los criterios y las lecturas fijadas de antemano, no se ha modificado.

**Fecha: 2026-09-24.** Escrito antes de lanzar A y B. Se ejecutan en este orden, en una sola
cadena secuencial (una GPU). Cualquier desviación se anota al final con fecha y motivo.

## 0. Por qué

- **Fuga del TAC de Libra.** El 97,6% de las filas de MMXU-test (2.920 de 2.992) son de
  pacientes del split oficial de train de MIMIC-CXR. El TAC de Libra se preentrenó sobre ese
  split, con informes y preguntas de cambio de Medical-Diff-VQA.
- **Fuga de RAD-DINO.** RAD-DINO también se preentrenó sobre ese split (paper, apéndice D.1:
  "their recommended training split"; checkpoint público: 368.960 imágenes).
- **El modelo actual tampoco sirve para el conjunto limpio.** `runs/full` se entrenó con
  `splits.json`, que contiene 2.733 filas de pacientes limpios, así que no puede evaluarse sobre
  el conjunto limpio.

Referencias: `clean_eval/CLEAN_SET_2026-09-24.md` y `PREREGISTRO_2026-09-21.md` (receta y
criterios originales).

## 1. Datos y poblaciones de evaluación

- **Entrenamiento** (A y B): `clean_eval/splits_clean_2026-09-24.json`. Train son 93.693 filas y
  13.037 pacientes, sin ningún paciente del split oficial validate/test.
- **Conjunto limpio** (lectura principal de A y de B): `clean_eval/CLEAN_SET_2026-09-24.csv`,
  filas `usable`.
  - Son 3.096 filas y 331 pacientes: 3.022 filas y 304 pacientes de MMXU-dev, y 74 filas y 27
    pacientes de MMXU-test.
  - Ni el TAC de Libra, ni RAD-DINO, ni nuestra LoRA han visto estos pacientes.
- **MMXU-test completo** (2.992 filas, 1.198 pacientes): lectura secundaria. Protocolo del ISBI
  sin cambios (`run_eval.py`, mismatch con el desplazamiento de una fila), comparable con todos
  los resultados anteriores. Se evalúa en B (pedido) y también en A, porque la comparación de B
  sobre MMXU-test completo necesita el modelo con TAC preentrenado entrenado con el mismo split.

### Condiciones sobre el conjunto limpio

`run_eval_clean.py` usa el mismo `oracle_evaluate.run()`, el mismo prompt (`prompts.build_prompt`)
y el mismo parseo. Condiciones: `full`, `blind` (forward solo texto, sin `<image>`) y `mismatch`.

**Mismatch sobre el conjunto limpio.** Se usa un derangement aleatorio por paciente (semilla 0):
- cada paciente recibe las previas de otro paciente distinto;
- cada fila recibe una previa elegida al azar entre las de ese paciente;
- una aserción comprueba que ninguna fila recibe una previa de su propio paciente.

No se usa el desplazamiento de una fila del protocolo de MMXU-test. Las filas de MMXU-dev vienen
agrupadas (~10 por paciente) y la mayoría recibiría una previa del mismo paciente.

Los `qid` de MMXU-dev se desplazan +1.000.000, porque colisionan con los de MMXU-test.

## 2. Criterios (los tres, idénticos al pre-registro original)

Se aplican sobre el conjunto limpio, que es la lectura principal. Bootstrap agrupado por
paciente, N=5000, semilla 0, IC95% percentil. TOST ±5pp con IC90%.

| # | criterio | umbral |
|---|---|---|
| 1 | accuracy(full) | IC95% enteramente por encima de 0,395 |
| 2 | gate de imagen | blind−full con IC95% < 0 |
| 3 | uso de la previa | mismatch−full ≤ −5pp, con IC95% < 0 |

Se calculan con `clean_eval/criteria_report_clean.py`. Sobre MMXU-test completo se usa
`criteria_report.py` sin cambios. Se reportan siempre accuracy(blind) y los desgloses por origen
y por `content_type`; son secundarios, no criterio.

**Potencia, fijada de antemano.**
- Con 331 pacientes, el IC95% de mismatch−full medirá ~5,0pp de ancho, es decir ~±2,5pp
  (P10–P90 del ancho: 4,5–5,4pp). Se estima submuestreando los pacientes del modelo actual.
- Un efecto real de −8pp daría un IC de ~[−10,5, −5,5]: cumple el criterio 3.
- Un efecto real de entre −5 y −7,5pp puede cumplir o no el criterio 3 según el ruido.
  Resultados en esa franja se reportarán como **no concluyentes**, no como fallo.

## 3. Ejecución A — TAC preentrenado de Libra, congelado (receta actual)

Idéntica a `runs/full` en todo salvo el split:
- LoRA r=16, alpha=32, dropout 0,05, sobre las lineales del LLM;
- lr 5e-5, coseno, warmup 3%, 1 época, batch 2 × accum 8;
- formatos 65/20/15 y permutación por (qid, época);
- TAC y RAD-DINO congelados, TAC en `.eval()`.

Salida en `runs/tac_pre_clean/`.

**Lectura pre-registrada de A:**
- **A cumple los 3 en el conjunto limpio:** el efecto del modelo actual no depende de haber visto
  a los pacientes de evaluación en el preentrenamiento. La afirmación del paper puede apoyarse en
  el conjunto limpio.
- **A cumple en MMXU-test completo pero no en el limpio (gate o mismatch):** el efecto dependía,
  al menos en parte, de la exposición previa a esos pacientes, es decir, de memorización.
- **A no cumple en ninguno:** el cambio de split (−2,8% de datos) no puede explicarlo por sí solo.
  Se investiga antes de interpretar B.

## 4. Ejecución B — TAC desde cero, entrenado junto con la LoRA

### Receta

**Qué cambia respecto a A:**
- **Inicialización del TAC:** `TAC(config)` nuevo con la init por defecto de PyTorch y
  `torch.manual_seed(0)`. Es la misma construcción que la etapa 1 de Libra (`libra_arch.py:101`,
  sin `pretrain_mm_mlp_adapter`). Una aserción comprueba que ninguno de los 54 tensores coincide
  con los de Libra.
- **TAC entrenable** con su propio grupo de LR, **tac_lr = 2e-5**. Es el LR que usa el propio
  Libra para entrenar este módulo desde cero (`scripts/pretrain.sh`). No se usa el 5e-6 de la
  ronda anterior: allí se trataba de no desestabilizar un módulo ya alineado, y aquí el módulo
  tiene que aprender desde cero.
- **TAC en modo train**, con su dropout.

**Qué no cambia respecto a A:** LoRA, datos, formatos, permutación, lr de la LoRA y RAD-DINO
congelado.

**Pasos: 1 época**, el mismo número que A. Así A y B difieren solo en el TAC: inicialización y
si se entrena o no.

**Por qué conjunto y no en dos etapas como Libra:**
- Nuestra única fuente de datos es MMXU-dev. Una etapa 1 separada entrenaría el TAC con el mismo
  objetivo de opción múltiple, así que no aportaría una alineación general.
- Duplicaría el coste.
- La pregunta ("¿basta la arquitectura sin preentrenamiento?") solo exige que el TAC no haya
  visto pesos preentrenados.

Salida: piloto en `runs/tac_scratch_clean_pilot/`, carrera completa en `runs/tac_scratch_clean/`
(`adapter/` + `tac_state_dict.bin`).

### Piloto y puerta, automática y fijada ahora

Piloto: 4.000 filas del train limpio, muestreadas por paciente con semilla 0 (misma función que
el piloto original), 1 época, y evaluación en `val_fast` (1.938 filas).

`pilot_gate.py` exige las tres condiciones siguientes. Si alguna falla, la cadena se detiene y no
se lanza la carrera completa.
- (a) loss finita, con al menos 10 registros;
- (b) media de los 5 últimos registros de loss < media de los 5 primeros;
- (c) parseo estricto en `val_fast` ≥ 95%.

La accuracy del piloto se reporta pero no se exige. Referencia: piloto con TAC preentrenado,
0,677 con 100% de parseo. Casi toda la accuracy puede venir del atajo textual, así que una
accuracy alta no demostraría que el TAC aprende.

### Lectura pre-registrada de B

B se evalúa en el conjunto limpio (principal) y en MMXU-test completo.

- **B cumple los 3 en el conjunto limpio:** la arquitectura hace que el modelo mire la imagen y
  use la previa sin necesidad de preentrenar el TAC.
  - Salvedad: RAD-DINO sigue preentrenado, pero no vio a estos pacientes.
  - Es la afirmación más limpia disponible: ningún componente vio a estos pacientes, y el TAC no
    vio nada en absoluto.
- **B no cumple el gate o el criterio 3 en el conjunto limpio** (y A sí): el efecto del modelo
  actual dependía del preentrenamiento del TAC.
  - **Se declara ahora la lectura alternativa, que no se podrá descartar:** con 1 época, el TAC
    aleatorio puede estar simplemente infraentrenado. Su preentrenamiento en Libra usó un corpus
    propio de alineación, de orden de magnitud igual o mayor, además de su etapa 2.
  - Si ocurre, el experimento que la separa es B con más pasos (p. ej. 2 épocas, ~24h). No se
    lanza sin aprobación.
- **B en la franja no concluyente** (§2): se reporta como tal.
- **Contraste A−B** (descriptivo, secundario): diferencias pareadas por paciente en accuracy,
  blind−full y mismatch−full sobre el conjunto limpio, con IC95%.

## 4b. Ejecución C — genérico Qwen2.5-VL-3B reentrenado con el split limpio

*(Añadida el 2026-09-24, con A ya entrenando. Antes de que C ni B produzcan ningún número.)*

**Por qué es imprescindible:**
- La comparación central del paper es el modelo con fusión temporal frente al genérico afinado
  igual.
- `qwen25vl_baseline_train/runs/full` se entrenó con `splits.json`, que contiene las 2.733 filas
  limpias de dev. No puede evaluarse en el conjunto limpio.

**Receta:** idéntica a `qwen25vl_baseline_train/runs/full` salvo el split.
- Qwen2.5-VL-3B-Instruct, con la torre visual congelada.
- LoRA r=16, alpha=32, dropout 0,05, lr 5e-5, 1 época, batch 2 × accum 8.
- Mismos formatos y permutación.
- Resolución de entrenamiento igualada a RAD-DINO: 268.324 px por imagen, 320 tokens.

Salida en `qwen25vl_baseline_train/runs/qwen_clean/`.

**Evaluación:**
- Conjunto limpio con `run_eval_clean.py --model qwen25vl_baseline`: mismo cargador, mismo
  derangement por paciente y mismos píxeles de evaluación del protocolo.
- MMXU-test completo con `run_eval_qwen25vl.py`.
- Condiciones: full, blind y mismatch en ambos.

**Criterios:** los mismos tres, reportados igual.

**Lectura esperada**, por el resultado previo del genérico: blind−full ≈ 0 y mismatch−full ≈ 0.
Si en el conjunto limpio el genérico sí usara la imagen, el contraste central del paper cambia y
se reporta así.

### Comparaciones pre-registradas (A frente a C; B frente a C como secundaria)

**1. Contraste central, sobre el conjunto limpio.** Diferencias pareadas por paciente entre
modelos (A − C), con IC95%:
- en blind−full;
- en mismatch−full;
- en accuracy.

**2. Fuga frente a dificultad (diferencia en diferencias).** La comparación limpio frente a
completo de un solo modelo mezcla dos cosas:
- la fuga;
- la diferencia de dificultad entre poblaciones. El conjunto limpio sale sobre todo de MMXU-dev,
  y MMXU-test se curó a mano.

El genérico sirve de referencia de dificultad. Se define

    DiD = (acc_A,limpio − acc_A,completo) − (acc_C,limpio − acc_C,completo)

- IC95% por bootstrap de pacientes, independiente en cada población y pareado entre modelos
  dentro de cada población.
- DiD ≈ 0: no hay indicio de fuga.
- DiD claramente negativo: el TAC rinde peor en pacientes no vistos más de lo que explica la
  dificultad, compatible con fuga.
- Lo mismo para blind−full y mismatch−full, como lecturas secundarias.

**Supuesto de esta referencia, declarado y no verificable con lo publicado:** se asume que
Qwen2.5-VL no vio MIMIC-CXR en su preentrenamiento. Su informe técnico no lista sus datos médicos
con detalle suficiente para descartarlo. Si los vio, la referencia también está contaminada y el
DiD solo mide la fuga *adicional* del TAC/RAD-DINO sobre la del genérico.

## 4c. Ejecución B2 — TAC desde cero, en dos etapas (LLM congelado primero)

*(Añadida el 2026-09-26, con A, B y C ya terminados. Antes de lanzar nada de B2.)*

### Por qué

En B, el TAC (aleatorio) y la LoRA se entrenaron **a la vez**. Hipótesis: la LoRA, que parte de
cero pero tiene acceso directo a los pesos del LLM, puede reducir la loss aprendiendo el atajo
textual (ya documentado en `shortcut_analysis/`) más rápido de lo que el TAC aleatorio puede
aportar señal útil de imagen — con lo que el LLM deja de necesitar al TAC antes de que este
aprenda nada, y el resultado de B (1/3 criterios, equivalencia TOST en mismatch) no distingue
"la arquitectura sin preentrenamiento no basta" de "el orden de entrenamiento no le dio ninguna
oportunidad".

B2 separa las dos preguntas congelando el LLM en la primera etapa: ahí, la única forma de que la
loss baje es que el TAC empiece a llevar información real de la imagen.

### Etapa 1 — TAC solo, LLM y RAD-DINO congelados, SIN LoRA

- Datos: train del split limpio (93.693 filas), mismos formatos y permutación que el resto.
- TAC con init aleatoria de Libra (`--tac-init scratch`), igual que en B.
- Único módulo entrenable: el TAC. LR = 2e-5, igual que en B (mismo razonamiento: es el LR que
  usa el propio Libra para este módulo desde cero).
- 1 época, mismos pasos que A y B en esta etapa — no dos épocas por defecto; si el piloto sugiere
  que la loss sigue cayendo con fuerza al final, se reporta antes de decidir si hacen falta más.
- Implementación: `train_lora.py --stage1-tac-only --unfreeze-tac --tac-init scratch --tac-lr 2e-5`.
  Sin envoltorio PEFT — no hay LoRA que construir. Se guarda solo `tac_state_dict.bin` (no hay
  `adapter/`).

**Piloto y puerta propia, distinta de la de B**, porque aquí no hay LoRA que garantice un parseo
limpio (eso es tarea de la etapa 2, el LLM sigue siendo el Llama-Instruct de base): 4.000 filas,
y `pilot_gate_stage1.py` exige (a) loss finita, ≥10 registros, y (b) media de los 5 últimos
registros ≤ 95% de la media de los 5 primeros (caída de al menos un 5%, más estricta que la
puerta de B porque no hay parseo con que contrastar). Si no pasa, la cadena se detiene y no se
lanza ni la etapa 1 completa ni la etapa 2.

Salida: piloto en `runs/tac_b2_stage1_pilot/`, completa en `runs/tac_b2_stage1/`.

### Etapa 2 — LoRA sobre el LLM, TAC de la etapa 1 congelado

Receta idéntica a A (mismos hiperparámetros, mismos pasos, misma época): LoRA r=16/alpha=32/
dropout=0,05, lr 5e-5, RAD-DINO y TAC congelados y en `.eval()`. Único cambio: el TAC se
inicializa desde `runs/tac_b2_stage1/tac_state_dict.bin` en vez de los pesos de Libra
(`train_lora.py --tac-state-init runs/tac_b2_stage1/tac_state_dict.bin`, sin `--unfreeze-tac`).

Salida: `runs/tac_b2_stage2/adapter/`. Evaluación con `run_eval.py`/`run_eval_clean.py --tac-state
runs/tac_b2_stage1/tac_state_dict.bin --adapter runs/tac_b2_stage2/adapter`, igual que A y B.

### Criterios y lectura (idénticos formato a A/B)

Los mismos tres criterios (§2), sobre el conjunto limpio (principal) y MMXU-test completo
(secundaria).

- **B2 pasa los 3 en el conjunto limpio:** el factor decisivo es el esquema de entrenamiento
  (congelar el LLM mientras el TAC aprende), no el preentrenamiento de Libra en sí. Un TAC
  aleatorio SÍ puede aprender a mirar la imagen con nuestros datos, si se le da la oportunidad
  sin competir con una LoRA que ya sabe adónde ir.
- **B2 falla igual que B:** el factor decisivo es el preentrenamiento — probablemente la tarea de
  generación de informes de Libra (etapa 1 real), donde el texto de entrada no contiene la
  respuesta y el TAC no tiene más remedio que aportar señal de imagen para bajar la loss. Nuestra
  etapa 1 usa el mismo formato de opción múltiple que el resto del proyecto, así que el atajo
  textual sigue estando disponible en el objetivo — congelar el LLM quita la LoRA como
  competidora, pero no quita el atajo del propio texto de las opciones (ver
  `shortcut_analysis/`). Esta salvedad se declara ahora: un fallo de B2 no aísla de forma limpia
  "preentrenamiento de Libra" de "el objetivo de MCQ tiene atajo textual incluso sin LoRA
  compitiendo".
- **Discrepancia entre poblaciones** (limpio vs. MMXU-test completo): misma lectura de fuga que
  en A/B, con la referencia de dificultad de C (DiD, §4b).

### Coste estimado

| paso | estimación |
|---|---|
| etapa 1: piloto (4.000 filas) | ~0,5 h |
| etapa 1: completa (93.693 filas, misma velocidad esperada que A/B: ~2,1-2,2 muestras/s) | ~12 h |
| etapa 2: completa (idéntica a A) | ~11,9 h |
| evaluación (limpio + MMXU-test, full/blind/mismatch) | ~1,2 h |
| **total B2** | **~25,6 h** |

La estimación de la etapa 1 asume el mismo coste por paso que A/B: el cuello de botella es el
forward+backward completo por el LLM de 3B (aunque esté congelado, el gradiente tiene que
atravesarlo para llegar al TAC), no el número de parámetros que se actualizan.

## 4d. Control de tokens constantes y subconjunto difícil

*(Añadido el 2026-09-28, antes de ningún resultado de estas condiciones.)*

### Control de tokens constantes (`both_other`), conjunto limpio: A, B2 y C

**Condición.** Las dos imágenes, previa y actual, vienen de otro paciente.
- Se usa la fila fuente del mismo derangement aleatorio por paciente (semilla 0) que `mismatch`.
- La previa es la misma que en `mismatch`; la actual es la de esa misma fila fuente.
- Mantiene el número de tokens visuales y quita toda la información correcta de la imagen.

**Se reportan:**
- accuracy(both_other);
- both_other−full y both_other−blind, pareados por fila, con IC95% por paciente.

**Lectura:**
- **both_other ≈ blind** (both_other−blind con IC que incluye cero, o dentro de ±5pp): el gate
  refleja uso de la información de la imagen, no la ausencia de tokens.
- **both_other ≈ full** (both_other−full cerca de cero y both_other−blind claramente positivo): el
  modelo no usa el contenido. El gate medía el cambio de distribución al quitar los tokens.
  - Es la hipótesis del *soft prompt* declarada para B2 al pasar su puerta del piloto.
- **Intermedio:** se reporta la fracción del gate que sobrevive con tokens presentes,
  (full − both_other) / (full − blind).

### Subconjunto difícil

**Definición:** las preguntas que falla el modelo solo-texto.
- Se reentrena el solo-texto con el split limpio (`qwen25vl_baseline_train/runs/textonly_clean`).
- El anterior (`runs/textonly`) se entrenó con `splits.json`, que contiene 2.733 filas limpias: en
  el conjunto limpio habría visto esas preguntas.

**Se reporta**, en el conjunto limpio y en MMXU-test completo:
- el tamaño del subconjunto;
- la accuracy de A, C y el solo-texto sobre él, con IC95% por paciente, frente al azar (0,25).

### Δpred y su nulo

- El conjunto limpio no admite `hole_random`: ninguna de las 3.022 filas limpias de MMXU-dev tiene
  bbox de la previa.
- El nulo se mide en MMXU-test completo (`hole_random`, protocolo del ISBI) para A, B y C, y se
  reporta como referencia en otra población. Es una salvedad declarada.
- El Δpred de blind, mismatch y both_other se mide en el conjunto limpio.

### Pendiente, sin lanzar (decisión después del ISBI)

**TAC desde cero con la etapa 1 de generación de informe:** prompt fijo, sin opciones, generando
`cur_report`.
- B2 usó en su etapa 1 preguntas con opciones, así que el atajo textual seguía disponible en el
  objetivo.
- Por eso B2 no separa "preentrenamiento de Libra" de "objetivo donde el texto no ayuda". Este
  experimento es el que lo separaría.

## 5. Coste estimado (una GPU, secuencial)

| paso | estimación |
|---|---|
| A: entrenamiento (93.693 filas, ~2,19 muestras/s medido) | ~11,9 h |
| A: evaluación limpio (full, blind, mismatch) + MMXU-test (ídem) | ~1,1 h |
| B: piloto (4.000 filas) + val_fast | ~0,7 h |
| B: entrenamiento (93.693 filas, ~2,15 muestras/s medido con TAC entrenable) | ~12,1 h |
| B: evaluación limpio + MMXU-test | ~1,1 h |
| C: entrenamiento (93.693 filas, ~1,63 muestras/s medido en `runs/full` del genérico) | ~16,0 h |
| C: evaluación limpio + MMXU-test (~39 min por condición con imagen) | ~2,8 h |
| **total** | **~46 h** |

## 6. Verificaciones hechas antes de lanzar (smoke tests, 2026-09-24)

- **B, carrera de humo** (32 filas, 2 pasos, init desde cero):
  - "TAC reinicializado desde cero: 54 tensores, ninguno igual a Libra";
  - 63,3M parámetros entrenables (TAC 39,0M + LoRA 24,3M);
  - loss finita y pico de VRAM de 14,0 GiB.
- **A, regresión:** 24,3M parámetros entrenables (solo LoRA), igual que `runs/full`.
  `train_stats.json` registra `splits_file = splits_clean_2026-09-24.json`.
- **`run_eval_clean.py`:**
  - 3.096 filas y 331 pacientes;
  - las tres condiciones corren, las respuestas parsean con el adaptador de `runs/full`;
  - las previas de `mismatch` vienen siempre de otro paciente (aserción).
- **`eval_valfast.py --tac-state` y `pilot_gate.py`:** funcionan. La puerta rechaza, con código
  de salida 1, un modelo de 2 pasos.

## Desviaciones

- **2026-09-24, verificación del conjunto limpio contra el split oficial.**
  - Fichero: `mimic_metadata/mimic-cxr-2.0.0-split.csv.gz`, SHA-256 `515997bd…bedb2f`.
  - 0 discrepancias en los 65.379 pacientes.
  - El conjunto limpio y el split de entrenamiento regenerados desde el oficial son idénticos
    byte a byte a los usados.
  - **El conjunto limpio queda verificado contra el split oficial.** No cambia nada del
    entrenamiento ni de la evaluación.
- **2026-09-24, adición antes de cualquier resultado.** Se añade la ejecución C (§4b) y las
  comparaciones A−C y DiD. Se lanza como cadena aparte (`runs/chain_generico_limpio.sh`) que
  espera a que termine la cadena A→B. No se toca el script en ejecución.
  `run_eval_clean.py` se generaliza con `--model`, sin cambiar la ruta del TAC.
  - La ruta del TAC ya estaba comprobada.
  - La del genérico no se pudo comprobar con A entrenando: la GPU no tiene memoria para los dos
    a la vez.
  - Por eso la cadena C empieza con una prueba de 3 filas por condición con el adaptador antiguo.
    Si falla, C se detiene antes de entrenar.
- **2026-09-26, adición de B2 (§4c), antes de cualquier resultado de B2.** `train_lora.py` gana
  `--stage1-tac-only` (sin envoltorio PEFT, solo el TAC entrenable) y `--tac-state-init` (cargar
  un TAC ya entrenado desde fichero, para la etapa 2). Puerta propia en `pilot_gate_stage1.py`,
  sin chequeo de parseo (no aplica sin LoRA). En cola detrás de C.
