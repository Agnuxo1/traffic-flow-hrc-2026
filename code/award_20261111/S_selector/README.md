# Cierre del ensayo selector-aligned

Convierte los modelos congelados en archivos candidatos con el formato de Kaggle, para que Francisco los revise. No envía nada.

## Qué hace

Para cada una de las 160 ventanas oficiales de `validation` y `private`:

1. Lee la capa enmascarada publicada y sobrepone la historia oficial de T-60 a T-5. La fila en T viene del archivo de estado, como en `x_t`.
2. Calcula las 95 características y aplica el ensemble de cada brazo: media de las tres probabilidades y umbral congelado.
3. Escribe `queue_pred` para todos los enlaces y horizontes. Los modelos cubren cada enlace directamente, con o sin sensor: la historia oficial los trae todos, y no hay interpolación.

Produce dos candidatos con la misma base `sub_v1.csv`, cambiando solo las celdas de cola:

| Brazo | Trabajo | Umbral | Papel |
|---|---|---:|---|
| S | `fd9c92c91852490d` | 0,5 | Pasó la puerta en filas alineadas |
| F | `93cf5c164e954d0c`, brazo A | 0,6 | Mejor control admitido, nunca medido en el marcador público |

Las tareas 1, 3 y 4 quedan byte a byte como en `sub_v1`, que tiene 0,72625 públicos. Por eso la nota pública de cada candidato mide directamente su `S_queue`: `(nota - 0,58512) / 0,30`.

## Qué no hace

- No busca umbrales, no reentrena y no elige entre brazos.
- No carga `train` en el job. Los días 258..272 no se abren. La paridad del preflight solo lee las fechas de las ventanas públicas de `train`.
- No toca `finalize_traffic.py`, su definición ni el congelado del controlador.
- No envía, no selecciona finales y no usa credenciales. Subir o elegir un archivo es decisión de Francisco.

## Comprobaciones

- `test_final.py`: pruebas unitarias en CPU.
- `preflight_final.py`, con datos reales en CPU: hashes; umbrales iguales a `choices-frozen.json`; probabilidades y puntuaciones iguales a las auditadas; paridad de características en las 80 ventanas públicas de `train`; identidad de enlaces; causalidad; recorrido completo sin escribir candidato.
- `verify_final.py`, tras el trabajo: recalcula todas las decisiones, comprueba por texto que nada más cambió y repite 16 ventanas por una ruta en numpy sin torch.

## Límites

El resultado local de S sigue condicionado a filas alineadas con un selector inferido, y S empeora en filas heredadas. Una nota pública se calcula sobre `validation`: dice cómo transfiere a ese mes, no a `private`, que se generó aparte.
