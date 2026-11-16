# Mejoras y trabajo futuro

Trabajo posterior al plazo de entrega (no forma parte de la entrega evaluada) sobre el #30 (MiniRocket + MLP en numba). Score oficial: $\sqrt{N/8}\cdot \text{gm}_f$, con $N$ = predicciones en 60 s en Colab CPU con 1 hilo.

## Carpetas

| Carpeta | Contenido |
|---|---|
| `mejor_actual/` | #30 + sesgos por clase «suave» (sin etiquetas). **Referencia elegida.** `motor_multi.py` + `modelo_numba.npz`. |
| `candidato_150/` | Alumno `pesos150_f64`: 150 rasgos MiniRocket elegidos por pesos, MLP 128 reentrenado (ens3, +8), float64. `motor_rapido.py` + `modelo_numba.npz` (= `artifacts/alumno_pesos30150_h128_ens3_mas8_s0_f64.npz`). |
| `modelos_harness/` | Todas las variantes probadas (base30, base30_sesgos, rapido*, pesos150_f32/f64, rank150_f32/f64). |
| `colab_comparar.zip` | `base30/`, `mejor_actual/` y `candidato_150/`, autocontenidos, para Colab. |
| `comparar_colab.ipynb` | Harness oficial en Colab (1 hilo, test_demo, 3 rondas por modelo con el orden intercalado): mediana de latencia, acierto, score400 y score estimado. |

## Tabla final: mejor_actual frente a candidato_150

| | base30 (#30) | **mejor_actual** | **candidato_150** |
|---|---|---|---|
| Rasgos MiniRocket | 253 | 253 | 150 |
| gm_f validación corrompida (14 250 × 4) | 0,520 | **0,511** | **0,499** |
| test_demo: aciertos / 400 (harness oficial) | 330 | **330** | **327** |
| test_demo: score400 | 3,488 | **3,507** | **3,385** |
| Harness oficial: status / excepciones / timeouts | ok / 0 / 0 | ok / 0 / 0 | ok / 0 / 0 |
| Latencia local 13/11, mediana de 3 rondas (µs) | 90 | 91 | 70 |
| Score estimado orientativo (latencia local × 140/90 µs) | ~120 | ~118 | ~131 |

- El candidato pierde un 2,4 % de gm_f (0,4987 frente a 0,5109). Solo compensa si en Colab su latencia es **≤ 0,954 ×** la de mejor_actual, porque $(0{,}499/0{,}511)^2 \approx 0{,}954$. En local, el 13/11 salió ×0,77 (70 µs frente a 91 µs), pero con mucho ruido: emparejando por ronda, ×0,93, ×0,95 y ×0,59.
- Las rondas locales del 16/11 (240–900 µs) no sirven, porque el equipo estaba cargado, y no se han usado.
- candidato_150 vuelve a pasar el harness oficial con test_demo (16/11, 1 ronda): 327/400, score400 3,385, sin incidencias.
- **La latencia definitiva se medirá en Colab** con `comparar_colab.ipynb`. Si en Colab el candidato no queda por debajo de ~0,95 × mejor_actual, la referencia sigue siendo `mejor_actual`.
