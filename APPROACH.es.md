# Cómo construimos esta entrega

[Read in English](APPROACH.md) · [Volver al README](README.es.md)

El concurso pide un clasificador de 8 clases para series de 4 × 256, evaluado muestra a muestra con una media geométrica por clase y un término de velocidad que crece con la raíz del número de respuestas en 60 s. Lo que más importó fue **generalizar a la distribución del test** y **responder en mucho menos de un milisegundo**.

## 1. Entender el cambio de distribución

El baseline oficial llegaba al 98 % en train, pero solo al 43 % en la demo. Comparar train con la demo explicó por qué:

- **Atajos que solo existen en train:** el metadato `level` estaba desplazado en las clases 3 y 5, el canal 3 tenía un offset propio en las clases 2 y 6, y cerca de la mitad de la clase 6 formaba un subgrupo aislado (el canal 2 copiaba al canal 1) que nunca aparece en test.
- **Un test más sucio:** alrededor de 1,3 veces más ruido, el doble de valores ausentes, más tramos congelados y picos en el 18 % de las muestras.
- **Lo que no cambia:** el papel de cada canal, los valores de los metadatos y que las clases se definen en Hz (`sample_rate` divide por dos la frecuencia en ciclos por muestra).

Solo con quitar los atajos, el baseline pasó del 43 % al 54 % en la demo.

## 2. Entradas robustas y un simulador tipo test

Cada muestra pasa por la misma limpieza robusta (picos y tramos congelados a NaN, interpolación, mediana/IQR por canal, rejilla común de 1 Hz). Para entrenar construimos un **simulador de corrupciones** calibrado con la demo (ruido extra, picos, NaN aislados y en bloque, tramos congelados), entrenamos con clases equilibradas, quitamos el subgrupo aislado de la clase 6 y validamos con un hold-out corrompido de 14 250 muestras × 4 versiones corrompidas.

## 3. Profesores en Kaggle, alumnos rápidos en Numba

Los modelos potentes son demasiado lentos para esta métrica, así que usamos **destilación de conocimiento**:

- **Profesores** (offline, GPU T4 de Kaggle, logits fuera de fold con 5 folds): InceptionTime con 5 semillas por fold (89,0 % en la demo) y ConvTran (85,8 %). Combinados llegaron al 90,0 %.
- **Alumnos:** características de MiniRocket, cuyos kernels fijos de ±1/2 son casi gratis en Numba, y un MLP pequeño. El alumno entregado usa 250 + 3 características y una capa oculta de 128, destilado de un ensemble de tres alumnos MLP de 256 con 8 pasadas corrompidas extra que InceptionTime etiquetó fuera de fold.
- **Motor de inferencia:** una llamada `@njit` por muestra con buffers reservados de antemano, compilada y calentada en `Model()`. Reproduce exactamente las predicciones de PyTorch.

## 4. Elegir según la métrica

Sin abstenciones, el término de cada clase solo depende de su recall, e ir el doble de rápido multiplica el score por ~1,41. Por eso comparamos los candidatos con `√(N/8)·gm_f`, usando la gm_f de la validación corrompida (la fiable) y la latencia medida en Colab con el harness oficial (33 modelos, rondas intercaladas). El modelo entregado fue el único entre los dos mejores tanto por validación como por la demo, su receta era estable con tres semillas (gm_f 0,518–0,523) y su latencia quedó confirmada en tres rondas en Colab.

## 5. Lo que no funcionó

| Idea | Por qué se descartó |
|---|---|
| BLAS en el MLP | Exacto, pero no más rápido: el MLP es solo el 2–5 % de la latencia |
| Sesgos por clase ajustados con la validación | Sin ganancia: los alumnos ya estaban equilibrados en datos derivados de train |
| Pooling extra de MiniRocket (MPV, MIPV) y selección de características guiada por el profesor | Mejor validación, peor demo, sobre todo en la clase 6 |
| Profesores de la validación cruzada anterior | Habían visto casi todos los datos, así que sus etiquetas blandas eran demasiado seguras |
| ConvTran como profesor adicional | El profesor mejoraba, pero el alumno con MiniRocket no podía aprender esa información extra |
| Alumno CNN diminuto (~2 k pesos) | No era más rápido que MiniRocket y acertaba claramente menos con ese tamaño |

En [Posibles mejoras y líneas futuras](README.es.md#posibles-mejoras-y-líneas-futuras) están las direcciones que exploraríamos después.
