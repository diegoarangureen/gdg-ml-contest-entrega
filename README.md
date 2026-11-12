# GDG UAM ML Contest: entrega

Clasificador de series temporales de 4 canales y 256 pasos en 8 clases.

## Contenido

| Archivo | Función |
| --- | --- |
| `model.py` | Clase `Model`: carga los pesos en `__init__` y `predict(x: Sample) -> int` devuelve una clase de 0 a 7 |
| `motor_multi.py` | Módulo auxiliar: limpieza de la señal, características y red en Numba |
| `modelo_numba.npz` | Modelo ya entrenado (se carga con una ruta relativa a `model.py`) |
| `requirements.txt` | Dependencias: `numpy` y `numba` |

## Modelo

- **Limpieza robusta** de la señal: picos y NaN, mediana y escala común, rejilla de 1 Hz según `sample_rate`.
- **253 características:** 250 de MiniRocket y 3 de asimetría temporal.
- **Red 253 → 128 → 8**, destilada de un InceptionTime entrenado con series corrompidas como las de test.

Todo se ejecuta en una sola llamada de Numba por muestra, en un hilo de CPU y sin acceso a internet. No usa `level` ni `group`, y nunca se abstiene.

## Comprobación

Desde un clon de [ml-contest-materials](https://github.com/GDG-UAM/ml-contest-materials) con los datos descargados:

```bash
git clone <URL de este repositorio> submission-check
python harness.py evaluate --model submission-check/model.py --source file:data/test_demo.npz --runs 1
```

Resultado esperado en la demo: accuracy 0,825 y score 3,488. `Model()` compila Numba al arrancar: unos 15 s la primera vez, dentro del límite de 120 s.
