"""Entrega del GDG on Campus UAM ML Contest: clasificador de series temporales (8 clases) en numba.

Modelo «A250, MLP 128 ← ensemble, 8 pasadas»:
- **Features:** 250 de MiniRocket (PPV) en 218 convoluciones, más 3 candidatos sin FFT, sobre la serie limpia.
- **Clasificador:** un MLP 253 → 128 → 8, destilado de un ensemble de 3 MLP 256 que a su vez se destilaron de un InceptionTime entrenado en Kaggle con series corrompidas.

Todo el pipeline (preprocesado, features, escalado y MLP) va en una sola llamada de numba por muestra
(``motor_multi.py``). Los pesos están en ``modelo_numba.npz``, junto a este fichero, y se cargan con rutas relativas
a ``__file__``.
"""
from pathlib import Path

from motor_multi import Model as _Motor

HERE = Path(__file__).resolve().parent


class Model(_Motor):
    def __init__(self):
        super().__init__(HERE / "modelo_numba.npz")
