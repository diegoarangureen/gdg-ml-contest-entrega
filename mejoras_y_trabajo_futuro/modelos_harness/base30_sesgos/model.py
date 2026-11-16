"""GDG on Campus UAM ML Contest: #30 + sesgos por clase sin etiquetas (método suave)

Pipeline completo (limpieza, MiniRocket, candidatos, escalado y MLP) en una llamada de numba por muestra
(``motor_multi.py``). Pesos en ``modelo_numba.npz``, junto a este fichero (rutas relativas a ``__file__``).
"""
from pathlib import Path

from motor_multi import Model as _Motor

HERE = Path(__file__).resolve().parent


class Model(_Motor):
    def __init__(self):
        super().__init__(HERE / "modelo_numba.npz")
