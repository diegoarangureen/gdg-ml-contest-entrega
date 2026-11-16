"""GDG on Campus UAM ML Contest: alumno rápido pesos30150_h128_ens3_mas8_s0_f32

Pipeline completo (limpieza, MiniRocket, candidatos, escalado y MLP) en una llamada de numba por muestra
(``motor_rapido.py``). Pesos en ``modelo_numba.npz``, junto a este fichero (rutas relativas a ``__file__``).
"""
from pathlib import Path

from motor_rapido import Model as _Motor

HERE = Path(__file__).resolve().parent


class Model(_Motor):
    def __init__(self):
        super().__init__(HERE / "modelo_numba.npz")
