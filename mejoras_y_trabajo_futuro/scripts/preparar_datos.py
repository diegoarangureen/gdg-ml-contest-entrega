"""Precalcula, en ``06_mejoras_sesgos_velocidad/cache``, lo que necesita la destilación del alumno rápido:

* features del plan del #30 (``A250_ppv``: 250 PPV + 3 candidatos) en las 13 versiones del pool de la receta del
  #30 (limpio, corrompido, pasada1–11) y en test_demo. Las features de cualquier subconjunto de esas 250 son sus
  columnas (cada PPV es independiente), así que no hace falta recalcular nada por plan;
* logits del profesor del #30 (el ensemble 3 × MLP 256 ``A250_ppv__ince_kl_a0.5_T2_h256_mas4_ens3``, de solo lectura
  en 05/opt/artifacts) en esas 13 versiones: ``cache/profesor_ens3_<versión>.npy``.

Las series corrompidas se borran al terminar cada versión (389 MB cada una) para no llenar el disco.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\preparar_datos.py
"""
from __future__ import annotations

import time

import numpy as np

import comun06 as c6
import planes

VERSIONES = list(c6.co.VERSIONES) + c6.EXTRA_30


def main():
    c6.co.no_suspender()
    plan = planes.plan("A250_ppv")
    P = c6.cargar_npz(c6.NPZ_PROFESOR_30)
    for v in VERSIONES + ["test"]:
        t0 = time.perf_counter()
        F = c6.co.features("A250_ppv", plan, v)
        ruta = c6.CACHE / f"profesor_ens3_{v}.npy"
        if not ruta.exists():
            np.save(ruta, c6.logits_lote(c6.mm, F, P["pi"], P["pf"]))
        serie = c6.CACHE / "series" / f"{v}.npy"
        if serie.exists():
            serie.unlink()
        c6.log(f"preparar_datos {v}: features {F.shape} y logits del profesor en {time.perf_counter() - t0:.0f} s")
    # control: el #30 sobre estas features reproduce sus logits de val guardados en 05/opt
    D = c6.co.datos()
    P30 = c6.cargar_npz(c6.NPZ_30)
    z = c6.cargar_npz(c6.OPT / "artifacts" / "val_logits_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0.npz")
    for v in c6.VERSIONES_VAL:
        lg = c6.logits_lote(c6.mm, c6.co.features("A250_ppv", plan, v)[D["i_va"]], P30["pi"], P30["pf"])
        c6.log(f"control {v}: |Δlogits| máx frente a 05/opt {np.abs(lg - z[v]).max():.2e}")


if __name__ == "__main__":
    main()
