"""Utilidades de 06_mejoras_sesgos_velocidad: reutiliza ``05_profesores_kaggle_destilacion/opt/comun_opt.py`` sin
escribir en las carpetas existentes.

* Las cachés (series corrompidas, features por plan) y los logs van a ``06_mejoras_sesgos_velocidad/cache`` y
  ``06_mejoras_sesgos_velocidad/artifacts`` (se redirigen ``comun_opt.CACHE`` y ``comun_opt.ART``).
* La caché de numba va a ``06_mejoras_sesgos_velocidad/cache/numba``.
* El modelo #30 es ``submission/modelo_numba.npz`` (idéntico a ``alumno_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0``) y su
  profesor, el ensemble 3 × MLP 256 ``A250_ppv__ince_kl_a0.5_T2_h256_mas4_ens3`` (de 05/opt/artifacts, solo lectura).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
OPT = ROOT / "05_profesores_kaggle_destilacion" / "opt"
CACHE = AQUI / "cache"
ART = AQUI / "artifacts"
os.environ["NUMBA_CACHE_DIR"] = str(CACHE / "numba")
sys.dont_write_bytecode = True
for _p in (str(OPT), str(AQUI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import comun_opt as co  # noqa: E402

co.CACHE = CACHE
co.ART = ART
CACHE.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

mm = co.mm                                    # motor del #30 (05/opt/motor_multi.py, idéntico a submission/)
K = 8
VERSIONES_VAL = co.VERSIONES_CORR
NPZ_30 = ROOT / "submission" / "modelo_numba.npz"
PROFESOR_30 = "A250_ppv__ince_kl_a0.5_T2_h256_mas4_ens3"
NPZ_PROFESOR_30 = OPT / "artifacts" / f"alumno_{PROFESOR_30}.npz"
EXTRA_30 = [f"pasada{i}" for i in range(4, 12)]   # las 8 pasadas extra de la receta del #30


def log(msg, fichero="progreso_06.log"):
    co.log(msg, fichero)


def cargar_npz(ruta):
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}


def recalls(y, p):
    return co.recalls(y, p)


def gm_f(rec):
    return co.gm_f(rec)


def metricas(y, p):
    m = co.cv2.cm.metrics(y, p)
    return {"acc": float(m["accuracy"]), "gm_f": float(m["gm_f"]), "recall": [float(r) for r in m["recall"]],
            "score_harness_400": float(m["harness_score"])}


def plegar(pi, pf, b):
    """pf con b sumado al bias de la última capa del MLP (logits + b, latencia idéntica)."""
    o_lay, off = int(pi[mm.H_OLAY]), int(pi[mm.H_FOW])
    n_capas = int(pi[mm.H_NL])
    for l in range(n_capas):
        n_in, n_out = int(pi[o_lay + 3 * l]), int(pi[o_lay + 3 * l + 1])
        ob = off + n_in * n_out
        if l == n_capas - 1:
            pf2 = pf.copy()
            pf2[ob:ob + n_out] += np.asarray(b, np.float32)
            return pf2
        off = ob + n_out
    raise ValueError("MLP sin capas")


def logits_lote(M, F, pi, pf):
    F = np.ascontiguousarray(F, np.float32)
    return M.logits_lote(F, pi, pf, np.zeros((len(F), K), np.float32))


def features_lote(M, X, sr, pi, pf, pd):
    return M.features_lote(np.ascontiguousarray(X, np.float32), np.asarray(sr, np.float64), pi, pf, pd,
                           np.zeros((len(X), int(pi[M.H_DIN])), np.float32))


def score_estimado(gm, us):
    return co.cv2.score_estimado(gm, us)
