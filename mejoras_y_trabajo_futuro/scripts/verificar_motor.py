"""Comprueba ``motor_rapido.py`` frente al motor del #30 y mide el preprocesado y el pipeline (1 hilo).

1. Con los pesos del #30 (``pi[H_PREP] = 0``), las features de test_demo y de 20 000 series corrompidas del pool
   deben ser idénticas bit a bit a las de ``motor_multi`` y las predicciones, iguales.
2. Con ``pi[H_PREP] = 1`` (preprocesado float32) se mide cuánto cambian las features.
3. Latencia por muestra (bucles dentro de numba, sin despacho), intercalando motores, mínimo de varias repeticiones.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\verificar_motor.py
"""
from __future__ import annotations

import os

os.environ["NUMBA_NUM_THREADS"] = "1"

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

import comun06 as c6  # noqa: E402
import motor_multi as mm  # noqa: E402
import motor_rapido as mr  # noqa: E402


def bucles(M):
    @njit
    def b_prep(X, ws64, z, reps):
        for _ in range(reps):
            for i in range(X.shape[0]):
                M._preprocesar(X[i], ws64, z)

    @njit
    def b_pred(X, sr, pi, pf, pd, ws64, ws32, reps):
        s = 0
        for _ in range(reps):
            for i in range(X.shape[0]):
                s += M.predecir(X[i], sr[i], pi, pf, pd, ws64, ws32)
        return s

    return b_prep, b_pred


def main():
    P = c6.cargar_npz(c6.NPZ_30)
    pi, pf, pd = P["pi"], P["pf"], P["pd"]
    pi32 = pi.copy()
    pi32[mr.H_PREP] = 1
    D = c6.co.datos()
    Xc = np.ascontiguousarray(c6.co.serie_version("corrompido")[:20000])
    src = np.ascontiguousarray(D["sr_pool"][:20000])
    res = {}
    for nombre, X, sr in (("test_demo", D["X_te"], D["sr_te"]), ("corrompido_20k", Xc, src)):
        f_ref = c6.features_lote(mm, X, sr, pi, pf, pd)
        f_new = c6.features_lote(mr, X, sr, pi, pf, pd)
        f_32 = c6.features_lote(mr, X, sr, pi32, pf, pd)
        p_ref = c6.logits_lote(mm, f_ref, pi, pf).argmax(1)
        p_32 = c6.logits_lote(mr, f_32, pi, pf).argmax(1)
        res[nombre] = {"identicas_f64": bool(np.array_equal(f_ref, f_new)),
                       "max_dif_f32": float(np.abs(f_ref - f_32).max()),
                       "frac_feats_distintas_f32": float((f_ref != f_32).mean()),
                       "pred_iguales_f32": float((p_ref == p_32).mean())}
        print(nombre, res[nombre], flush=True)
    # predecir muestra a muestra con el Model del motor rápido = lote del motor del #30
    ws64, ws32 = mr.crear_workspace(pi)
    p_uno = np.array([mr.predecir(D["X_te"][i], D["sr_te"][i], pi, pf, pd, ws64, ws32) for i in range(400)])
    p_ref = np.load(c6.OPT / "artifacts" / "pred_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0_test.npy")
    res["predecir_rapido_igual_que_30"] = int((p_uno == p_ref).sum())
    print("predecir (rápido, pesos #30) = predicciones guardadas del #30:", res["predecir_rapido_igual_que_30"], "/ 400")

    # latencias
    X, sr = D["X_te"], D["sr_te"]
    z = np.zeros((4, 256), np.float32)
    bm_prep, bm_pred = bucles(mm)
    br_prep, br_pred = bucles(mr)
    w64m, w32m = mm.crear_workspace(pi)
    w64r, w32r = mr.crear_workspace(pi)
    w64r32 = np.zeros(len(w64r), np.float32)
    casos = {
        "prep #30 (f64)": lambda r: bm_prep(X, w64m, z, r),
        "prep rápido f64": lambda r: br_prep(X, w64r, z, r),
        "prep rápido f32": lambda r: br_prep(X, w64r32, z, r),
        "predecir #30": lambda r: bm_pred(X, sr, pi, pf, pd, w64m, w32m, r),
        "predecir rápido f64": lambda r: br_pred(X, sr, pi, pf, pd, w64r, w32r, r),
        "predecir rápido f32": lambda r: br_pred(X, sr, pi32, pf, pd, w64r, w32r, r),
    }
    for f in casos.values():
        f(1)
    ts = {k: [] for k in casos}
    for _ in range(12):
        for k, f in casos.items():
            t0 = time.perf_counter()
            f(10)
            ts[k].append((time.perf_counter() - t0) / (10 * len(X)) * 1e6)
    res["latencia_us"] = {k: {"min": float(np.min(v)), "mediana": float(np.median(v))} for k, v in ts.items()}
    for k, v in res["latencia_us"].items():
        print(f"{k:22s} min {v['min']:6.1f} µs | mediana {v['mediana']:6.1f} µs")
    (c6.ART / "verificar_motor.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
