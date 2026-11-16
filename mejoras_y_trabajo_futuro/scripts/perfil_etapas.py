"""Perfil de latencia por etapa del motor del #30 (1 hilo), con bucles dentro de numba (sin coste de despacho).

Etapas: preprocesado, ventanas + MiniRocket (+ candidatos), escalado + MLP. Se mide sobre las 400 series de
test_demo, varias repeticiones, separando sample_rate 1 (2 ventanas) y 2 (1 ventana).

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\perfil_etapas.py [--npz ruta] [--motor motor_multi|motor_rapido]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import importlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

import comun06 as c6  # noqa: E402


def construir_bucles(M):
    @njit(cache=False)
    def b_prep(X, ws64, z, reps):
        for _ in range(reps):
            for i in range(X.shape[0]):
                M._preprocesar(X[i], ws64, z)

    @njit(cache=False)
    def b_feat(X, sr, pi, pf, pd, ws64, ws32, reps):
        for _ in range(reps):
            for i in range(X.shape[0]):
                M._features(X[i], sr[i], pi, pf, pd, ws64, ws32)

    @njit(cache=False)
    def b_pred(X, sr, pi, pf, pd, ws64, ws32, reps):
        s = 0
        for _ in range(reps):
            for i in range(X.shape[0]):
                s += M.predecir(X[i], sr[i], pi, pf, pd, ws64, ws32)
        return s

    return b_prep, b_feat, b_pred


def medir(fn, args, n, reps=20):
    fn(*args[:-1], 1)
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        fn(*args[:-1], reps)
        ts.append((time.perf_counter() - t0) / (reps * n) * 1e6)
    return float(np.min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(c6.NPZ_30))
    ap.add_argument("--motor", default="motor_multi")
    args = ap.parse_args()
    M = importlib.import_module(args.motor)
    P = c6.cargar_npz(args.npz)
    pi, pf, pd = P["pi"], P["pf"], P["pd"]
    D = c6.co.datos()
    X, sr = D["X_te"], D["sr_te"]
    ws64, ws32 = M.crear_workspace(pi)
    z = np.zeros((4, 256), np.float32)
    b_prep, b_feat, b_pred = construir_bucles(M)
    print(f"{args.npz}: {int(pi[M.H_NM])} convoluciones, {int(pi[M.H_NF])} feats MiniRocket, d_in {int(pi[M.H_DIN])}")
    for nombre, m in (("sr=1", sr < 1.5), ("sr=2", sr > 1.5), ("todas", np.ones(len(sr), bool))):
        Xm, srm = np.ascontiguousarray(X[m]), np.ascontiguousarray(sr[m])
        n = len(Xm)
        t_prep = medir(b_prep, (Xm, ws64, z, 0), n)
        t_feat = medir(b_feat, (Xm, srm, pi, pf, pd, ws64, ws32, 0), n)
        t_pred = medir(b_pred, (Xm, srm, pi, pf, pd, ws64, ws32, 0), n)
        print(f"{nombre:6s} (n={n:3d}) | preprocesado {t_prep:6.1f} µs | ventanas+MiniRocket+cand {t_feat - t_prep:6.1f} | "
              f"escalado+MLP {t_pred - t_feat:6.1f} | total {t_pred:6.1f} µs")
    print(f"sample_rate en test_demo: sr=1 {np.mean(sr < 1.5):.2f}, sr=2 {np.mean(sr > 1.5):.2f}; "
          f"en train: sr=1 {np.mean(D['sr_pool'] < 1.5):.2f}")


if __name__ == "__main__":
    main()
