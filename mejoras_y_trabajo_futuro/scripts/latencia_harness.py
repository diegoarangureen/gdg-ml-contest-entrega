"""Latencia en el harness oficial (local, 1 hilo): rondas intercaladas de las 4 variantes en orden aleatorio.

Cada ronda es una invocación de ``materials/harness.py evaluate`` con los 4 modelos, ``--runs 1`` y
``time_budget_s`` corto, sobre el generador por defecto (train), con ``NUMBA_NUM_THREADS=1`` y demás hilos a 1. Se
guarda la latencia media de ``predict`` de cada modelo en cada ronda; el resumen es la mediana por modelo y la razón
emparejada (misma ronda) frente a ``base30``.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\latencia_harness.py [--rondas 6] [--presupuesto 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

import comun06 as c6

MODELOS = ("base30", "base30_sesgos", "rapido", "rapido_sesgos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rondas", type=int, default=6)
    ap.add_argument("--presupuesto", type=float, default=8.0)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    c6.co.no_suspender()
    env = {**os.environ, "NUMBA_CACHE_DIR": str(c6.CACHE / "numba_harness")}
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[v] = "1"
    rng = np.random.default_rng(a.semilla)
    ruta_j = c6.ART / "latencia_harness.json"
    todo = json.loads(ruta_j.read_text(encoding="utf-8")) if ruta_j.exists() else {"rondas": []}
    for r in range(a.rondas):
        orden = [MODELOS[i] for i in rng.permutation(len(MODELOS))]
        out = c6.ART / "harness" / f"lat_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = [sys.executable, "harness.py", "evaluate", "--runs", "1", "--set", f"time_budget_s={a.presupuesto}",
               "--out", str(out)]
        for m in orden:
            cmd += ["--model", str(c6.AQUI / "modelos_harness" / m / "model.py")]
        p = subprocess.run(cmd, cwd=c6.ROOT / "materials", env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-3000:])
        fila = {"orden": orden, "out": str(out.relative_to(c6.ROOT))}
        with open(out / "leaderboard.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                nombre = row["project"].removesuffix("_model")
                fila[nombre] = {"lat_us": float(row["mean_latency_s"]) * 1e6, "n": float(row["mean_n_scored"]),
                                "acc": float(row["mean_accuracy_all"]), "score": float(row["mean"])}
        todo["rondas"].append(fila)
        c6.log("ronda harness: " + " | ".join(f"{m} {fila[m]['lat_us']:.1f} µs" for m in MODELOS), "progreso_latencia.log")
        ruta_j.write_text(json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")
    R = todo["rondas"]
    res = {}
    for m in MODELOS:
        lat = np.array([f[m]["lat_us"] for f in R])
        raz = np.array([f[m]["lat_us"] / f["base30"]["lat_us"] for f in R])
        res[m] = {"rondas": len(R), "lat_mediana_us": float(np.median(lat)), "lat_media_us": float(lat.mean()),
                  "lat_min_us": float(lat.min()), "razon_vs_base30_mediana": float(np.median(raz)),
                  "razon_vs_base30_media": float(raz.mean()), "razon_desv": float(raz.std())}
        c6.log(f"{m:14s} mediana {res[m]['lat_mediana_us']:6.1f} µs | ×base30 {res[m]['razon_vs_base30_mediana']:.3f} "
               f"(media {res[m]['razon_vs_base30_media']:.3f} ± {res[m]['razon_desv']:.3f})", "progreso_latencia.log")
    todo["resumen"] = res
    ruta_j.write_text(json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
