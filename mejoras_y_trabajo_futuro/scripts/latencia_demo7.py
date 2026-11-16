"""Latencia en el harness oficial sobre test_demo de 7 variantes, en las mismas rondas (orden aleatorio por ronda).

Variantes (carpetas en ``modelos_harness/``; las que faltan se crean aquí):
- ``base30`` (#30), ``mejor_actual`` (#30 + sesgos «suave», referencia);
- ``rapido253_f32``: pesos del #30 en ``motor_rapido`` con preprocesado float32;
- ``pesos150_f64`` / ``pesos150_f32``, ``rank150_f64`` / ``rank150_f32``: alumnos de 150 rasgos.

Cada ronda: ``harness.py evaluate --source file:data/test_demo.npz --runs 1`` con los 7 modelos, 1 hilo. Por modelo
se guarda la media y la mediana de latencia por muestra de cada ronda; el resumen da la mediana y el mínimo entre
rondas y el score estimado en Colab: latencia local × (140 µs / latencia local del #30) y √(N/8)·gm_f(val-corr).

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\latencia_demo7.py [--rondas 3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import comun06 as c6
import crear_harness06 as ch
import motor_rapido as mr

COLAB_30_US = 140.0
MODELOS = ("base30", "mejor_actual", "rapido253_f32", "pesos150_f64", "pesos150_f32", "rank150_f64", "rank150_f32")
ALUMNOS = {"pesos150_f64": "pesos30150_h128_ens3_mas8_s0_f64", "pesos150_f32": "pesos30150_h128_ens3_mas8_s0_f32",
           "rank150_f64": "rank150_h128_ens3_mas8_s0_f64", "rank150_f32": "rank150_h128_ens3_mas8_s0_f32"}


def carpetas():
    MH = c6.AQUI / "modelos_harness"
    z30 = c6.cargar_npz(c6.OPT / "artifacts" / "val_logits_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0.npz")
    dst = MH / "mejor_actual"
    if not dst.exists():
        shutil.copytree(c6.AQUI / "mejor_actual", dst, ignore=shutil.ignore_patterns("__pycache__"))
    if not (MH / "rapido253_f32").exists():
        P = c6.cargar_npz(c6.NPZ_30)
        P["pi"] = P["pi"].copy()
        P["pi"][mr.H_PREP] = 1
        ruta = c6.ART / "alumno_base30_f32.npz"
        np.savez(ruta, **P)
        ch.crear("rapido253_f32", ruta, "motor_rapido", None, "pesos del #30 con preprocesado float32", z30["test"])
    for m, a in ALUMNOS.items():
        if not (MH / m).exists():
            zr = c6.cargar_npz(c6.ART / f"val_logits_{a}.npz")
            ch.crear(m, c6.ART / f"alumno_{a}.npz", "motor_rapido", None, f"alumno rápido {a}", zr["test"])


def gm_val():
    """gm_f en la validación corrompida («corrompido», 14 250 filas) de cada variante."""
    S = json.loads((c6.ART / "sesgos_sin_etiquetas.json").read_text(encoding="utf-8"))
    g = {"base30": S["base30"]["sin"]["val"]["gm_f_corrompido"],
         "mejor_actual": S["base30"]["suave"]["val"]["gm_f_corrompido"]}
    # rapido253_f32: mismas predicciones que el #30 en 20 000 filas corrompidas (verificar_motor.json); se mide en val
    D = c6.co.datos()
    X = c6.co.serie_version("corrompido")[D["i_va"]]
    P = c6.cargar_npz(c6.ART / "alumno_base30_f32.npz")
    F = c6.features_lote(mr, X, D["sr_pool"][D["i_va"]], P["pi"], P["pf"], P["pd"])
    g["rapido253_f32"] = float(c6.gm_f(c6.recalls(D["y_pool"][D["i_va"]], c6.logits_lote(mr, F, P["pi"], P["pf"]).argmax(1))))
    for m, a in ALUMNOS.items():
        lg = c6.cargar_npz(c6.ART / f"val_logits_{a}.npz")["corrompido"]
        g[m] = float(c6.gm_f(c6.recalls(D["y_pool"][D["i_va"]], lg.argmax(1))))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rondas", type=int, default=3)
    a = ap.parse_args()
    c6.co.no_suspender()
    carpetas()
    env = {**os.environ, "NUMBA_CACHE_DIR": str(c6.CACHE / "numba_harness")}
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[v] = "1"
    ruta_j = c6.ART / "latencias_harness.json"
    rondas = json.loads(ruta_j.read_text(encoding="utf-8"))["rondas"] if ruta_j.exists() else []
    for r in range(a.rondas):
        k = len(rondas)
        orden = list(MODELOS[k % len(MODELOS):] + MODELOS[:k % len(MODELOS)])   # orden rotado y alterno
        if k % 2:
            orden = orden[::-1]
        out = c6.ART / "harness" / f"demo7_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = [sys.executable, "harness.py", "evaluate", "--source", "file:data/test_demo.npz", "--runs", "1",
               "--out", str(out)]
        for m in orden:
            cmd += ["--model", str(c6.AQUI / "modelos_harness" / m / "model.py")]
        p = subprocess.run(cmd, cwd=c6.ROOT / "materials", env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-3000:])
        fila = {"orden": orden}
        for m in MODELOS:
            j = json.loads((out / f"{m}_model" / "run_0.json").read_text())
            fila[m] = {"media_us": j["latency_s"]["mean"] * 1e6, "mediana_us": j["latency_s"]["median"] * 1e6,
                       "acc": j["accuracy_all"], "score400": j["score"], "n_correct": j["n_correct"]}
        rondas.append(fila)
        c6.log(f"ronda demo7 {r}: " + " | ".join(f"{m} {fila[m]['media_us']:.1f}" for m in MODELOS), "progreso_latencia.log")
        ruta_j.write_text(json.dumps({"rondas": rondas}, indent=1, ensure_ascii=False), encoding="utf-8")
    g = gm_val()
    med30 = float(np.median([f["base30"]["mediana_us"] for f in rondas]))
    ratio = COLAB_30_US / med30
    res = {}
    for m in MODELOS:
        media = np.array([f[m]["media_us"] for f in rondas])
        mediana = np.array([f[m]["mediana_us"] for f in rondas])
        raz = np.array([f[m]["mediana_us"] / f["base30"]["mediana_us"] for f in rondas])
        lat = float(np.median(mediana))          # mediana (entre rondas) de la mediana por muestra
        colab = lat * ratio
        res[m] = {"lat_mediana_us": lat, "lat_mediana_us_min": float(mediana.min()),
                  "lat_media_us_mediana": float(np.median(media)), "lat_media_us_min": float(media.min()), "razon_vs_base30": float(np.median(raz)),
                  "colab_us_est": colab, "gm_f_val_corr": g[m],
                  "score_colab_est": c6.score_estimado(g[m], colab),
                  "score_est_rel30": 120 * np.sqrt(med30 / lat) * g[m] / 0.520, "acc_test_demo": rondas[0][m]["acc"],
                  "score400": rondas[0][m]["score400"], "n_correct": rondas[0][m]["n_correct"]}
        c6.log(f"{m:14s} mediana {lat:6.1f} µs (mín {mediana.min():6.1f}; media {np.median(media):6.1f}) "
               f"×#30 {res[m]['razon_vs_base30']:.3f} | Colab est {colab:6.1f} µs | gm_f val {g[m]:.4f} | score est "
               f"{res[m]['score_colab_est']:.1f} (rel #30 {res[m]['score_est_rel30']:.1f}) | demo {res[m]['n_correct']}/400 score400 {res[m]['score400']:.3f}",
               "progreso_latencia.log")
    ruta_j.write_text(json.dumps({"rondas": rondas, "ratio_colab_local": ratio, "resumen": res}, indent=1, default=float,
                                 ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
