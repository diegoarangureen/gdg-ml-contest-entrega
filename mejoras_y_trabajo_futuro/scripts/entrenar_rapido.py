"""Mejora 4: alumno rápido. MLP 128 destilado del mismo profesor que el #30 (ensemble 3 × MLP 256 ← InceptionTime),
con ~150 de las 250 features MiniRocket del #30 (+ los 3 candidatos), exportado a ``motor_rapido.py``.

**Selección de features** (``--seleccion``), siempre un subconjunto de las 250 del #30, sin mirar test_demo:
- ``rank``: las N primeras del ranking de importancia de v2 con el que se eligieron las 250 (A250 = las 250 primeras);
- ``pesos30``: las N con mayor norma de su columna en la primera capa del MLP del #30 (entradas estandarizadas).

**Receta**: la del #30 (``05/opt/entrenar.py``, cuyo bucle ``entrenar`` se reutiliza sin cambios): KL con α 0,5 y
T 2 + CE con pesos por clase, AdamW, coseno, parada temprana con la gm_f media de las 4 versiones corrompidas de la
validación; 13 versiones del pool (limpio, corrompido, pasada1–11) con los logits del ensemble profesor
(``preparar_datos.py``).

**Exportación** a ``motor_rapido`` con ``prep`` 0 (preprocesado float64) o 1 (float32). Se comprueba que las features
del motor sobre test_demo son las columnas de las del #30 y que ``predecir`` muestra a muestra coincide con el lote.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\entrenar_rapido.py --n 150 --seleccion rank --semilla 0

Salidas en ``artifacts/``: ``alumno_<nombre>.npz``, ``val_logits_<nombre>.npz`` y ``alumnos_06.json``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import comun06 as c6  # noqa: E402
import entrenar as ent05  # noqa: E402  (05/opt/entrenar.py: solo se usa su función entrenar)
import motor_rapido as mr  # noqa: E402
import planes  # noqa: E402

VERSIONES = list(c6.co.VERSIONES) + c6.EXTRA_30


def seleccion(n, criterio):
    """Índices (de aeon, crecientes) de las n features elegidas y sus columnas dentro de las 250 del #30."""
    sel250 = np.asarray(planes.plan("A250_ppv")["seleccion"])
    if criterio == "rank":
        ranking = np.asarray(json.loads((c6.co.cv2.ART / "importancias.json").read_text(encoding="utf-8"))
                             ["ranking_minirocket"])
        assert np.array_equal(np.sort(ranking[:250]), sel250)
        elegidas = np.sort(ranking[:n])
    elif criterio == "pesos30":
        P = c6.cargar_npz(c6.NPZ_30)
        pi, pf = P["pi"], P["pf"]
        d_in, h = int(pi[c6.mm.H_DIN]), int(pi[int(pi[c6.mm.H_OLAY]) + 1])
        W1 = pf[int(pi[c6.mm.H_FOW]):int(pi[c6.mm.H_FOW]) + d_in * h].reshape(h, d_in)
        imp = np.linalg.norm(W1[:, :250], axis=0)
        elegidas = np.sort(sel250[np.argsort(-imp)[:n]])
    else:
        raise ValueError(criterio)
    cols = np.searchsorted(sel250, elegidas)
    assert np.array_equal(sel250[cols], elegidas)
    return elegidas, np.concatenate([cols, [250, 251, 252]])


def n_convoluciones(elegidas):
    pi, _, _ = construir(elegidas, np.zeros(len(elegidas) + 3), np.ones(len(elegidas) + 3),
                         [np.zeros((8, len(elegidas) + 3), np.float32)], [np.zeros(8, np.float32)])
    return int(pi[mr.H_NM])


def construir(elegidas, mu, sd, Ws, bs, prep=0):
    mrk = c6.co.minirocket("672")
    return mr.construir(mrk.parameters, elegidas, "baratos", mu, sd, Ws, bs, "relu", indices=type(mrk)._indices,
                        mascaras=np.ones(len(elegidas), np.int64), prep=prep)


def preparar(cols):
    D = c6.co.datos()
    i_tr, i_va, y = D["i_tr"], D["i_va"], D["y_pool"]
    plan = planes.plan("A250_ppv")
    n_tr, d = len(i_tr), len(cols)
    X_tr = np.empty((len(VERSIONES) * n_tr, d), np.float16)
    T_tr = np.empty((len(VERSIONES) * n_tr, c6.K), np.float32)
    X_va = {}
    for j, v in enumerate(VERSIONES):
        F = np.asarray(c6.co.features("A250_ppv", plan, v))
        X_tr[j * n_tr:(j + 1) * n_tr] = F[i_tr][:, cols]
        T_tr[j * n_tr:(j + 1) * n_tr] = np.load(c6.CACHE / f"profesor_ens3_{v}.npy")[i_tr]
        if v in c6.VERSIONES_VAL:
            X_va[v] = np.ascontiguousarray(F[i_va][:, cols], np.float32)
    Xd = X_tr.astype(np.float64)
    mu = Xd.mean(0)
    sd = Xd.std(0)
    sd = np.where(sd > 1e-7, sd, 1.0)
    del Xd
    X_te = np.ascontiguousarray(np.asarray(c6.co.features("A250_ppv", plan, "test"))[:, cols], np.float32)
    return {"X_tr": X_tr, "T_tr": T_tr, "y_tr": np.tile(y[i_tr], len(VERSIONES)), "X_va": X_va, "y_va": y[i_va],
            "mu": mu, "sd": sd, "X_te": X_te, "y_te": D["y_te"], "d": d}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seleccion", default="rank", choices=["rank", "pesos30"])
    ap.add_argument("--oculta", type=int, default=128)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--paciencia", type=int, default=8)
    a = ap.parse_args()
    nombre = f"{a.seleccion}{a.n}_h{a.oculta}_ens3_mas8_s{a.semilla}"
    c6.co.no_suspender()
    elegidas, cols = seleccion(a.n, a.seleccion)
    n_conv = n_convoluciones(elegidas)
    c6.log(f"{nombre}: {a.n} features MiniRocket en {n_conv} convoluciones (#30: 250 en 218)", "progreso_entrenar.log")
    t0 = time.perf_counter()
    P = preparar(cols)
    c6.log(f"{nombre}: datos {P['X_tr'].shape} en {time.perf_counter() - t0:.0f} s", "progreso_entrenar.log")
    args = SimpleNamespace(oculta=a.oculta, oculta2=0, perdida="kl", alfa=0.5, temp=2.0, refuerzo=1.0,
                           semilla=a.semilla, epocas=a.epocas, paciencia=a.paciencia, nombre=nombre)
    Ws, bs, mejor_ep, hist, minutos = ent05.entrenar(P, args)
    D = c6.co.datos()
    res = {"n": a.n, "seleccion": a.seleccion, "seleccion_aeon": elegidas.tolist(), "n_convoluciones": n_conv,
           "oculta": a.oculta, "semilla": a.semilla, "mejor_epoca": mejor_ep, "epocas": len(hist), "minutos": minutos}
    for prep in (0, 1):
        pi, pf, pd_ = construir(elegidas, P["mu"], P["sd"], Ws, bs, prep=prep)
        etiqueta = "f64" if prep == 0 else "f32"
        # features del motor rápido sobre test_demo (con este preprocesado) y comprobación frente a las columnas del #30
        F_te = c6.features_lote(mr, D["X_te"], D["sr_te"], pi, pf, pd_)
        dif = float(np.abs(F_te - P["X_te"]).max())
        lg_te = c6.logits_lote(mr, F_te, pi, pf)
        ws64, ws32 = mr.crear_workspace(pi)
        pred_uno = np.array([mr.predecir(D["X_te"][i], D["sr_te"][i], pi, pf, pd_, ws64, ws32) for i in range(400)])
        lg_val = {v: c6.logits_lote(mr, x, pi, pf) for v, x in P["X_va"].items()}
        gm = {v: float(c6.gm_f(c6.recalls(P["y_va"], lg.argmax(1)))) for v, lg in lg_val.items()}
        m_te = c6.metricas(P["y_te"], lg_te.argmax(1))
        np.savez(c6.ART / f"alumno_{nombre}_{etiqueta}.npz", pi=pi, pf=pf, pd=pd_, clase_defecto=np.int64(0))
        np.savez(c6.ART / f"val_logits_{nombre}_{etiqueta}.npz", **lg_val, test=lg_te)
        res[etiqueta] = {"val_gm_f_corrompido": gm["corrompido"], "val_gm_f_media4": float(np.mean(list(gm.values()))),
                         "val_por_version": gm, "test_demo": m_te, "max_dif_features_test_vs_cols30": dif,
                         "predecir_igual_a_lote": int((pred_uno == lg_te.argmax(1)).sum())}
        c6.log(f"{nombre} [{etiqueta}]: época {mejor_ep}/{len(hist)} | val-corr gm_f {gm['corrompido']:.4f} (media 4 "
               f"{res[etiqueta]['val_gm_f_media4']:.4f}) | test_demo acc {m_te['acc']:.4f} gm_f {m_te['gm_f']:.4f} "
               f"score400 {m_te['score_harness_400']:.3f} | recall {np.round(m_te['recall'], 2).tolist()} | "
               f"Δfeat {dif:.1e} | predecir = lote {res[etiqueta]['predecir_igual_a_lote']}/400 | {minutos:.1f} min",
               "progreso_entrenar.log")
    res["historial"] = hist
    ruta = c6.ART / "alumnos_06.json"
    todo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    todo[nombre] = res
    ruta.write_text(json.dumps(todo, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
