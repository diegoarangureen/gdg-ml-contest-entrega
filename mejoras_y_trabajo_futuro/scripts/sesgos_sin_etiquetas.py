"""Mejora 1: sesgos por clase ajustados SIN etiquetas sobre test_demo (coste de latencia cero).

Se ajusta un vector b (8 sesgos, plegado en el último bias del MLP) solo con los logits del modelo sobre las 400
series de test_demo, sin mirar sus etiquetas, para que el reparto de predicciones se acerque al equilibrado (1/8 por
clase) que se espera en test. Tres métodos:

* ``em_saerens``: EM de Saerens et al. (2002). Prior «de entrenamiento» π₀ = media de softmax del modelo en la
  validación corrompida (sin etiquetas); el EM sobre test_demo estima su prior π̂ y los posteriores se reajustan a
  él: b = log(π̂ / π₀).
* ``suave``: desplazamiento iterativo de logits b ← b − log(media softmax(lg + b) · 8) hasta que la media de las
  probabilidades en test_demo es uniforme.
* ``duro``: desplazamiento iterativo con paso decreciente sobre el recuento de argmax, hasta que cada clase reciba
  ~50 de las 400 predicciones (se queda con el b de menor desviación).

**Evaluación.**
- test_demo, ajuste completo (transductivo, sin etiquetas): acierto, gm_f y score del harness.
- test_demo con *cross-fitting*: mitades aleatorias (sin estratificar: no se usan etiquetas), 20 repeticiones; b se ajusta en una mitad y se mide en la otra. Estima la ganancia sin sobreajustar a las mismas 400.
- validación corrompida (4 versiones, 57 000 filas): el mismo b (ajustado en test_demo) aplicado a sus logits; mide
  si los recall por clase mejoran en datos que tienen etiquetas.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\sesgos_sin_etiquetas.py [--nombre base30] [--val-logits ruta.npz]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import comun06 as c6

METODOS = ("em_saerens", "suave", "duro")


def softmax(lg):
    lg = np.asarray(lg, np.float64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def b_em_saerens(lg_te, pi0, iters=200):
    """EM de Saerens: π̂ del test y b = log(π̂/π₀) (posteriores reajustados al prior estimado)."""
    p = softmax(lg_te)
    pi = pi0.copy()
    for _ in range(iters):
        q = p * (pi / pi0)
        q /= q.sum(1, keepdims=True)
        nuevo = q.mean(0)
        if np.abs(nuevo - pi).max() < 1e-8:
            pi = nuevo
            break
        pi = nuevo
    b = np.log(pi / pi0)
    return b - b.mean(), pi


def b_suave(lg_te, iters=500):
    b = np.zeros(lg_te.shape[1])
    for _ in range(iters):
        m = softmax(lg_te + b).mean(0)
        paso = np.log(m * len(m))
        b -= paso
        if np.abs(paso).max() < 1e-7:
            break
    return b - b.mean()


def b_duro(lg_te, iters=400):
    K = lg_te.shape[1]
    objetivo = len(lg_te) / K
    b = np.zeros(K)
    mejor_b, mejor_d = b.copy(), np.inf
    escala = float(np.std(lg_te))
    for it in range(iters):
        n = np.bincount((lg_te + b).argmax(1), minlength=K)
        d = np.abs(n - objetivo).max()
        if d < mejor_d:
            mejor_b, mejor_d = b.copy(), d
        eta = 0.05 * escala / (1 + it / 50)
        b -= eta * np.log((n + 1) / (objetivo + 1))
    return mejor_b - mejor_b.mean()


def ajustar(metodo, lg_te, pi0):
    if metodo == "em_saerens":
        return b_em_saerens(lg_te, pi0)[0]
    if metodo == "suave":
        return b_suave(lg_te)
    return b_duro(lg_te)


def evaluar(lg_val, lg_te, y_va, y_te, rep=20, semilla=0):
    """Resultados de los 3 métodos (y sin sesgos) para un modelo con logits de val (4 versiones) y de test_demo."""
    LG = np.concatenate([lg_val[v] for v in c6.VERSIONES_VAL])
    Y = np.tile(y_va, len(c6.VERSIONES_VAL))
    pi0 = softmax(LG).mean(0)
    rng = np.random.default_rng(semilla)
    mitades = [rng.permutation(len(y_te)) for _ in range(rep)]
    out = {"pi0_val": pi0.round(4).tolist()}

    def val_m(b):
        pred = (LG + b).argmax(1)
        rec = c6.recalls(Y, pred)
        por_v = {v: float(c6.gm_f(c6.recalls(y_va, (lg_val[v] + b).argmax(1)))) for v in c6.VERSIONES_VAL}
        return {"gm_f_4v": float(c6.gm_f(rec)), "gm_f_corrompido": por_v["corrompido"],
                "gm_f_media4": float(np.mean(list(por_v.values()))), "recall_4v": rec.round(4).tolist()}

    b0 = np.zeros(lg_te.shape[1])
    out["sin"] = {"b": b0.tolist(), "test_demo": c6.metricas(y_te, lg_te.argmax(1)), "val": val_m(b0),
                  "reparto_pred_test": np.bincount(lg_te.argmax(1), minlength=8).tolist()}
    for metodo in METODOS:
        b = ajustar(metodo, lg_te, pi0)
        pred = (lg_te + b).argmax(1)
        # cross-fitting: ajusta en una mitad (sin etiquetas) y mide en la otra; las dos mitades juntas dan 400 preds
        accs, gms, d_acc, d_gm = [], [], [], []
        for perm in mitades:
            a, c = perm[:200], perm[200:]
            p_cf = np.empty(len(y_te), np.int64)
            p_cf[c] = (lg_te[c] + ajustar(metodo, lg_te[a], pi0)).argmax(1)
            p_cf[a] = (lg_te[a] + ajustar(metodo, lg_te[c], pi0)).argmax(1)
            m = c6.metricas(y_te, p_cf)
            accs.append(m["acc"])
            gms.append(m["gm_f"])
        r = {"b": b.round(4).tolist(), "test_demo": c6.metricas(y_te, pred), "val": val_m(b),
             "reparto_pred_test": np.bincount(pred, minlength=8).tolist(),
             "test_demo_crossfit": {"acc_media": float(np.mean(accs)), "acc_desv": float(np.std(accs)),
                                    "gm_f_media": float(np.mean(gms)), "gm_f_desv": float(np.std(gms))}}
        if metodo == "em_saerens":
            r["pi_test_estimado"] = b_em_saerens(lg_te, pi0)[1].round(4).tolist()
        out[metodo] = r
    return out


def imprimir(nombre, R):
    s = R["sin"]
    print(f"== {nombre} | π₀ (val, sin etiquetas) {R['pi0_val']}")
    print(f"   sin sesgos: test acc {s['test_demo']['acc']:.4f} gm_f {s['test_demo']['gm_f']:.4f} score400 "
          f"{s['test_demo']['score_harness_400']:.3f} | val gm_f corr {s['val']['gm_f_corrompido']:.4f} media4 "
          f"{s['val']['gm_f_media4']:.4f} | reparto {s['reparto_pred_test']}")
    for m in METODOS:
        r = R[m]
        cf = r["test_demo_crossfit"]
        print(f"   {m:10s}: b {np.round(r['b'], 3).tolist()}\n"
              f"      test acc {r['test_demo']['acc']:.4f} gm_f {r['test_demo']['gm_f']:.4f} score400 "
              f"{r['test_demo']['score_harness_400']:.3f} | crossfit acc {cf['acc_media']:.4f}±{cf['acc_desv']:.4f} gm_f "
              f"{cf['gm_f_media']:.4f} | val gm_f corr {r['val']['gm_f_corrompido']:.4f} media4 {r['val']['gm_f_media4']:.4f}"
              f" | reparto {r['reparto_pred_test']}" + (f" | π̂ test {r['pi_test_estimado']}" if 'pi_test_estimado' in r else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="base30")
    ap.add_argument("--val-logits", default=str(c6.OPT / "artifacts" / "val_logits_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0.npz"))
    args = ap.parse_args()
    D = c6.co.datos()
    y_va, y_te = D["y_pool"][D["i_va"]], D["y_te"]
    z = c6.cargar_npz(args.val_logits)
    lg_val = {v: z[v].astype(np.float64) for v in c6.VERSIONES_VAL}
    R = evaluar(lg_val, z["test"].astype(np.float64), y_va, y_te)
    R["reparto_val_real"] = np.bincount(y_va, minlength=8).tolist()
    imprimir(args.nombre, R)
    ruta = c6.ART / "sesgos_sin_etiquetas.json"
    todo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    todo[args.nombre] = R
    ruta.write_text(json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
