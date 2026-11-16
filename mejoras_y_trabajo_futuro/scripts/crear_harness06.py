"""Crea carpetas autocontenidas para el harness en ``06_mejoras_sesgos_velocidad/modelos_harness/<nombre>/``
(``model.py`` + motor + ``modelo_numba.npz``) y comprueba que ``Model().predict`` da, sobre test_demo, las mismas
clases que los logits del lote (con los sesgos sumados).

Variantes:
- ``base30``: los pesos del #30 (``submission/modelo_numba.npz``) con su motor (``motor_multi.py``), copiados;
- ``base30_sesgos``: lo mismo con los sesgos sin etiquetas (método ``suave``) plegados en el último bias;
- ``rapido`` / ``rapido_sesgos``: el alumno rápido elegido (``--rapido <nombre de artifacts/alumno_<…>.npz>``) con
  ``motor_rapido.py``.

    .\\.venv\\Scripts\\python.exe 06_mejoras_sesgos_velocidad\\crear_harness06.py --rapido rank150_h128_ens3_mas8_s0_f64
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

import numpy as np

import comun06 as c6

MODEL_PY = '''"""GDG on Campus UAM ML Contest: {titulo}

Pipeline completo (limpieza, MiniRocket, candidatos, escalado y MLP) en una llamada de numba por muestra
(``{motor}.py``). Pesos en ``modelo_numba.npz``, junto a este fichero (rutas relativas a ``__file__``).
"""
from pathlib import Path

from {motor} import Model as _Motor

HERE = Path(__file__).resolve().parent


class Model(_Motor):
    def __init__(self):
        super().__init__(HERE / "modelo_numba.npz")
'''

COMPROBAR = r'''
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from model import Model
from competition_api import load_dataset, Sample
te = load_dataset(sys.argv[3])
m = Model()
pred = [m.predict(Sample(np.frombuffer(np.asarray(te.series[i]).tobytes(), dtype=np.float32).reshape(4, 256), te.meta_at(i))) for i in range(len(te.labels))]
print(json.dumps(pred))
'''


def crear(nombre, npz, motor, b, titulo, lg_te):
    dst = c6.AQUI / "modelos_harness" / nombre
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    P = c6.cargar_npz(npz)
    if b is not None:
        P["pf"] = c6.plegar(P["pi"], P["pf"], b)
    np.savez(dst / "modelo_numba.npz", **P)
    origen = c6.ROOT / "submission" / "motor_multi.py" if motor == "motor_multi" else c6.AQUI / "motor_rapido.py"
    shutil.copy2(origen, dst / f"{motor}.py")
    (dst / "model.py").write_text(MODEL_PY.format(titulo=titulo, motor=motor), encoding="utf-8")
    (dst / "requirements.txt").write_text("numpy\nnumba\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", COMPROBAR, str(dst), str(c6.ROOT / "materials"),
                        str(c6.ROOT / "materials" / "data" / "test_demo.npz")], capture_output=True, text=True,
                       env={**__import__("os").environ, "NUMBA_CACHE_DIR": str(c6.CACHE / "numba_harness")})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    pred = np.array(json.loads(r.stdout.strip().splitlines()[-1]))
    esperado = (lg_te + (0 if b is None else b)).argmax(1)
    iguales = int((pred == esperado).sum())
    c6.log(f"crear_harness06 {nombre}: Model().predict = lote en {iguales}/400 | acc test_demo "
           f"{(pred == c6.co.datos()['y_te']).mean():.4f}")
    assert iguales == 400, iguales
    shutil.rmtree(dst / "__pycache__", ignore_errors=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", required=True, help="nombre del alumno rápido (artifacts/alumno_<nombre>.npz)")
    ap.add_argument("--metodo", default="suave")
    a = ap.parse_args()
    S = json.loads((c6.ART / "sesgos_sin_etiquetas.json").read_text(encoding="utf-8"))
    z30 = c6.cargar_npz(c6.OPT / "artifacts" / "val_logits_A250_ppv__ens3_kl_a0.5_T2_h128_mas8_s0.npz")
    zr = c6.cargar_npz(c6.ART / f"val_logits_{a.rapido}.npz")
    b30 = np.asarray(S["base30"][a.metodo]["b"], np.float32)
    br = np.asarray(S[a.rapido][a.metodo]["b"], np.float32)
    crear("base30", c6.NPZ_30, "motor_multi", None, "modelo #30 (copia de submission/)", z30["test"])
    crear("base30_sesgos", c6.NPZ_30, "motor_multi", b30,
          f"#30 + sesgos por clase sin etiquetas (método {a.metodo})", z30["test"])
    ruta_r = c6.ART / f"alumno_{a.rapido}.npz"
    crear("rapido", ruta_r, "motor_rapido", None, f"alumno rápido {a.rapido}", zr["test"])
    crear("rapido_sesgos", ruta_r, "motor_rapido", br,
          f"alumno rápido {a.rapido} + sesgos sin etiquetas (método {a.metodo})", zr["test"])


if __name__ == "__main__":
    main()
