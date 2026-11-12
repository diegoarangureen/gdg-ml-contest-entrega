"""MiniRocket con pooling tipo MultiRocket + MLP en numba: una muestra por llamada, todo el pipeline en una función.

Copia autocontenida de ``v2/model_numba.py`` (generada por ``v3/opt/generar_motor.py``) con un cambio: tras cada
convolución de MiniRocket, para cada sesgo seleccionado, se calculan los poolings de su máscara en el mismo bucle:

* PPV (1): proporción de valores positivos de la convolución menos el sesgo (la feature de MiniRocket);
* MPV (2): media de los valores positivos;
* LSPV (4): la racha más larga de valores positivos, dividida por la longitud;
* MIPV (8): media de los índices de los valores positivos, dividida por la longitud (−1 si no hay ninguno).

El orden de las features es el de la selección de aeon y, dentro de cada una, PPV, MPV, LSPV y MIPV (los bits
activos). Sin máscaras (solo PPV), el plan y las features son idénticos a los de v2. El resto del pipeline
(preprocesado de v1, ventanas a 1 Hz, candidatos, scaler y MLP) es el de v2, copiado tal cual.

Compatible con el harness: ``class Model`` con ``predict(x) -> int``; pesos en un ``.npz`` (``pi``, ``pf``, ``pd``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
PESOS = "modelo_numba.npz"     # pesos por defecto, junto a este fichero

SPIKE_Z, CLIP_Z, VENTANA = 8.0, 8.0, 10
N_CAND = 5                     # dskew_1, env_ac10_1, env_ac10_0, irrev_1, dskew_0 (orden de rasgos_6v7.CANDIDATOS)
ACT_NINGUNA, ACT_RELU, ACT_SILU = 0, 1, 2

# Cabecera de `pi`
H_L, H_C, H_WL, H_ND, H_NM, H_NF, H_CAND, H_NL, H_PAD = 0, 1, 2, 3, 4, 5, 6, 7, 8
H_ODIL, H_OCOMBO, H_OCH, H_OKIDX, H_OLAY, H_DIN = 9, 10, 11, 12, 13, 14
H_FOB, H_FOM, H_FOS, H_FOW, H_MAXW, H_OREV = 15, 16, 17, 18, 19, 20
H_OSLOT = 21                   # inicio de los slots: [máscara, primera salida] por sesgo
P_PPV, P_MPV, P_LSPV, P_MIPV = 1, 2, 4, 8
N_HEAD = 24
COMBO_W = 7                    # [posición de la dilación, kernel, relleno, inicio canales, nº canales, inicio feats, nº feats]


# ====================================================================== construcción de los pesos (Python)


def construir(mr_params, seleccion, usa_candidatos, sc_mean, sc_scale, Ws, bs, activacion, L=256, C=4, WL=128,
              indices=None, mascaras=None):
    """Bloques ``pi``/``pf``/``pd`` a partir de un MiniRocket de aeon ajustado y un MLP exportado.

    mr_params: ``mr.parameters`` (n_ch_comb, ch_idx, dilaciones, n_feat_dil, sesgos); seleccion: índices (orden de
    aeon, crecientes) de las features de MiniRocket que usa el modelo; mascaras: poolings de cada una (bits P_PPV,
    P_MPV, P_LSPV y P_MIPV; por defecto solo PPV). Las entradas del scaler son, por cada índice seleccionado, sus
    poolings en el orden PPV, MPV, LSPV, MIPV, y después los candidatos. sc_mean/sc_scale: StandardScaler (en
    float32). Ws/bs: capas del MLP (W con forma (salidas, entradas)).
    """
    n_ch_comb, ch_idx, dil, n_feat_dil, sesgos = (np.asarray(a) for a in mr_params)
    if n_ch_comb.max() > 4 or C != 4:
        raise ValueError("el motor asume 4 canales y como mucho 4 canales por combinación (como aeon con C = 4)")
    if indices is None:
        from itertools import combinations
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
    n_k = len(indices)
    seleccion = np.asarray(seleccion, dtype=np.int64)
    if np.any(np.diff(seleccion) <= 0):
        raise ValueError("la selección debe ir en el orden de aeon (creciente y sin repetidos)")
    mascaras = np.full(len(seleccion), P_PPV, np.int64) if mascaras is None else np.asarray(mascaras, np.int64)
    if len(mascaras) != len(seleccion) or np.any(mascaras < 1) or np.any(mascaras > 15):
        raise ValueError("una máscara (1..15) por feature seleccionada")
    pop = np.array([bin(int(m)).count("1") for m in mascaras], np.int64)
    salida0 = np.concatenate([[0], np.cumsum(pop)])[:-1]
    # posición de cada feature de aeon: (dilación j, kernel k) y orden dentro del combo
    f_inicio = np.concatenate([[0], np.cumsum(np.repeat(n_feat_dil, n_k))])
    ch_inicio = np.concatenate([[0], np.cumsum(n_ch_comb)])
    sel_set = {int(f): i for i, f in enumerate(seleccion)}
    dil_usadas, combos, canales, sesgos_sel, orden, slots = [], [], [], [], [], []
    for j in range(len(dil)):
        for k in range(n_k):
            comb = j * n_k + k
            fs = [f for f in range(f_inicio[comb], f_inicio[comb + 1]) if f in sel_set]
            if not fs:
                continue
            if not dil_usadas or dil_usadas[-1] != int(dil[j]):
                dil_usadas.append(int(dil[j]))
            chs = ch_idx[ch_inicio[comb]:ch_inicio[comb + 1]]
            combos.append([len(dil_usadas) - 1, k, (j + k) % 2, len(canales), len(chs), len(sesgos_sel), len(fs)])
            canales.extend(int(c) for c in chs)
            sesgos_sel.extend(float(sesgos[f]) for f in fs)
            for f in fs:
                i = sel_set[f]
                orden.append(i)
                slots.extend([int(mascaras[i]), int(salida0[i])])
    if orden != list(range(len(seleccion))):
        raise AssertionError("el plan no respeta el orden de la selección")
    modo_cand = {False: 0, True: 1, "todos": 1, "baratos": 2}[usa_candidatos]
    n_f = int(pop.sum())
    d_in = n_f + (0, N_CAND, 3)[modo_cand]
    if len(sc_mean) != d_in or Ws[0].shape[1] != d_in:
        raise ValueError(f"dimensiones incoherentes: d_in={d_in}, scaler={len(sc_mean)}, W0={Ws[0].shape}")
    act = {"ninguna": ACT_NINGUNA, "relu": ACT_RELU, "silu": ACT_SILU}[activacion]
    capas = [[W.shape[1], W.shape[0], act if i < len(Ws) - 1 else ACT_NINGUNA] for i, W in enumerate(Ws)]
    pad = 4 * max(dil_usadas)

    head = np.zeros(N_HEAD, np.int64)
    bits = int(np.log2(WL))
    if 2 ** bits != WL:
        raise ValueError("la longitud de ventana debe ser potencia de 2")
    rev = np.array([int(format(i, f"0{bits}b")[::-1], 2) for i in range(WL)], np.int64)
    partes = [np.asarray(dil_usadas, np.int64), np.asarray(combos, np.int64).ravel(), np.asarray(canales, np.int64),
              np.asarray(indices, np.int64).ravel(), np.asarray(capas, np.int64).ravel(), rev,
              np.asarray(slots, np.int64)]
    off = N_HEAD
    offs = []
    for p in partes:
        offs.append(off)
        off += len(p)
    head[[H_L, H_C, H_WL, H_ND, H_NM, H_NF, H_CAND, H_NL, H_PAD]] = [L, C, WL, len(dil_usadas), len(combos), n_f,
                                                                     modo_cand, len(Ws), pad]
    head[[H_ODIL, H_OCOMBO, H_OCH, H_OKIDX, H_OLAY, H_OREV, H_OSLOT]] = offs
    head[H_DIN] = d_in
    pf_partes = [np.asarray(sesgos_sel, np.float32), np.asarray(sc_mean, np.float64).astype(np.float32),
                 np.asarray(sc_scale, np.float64).astype(np.float32)]
    pf_partes += [a for W, b in zip(Ws, bs) for a in (np.asarray(W, np.float32).ravel(), np.asarray(b, np.float32))]
    pf_off = np.concatenate([[0], np.cumsum([len(p) for p in pf_partes])])
    head[[H_FOB, H_FOM, H_FOS, H_FOW]] = pf_off[:4]
    head[H_MAXW] = max(max(W.shape) for W in Ws)
    pi = np.concatenate([head] + partes).astype(np.int32)
    pf = np.concatenate(pf_partes).astype(np.float32)
    m = WL // 2
    pd = np.concatenate([np.cos(-2 * np.pi * np.arange(m) / WL), np.sin(-2 * np.pi * np.arange(m) / WL)])
    return pi, pf, pd.astype(np.float64)


def crear_workspace(pi):
    """Buffers de trabajo (se reservan una vez y se reutilizan en cada llamada)."""
    L, C, WL, pad, n_f, maxw = int(pi[H_L]), int(pi[H_C]), int(pi[H_WL]), int(pi[H_PAD]), int(pi[H_NF]), int(pi[H_MAXW])
    n32 = C * L + C * WL + 2 * C * (WL + 2 * pad) + C * WL + WL + 2 * (n_f + N_CAND) + 2 * maxw + 16
    n64 = C * L + 4 * (L + VENTANA) + 12 * WL + 16
    return np.zeros(n64, np.float64), np.zeros(n32, np.float32)


# ====================================================================== preprocesado (exacto)


@njit(cache=True, error_model="numpy")
def _part_menor(a, lo, hi, piv):
    """Lomuto sin saltos: deja a[lo:s] < piv y a[s:hi] >= piv (mismo multiconjunto); devuelve s."""
    s = lo
    for i in range(lo, hi):
        v = a[i]
        a[i] = a[s]
        a[s] = v
        s += v < piv
    return s


@njit(cache=True, error_model="numpy")
def _part_menor_igual(a, lo, hi, piv):
    s = lo
    for i in range(lo, hi):
        v = a[i]
        a[i] = a[s]
        a[s] = v
        s += v <= piv
    return s


@njit(cache=True, error_model="numpy")
def _select(a, n, k):
    """k-ésimo menor de a[0:n] (quickselect sin saltos, in situ; deja a[0:k] <= a[k] <= a[k+1:n]).

    El reparto en tres partes (< pivote, = pivote, > pivote) evita bucles infinitos con valores repetidos
    (tramos congelados).
    """
    lo, hi = 0, n
    while hi - lo > 1:
        x0, x1, x2 = a[lo], a[(lo + hi) // 2], a[hi - 1]
        bajo = x0 if x0 < x1 else x1
        alto = x1 if x0 < x1 else x0
        piv = alto if alto < x2 else (x2 if x2 > bajo else bajo)   # mediana de tres
        s1 = _part_menor(a, lo, hi, piv)
        if k < s1:
            hi = s1
            continue
        s2 = _part_menor_igual(a, s1, hi, piv)
        if k < s2:
            return piv
        lo = s2
    return a[k]


@njit(cache=True, error_model="numpy")
def _min_desde(a, i0, n):
    m = a[i0]
    for i in range(i0 + 1, n):
        if a[i] < m:
            m = a[i]
    return m


@njit(cache=True, error_model="numpy")
def _max_hasta(a, n):
    m = a[0]
    for i in range(1, n):
        if a[i] > m:
            m = a[i]
    return m


@njit(cache=True, error_model="numpy")
def _mediana_nan(src, buf):
    """rasgos_6v7.mediana_nan para un canal: media de los dos valores centrales de los válidos (NaN si no hay)."""
    nv = 0
    for t in range(src.shape[0]):
        v = src[t]
        if not np.isnan(v):
            buf[nv] = v
            nv += 1
    if nv == 0:
        return np.nan
    lo, hi = (nv - 1) // 2, nv // 2
    a = _select(buf, nv, lo)
    b = a if hi == lo else _min_desde(buf, lo + 1, nv)
    return (a + b) / 2


@njit(cache=True, error_model="numpy")
def _preprocesar(x, ws64, z):
    """rasgos_6v7.preprocesar(x[None], ventana=10, spike_z=8, clip_z=8, restar_mediana=True)[0] → z (float32)."""
    C, L = x.shape
    xw = ws64[:C * L].reshape((C, L))
    buf = ws64[C * L:C * L + L]
    dif = ws64[C * L + L:C * L + 2 * L]
    il = ws64[C * L + 2 * L:C * L + 3 * L + VENTANA]
    rn = ws64[C * L + 3 * L + VENTANA:C * L + 4 * L + 2 * VENTANA]
    hay_nan = False
    for c in range(C):
        for t in range(L):
            xw[c, t] = x[c, t]
        med = _mediana_nan(xw[c], buf)
        for t in range(L):
            dif[t] = abs(xw[c, t] - med)
        mad = 1.4826 * _mediana_nan(dif, buf) + 1e-9
        for t in range(L):
            if abs(xw[c, t] - med) / mad > SPIKE_Z:
                xw[c, t] = np.nan
            if np.isnan(xw[c, t]):
                hay_nan = True
    if hay_nan:
        # rellenar_media_movil: uniform_filter1d(·, 10, mode="constant") = suma de x[t-5 .. t+4] / 10, calculada como
        # scipy (suma acumulada y división al final); NaN con algún válido en la ventana -> media de los válidos.
        s1 = VENTANA // 2
        s2 = VENTANA - s1 - 1
        for c in range(C):
            alguno = False
            for t in range(L):
                if np.isnan(xw[c, t]):
                    alguno = True
                    break
            if not alguno:
                continue
            for t in range(s1):
                il[t] = 0.0
                rn[t] = 0.0
            for t in range(L):
                v = xw[c, t]
                if np.isnan(v):
                    il[s1 + t] = 0.0
                    rn[s1 + t] = 0.0
                else:
                    il[s1 + t] = v
                    rn[s1 + t] = 1.0
            for t in range(s2):
                il[s1 + L + t] = 0.0
                rn[s1 + L + t] = 0.0
            ts, tn = 0.0, 0.0
            for t in range(VENTANA):
                ts += il[t]
                tn += rn[t]
            # el valor de cada posición solo se usa si era NaN; los válidos no cambian
            for t in range(L):
                if t > 0:
                    ts += il[t + VENTANA - 1] - il[t - 1]
                    tn += rn[t + VENTANA - 1] - rn[t - 1]
                if np.isnan(xw[c, t]):
                    nval = tn / VENTANA
                    if nval > 1e-9:
                        xw[c, t] = (ts / VENTANA) / max(nval, 1e-9)
            # huecos más largos que la ventana: interpolación lineal (np.interp); canal sin válidos -> 0
            prev = -1
            t = 0
            while t < L:
                if np.isnan(xw[c, t]):
                    nxt = t + 1
                    while nxt < L and np.isnan(xw[c, nxt]):
                        nxt += 1
                    if prev == -1 and nxt == L:
                        for u in range(t, L):
                            xw[c, u] = 0.0
                    elif prev == -1:
                        for u in range(t, nxt):
                            xw[c, u] = xw[c, nxt]
                    elif nxt == L:
                        for u in range(t, L):
                            xw[c, u] = xw[c, prev]
                    else:
                        slope = (xw[c, nxt] - xw[c, prev]) / (nxt - prev)
                        for u in range(t, nxt):
                            xw[c, u] = slope * (u - prev) + xw[c, prev]
                    t = nxt
                else:
                    prev = t
                    t += 1
    # centro = mediana; IQR = percentil 75 - percentil 25 (interpolación lineal de numpy)
    centro = ws64[C * L + 4 * L + 2 * VENTANA:C * L + 4 * L + 2 * VENTANA + C]
    iqr = ws64[C * L + 4 * L + 2 * VENTANA + C:C * L + 4 * L + 2 * VENTANA + 2 * C]
    for c in range(C):
        for t in range(L):
            buf[t] = xw[c, t]
        k75 = (3 * (L - 1)) // 4                       # 191 con L = 256
        g75 = 3 * (L - 1) / 4 - k75                    # 0,25
        k25 = (L - 1) // 4                             # 63
        g25 = (L - 1) / 4 - k25                        # 0,75
        b75 = _select(buf, L, k75 + 1)
        a75 = _max_hasta(buf, k75 + 1)
        km = L // 2                                    # mediana de L par: (s[L/2-1] + s[L/2]) / 2
        b_m = _select(buf, k75 + 1, km)
        a_m = _max_hasta(buf, km)
        b25 = _select(buf, km, k25 + 1)
        a25 = _max_hasta(buf, k25 + 1)
        centro[c] = (a_m + b_m) / 2
        d75 = b75 - a75
        p75 = a75 + d75 * g75 if g75 < 0.5 else b75 - d75 * (1 - g75)
        d25 = b25 - a25
        p25 = a25 + d25 * g25 if g25 < 0.5 else b25 - d25 * (1 - g25)
        iqr[c] = p75 - p25
    # escala = mediana de los IQR de los canales (C = 4: media de los dos centrales)
    for c in range(C):
        buf[c] = iqr[c]
    hi = C // 2
    b = _select(buf, C, hi)
    a = _max_hasta(buf, hi) if C % 2 == 0 else b
    escala = (a + b) / 2 + 1e-9
    for c in range(C):
        for t in range(L):
            v = (xw[c, t] - centro[c]) / escala
            if v < -CLIP_Z:
                v = -CLIP_Z
            elif v > CLIP_Z:
                v = CLIP_Z
            z[c, t] = np.float32(v)


# ====================================================================== MiniRocket (subconjunto de features)


@njit(cache=True, error_model="numpy")
def _minirocket(win, pi, pf, A1, G1, ca1, cbuf, out):
    """Features MiniRocket seleccionadas (con sus poolings) de una ventana ``win`` (C, WL) -> ``out`` (n_f).

    Núcleo de eda/09_latencia_minirocket.minirocket_una (mismo orden de las sumas en float32 que aeon), pero
    recorriendo solo los (dilación, kernel) del plan. Índices uint64 para que LLVM vectorice.
    """
    C, WL, pad = pi[H_C], pi[H_WL], pi[H_PAD]
    W = WL + 2 * pad
    o_dil, o_combo, o_ch, o_k = pi[H_ODIL], pi[H_OCOMBO], pi[H_OCH], pi[H_OKIDX]
    o_b, o_sl = pi[H_FOB], pi[H_OSLOT]
    for c in range(C):
        base = c * W + pad
        for t in range(WL):
            v = win[c, t]
            A1[np.uint64(base + t)] = -v
            G1[np.uint64(base + t)] = np.float32(3.0) * v
    u_actual = -1
    for m in range(pi[H_NM]):
        r = o_combo + COMBO_W * m
        u, k, relleno = pi[r], pi[r + 1], pi[r + 2]
        ch0, nch, f0, nf = pi[r + 3], pi[r + 4], pi[r + 5], pi[r + 6]
        d = pi[o_dil + u]
        if u != u_actual:   # suma de las 9 posiciones con peso alfa = -1 para esta dilación, todos los canales
            u_actual = u
            for c in range(C):
                b = c * W + pad
                a0, m4, m3 = np.uint64(b), np.uint64(b - 4 * d), np.uint64(b - 3 * d)
                m2, m1, p1 = np.uint64(b - 2 * d), np.uint64(b - d), np.uint64(b + d)
                p2, p3, p4 = np.uint64(b + 2 * d), np.uint64(b + 3 * d), np.uint64(b + 4 * d)
                cc = np.uint64(c * WL)
                for t in range(WL):
                    q = np.uint64(t)
                    ca1[cc + q] = (((((((A1[a0 + q] + A1[m4 + q]) + A1[m3 + q]) + A1[m2 + q]) + A1[m1 + q])
                                    + A1[p1 + q]) + A1[p2 + q]) + A1[p3 + q]) + A1[p4 + q]
        if relleno == 0:
            t0, n = 0, WL
        else:
            t0, n = 4 * d, WL - 8 * d
        i0, i1, i2 = pi[o_k + 3 * k] - 4, pi[o_k + 3 * k + 1] - 4, pi[o_k + 3 * k + 2] - 4
        # C = suma sobre los canales de la combinación de ((alfa + gamma_i0) + gamma_i1) + gamma_i2, en el mismo
        # orden que aeon ((v0 + v1) + v2) + v3, en un solo bucle por combinación (1 a 4 canales).
        c = pi[o_ch + ch0]
        x0 = np.uint64(c * WL + t0)
        gb = c * W + pad + t0
        a0, a1, a2 = np.uint64(gb + i0 * d), np.uint64(gb + i1 * d), np.uint64(gb + i2 * d)
        if nch == 1:
            for t in range(n):
                s = np.uint64(t)
                cbuf[t] = ((ca1[x0 + s] + G1[a0 + s]) + G1[a1 + s]) + G1[a2 + s]
        else:
            c = pi[o_ch + ch0 + 1]
            x1 = np.uint64(c * WL + t0)
            gb = c * W + pad + t0
            b0, b1, b2 = np.uint64(gb + i0 * d), np.uint64(gb + i1 * d), np.uint64(gb + i2 * d)
            if nch == 2:
                for t in range(n):
                    s = np.uint64(t)
                    cbuf[t] = (((ca1[x0 + s] + G1[a0 + s]) + G1[a1 + s]) + G1[a2 + s]) \
                        + (((ca1[x1 + s] + G1[b0 + s]) + G1[b1 + s]) + G1[b2 + s])
            else:
                c = pi[o_ch + ch0 + 2]
                x2 = np.uint64(c * WL + t0)
                gb = c * W + pad + t0
                e0, e1, e2 = np.uint64(gb + i0 * d), np.uint64(gb + i1 * d), np.uint64(gb + i2 * d)
                if nch == 3:
                    for t in range(n):
                        s = np.uint64(t)
                        cbuf[t] = ((((ca1[x0 + s] + G1[a0 + s]) + G1[a1 + s]) + G1[a2 + s])
                                   + (((ca1[x1 + s] + G1[b0 + s]) + G1[b1 + s]) + G1[b2 + s])) \
                            + (((ca1[x2 + s] + G1[e0 + s]) + G1[e1 + s]) + G1[e2 + s])
                else:
                    c = pi[o_ch + ch0 + 3]
                    x3 = np.uint64(c * WL + t0)
                    gb = c * W + pad + t0
                    h0, h1, h2 = np.uint64(gb + i0 * d), np.uint64(gb + i1 * d), np.uint64(gb + i2 * d)
                    for t in range(n):
                        s = np.uint64(t)
                        cbuf[t] = (((((ca1[x0 + s] + G1[a0 + s]) + G1[a1 + s]) + G1[a2 + s])
                                    + (((ca1[x1 + s] + G1[b0 + s]) + G1[b1 + s]) + G1[b2 + s]))
                                   + (((ca1[x2 + s] + G1[e0 + s]) + G1[e1 + s]) + G1[e2 + s])) \
                            + (((ca1[x3 + s] + G1[h0 + s]) + G1[h1 + s]) + G1[h2 + s])
        for ff in range(nf):
            si = f0 + ff
            sesgo = pf[o_b + si]
            masc, k0 = pi[o_sl + 2 * si], pi[o_sl + 2 * si + 1]
            cnt = 0
            for t in range(n):     # PPV: el bucle de v2, idéntico
                cnt += cbuf[t] > sesgo
            if masc == P_PPV:
                out[k0] = cnt / n
                continue
            k = k0
            if masc & P_PPV:
                out[k] = cnt / n
                k += 1
            if masc & P_MPV:
                out[k] = _suma_pos(cbuf, n, sesgo) / cnt if cnt > 0 else np.float32(0.0)
                k += 1
            if masc & P_LSPV:
                out[k] = _racha_pos(cbuf, n, sesgo) / n
                k += 1
            if masc & P_MIPV:
                out[k] = _suma_idx(cbuf, n, sesgo) / (cnt * n) if cnt > 0 else np.float32(-1.0)
    return out


@njit(cache=True, fastmath=True, error_model="numpy")
def _suma_pos(cbuf, n, sesgo):
    """Σ max(C − sesgo, 0) (fastmath: suma vectorizada; se usa igual al entrenar y al predecir)."""
    s = np.float32(0.0)
    for t in range(n):
        s += max(cbuf[t] - sesgo, np.float32(0.0))
    return s


@njit(cache=True, error_model="numpy")
def _suma_idx(cbuf, n, sesgo):
    """Σ de los índices t con C[t] > sesgo."""
    s = 0
    for t in range(n):
        s += t if cbuf[t] > sesgo else 0
    return s


@njit(cache=True, error_model="numpy")
def _racha_pos(cbuf, n, sesgo):
    """Longitud de la racha más larga de C[t] > sesgo."""
    racha, larga = 0, 0
    for t in range(n):
        racha = racha + 1 if cbuf[t] > sesgo else 0
        larga = racha if racha > larga else larga
    return larga


# ====================================================================== rasgos «candidatos» (v0/05, 8.3)


@njit(cache=True, error_model="numpy")
def _fft(re, im, pd, rev, inversa):
    """FFT radix 2 in situ (N = len(re), potencia de 2), sin normalizar.

    pd: [cos(-2πk/N) | sin(-2πk/N)] para k < N/2; rev: permutación de inversión de bits. Cada factor de giro se
    carga una vez y se aplica a todas las mariposas que lo comparten.
    """
    n = re.shape[0]
    for i in range(n):
        j = rev[i]
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    m = n // 2
    signo = -1.0 if inversa else 1.0
    largo = 2
    while largo <= n:
        paso = n // largo
        mitad = largo // 2
        for k in range(mitad):
            wr = pd[k * paso]
            wi = signo * pd[m + k * paso]
            for a in range(k, n, largo):
                b = a + mitad
                xr = re[b] * wr - im[b] * wi
                xi = re[b] * wi + im[b] * wr
                re[b] = re[a] - xr
                im[b] = im[a] - xi
                re[a] += xr
                im[a] += xi
        largo *= 2


@njit(cache=True, error_model="numpy")
def _media(a, n):
    s = 0.0
    for i in range(n):
        s += a[i]
    return s / n


@njit(cache=True, error_model="numpy")
def _skew_diff(w, c, WL):
    """skew(diff(w[c])) con bias=True (rasgos_6v7.skew)."""
    n = WL - 1
    s = 0.0
    for t in range(n):
        s += np.float64(w[c, t + 1]) - np.float64(w[c, t])
    mu = s / n
    m2, m3 = 0.0, 0.0
    for t in range(n):
        d = (np.float64(w[c, t + 1]) - np.float64(w[c, t])) - mu
        m2 += d * d
        m3 += d * d * d
    m2 /= n
    m3 /= n
    return m3 / m2 ** 1.5


@njit(cache=True, error_model="numpy")
def _candidatos(w, ws64, pd, rev, out, completos):
    """[dskew_1, env_ac10_1, env_ac10_0, irrev_1, dskew_0] de una ventana w (C, WL) float32, en float64.

    La envolvente de Hilbert de ch0 y ch1 (como scipy.signal.hilbert) sale de una sola FFT directa de n0 + i·n1
    (se separan los dos espectros por simetría) y una FFT inversa por canal.
    completos=False: solo los tres baratos, sin FFT, en el orden [dskew_1, irrev_1, dskew_0].
    """
    WL = w.shape[1]
    o = ws64.shape[0] - 12 * WL - 16
    nrm = ws64[o:o + 2 * WL].reshape((2, WL))
    zr = ws64[o + 2 * WL:o + 3 * WL]
    zi = ws64[o + 3 * WL:o + 4 * WL]
    re = ws64[o + 4 * WL:o + 5 * WL]
    im = ws64[o + 5 * WL:o + 6 * WL]
    env = ws64[o + 6 * WL:o + 7 * WL]
    for c in range(0 if completos else 1, 2):   # normalizar(w[:, :2]): (w - media) / (std + 1e-9)
        mu = 0.0
        for t in range(WL):
            mu += np.float64(w[c, t])
        mu /= WL
        v = 0.0
        for t in range(WL):
            d = np.float64(w[c, t]) - mu
            v += d * d
        sd = np.sqrt(v / WL)
        for t in range(WL):
            nrm[c, t] = (np.float64(w[c, t]) - mu) / (sd + 1e-9)
    irr = 0.0
    for t in range(WL - 2):
        a, b = nrm[1, t], nrm[1, t + 2]
        irr += a * b * b - a * a * b
    if not completos:
        vals3 = (_skew_diff(w, 1, WL), irr / (WL - 2), _skew_diff(w, 0, WL))
        for i in range(3):
            v = np.float32(vals3[i])
            out[i] = v if np.isfinite(v) else np.float32(0.0)
        return
    for t in range(WL):
        zr[t] = nrm[0, t]
        zi[t] = nrm[1, t]
    _fft(zr, zi, pd, rev, False)
    h = WL // 2
    env_ac0, env_ac1 = 0.0, 0.0
    for c in range(2):   # envolvente de Hilbert y autocorrelación a 10 de la envolvente centrada
        for k in range(WL):
            if k > h:
                re[k] = 0.0
                im[k] = 0.0
                continue
            kk = (WL - k) % WL
            a, b, cr, ci = zr[k], zi[k], zr[kk], zi[kk]
            if c == 0:
                xr, xi = (a + cr) / 2, (b - ci) / 2
            else:
                xr, xi = (b + ci) / 2, (cr - a) / 2
            g = 1.0 if (k == 0 or k == h) else 2.0
            re[k] = g * xr
            im[k] = g * xi
        _fft(re, im, pd, rev, True)
        for t in range(WL):
            env[t] = np.sqrt(re[t] * re[t] + im[t] * im[t]) / WL
        mu = _media(env, WL)
        for t in range(WL):
            env[t] -= mu
        mu2 = _media(env, WL)
        var = 0.0
        for t in range(WL):
            d = env[t] - mu2
            var += d * d
        var /= WL
        s = 0.0
        for t in range(WL - 10):
            s += env[t + 10] * env[t]
        if c == 0:
            env_ac0 = (s / (WL - 10)) / (var + 1e-9)
        else:
            env_ac1 = (s / (WL - 10)) / (var + 1e-9)
    vals = (_skew_diff(w, 1, WL), env_ac1, env_ac0, irr / (WL - 2), _skew_diff(w, 0, WL))
    for i in range(N_CAND):
        v = np.float32(vals[i])
        out[i] = v if np.isfinite(v) else np.float32(0.0)


# ====================================================================== MLP


@njit(cache=True, fastmath=True, error_model="numpy")
def _capas(pi, pf, h0, h1):
    """Propaga h0 (entrada ya escalada) por las capas; devuelve (buffer con los logits, nº de logits).

    Cada fila de W se recorre como vista contigua (índices desde 0): así LLVM vectoriza el producto escalar.
    """
    o_lay, off = pi[H_OLAY], pi[H_FOW]
    a, b = h0, h1
    n_out = 0
    for l in range(pi[H_NL]):
        n_in, n_out, act = pi[o_lay + 3 * l], pi[o_lay + 3 * l + 1], pi[o_lay + 3 * l + 2]
        ob = off + n_in * n_out
        x = a[:n_in]
        for o in range(n_out):
            w = pf[off + o * n_in:off + (o + 1) * n_in]
            s = np.float32(0.0)
            for i in range(n_in):
                s += w[i] * x[i]
            s += pf[ob + o]
            if act == ACT_RELU:
                s = max(s, np.float32(0.0))
            elif act == ACT_SILU:
                s = s / (np.float32(1.0) + np.exp(-s))
            b[o] = s
        off = ob + n_out
        a, b = b, a
    return a, n_out


@njit(cache=True, error_model="numpy")
def _mlp(pi, pf, h0, h1):
    """MLP sobre h0 (entrada ya escalada). Devuelve la clase (argmax de los logits)."""
    a, n_out = _capas(pi, pf, h0, h1)
    mejor, mejor_v = 0, a[0]
    for k in range(1, n_out):
        if a[k] > mejor_v:
            mejor, mejor_v = k, a[k]
    return mejor


@njit(cache=True, error_model="numpy")
def _logits(pi, pf, h0, h1, out):
    a, n_out = _capas(pi, pf, h0, h1)
    for k in range(n_out):
        out[k] = a[k]


# ====================================================================== pipeline completo


@njit(cache=True, error_model="numpy")
def _features(x, sr, pi, pf, pd, ws64, ws32):
    """Features (sin escalar) de una muestra: devuelve la vista de ws32 donde quedan (n_f [+5])."""
    L, C, WL, pad, n_f = pi[H_L], pi[H_C], pi[H_WL], pi[H_PAD], pi[H_NF]
    modo_cand = pi[H_CAND]
    n_cand = N_CAND if modo_cand == 1 else (3 if modo_cand == 2 else 0)
    n_tot = n_f + n_cand
    W = WL + 2 * pad
    o = 0
    z = ws32[o:o + C * L].reshape((C, L))
    o += C * L
    win = ws32[o:o + C * WL].reshape((C, WL))
    o += C * WL
    A1 = ws32[o:o + C * W]
    o += C * W
    G1 = ws32[o:o + C * W]
    o += C * W
    ca1 = ws32[o:o + C * WL]
    o += C * WL
    cbuf = ws32[o:o + WL]
    o += WL
    fw = ws32[o:o + n_tot]
    o += n_tot
    facc = ws32[o:o + n_tot]
    rev = pi[pi[H_OREV]:pi[H_OREV] + WL]
    _preprocesar(x, ws64, z)
    n_win = 1 if sr > 1.5 else L // WL
    for v in range(n_win):
        if sr > 1.5:   # a_1hz con sample_rate 2: media de pares de puntos (float32)
            for c in range(C):
                for t in range(WL):
                    win[c, t] = (z[c, 2 * t] + z[c, 2 * t + 1]) / np.float32(2.0)
        else:
            for c in range(C):
                for t in range(WL):
                    win[c, t] = z[c, v * WL + t]
        _minirocket(win, pi, pf, A1, G1, ca1, cbuf, fw)
        if n_cand > 0:
            _candidatos(win, ws64, pd, rev, fw[n_f:], modo_cand == 1)
        if v == 0:
            for i in range(n_tot):
                facc[i] = fw[i]
        else:
            for i in range(n_tot):
                facc[i] += fw[i]
    if n_win > 1:
        for i in range(n_tot):
            facc[i] = facc[i] / np.float32(n_win)
    return facc


@njit(cache=True, error_model="numpy")
def _escalar(pi, pf, f, h0):
    om, osc = pi[H_FOM], pi[H_FOS]
    for i in range(pi[H_DIN]):
        h0[i] = (f[i] - pf[om + i]) / pf[osc + i]


@njit(cache=True, error_model="numpy")
def predecir(x, sr, pi, pf, pd, ws64, ws32):
    """Clase (0..7) de una muestra x (C, L) float32 con sample_rate sr: todo el pipeline en una llamada."""
    f = _features(x, sr, pi, pf, pd, ws64, ws32)
    n_tot, maxw = pi[H_DIN], pi[H_MAXW]
    o = ws32.shape[0] - 2 * maxw - 16
    h0 = ws32[o:o + maxw]
    h1 = ws32[o + maxw:o + 2 * maxw]
    _escalar(pi, pf, f, h0)
    return _mlp(pi, pf, h0, h1)


@njit(cache=True, error_model="numpy")
def features_lote(X, sr, pi, pf, pd, out):
    """Features sin escalar de un lote (N, C, L) -> out (N, n_f [+5]); para entrenar y verificar."""
    ws64 = np.zeros(pi[H_C] * pi[H_L] + 4 * (pi[H_L] + VENTANA) + 12 * pi[H_WL] + 16)
    C, WL, pad, n_f, maxw = pi[H_C], pi[H_WL], pi[H_PAD], pi[H_NF], pi[H_MAXW]
    ws32 = np.zeros(C * pi[H_L] + C * WL + 2 * C * (WL + 2 * pad) + C * WL + WL + 2 * (n_f + N_CAND) + 2 * maxw + 16,
                    np.float32)
    for i in range(X.shape[0]):
        f = _features(X[i], sr[i], pi, pf, pd, ws64, ws32)
        for j in range(f.shape[0]):
            out[i, j] = f[j]
    return out


@njit(cache=True, error_model="numpy")
def logits_lote(F, pi, pf, out):
    """Logits del MLP para features sin escalar F (N, d_in) -> out (N, 8)."""
    maxw = pi[H_MAXW]
    h0 = np.zeros(maxw, np.float32)
    h1 = np.zeros(maxw, np.float32)
    for i in range(F.shape[0]):
        _escalar(pi, pf, F[i], h0)
        _logits(pi, pf, h0, h1, out[i])
    return out


@njit(cache=True, parallel=True, error_model="numpy")
def features_paralelo(X, sr, pi, pf, pd, out, n_trozos):
    """Como ``features_lote``, en paralelo por trozos de muestras (cada trozo con sus buffers)."""
    N = X.shape[0]
    C, WL, pad, n_f, maxw = pi[H_C], pi[H_WL], pi[H_PAD], pi[H_NF], pi[H_MAXW]
    for b in prange(n_trozos):
        ws64 = np.zeros(pi[H_C] * pi[H_L] + 4 * (pi[H_L] + VENTANA) + 12 * pi[H_WL] + 16)
        ws32 = np.zeros(C * pi[H_L] + C * WL + 2 * C * (WL + 2 * pad) + C * WL + WL + 2 * (n_f + N_CAND)
                        + 2 * maxw + 16, np.float32)
        for i in range(b * N // n_trozos, (b + 1) * N // n_trozos):
            f = _features(X[i], sr[i], pi, pf, pd, ws64, ws32)
            for j in range(f.shape[0]):
                out[i, j] = f[j]
    return out


@njit(cache=True, error_model="numpy")
def ventanas_lote(X, sr, WL, out, dueno):
    """Ventanas preprocesadas de 128 puntos a 1 Hz (como ``_features``) de un lote: out (≥ 2N, C, WL), dueño de cada
    una; devuelve cuántas hay. Sirve para ajustar un MiniRocket nuevo sobre la misma entrada que ve el motor."""
    C, L = X.shape[1], X.shape[2]
    ws64 = np.zeros(C * L + 4 * (L + VENTANA) + 12 * WL + 16)
    z = np.zeros((C, L), np.float32)
    n = 0
    for i in range(X.shape[0]):
        _preprocesar(X[i], ws64, z)
        if sr[i] > 1.5:
            for c in range(C):
                for t in range(WL):
                    out[n, c, t] = (z[c, 2 * t] + z[c, 2 * t + 1]) / np.float32(2.0)
            dueno[n] = i
            n += 1
        else:
            for v in range(L // WL):
                for c in range(C):
                    for t in range(WL):
                        out[n, c, t] = z[c, v * WL + t]
                dueno[n] = i
                n += 1
    return n


# ====================================================================== clase para el harness


class Model:
    """Modelo para el harness: ``Model()`` carga ``PESOS`` (junto a este fichero) y compila/calienta numba."""

    def __init__(self, pesos=None):
        ruta = Path(pesos) if pesos is not None else HERE / PESOS
        with np.load(ruta) as z:
            self.pi = np.ascontiguousarray(z["pi"], dtype=np.int32)
            self.pf = np.ascontiguousarray(z["pf"], dtype=np.float32)
            self.pd = np.ascontiguousarray(z["pd"], dtype=np.float64)
            self.clase_defecto = int(z["clase_defecto"]) if "clase_defecto" in z.files else 0
        self.forma = (int(self.pi[H_C]), int(self.pi[H_L]))
        self.ws64, self.ws32 = crear_workspace(self.pi)
        # La muestra se copia a este buffer: numba ve siempre el mismo tipo de array (una sola compilación) y da
        # igual que x.series sea de solo lectura, no contigua o de otro dtype.
        self._x = np.zeros(self.forma, np.float32)
        # Compilación y calentamiento fuera del tiempo cronometrado: NaN, picos, tramos congelados, un canal
        # entero NaN y los dos sample_rate.
        rng = np.random.default_rng(0)
        for i in range(8):
            a = rng.normal(size=self.forma).astype(np.float32)
            a[i % self.forma[0], 10 * i:10 * i + 3 + 4 * i] = np.nan
            a[(i + 1) % self.forma[0], 100 + i] = 50.0
            a[(i + 2) % self.forma[0], 200:210] = a[(i + 2) % self.forma[0], 200]
            if i == 7:
                a[0, :] = np.nan
            self._x[...] = a
            for sr in (1.0, 2.0):
                predecir(self._x, sr, self.pi, self.pf, self.pd, self.ws64, self.ws32)
        # Y el camino completo de predict, con una serie de solo lectura (np.frombuffer, como el harness) y con NaN
        from types import SimpleNamespace
        ro = np.frombuffer(a.tobytes(), dtype=np.float32).reshape(self.forma)
        for sr in (1, 2):
            self.predict(SimpleNamespace(series=ro, meta={"sample_rate": sr}))

    def predict(self, x) -> int:
        try:
            s = x.series
            if s.shape != self.forma:
                return self.clase_defecto
            self._x[...] = s
            sr = float(x.meta.get("sample_rate", 1.0))
            return int(predecir(self._x, sr, self.pi, self.pf, self.pd, self.ws64, self.ws32))
        except Exception:
            return self.clase_defecto
