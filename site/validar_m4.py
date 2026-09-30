"""Confere a reimplementação em NumPy do M4 contra o treino.py: NLL (gap para o oráculo) e erro de direção
no conjunto de teste do ruído vetorial, para as redes GL e GBL."""
import json, sys, time
import numpy as np
import os
from m4np import M4, wrap
# uso: python3 site/validar_m4.py <pasta do dataset>   (roda da raiz do repositório)
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'
DADOS = sys.argv[1] if len(sys.argv) > 1 else B + 'vicsek-dados/dataset_bloco2'

def nll_oraculo(meta, y, tb, C, n):
    psi = wrap(y.astype(np.float64) - tb.astype(np.float64)); eta = meta['eta']
    n = n.astype(np.float64)
    r = eta * (n if meta['gcnorm'] else n + 1) / (C.astype(np.float64) * n)
    s2 = np.sin(psi) ** 2
    root = np.sqrt(np.maximum(r ** 2 - s2, 1e-12 * r ** 2))
    lp = np.full(psi.shape, -np.inf)
    big = r > 1
    lp[big] = np.log1p(np.cos(psi[big]) / root[big]) - np.log(2 * np.pi)
    small = (~big) & (s2 < r ** 2 * (1 + 1e-5)) & (np.cos(psi) > 0)
    lp[small] = np.log(np.cos(psi[small]) / (np.pi * root[small]))
    return -lp

esperado = {('GL', 'G'): (0.20594, 0.97), ('GL', 'B'): (0.19657, 0.67), ('GBL', 'G'): (0.2014, 0.93), ('GBL', 'B'): (0.1917, 0.62)}
for nome, rot in [('vetorial_m4_hist_GL_e60', 'GL'), ('vetorial_m4_hist_GBL_e60', 'GBL')]:
    m = M4(B + 'runs/' + nome)
    for arq, est in [('n1G_v0.500_r0.125_e0.5500_L256x256_s200040.npz', 'G'), ('n1G_v0.500_r1.000_e0.5500_L1024x128_s200050.npz', 'B')]:
        z = np.load(os.path.join(DADOS, arq))
        meta = json.loads(str(z['meta']))
        feat, ptr = z['feat'].astype(np.float64), z['ptr']
        S = len(z['y'])
        seg = np.repeat(np.arange(S), np.diff(ptr))
        t0 = time.time()
        out = np.concatenate([m.saida(feat[ptr[a]:ptr[b]], seg[ptr[a]:ptr[b]] - a, b - a) for a, b in
                              zip(range(0, S, 8192), [min(S, k + 8192) for k in range(0, S, 8192)])])
        lp = m.logp(out, z['y'].astype(np.float64))
        orac = nll_oraculo(meta, z['y'], z['thbar_rel'], z['C'], z['n'])
        mu = np.arctan2(out[:, 1], out[:, 0])
        err = np.degrees(np.abs(wrap(mu - z['thbar_rel'].astype(np.float64)))).mean()
        gap = (-lp).mean() - orac.mean()
        e = esperado[(rot, est)]
        print(f'{nome:28s} estado {est}: S={S} gap {gap:.5f} (treino.py {e[0]:.4f})  erro {err:.2f}° (treino.py {e[1]:.2f}°)  [{time.time()-t0:.1f}s]  meta.conjunto={meta["conjunto"]} robs={meta.get("robs")}')
