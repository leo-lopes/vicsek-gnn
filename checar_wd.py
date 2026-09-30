#!/usr/bin/env python3
"""
Confere o w(d) de modelos M4 antes de rodar o rollout com --robs menor que o do treino.

  python3 checar_wd.py runs/vetorial_m4_hist_GL_e60 [runs/vetorial_m4_hist_GBL_e60 ...] [--robs 1.2]

O M4 agrega por media ponderada normalizada: sum_j w(d_j) m_j / sum_j w(d_j). Cortar as arestas com
d > robs muda a politica no maximo pela fracao do peso que elas carregam. Pior caso: particula sem
vizinhos dentro de robs (so o laco i -> i, peso w(0)) e a coroa robs < d < 2 cheia na densidade
local maxima de uma banda (rho_loc = 8, folgado: o pico medido nas bandas e ~5).
Roda na CPU, precisa do treino.py na mesma pasta.
"""
import json
import math
import os
import sys

import torch

import treino as T

args = sys.argv[1:]
robs = 1.2
if '--robs' in args:
    k = args.index('--robs')
    robs = float(args[k + 1])
    del args[k:k + 2]
if not args:
    raise SystemExit(__doc__)

RHO_LOC, ROBS_TREINO = 8.0, 2.0
D_TAB = [0.0, 0.5, 0.9, 0.95, 1.0, 1.03, 1.05, 1.08, 1.1, 1.2, 1.5, 2.0]
d = torch.linspace(0, ROBS_TREINO, 4001)
res = []
for pasta in args:
    with open(os.path.join(pasta, 'resultado.json')) as f:
        a = json.load(f)['args']
    if a['modelo'] != 'm4':
        print('%s: modelo %s (este teste e so para o M4)' % (pasta, a['modelo']))
        continue
    cab = {'hist': lambda: T.CabecaHist(a['bins']), 'hlivre': lambda: T.CabecaHistLivre(a['bins']),
           'vm': T.CabecaVM}[a['cabeca']]()
    m = T.MODELOS['m4'](a['oculto'], cab.nout)
    m.load_state_dict(torch.load(os.path.join(pasta, 'modelo.pt'), map_location='cpu'))
    m.eval()
    with torch.no_grad():
        w = m.peso(d).double()
        wt = m.peso(torch.tensor(D_TAB)).double()
    w0 = float(w[0])
    dentro = float(w[(d >= 0.3) & (d <= 0.9)].mean())
    meia = float(d[torch.nonzero(w < dentro / 2)[0, 0]]) if (w < dentro / 2).any() else float('nan')
    wfora = float(w[d >= robs].max())
    nfora = RHO_LOC * math.pi * (ROBS_TREINO ** 2 - robs ** 2)
    pior = nfora * wfora / (w0 + nfora * wfora)
    res.append((os.path.basename(os.path.normpath(pasta)), a['estados'], wt, dentro, meia, wfora, pior))

if res:
    print('\nw(d) bruto (softplus + 1e-6 do codigo)')
    print('%6s ' % 'd' + ' '.join('%22s' % ('%s [%s]' % (r[0].replace('vetorial_', 'vet_').replace('angular_', 'ang_')
                                                          .replace('_hist', ''), r[1]))[-22:] for r in res))
    for i, dd in enumerate(D_TAB):
        print('%6.2f ' % dd + ' '.join('%22.3g' % float(r[2][i]) for r in res))
    print()
    for nome, est, _, dentro, meia, wfora, pior in res:
        ok = pior < 1e-3
        print('%s [treino %s]: w medio em 0.3-0.9 = %.3g; cai a metade em d = %.3f; max w(d >= %.2f) = %.2g'
              % (nome, est, dentro, meia, robs, wfora))
        print('   pior caso com --robs %.2f: %.1e da agregacao (coroa com %.0f vizinhos a rho_loc = %g)  -> %s'
              % (robs, pior, RHO_LOC * math.pi * (ROBS_TREINO ** 2 - robs ** 2), RHO_LOC,
                 'OK, mesma politica' if ok else 'NAO use --robs %.2f com este modelo' % robs))
