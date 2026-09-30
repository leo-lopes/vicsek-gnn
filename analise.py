#!/usr/bin/env python3
"""
Experimento I -- o que a rede aprendeu num passo.

  python3 analise.py runs/angular_m4_hist_GBL [runs/outra_rodada ...]
  python3 analise.py runs/... --dados /outro/dir     (se o dataset mudou de lugar)

Gera em cada runs/<nome>/ e imprime as tabelas:

  concentracao_C.png  <cos psi> em funcao do alinhamento local C:
                      - verdade: cos(dtheta - thbar) medido nos dados
                      - teoria: sin(pi eta)/(pi eta) no angular; E[cos arg(1 + r e^{i phi})], r = eta/C, no vetorial
                      - rede: a concentracao R que ela preve, e cos(dtheta - mu_rede) que ela de fato obtem.
                      No angular a verdade e plana (teste nulo: a forma do ruido nao depende de C);
                      no vetorial ela cresce com C. A rede aprendeu isso?
  pesos_d.png         so m4 e m5. m4: w(d)/w(0.5). m5: peso de atencao relativo (alpha x n_arestas;
                      1 = uniforme) medio por distancia. A regra verdadeira e um degrau em d = 1.

Use no mesmo diretorio do treino.py. Precisa de numpy, torch e matplotlib.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import treino as T


def construir(pasta, dados, dev):
    with open(os.path.join(pasta, 'resultado.json')) as f:
        res = json.load(f)
    a = res['args']
    if a['modelo'] == 'm0':
        raise SystemExit('%s: o m0 nao tem rede para analisar' % pasta)
    cab = {'hist': lambda: T.CabecaHist(a['bins']), 'hlivre': lambda: T.CabecaHistLivre(a['bins']),
           'vm': T.CabecaVM}[a['cabeca']]().to(dev)
    model = T.MODELOS[a['modelo']](a['oculto'], cab.nout).to(dev)
    model.load_state_dict(torch.load(os.path.join(pasta, 'modelo.pt'), map_location=dev))
    model.eval()
    D = T.Dados(T.listar(dados or a['dados'], a['ruido']), 'validacao', 'GBL', dev, model.precisa)
    return a, cab, model, D


def concentracao(cab, out):
    """R = <cos psi> da distribuicao prevista, em torno da direcao prevista."""
    if isinstance(cab, T.CabecaVM):
        k = F.softplus(out[:, 2]) + 1e-3
        return torch.special.i1e(k) / torch.special.i0e(k)
    if isinstance(cab, T.CabecaHist):
        h = out[:, 2:] + cab.prior
        p = torch.softmax(torch.cat([h.flip(1), h], 1), 1)
        c = -math.pi + (torch.arange(cab.K, device=out.device) + 0.5) * cab.w
        return (p * torch.cos(c)).sum(1)
    p = torch.softmax(out, 1)                                   # histograma livre
    return torch.hypot((p * torch.cos(cab.c)).sum(1), (p * torch.sin(cab.c)).sum(1))


def teoria_cos(C, eta, noise, gcnorm):
    if noise == 0:
        return np.full_like(C, math.sin(math.pi * eta) / (math.pi * eta))
    if not gcnorm:                                               # -G 0: depende tambem de n
        return np.full_like(C, np.nan)
    phi = np.linspace(0, 2 * np.pi, 2001)[:-1]
    return np.array([np.cos(np.angle(1 + (eta / c) * np.exp(1j * phi))).mean() for c in C])


@torch.no_grad()
def coletar(model, cab, D, lote=16384, max_arestas=3000000):
    col = {k: [] for k in ('C', 'y', 'tb', 'mu', 'R', 'est')}
    ar_d, ar_w, n_ar = [], [], 0
    for i in range(0, D.S, lote):
        idx = torch.arange(i, min(i + lote, D.S), device=D.y.device)
        out, mu = model(D, idx)
        mu = cab.mu(out) if mu is None else mu
        col['C'].append(D.C[idx]); col['y'].append(D.y[idx]); col['tb'].append(D.tb[idx])
        col['mu'].append(mu); col['R'].append(concentracao(cab, out)); col['est'].append(D.est[idx])
        if isinstance(model, T.M5) and n_ar < max_arestas:
            feat, seg, cnt = D.arestas(idx)
            a = model.atencao(feat, seg, len(idx))
            ar_d.append(feat[:, 2]); ar_w.append(a * cnt[seg].float()); n_ar += len(seg)
    col = {k: torch.cat(v).cpu().numpy() for k, v in col.items()}
    if ar_d:
        col['ar_d'] = torch.cat(ar_d).cpu().numpy()
        col['ar_w'] = torch.cat(ar_w).cpu().numpy()
    return col


def fig_concentracao(pasta, a, meta, col):
    C = col['C'].astype(float)
    verdade = np.cos(T.wrap_np(col['y'] - col['tb']))
    obtido = np.cos(T.wrap_np(col['y'] - col['mu']))
    bordas = np.linspace(0, 1, 21)
    b = np.clip(np.digitize(C, bordas) - 1, 0, 19)
    cen, v, R, o, nb = [], [], [], [], []
    for k in range(20):
        m = b == k
        if m.sum() < 300:
            continue
        cen.append(C[m].mean()); v.append(verdade[m].mean()); R.append(col['R'][m].mean())
        o.append(obtido[m].mean()); nb.append(int(m.sum()))
    cen = np.array(cen)
    th = teoria_cos(cen, meta['eta'], meta['noise'], meta['gcnorm'])
    print('\n  <cos psi> por faixa de C (validacao)')
    print('  %6s %8s %8s %9s %9s %8s' % ('C', 'teoria', 'verdade', 'rede_R', 'rede_obt', 'n'))
    for i in range(len(cen)):
        print('  %6.3f %8.4f %8.4f %9.4f %9.4f %8d' % (cen[i], th[i], v[i], R[i], o[i], nb[i]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cen, th, '-', color='0.5', label='teoria')
    ax.plot(cen, v, 'o', mfc='none', color='k', label='verdade (dados)')
    ax.plot(cen, R, 's', color='C0', label='rede: R previsto')
    ax.plot(cen, o, '^', color='C1', label='rede: cos(dθ − μ_rede) obtido')
    ax.set_xlabel('alinhamento local C')
    ax.set_ylabel('<cos ψ>')
    ax.set_title('%s  (%s, η=%.2f)' % (os.path.basename(pasta), a['ruido'], meta['eta']), fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, 'concentracao_C.png'), dpi=120)
    plt.close(fig)


def fig_pesos(pasta, model, col, dev):
    dd = np.linspace(0, 2, 401)
    if isinstance(model, T.M4):
        with torch.no_grad():
            w = model.peso(torch.tensor(dd, dtype=torch.float32, device=dev)).cpu().numpy()
        w = w / w[np.argmin(np.abs(dd - 0.5))]
        ylab, x, y = 'w(d) / w(0,5)', dd, w
    else:
        bordas = np.linspace(0, 2, 41)
        b = np.clip(np.digitize(col['ar_d'], bordas) - 1, 0, 39)
        x = 0.5 * (bordas[1:] + bordas[:-1])
        y = np.array([col['ar_w'][b == k].mean() if (b == k).any() else np.nan for k in range(40)])
        lam = float(F.softplus(model.lam.detach()))
        ylab = 'atenção relativa (1 = uniforme)   λ = %.3f' % lam
    print('\n  peso por distancia (%s)' % ('m4: w(d)/w(0.5)' if isinstance(model, T.M4) else 'm5: alpha x n'))
    for d0 in (0.25, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 1.9):
        print('  d = %.2f   %.4f' % (d0, np.interp(d0, x, y)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, '-o' if len(x) < 100 else '-', ms=3)
    ax.axvline(1.0, color='0.6', ls='--', lw=1)
    ax.set_xlabel('distância d (R = 1)')
    ax.set_ylabel(ylab)
    ax.set_title(os.path.basename(pasta), fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, 'pesos_d.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='experimento I: concentracao x C e pesos x distancia')
    ap.add_argument('pastas', nargs='+')
    ap.add_argument('--dados', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pasta in a.pastas:
        print('== %s' % pasta)
        args, cab, model, D = construir(pasta, a.dados, dev)
        meta = next(m for f, m in T.listar(a.dados or args['dados'], args['ruido']))
        col = coletar(model, cab, D)
        fig_concentracao(pasta, args, meta, col)
        if isinstance(model, (T.M4, T.M5)):
            fig_pesos(pasta, model, col, dev)
        print('  figuras em %s/' % pasta)


if __name__ == '__main__':
    main()
