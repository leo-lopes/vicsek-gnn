#!/usr/bin/env python3
"""
Bandas em caixa pequena (ex. 64x64) perto da transicao: rede x oraculo.

  python3 bandas64.py <pasta> [<pasta> ...]        (le os .npz do rollout.py gravados com --quadros)

Para cada corrida:
  - Phi da producao: media, chi, Binder e histograma P(Phi) (bimodal perto da transicao descontinua);
  - amplitude de banda por quadro: A = |<exp(i k.r)>| do modo de Fourier dominante entre os mais
    baixos (|m|, |n| <= 2). Num gas homogeneo A ~ 0.886/sqrt(N); aglomerados do gas chegam a ~0.2;
  - banda presente no quadro se A > A_MIN e a normal da banda (k) estiver alinhada com a direcao do
    bando (|cos| > COS_MIN): banda de verdade anda ao longo da normal; aglomerado nao tem orientacao;
  - perfil medio da banda: particulas projetadas na normal do modo dominante, alinhadas pela fase do
    modo em cada quadro com banda, movimento para a direita;
  - kimografo alinhado pela fase (a banda fica parada no centro; some quando a banda se desfaz).

Figuras e resumo em <primeira pasta>/figuras64/. So precisa de numpy e matplotlib.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

A_MIN, COS_MIN = 0.25, 0.8
# fonte = 'oraculo' ou 'rede_<estados do treino>' (tirado da origem, ex. vetorial_m4_hist_GL_e60 -> rede_GL)
ORDEM = ('oraculo', 'rede_GBL', 'rede_GL')
COR = {'oraculo': '#2a78d6', 'rede_GBL': '#eb6834', 'rede_GL': '#1baf7a'}    # slots 1-3 da paleta (validada)
NOME = {'oraculo': 'oráculo', 'rede_GBL': 'rede M4 GBL', 'rede_GL': 'rede M4 GL'}
MARCA = {'oraculo': 'o', 'rede_GBL': 's', 'rede_GL': '^'}
AVISO = 'GBL = rede treinada com G+B+L (viu bandas) · GL = rede treinada só com G+L (nunca viu bandas)'


def fonte_de(origem):
    if origem.startswith('oraculo'):
        return 'oraculo'
    m = re.search(r'_([GBL]+)_e\d+', origem)
    return 'rede_' + (m.group(1) if m else origem)


def em_ordem(fontes):
    fontes = set(fontes)
    return [f for f in ORDEM if f in fontes] + sorted(fontes - set(ORDEM))
INIT = {'aleatorio': 'aleatório', 'alinhado': 'alinhado'}
TINTA, TINTA2, MUDO, GRADE, EIXO, FUNDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('azul', [FUNDO, '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                                                 '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({
    'figure.facecolor': FUNDO, 'axes.facecolor': FUNDO, 'savefig.facecolor': FUNDO,
    'axes.edgecolor': EIXO, 'axes.labelcolor': TINTA2, 'axes.titlecolor': TINTA,
    'xtick.color': EIXO, 'ytick.color': EIXO, 'xtick.labelcolor': TINTA2, 'ytick.labelcolor': TINTA2,
    'text.color': TINTA, 'grid.color': GRADE, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9.5,
    'axes.titlesize': 9.5, 'axes.titlelocation': 'left', 'legend.frameon': False,
    'legend.fontsize': 8.5, 'mathtext.default': 'regular'})

MODOS = [(m, n) for m in range(-2, 3) for n in range(0, 3) if (n > 0 or m > 0)]   # meio plano, sem (0,0)
NS = 64                                                                          # caixas do perfil / kimografo


def suaviza2(H, sig):
    ky, kx = np.fft.fftfreq(H.shape[0]), np.fft.fftfreq(H.shape[1])
    return np.real(np.fft.ifft2(np.fft.fft2(H) * np.exp(-2 * (np.pi * sig) ** 2 * (ky[:, None] ** 2 + kx[None, :] ** 2))))


def analisar(arq):
    d = np.load(arq)
    if 'quadros' not in d.files:
        return None
    p = json.loads(str(d['params']))
    q = d['quadros'].astype(np.float64)                   # (T, 3, nsub)
    Lx, Ly = float(p['L']), float(p['Y'] or p['L'])
    N, T, nsub = int(p['N']), q.shape[0], q.shape[2]
    phi = d['phi'].astype(np.float64)
    phit = d['phi_transiente'].astype(np.float64)
    m1, m2, m4 = phi.mean(), (phi ** 2).mean(), (phi ** 4).mean()
    r = dict(arq=os.path.basename(arq), ruido=p['ruido'], rho=float(p['rho']), init=p['init'],
             semente=int(p['semente']), N=N, Lx=Lx, Ly=Ly, dt=int(p['quadros']), T=T, nsub=nsub,
             fonte=fonte_de(p['origem']), robs=float(p['robs']),
             transiente=int(p['transiente']), passos=int(p['passos']),
             phi=float(m1), chi=float(N * (m2 - m1 * m1)), binder=float(1 - m4 / (3 * m2 * m2)))
    x, y, th = q[:, 0, :], q[:, 1, :], q[:, 2, :]
    Th = np.arctan2(np.sin(th).mean(1), np.cos(th).mean(1))
    Z = np.zeros((T, len(MODOS)), complex)
    for j, (m, n) in enumerate(MODOS):
        ph = 2 * np.pi * (m * x / Lx + n * y / Ly)
        Z[:, j] = np.cos(ph).mean(1) + 1j * np.sin(ph).mean(1)
    A = np.abs(Z)
    jd = A.argmax(1)
    Ad = A[np.arange(T), jd]
    kv = np.array([[MODOS[j][0] / Lx, MODOS[j][1] / Ly] for j in jd])
    cosk = np.cos(np.arctan2(kv[:, 1], kv[:, 0]) - Th)
    banda = (Ad > A_MIN) & (np.abs(cosk) > COS_MIN)
    # perfil medio alinhado pela fase do modo dominante (so quadros com banda)
    prof, nq = np.zeros(NS), 0
    for f in np.where(banda)[0]:
        m, n = MODOS[jd[f]]
        s = np.mod((m * x[f] / Lx + n * y[f] / Ly) - np.angle(Z[f, jd[f]]) / (2 * np.pi) + 0.5, 1.0)   # fase 0.5 = centro
        if cosk[f] < 0:
            s = 1.0 - s                                   # movimento sempre para +s
        prof += np.bincount((s * NS).astype(int) % NS, minlength=NS)
        nq += 1
    prof = prof / max(nq, 1) * NS / nsub                  # rho / rho0
    # kimografo: cada quadro projetado na normal do SEU modo dominante e alinhado pela fase dele
    # (a banda fica no centro em qualquer orientacao; sem banda sobra so o ruido)
    K = np.zeros((T, NS))
    for f in range(T):
        mf, nf = MODOS[jd[f]]
        s = np.mod((mf * x[f] / Lx + nf * y[f] / Ly) - np.angle(Z[f, jd[f]]) / (2 * np.pi) + 0.5, 1.0)
        if cosk[f] < 0:
            s = 1.0 - s
        K[f] = np.bincount((s * NS).astype(int) % NS, minlength=NS) * NS / nsub
    jm = np.bincount(jd[banda], minlength=len(MODOS)).argmax() if banda.any() else int(A.mean(0).argmax())
    m, n = MODOS[jm]
    lam = 1.0 / math.hypot(m / Lx, n / Ly)
    # quadro tipico com banda: A mais proximo da mediana entre os quadros com banda
    if banda.any():
        fb = np.where(banda)[0]
        ft = fb[np.argmin(np.abs(Ad[fb] - np.median(Ad[fb])))]
    else:
        ft = T - 1
    r.update(A_media=float(Ad.mean()), frac_banda=float(banda.mean()), modo='(%d,%d)' % (m, n), lam_modo=lam,
             cos_medio=float(np.abs(cosk[banda]).mean()) if banda.any() else float('nan'),
             A_banda=float(Ad[banda].mean()) if banda.any() else float('nan'))
    r.update(Ad=Ad, banda=banda, prof=prof, K=K, phi_prod=phi, phi_all=np.concatenate([phit, phi]),
             tipico=q[ft], Th_tipico=float(Th[ft]), f_tipico=int(ft), ntrans=len(phit))
    return r


def carregar(pastas):
    R = []
    for pa in pastas:
        for arq in sorted(glob.glob(os.path.join(pa, '*.npz'))):
            r = analisar(arq)
            if r is not None:
                R.append(r)
    return R


def mapa(fr, Lx, Ly, b=2.0, sig=1.0):
    x, y = fr[0], fr[1]
    nx, ny = int(round(Lx / b)), int(round(Ly / b))
    H = np.histogram2d(y, x, bins=[ny, nx], range=[[0, Ly], [0, Lx]])[0]
    return suaviza2(H, sig) * (Lx * Ly) / (len(x) * b * b)


def principais(R, init='aleatorio', semente=1):
    """uma corrida por (rho, fonte): a da semente pedida"""
    out = {}
    for r in R:
        if r['init'] == init and r['semente'] == semente:
            out[(r['rho'], r['fonte'])] = r
    return out


def fig_varredura(R, saida):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    for fo in em_ordem(r['fonte'] for r in R):
        mk = MARCA.get(fo, 'D')
        for init, ls in (('aleatorio', '-'), ('alinhado', (0, (4, 3)))):
            rs = [r for r in R if r['fonte'] == fo and r['init'] == init]
            if not rs:
                continue
            rhos = sorted({r['rho'] for r in rs})
            med = {k: [np.mean([r[k] for r in rs if r['rho'] == rh]) for rh in rhos] for k in ('phi', 'frac_banda', 'binder')}
            lab = '%s, início %s' % (NOME[fo], INIT.get(init, init))
            for ax, k in zip(axs, ('phi', 'frac_banda', 'binder')):
                ax.plot(rhos, med[k], color=COR[fo], ls=ls, marker=mk, ms=4.5, lw=1.5, label=lab)
                for r in rs:                                           # sementes individuais
                    ax.plot(r['rho'], r[k], marker=mk, ms=3, color=COR[fo], alpha=0.35, ls='none')
    axs[0].set_ylabel('⟨Φ⟩ na produção'); axs[1].set_ylabel('fração do tempo com banda')
    axs[2].set_ylabel('Binder U₄')
    axs[2].axhline(2 / 3, color=EIXO, lw=0.8); axs[2].axhline(1 / 3, color=EIXO, lw=0.8)
    for ax in axs:
        ax.set_xlabel('ρ'); ax.grid(True, axis='y')
    axs[0].legend(loc='lower right')
    r0 = R[0]
    fig.suptitle('Caixa %d × %d, ruído %s (η = 0.55): varredura em densidade perto da transição. '
                 'Linha = média das sementes; pontos claros = sementes\n%s' % (r0['Lx'], r0['Ly'], r0['ruido'], AVISO),
                 x=0.01, ha='left', fontsize=10.5)
    fig.savefig(os.path.join(saida, 'varredura.png'), dpi=170)
    plt.close(fig)


def fig_grade(R, saida, rhos, init='aleatorio'):
    P = principais(R, init)
    fontes = em_ordem(k[1] for k in P)
    rhos = [rh for rh in rhos if any((rh, fo) in P for fo in fontes)]
    if not rhos:
        return
    nl = 2 * len(fontes) + 2
    fig = plt.figure(figsize=(2.35 * len(rhos) + 0.9, 2.2 * nl), constrained_layout=True)
    gs = fig.add_gridspec(nl, len(rhos), height_ratios=[1.0] * len(fontes) + [0.9] * len(fontes) + [0.75, 0.75])
    mapas = {k: mapa(r['tipico'], r['Lx'], r['Ly']) for k, r in P.items() if k[0] in rhos}
    vmax = np.percentile(np.concatenate([m.ravel() for m in mapas.values()]), 99.5)
    Ks = [P[k]['K'] for k in P if k[0] in rhos]
    kmax = np.percentile(np.concatenate([k.ravel() for k in Ks]), 99.5)
    im = imk = None
    for j, rh in enumerate(rhos):
        for i, fo in enumerate(fontes):
            r = P.get((rh, fo))
            ax = fig.add_subplot(gs[i, j])
            axk = fig.add_subplot(gs[len(fontes) + i, j])
            if r is None:
                ax.axis('off'); axk.axis('off'); continue
            L = r['Lx']
            im = ax.imshow(mapas[(rh, fo)], origin='lower', extent=[0, L, 0, r['Ly']], cmap=SEQ, vmin=0, vmax=vmax,
                           interpolation='bilinear')
            ax.annotate('', xy=(L / 2 + 0.3 * L * math.cos(r['Th_tipico']), L / 2 + 0.3 * L * math.sin(r['Th_tipico'])),
                        xytext=(L / 2, L / 2), arrowprops=dict(arrowstyle='-|>', color=TINTA, lw=1.3))
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(EIXO)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title('ρ = %g · %s\nΦ = %.2f · banda %.0f%% do tempo' % (rh, NOME[fo], r['phi'], 100 * r['frac_banda']),
                         fontsize=8.3)
            tmax = r['T'] * r['dt']
            imk = axk.imshow(r['K'], origin='lower', aspect='auto', extent=[-0.5, 0.5, 0, tmax / 1000], cmap=SEQ,
                             vmin=0, vmax=kmax, interpolation='nearest')
            for s in axk.spines.values():
                s.set_visible(True); s.set_color(EIXO)
            axk.set_xticks([-0.5, 0, 0.5]); axk.set_xticklabels(['−λ/2', '0', 'λ/2'])
            if j == 0:
                axk.set_ylabel('mil passos (produção)')
            axk.set_title('kimógrafo alinhado · %s' % NOME[fo], fontsize=8)
        # perfil medio e P(Phi)
        axp = fig.add_subplot(gs[2 * len(fontes), j])
        axh = fig.add_subplot(gs[2 * len(fontes) + 1, j])
        for fo in fontes:
            r = P.get((rh, fo))
            if r is None:
                continue
            s = (np.arange(NS) + 0.5) / NS - 0.5
            if r['frac_banda'] > 0.02:
                axp.plot(s, r['prof'], color=COR[fo], lw=1.5, label=NOME[fo])
            axh.hist(r['phi_prod'], bins=np.linspace(0, 1, 51), density=True, histtype='step', color=COR[fo], lw=1.4,
                     label=NOME[fo])
        axp.axhline(1, color=EIXO, lw=0.8); axp.grid(True, axis='y')
        axp.set_xticks([-0.5, 0, 0.5]); axp.set_xticklabels(['−λ/2', '0', 'λ/2'])
        axp.set_title('perfil médio da banda (→ movimento)', fontsize=8)
        axh.set_title('P(Φ) na produção', fontsize=8); axh.set_xlabel('Φ')
        if j == 0:
            axp.set_ylabel(r'$\rho/\rho_0$'); axh.set_ylabel('densidade de prob.')
            axp.legend(loc='upper left', fontsize=7.5)
    ax_mapas = [a for a in fig.axes if a.get_images() and a.get_images()[0].get_extent()[1] > 1]
    ax_kimo = [a for a in fig.axes if a.get_images() and a.get_images()[0].get_extent()[1] <= 1]
    for ims, axl in ((im, ax_mapas), (imk, ax_kimo)):
        if ims is not None and axl:
            cb = fig.colorbar(ims, ax=axl, location='right', fraction=0.02, pad=0.01, shrink=0.9)
            cb.set_label(r'$\rho/\rho_0$'); cb.outline.set_edgecolor(EIXO)
    fig.suptitle('Caixa %d × %d, vetorial η = 0.55, início %s, semente 1\n'
                 'Mapas: quadro típico com banda (seta = direção do bando)\n'
                 'Kimógrafos: densidade ao longo da normal da banda, alinhada pela fase dela\n%s'
                 % (R[0]['Lx'], R[0]['Ly'], INIT.get(init, init), AVISO), x=0.01, ha='left', fontsize=10)
    fig.savefig(os.path.join(saida, 'grade_%s.png' % init), dpi=150)
    plt.close(fig)


def main():
    pastas = sys.argv[1:] or ['varredura64']
    saida = os.path.join(pastas[0], 'figuras64')
    os.makedirs(saida, exist_ok=True)
    R = carregar(pastas)
    R.sort(key=lambda r: (r['init'], r['rho'], r['fonte'], r['semente']))
    cols = ['arq', 'fonte', 'init', 'semente', 'rho', 'N', 'robs', 'transiente', 'passos', 'phi', 'chi', 'binder',
            'A_media', 'frac_banda', 'A_banda', 'cos_medio', 'modo']
    with open(os.path.join(saida, 'resumo64.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in R:
            f.write('\t'.join(('%.4g' % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
    print('%-8s %-9s %3s %5s %6s %7s %6s %6s %6s %6s %6s' % ('fonte', 'init', 'sem', 'rho', 'Phi', 'chi', 'U4',
                                                             'A', 'banda', 'A_bd', 'modo'))
    for r in R:
        print('%-8s %-9s %3d %5g %6.3f %7.2f %6.3f %6.3f %5.0f%% %6.3f %6s' % (
            r['fonte'], r['init'], r['semente'], r['rho'], r['phi'], r['chi'], r['binder'], r['A_media'],
            100 * r['frac_banda'], r['A_banda'], r['modo']))
    fig_varredura(R, saida)
    for init in sorted({r['init'] for r in R}):
        P = principais(R, init)
        ambos = sorted({k[0] for k in P if (k[0], 'oraculo') in P and any(f != 'oraculo' and (k[0], f) in P
                                                                              for f in em_ordem(q[1] for q in P))})
        fig_grade(R, saida, ambos or sorted({k[0] for k in P}), init)


if __name__ == '__main__':
    main()
