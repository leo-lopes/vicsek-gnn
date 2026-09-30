#!/usr/bin/env python3
"""
Bandas na caixa 1024 x 128 com as tres fontes: oraculo x rede M4 GBL (viu bandas no treino) x
rede M4 GL (nunca viu bandas).

  python3 bandas1024.py                      (padrao: pastas bandas e bandas_GL, ruido vetorial)
  python3 bandas1024.py bandas bandas_GL --ruido vetorial --saida bandas_GL/figuras

Reusa o analisar() do bandas_analise.py (modo dominante, velocidade pela fase, kimografo no
referencial das bandas; cache em <pasta>/figuras/.cache) e separa a fonte pela origem do arquivo,
como o bandas64.py: oraculo / rede_GBL / rede_GL.

O perfil medio e refeito com a MESMA resolucao ao longo da normal (--dn, padrao 1.0) para as tres
fontes. O bandas_analise.py usa bins de 2 na coordenada "desinclinada", que ao longo da normal
viram 2 cos(alpha): 2.0 nas bandas perpendiculares e 1.4 nas obliquas a 45 graus, e o pico medido
dependia disso. Tambem mede o excesso de massa por banda, integral de (rho - rho_gas) ao longo da
normal, e compara com a regra da alavanca (1 - rho_gas) x espacamento.

Grava em <saida>: configuracoes_, densidade_, kimografos_, phi_ e perfis_oraculo_GBL_GL.png e
resumo1024_oraculo_GBL_GL.tsv. So precisa de numpy e matplotlib.
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np

ap = argparse.ArgumentParser(description='bandas 1024x128: oraculo x GBL x GL')
ap.add_argument('pastas', nargs='*', default=['bandas', 'bandas_GL'])
ap.add_argument('--ruido', default='vetorial', choices=['vetorial', 'angular'])
ap.add_argument('--saida', default=None, help='padrao: <ultima pasta>/figuras')
ap.add_argument('--dn', type=float, default=1.0, help='resolucao do perfil ao longo da normal')
A = ap.parse_args()
sys.argv = sys.argv[:1]                       # o bandas_analise le sys.argv ao ser importado
import bandas_analise as B                    # noqa: E402  (tambem fixa o estilo do matplotlib)
import matplotlib.pyplot as plt               # noqa: E402

SAIDA = A.saida or os.path.join(A.pastas[-1], 'figuras')
SUF = 'oraculo_GBL_GL'
ORDEM = ('oraculo', 'rede_GBL', 'rede_GL')
COR = {'oraculo': '#2a78d6', 'rede_GBL': '#eb6834', 'rede_GL': '#1baf7a'}   # slots 1-3 da paleta validada
NOME = {'oraculo': 'oráculo (Vicsek exato)', 'rede_GBL': 'rede M4 GBL (viu bandas)',
        'rede_GL': 'rede M4 GL (nunca viu bandas)'}
CURTO = {'oraculo': 'oráculo', 'rede_GBL': 'GBL', 'rede_GL': 'GL'}


def fonte_de(arq):
    nome = os.path.basename(arq)
    if nome.startswith('oraculo'):
        return 'oraculo'
    m = re.search(r'_([GBL]+)_e\d+', nome)
    return 'rede_' + (m.group(1) if m else '?')


def carregar():
    R = {}
    for pasta in A.pastas:
        for arq in sorted(glob.glob(os.path.join(pasta, '*.npz'))):
            B.CACHE = os.path.join(pasta, 'figuras', '.cache')
            r = B.analisar(arq)
            if r['ruido'] != A.ruido:
                continue
            f = fonte_de(arq)
            if f in R:
                print('aviso: mais de uma corrida de %s; fico com %s' % (f, R[f]['arq']))
                continue
            r['fonte'], r['caminho'] = f, arq
            R[f] = r
            print('ok', f, r['arq'], flush=True)
    return R


# ------------------------------------------------------------------ perfil com resolucao fixa na normal
def coordenada(q, r):
    """a mesma coordenada "desinclinada" do analisar(), com as bandas andando para +S"""
    Lx, Ly, m, n = r['Lx'], r['Ly'], r['m'], r['n']
    if m > 0:
        P = Lx
        S = np.mod(q[:, 0, :] + q[:, 1, :] * np.float32(n * Lx / (m * Ly)), np.float32(Lx))
    else:
        P = Ly
        S = q[:, 1, :].copy()
    alpha = math.atan2(n / Ly, m / Lx)
    if np.mean(np.cos(r['th_dir'] - alpha)) < 0:
        S = np.mod(np.float32(P) - S, np.float32(P))
    return S, P


def perfil_normal(r, dn):
    with np.load(r['caminho']) as d:
        q = d['quadros']
        phi = d['phi'].astype(np.float64)
    S, P = coordenada(q, r)
    del q
    T, nsub = S.shape
    cnt, cosang = r['cnt'], r['cosang']
    nb = int(round(P * cosang / dn))
    bw = P / nb                                    # bin na coordenada S (= dn ao longo da normal)
    sig_b = min(20.0, max(3.0, 0.05 * P / cnt)) / bw
    W = min(int(round(0.5 * (P / cnt) / bw)), nb // 2 - 1)
    acum, nseg = np.zeros(2 * W + 1), 0
    for f in range(T):
        h = np.bincount((S[f] / bw).astype(np.int64) % nb, minlength=nb).astype(np.float64)
        hs = B.suaviza(h, sig_b)
        for c in B.picos_regioes(hs, hs.mean() + 0.35 * (hs.max() - hs.mean())):
            acum += np.roll(h, nb // 2 - c)[nb // 2 - W: nb // 2 + W + 1]
            nseg += 1
    passo_n = bw * cosang
    perfil = acum / max(nseg, 1) * (P / (nsub * bw))          # rho / rho0
    xn = np.arange(-W, W + 1) * passo_n
    ps = np.convolve(perfil, np.ones(3) / 3, mode='same')
    imax = W - 3 + int(np.argmax(ps[W - 3:W + 4]))
    rmax, rmin = ps[imax], ps[2:-2].min()
    meia = 0.5 * (rmax + rmin)
    fr_ = np.where(ps[imax:] < meia)[0]
    ca_ = np.where(ps[:imax + 1][::-1] < meia)[0]
    esp = P * cosang / max(int(r['nbandas_moda']), 1)          # espacamento medio ao longo da normal
    r.update(perfil_n=perfil, xn_n=xn, dn_real=passo_n, pico_n=float(rmax), gas_n=float(rmin),
             frente_n=float(fr_[0] * passo_n) if len(fr_) else float('nan'),
             cauda_n=float(ca_[0] * passo_n) if len(ca_) else float('nan'),
             excesso=float(np.clip(perfil - rmin, 0, None).sum() * passo_n),
             alavanca=float((1.0 - rmin) * esp), espacamento=float(esp), phi_serie=phi)


# ------------------------------------------------------------------ figuras
def rotulo(r):
    return '%s — %s, bando a %.0f°, ⟨Φ⟩ = %.4f' % (NOME[r['fonte']], B.bandas_txt(r), r['dir_media'], r['phi'])


def fig_configuracoes(R):
    fig, axs = plt.subplots(3, 1, figsize=(13.5, 6.9), constrained_layout=True)
    sc = None
    for ax, f in zip(axs, ORDEM):
        r = R.get(f)
        if r is None:
            ax.axis('off')
            continue
        x, y, th = r['ult']
        psi = np.angle(np.exp(1j * (th - float(r['dir_ult']))))
        sc = ax.scatter(x, y, c=psi, s=0.8, cmap='twilight', vmin=-np.pi, vmax=np.pi, linewidths=0,
                        rasterized=True)
        ax.set_xlim(0, r['Lx']); ax.set_ylim(0, r['Ly']); ax.set_aspect('equal')
        ax.set_yticks([0, 64, 128]); B.moldura(ax)
        ax.set_title(rotulo(r))
    axs[-1].set_xlabel('x')
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, fraction=0.012, pad=0.01, aspect=45)
        cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(['−π', '0', 'π'])
        cb.set_label(r'$\theta - \bar\theta$  (escuro = alinhado com o bando)')
        cb.outline.set_edgecolor(B.EIXO)
    fig.suptitle('Vetorial (η = 0.55), 1024 × 128, ρ = 1, início alinhado: configuração no fim da produção '
                 '(t = 130 000) — 20 000 das 131 072 partículas', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'configuracoes_%s.png' % SUF), dpi=200)
    plt.close(fig)


def fig_densidade(R):
    mapas = {f: B.mapa_densidade(r) for f, r in R.items()}
    vmax = np.percentile(np.concatenate([m.ravel() for m in mapas.values()]), 99.7)
    fig, axs = plt.subplots(3, 1, figsize=(13.5, 6.9), constrained_layout=True)
    im = None
    for ax, f in zip(axs, ORDEM):
        r = R.get(f)
        if r is None:
            ax.axis('off')
            continue
        im = ax.imshow(mapas[f], origin='lower', extent=[0, r['Lx'], 0, r['Ly']], cmap=B.SEQ, vmin=0,
                       vmax=vmax, interpolation='bilinear')
        ax.set_yticks([0, 64, 128]); B.moldura(ax)
        ax.set_title(rotulo(r))
    axs[-1].set_xlabel('x')
    if im is not None:
        cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01, aspect=45)
        cb.set_label(r'$\rho/\rho_0$'); cb.outline.set_edgecolor(B.EIXO)
    fig.suptitle('Densidade local no fim da produção (caixas de 4 × 4, suavizada) — mesma escala de cor '
                 'nas três fontes', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'densidade_%s.png' % SUF), dpi=200)
    plt.close(fig)


def fig_kimografos(R):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9), constrained_layout=True)
    vmax = np.percentile(np.concatenate([r['K'].ravel() for r in R.values()]), 99.5)
    im = None
    for j, (ax, f) in enumerate(zip(axs, ORDEM)):
        r = R.get(f)
        if r is None:
            ax.axis('off')
            continue
        im = ax.imshow(r['K'], origin='lower', aspect='auto', extent=[0, r['P'], 0, r['T'] * r['dt']],
                       cmap=B.SEQ, vmin=0, vmax=vmax, interpolation='nearest')
        B.moldura(ax)
        ax.set_title('%s\n%d bandas · c = %.3f por passo (ao longo da normal)' % (
            NOME[f], r['nbandas_moda'], r['c_normal']))
        ax.set_xlabel("x' − c t  (referencial das bandas%s)" % (', desinclinadas' if r['n'] else ''))
        if j == 0:
            ax.set_ylabel('passos de produção')
    if im is not None:
        cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
        cb.set_label(r'$\rho/\rho_0$'); cb.outline.set_edgecolor(B.EIXO)
    fig.suptitle('Kimógrafos no referencial que anda com as bandas: faixas verticais = bandas estáveis '
                 'durante os 30 000 passos de produção', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'kimografos_%s.png' % SUF), dpi=180)
    plt.close(fig)


def fig_phi(R):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 4.4), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1.6, 1]))
    for f in ORDEM:
        r = R.get(f)
        if r is not None:
            ax.plot(r['tl'], r['pl'], color=COR[f], lw=1.6, label='%s: ⟨Φ⟩ = %.4f' % (NOME[f], r['phi']))
    r0 = next(iter(R.values()))
    ax.axvline(r0['ntrans'], color=B.MUDO, lw=1, ls=(0, (4, 3)))
    ax.text(r0['ntrans'] * 0.93, 0.03, 'início da\nprodução', transform=ax.get_xaxis_transform(),
            ha='right', va='bottom', color=B.TINTA2, fontsize=8.5)
    ax.set_xscale('log'); ax.set_xlim(1, r0['tl'][-1] * 1.05)
    ax.grid(True, axis='y')
    ax.set_xlabel('passo (desde o início alinhado)'); ax.set_ylabel('Φ')
    ax.set_title('Φ(t), médias em janelas logarítmicas'); ax.legend(loc='upper right')
    todos = np.concatenate([r['phi_serie'] for r in R.values()])
    bordas = np.linspace(todos.min(), todos.max(), 61)
    centros = 0.5 * (bordas[1:] + bordas[:-1])
    for f in ORDEM:
        r = R.get(f)
        if r is None:
            continue
        h, _ = np.histogram(r['phi_serie'], bordas, density=True)
        bx.plot(centros, h, color=COR[f], lw=1.6, drawstyle='steps-mid', label=CURTO[f])
        bx.axvline(r['phi'], color=COR[f], lw=1, ls=(0, (2, 2)))
    bx.set_yticks([]); bx.spines['left'].set_visible(False)
    bx.set_xlabel('Φ'); bx.set_title('P(Φ) na produção (30 000 passos; tracejado = média)')
    bx.legend(loc='upper left')
    fig.suptitle('Parâmetro de ordem — vetorial (η = 0.55), 1024 × 128, ρ = 1', x=0.01, ha='left',
                 fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'phi_%s.png' % SUF), dpi=180)
    plt.close(fig)


def fig_perfis(R):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    for f in ORDEM:
        r = R.get(f)
        if r is None:
            continue
        ax.plot(r['xn_n'], r['perfil_n'], color=COR[f], lw=1.6, label='%s: %d bandas, esp. %.0f, pico %.2f, gás %.3f'
                % (CURTO[f], r['nbandas_moda'], r['espacamento'], r['pico_n'], r['gas_n']))
        z = (r['perfil_n'] - r['gas_n']) / (r['pico_n'] - r['gas_n'])
        bx.plot(r['xn_n'], z, color=COR[f], lw=1.6, label='%s: meia-largura frente %.1f / cauda %.1f'
                % (CURTO[f], r['frente_n'], r['cauda_n']))
    topo = max(r['perfil_n'].max() for r in R.values())
    ax.set_ylim(0, 1.35 * topo); ax.axhline(1, color=B.EIXO, lw=0.8)
    ax.set_ylabel(r'$\rho/\rho_0$'); ax.set_title('Perfil médio de uma banda')
    bx.set_ylim(-0.1, 1.45); bx.axhline(0, color=B.EIXO, lw=0.8); bx.axhline(0.5, color=B.EIXO, lw=0.8, ls=(0, (3, 3)))
    bx.set_ylabel(r'$(\rho - \rho_{gás}) / (\rho_{pico} - \rho_{gás})$'); bx.set_title('Mesma curva normalizada: forma da banda')
    lim = min(r['xn_n'][-1] for r in R.values())
    for e in (ax, bx):
        e.set_xlim(-lim, lim); e.grid(True, axis='y'); e.legend(loc='upper left')
        e.set_xlabel('distância ao pico ao longo da normal  (movimento →)')
    fig.suptitle('Perfil médio de uma banda (alinhado no pico de cada banda, 300 quadros; bins de %.1f ao longo '
                 'da normal nas três fontes)' % A.dn, x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'perfis_%s.png' % SUF), dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(SAIDA, exist_ok=True)
    R = carregar()
    if not R:
        raise SystemExit('nenhuma corrida de ruido %s em %s' % (A.ruido, A.pastas))
    for r in R.values():
        perfil_normal(r, A.dn)
    cols = ['arq', 'fonte', 'phi', 'chi', 'binder', 'm', 'n', 'ang_normal', 'dir_media', 'nbandas_moda',
            'nbandas_hist', 'espacamento', 'c_normal', 'coerencia', 'gas_n', 'pico_n', 'frente_n', 'cauda_n',
            'excesso', 'alavanca', 'rho_pico', 'rho_fundo', 'meia_frente', 'meia_cauda']
    fontes = [f for f in ORDEM if f in R]
    with open(os.path.join(SAIDA, 'resumo1024_%s.tsv' % SUF), 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for f in fontes:
            fh.write('\t'.join(('%.4g' % R[f][c]) if isinstance(R[f][c], float) else str(R[f][c])
                               for c in cols) + '\n')
    print('\n%-14s' % '' + ''.join('%14s' % CURTO[f] for f in fontes))
    for c in cols[2:]:
        print('%-14s' % c + ''.join(('%14.4g' % R[f][c]) if isinstance(R[f][c], float) else '%14s' % R[f][c]
                                    for f in fontes))
    for fn in (fig_configuracoes, fig_densidade, fig_kimografos, fig_phi, fig_perfis):
        fn(R)
        print('figura:', fn.__name__, flush=True)
    print('em', SAIDA)


if __name__ == '__main__':
    main()
