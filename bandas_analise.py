#!/usr/bin/env python3
"""
Bandas na caixa alongada (saida do rollout.py com --quadros): rede x oraculo.

  python3 bandas_analise.py [pasta]          (padrao: bandas)

Para cada .npz da pasta:
  - modo dominante da densidade (m, n): m bandas ao longo de x; n != 0 = bandas obliquas;
  - comprimento de onda lambda (ao longo da normal) e angulo da normal;
  - numero de bandas por quadro (regioes acima do limiar no perfil suavizado);
  - velocidade das bandas pela fase do modo dominante. Com quadros a cada 100 passos o
    kimografo no referencial do laboratorio tem aliasing sempre que a banda anda mais
    que meio comprimento de onda por quadro (vetorial: ~50 contra lambda/2 ~ 35), por isso
    o kimografo e mostrado no referencial que anda com as bandas;
  - perfil medio de uma banda (alinhado no pico, movimento para a direita): densidade de
    pico e de fundo, meia-largura na frente e na cauda.

Grava em <pasta>/figuras/: configuracoes.png, densidade.png, kimografos.png, phi_t.png,
perfis.png e resumo.tsv (cache dos calculos em figuras/.cache). So precisa de numpy e matplotlib.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PASTA = sys.argv[1] if len(sys.argv) > 1 else 'bandas'
SAIDA = os.path.join(PASTA, 'figuras')
CACHE = os.path.join(SAIDA, '.cache')

# ------------------------------------------------------------------ estilo (paleta validada)
COR = {'oraculo': '#2a78d6', 'rede': '#eb6834'}
NOME = {'oraculo': 'oráculo (Vicsek exato)', 'rede': 'rede (M4)'}
RUIDO = {'vetorial': 'Vetorial (η = 0.55)', 'angular': 'Angular (η = 0.40)'}
TINTA, TINTA2, MUDO, GRADE, EIXO, FUNDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('azul', [FUNDO, '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                                                 '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({
    'figure.facecolor': FUNDO, 'axes.facecolor': FUNDO, 'savefig.facecolor': FUNDO,
    'axes.edgecolor': EIXO, 'axes.labelcolor': TINTA2, 'axes.titlecolor': TINTA,
    'xtick.color': EIXO, 'ytick.color': EIXO, 'xtick.labelcolor': TINTA2, 'ytick.labelcolor': TINTA2,
    'text.color': TINTA, 'grid.color': GRADE, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 10.5, 'axes.titlelocation': 'left', 'legend.frameon': False,
    'legend.fontsize': 9, 'mathtext.default': 'regular'})
ORDEM = [('vetorial', 'oraculo'), ('vetorial', 'rede'), ('angular', 'oraculo'), ('angular', 'rede')]


# ------------------------------------------------------------------ utilidades
def suaviza(r, sigma_bins):
    """convolucao gaussiana periodica (1D)"""
    k = np.fft.fftfreq(len(r))
    return np.real(np.fft.ifft(np.fft.fft(r) * np.exp(-2 * (np.pi * k * sigma_bins) ** 2)))


def suaviza2(H, sigma_bins):
    ky, kx = np.fft.fftfreq(H.shape[0]), np.fft.fftfreq(H.shape[1])
    g = np.exp(-2 * (np.pi * sigma_bins) ** 2 * (ky[:, None] ** 2 + kx[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(H) * g))


def modo_dominante(q, Lx, Ly, b=8.0):
    """(m, n) do maior pico do espectro de potencia da densidade, somado sobre os quadros"""
    nx, ny = int(round(Lx / b)), int(round(Ly / b))
    P = np.zeros((ny, nx))
    for fr in q:
        H = np.histogram2d(fr[1], fr[0], bins=[ny, nx], range=[[0, Ly], [0, Lx]])[0]
        P += np.abs(np.fft.fft2(H - H.mean())) ** 2
    P[0, 0] = 0.0
    iy, ix = np.unravel_index(np.argmax(P), P.shape)
    m = ix if ix <= nx // 2 else ix - nx
    n = iy if iy <= ny // 2 else iy - ny
    if m < 0 or (m == 0 and n < 0):
        m, n = -m, -n
    return int(m), int(n)


def picos_regioes(hs, thr):
    """um pico (maximo) por regiao contigua acima do limiar, vetor periodico"""
    acima = hs > thr
    if not acima.any():
        return []
    if acima.all():
        return [int(np.argmax(hs))]
    k0 = int(np.argmin(acima))
    a, v = np.roll(acima, -k0), np.roll(hs, -k0)
    d = np.diff(np.r_[0, a.astype(np.int8), 0])
    ini, fim = np.where(d == 1)[0], np.where(d == -1)[0]
    return [int((i + np.argmax(v[i:j]) + k0) % len(hs)) for i, j in zip(ini, fim)]


# ------------------------------------------------------------------ analise de uma corrida
def analisar(arq):
    cache = os.path.join(CACHE, os.path.basename(arq))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(arq):
        c = np.load(cache)
        r = json.loads(str(c['meta']))
        r.update({k: c[k] for k in c.files if k != 'meta'})
        return r
    d = np.load(arq)
    p = json.loads(str(d['params']))
    q = d['quadros']                               # (T, 3, nsub): x, y, theta a cada p['quadros'] passos
    T, _, nsub = q.shape
    Lx, Ly = float(p['L']), float(p['Y'] or p['L'])
    N, dt = int(p['N']), int(p['quadros'])
    r = dict(arq=os.path.basename(arq), ruido=p['ruido'], eta=p['eta'], robs=p['robs'], N=N, Lx=Lx, Ly=Ly,
             dt=dt, T=T, nsub=nsub, ntrans=int(len(d['phi_transiente'])),
             fonte='oraculo' if p['origem'].startswith('oraculo') else 'rede')

    phi = d['phi'].astype(np.float64)
    m1, m2, m4 = phi.mean(), (phi ** 2).mean(), (phi ** 4).mean()
    r.update(phi=float(m1), chi=float(N * (m2 - m1 * m1)), binder=float(1 - m4 / (3 * m2 * m2)))
    phi_all = np.concatenate([d['phi_transiente'], d['phi']]).astype(np.float64)
    bordas = np.unique(np.round(np.logspace(0, np.log10(len(phi_all) + 1), 320)).astype(int))
    tl = np.sqrt(bordas[:-1] * bordas[1:].astype(float))
    pl = np.array([phi_all[a - 1:b - 1].mean() for a, b in zip(bordas[:-1], bordas[1:])])

    th = q[:, 2, :]
    th_dir = np.arctan2(np.sin(th).mean(1), np.cos(th).mean(1)).astype(np.float64)

    m, n = modo_dominante(q, Lx, Ly)
    if m > 0:
        P, cnt = Lx, m
        S = np.mod(q[:, 0, :] + q[:, 1, :] * np.float32(n * Lx / (m * Ly)), np.float32(Lx))
    else:                                          # bandas "deitadas": o padrao varia so em y
        P, cnt = Ly, n
        S = q[:, 1, :].copy()
    kx, ky = m / Lx, n / Ly
    lam = 1.0 / math.hypot(kx, ky)
    alpha = math.atan2(ky, kx)                     # normal das bandas (fase crescente)
    cosang = lam * cnt / P                         # deslocamento em S -> deslocamento ao longo da normal
    sgn = 1.0 if np.mean(np.cos(th_dir - alpha)) >= 0 else -1.0
    if sgn < 0:                                    # sempre com as bandas andando para +S
        S = np.mod(np.float32(P) - S, np.float32(P))

    # velocidade pela fase do modo dominante (valida enquanto a banda anda < 1 periodo por quadro)
    ph = (2 * np.pi * cnt / P) * S
    F = np.cos(ph).mean(1) + 1j * np.sin(ph).mean(1)
    z = F[1:] * np.conj(F[:-1])
    z = z / np.abs(z)
    zm = z.mean()
    adv = np.mod(np.angle(zm), 2 * np.pi) / (2 * np.pi) * P / cnt
    c_s = adv / dt
    r.update(m=m, n=n, lam=lam, ang_normal=math.degrees(alpha), cosang=cosang, P=P, cnt=cnt,
             c_normal=c_s * cosang, c_s=c_s, coerencia=float(abs(zm)),
             dir_media=float(math.degrees(np.angle(np.exp(1j * th_dir).mean()))))

    # perfis por quadro: numero de bandas e perfil medio alinhado no pico
    nb = 512
    bw = P / nb
    sig_b = min(20.0, max(3.0, 0.05 * P / cnt)) / bw
    W = min(int(round(0.5 * (P / cnt) / bw)), nb // 2 - 1)
    acum, nseg, npic = np.zeros(2 * W + 1), 0, []
    for f in range(T):
        h = np.bincount((S[f] / bw).astype(np.int64) % nb, minlength=nb).astype(np.float64)
        hs = suaviza(h, sig_b)
        pk = picos_regioes(hs, hs.mean() + 0.35 * (hs.max() - hs.mean()))
        npic.append(len(pk))
        for c in pk:
            acum += np.roll(h, nb // 2 - c)[nb // 2 - W: nb // 2 + W + 1]
            nseg += 1
    perfil = acum / max(nseg, 1) * (P / (nsub * bw))          # rho / rho0
    xn = np.arange(-W, W + 1) * bw * cosang
    ps = np.convolve(perfil, np.ones(3) / 3, mode='same')
    imax = W - 3 + int(np.argmax(ps[W - 3:W + 4]))
    rmax, rmin = ps[imax], ps[2:-2].min()
    meia = 0.5 * (rmax + rmin)
    fr_ = np.where(ps[imax:] < meia)[0]
    ca_ = np.where(ps[:imax + 1][::-1] < meia)[0]
    npic = np.array(npic)
    vals, cont = np.unique(npic, return_counts=True)
    r.update(rho_pico=float(rmax), rho_fundo=float(rmin),
             meia_frente=float(fr_[0] * bw * cosang) if len(fr_) else float('nan'),
             meia_cauda=float(ca_[0] * bw * cosang) if len(ca_) else float('nan'),
             nbandas_media=float(npic.mean()), nbandas_moda=int(vals[np.argmax(cont)]),
             nbandas_hist=' '.join('%d:%d' % (v, c) for v, c in zip(vals, cont)))

    # kimografo no referencial das bandas
    nbk = 256
    bwk = P / nbk
    K = np.empty((T, nbk), np.float32)
    for f in range(T):
        s = np.mod(S[f] - c_s * dt * (f + 1), P)
        K[f] = np.bincount((s / bwk).astype(np.int64) % nbk, minlength=nbk) * (P / (nsub * bwk))

    r = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}   # escalares numpy -> Python
    arrays = dict(perfil=perfil, xn=xn, K=K, npic=npic, tl=tl, pl=pl, th_dir=th_dir,
                  ult=q[-1].copy(), dir_ult=np.array(th_dir[-1]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(cache, meta=np.array(json.dumps(r)), **arrays)
    r.update(arrays)
    return r


# ------------------------------------------------------------------ figuras
def bandas_txt(r):
    nb = int(r['nbandas_moda'])
    esp = r['P'] * r['cosang'] / max(nb, 1)          # espacamento medio ao longo da normal
    tipo = ('banda' if nb == 1 else 'bandas') + (' oblíquas (normal a %.0f°)' % r['ang_normal'] if r['n'] else '')
    return '%d %s, espaçamento %.0f' % (nb, tipo, esp)


def rotulo(r):
    return '%s · %s — %s, ⟨Φ⟩ = %.3f' % (RUIDO[r['ruido']].split(' ')[0], NOME[r['fonte']], bandas_txt(r), r['phi'])


def moldura(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(EIXO)


def fig_configuracoes(R):
    fig, axs = plt.subplots(4, 1, figsize=(13.5, 8.8), constrained_layout=True)
    sc = None
    for ax, chave in zip(axs, ORDEM):
        r = R.get(chave)
        if r is None:
            ax.axis('off')
            continue
        x, y, th = r['ult']
        psi = np.angle(np.exp(1j * (th - float(r['dir_ult']))))
        sc = ax.scatter(x, y, c=psi, s=0.8, cmap='twilight', vmin=-np.pi, vmax=np.pi, linewidths=0,
                        rasterized=True)
        ax.set_xlim(0, r['Lx']); ax.set_ylim(0, r['Ly']); ax.set_aspect('equal')
        ax.set_yticks([0, 64, 128]); moldura(ax)
        ax.set_title(rotulo(r))
    axs[-1].set_xlabel('x')
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, fraction=0.012, pad=0.01)
        cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(['−π', '0', 'π'])
        cb.set_label(r'$\theta - \bar\theta$  (escuro = alinhado com o bando)')
        cb.outline.set_edgecolor(EIXO)
    fig.suptitle('Configurações no fim da produção (t = 130 000), caixa 1024 × 128, ρ = 1 — '
                 '20 000 das 131 072 partículas', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'configuracoes.png'), dpi=200)
    plt.close(fig)


def mapa_densidade(r, b=4.0, sig=1.2):
    x, y, _ = r['ult']
    nx, ny = int(round(r['Lx'] / b)), int(round(r['Ly'] / b))
    H = np.histogram2d(y, x, bins=[ny, nx], range=[[0, r['Ly']], [0, r['Lx']]])[0]
    return suaviza2(H, sig) * (r['Lx'] * r['Ly']) / (len(x) * b * b)


def fig_densidade(R):
    mapas = {k: mapa_densidade(r) for k, r in R.items()}
    vmax = {ru: np.percentile(np.concatenate([mapas[k].ravel() for k in mapas if k[0] == ru]), 99.7)
            for ru in {k[0] for k in mapas}}
    fig, axs = plt.subplots(4, 1, figsize=(13.5, 8.8), constrained_layout=True)
    ims = {}
    for ax, chave in zip(axs, ORDEM):
        r = R.get(chave)
        if r is None:
            ax.axis('off')
            continue
        ims[chave] = ax.imshow(mapas[chave], origin='lower', extent=[0, r['Lx'], 0, r['Ly']], cmap=SEQ,
                               vmin=0, vmax=vmax[chave[0]], interpolation='bilinear')
        ax.set_yticks([0, 64, 128]); moldura(ax)
        ax.set_title(rotulo(r))
    axs[-1].set_xlabel('x')
    for ru, par in (('vetorial', axs[:2]), ('angular', axs[2:])):
        k = (ru, 'oraculo') if (ru, 'oraculo') in ims else (ru, 'rede')
        if k in ims:
            cb = fig.colorbar(ims[k], ax=par, fraction=0.012, pad=0.01)
            cb.set_label(r'$\rho/\rho_0$'); cb.outline.set_edgecolor(EIXO)
    fig.suptitle('Densidade local no fim da produção (caixas de 4 × 4, suavizada) — mesma escala de cor '
                 'para rede e oráculo em cada ruído', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'densidade.png'), dpi=200)
    plt.close(fig)


def fig_kimografos(R):
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    for i, ru in enumerate(('vetorial', 'angular')):
        Ks = [R[(ru, fo)]['K'] for fo in ('oraculo', 'rede') if (ru, fo) in R]
        if not Ks:
            continue
        vmax = np.percentile(np.concatenate([k.ravel() for k in Ks]), 99.5)
        im = None
        for j, fo in enumerate(('oraculo', 'rede')):
            ax = axs[i, j]
            r = R.get((ru, fo))
            if r is None:
                ax.axis('off')
                continue
            tmax = r['T'] * r['dt']
            im = ax.imshow(r['K'], origin='lower', aspect='auto', extent=[0, r['P'], 0, tmax], cmap=SEQ,
                           vmin=0, vmax=vmax, interpolation='nearest')
            moldura(ax)
            ax.set_title('%s · %s — c = %.3f por passo (ao longo da normal)' % (
                RUIDO[ru].split(' ')[0], NOME[fo], r['c_normal']))
            ax.set_xlabel("x' − c t  (referencial das bandas%s)" % (', bandas desinclinadas' if r['n'] else ''))
            if j == 0:
                ax.set_ylabel('passos de produção')
        cb = fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01)
        cb.set_label(r'$\rho/\rho_0$'); cb.outline.set_edgecolor(EIXO)
    fig.suptitle('Kimógrafos: densidade ao longo da caixa × tempo, no referencial que anda com as bandas '
                 '(faixas verticais = bandas estáveis)', x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'kimografos.png'), dpi=180)
    plt.close(fig)


def fig_phi(R):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.4), constrained_layout=True)
    for ax, ru in zip(axs, ('vetorial', 'angular')):
        for fo in ('oraculo', 'rede'):
            r = R.get((ru, fo))
            if r is None:
                continue
            ax.plot(r['tl'], r['pl'], color=COR[fo], lw=1.6, label='%s: ⟨Φ⟩ = %.3f' % (NOME[fo], r['phi']))
        r = R.get((ru, 'oraculo')) or R.get((ru, 'rede'))
        ax.axvline(r['ntrans'], color=MUDO, lw=1, ls=(0, (4, 3)))
        ax.text(r['ntrans'] * 0.93, 0.02, 'início da\nprodução', transform=ax.get_xaxis_transform(),
                ha='right', va='bottom', color=TINTA2, fontsize=8.5)
        ax.set_xscale('log'); ax.set_xlim(1, r['tl'][-1] * 1.05)
        ax.grid(True, axis='y')
        ax.set_xlabel('passo (desde o início alinhado)'); ax.set_ylabel('Φ')
        ax.set_title(RUIDO[ru]); ax.legend(loc='upper right')
    fig.suptitle('Parâmetro de ordem desde o início alinhado (médias em janelas logarítmicas)',
                 x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'phi_t.png'), dpi=180)
    plt.close(fig)


def fig_perfis(R):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, ru in zip(axs, ('vetorial', 'angular')):
        for fo in ('oraculo', 'rede'):
            r = R.get((ru, fo))
            if r is None:
                continue
            ax.plot(r['xn'], r['perfil'], color=COR[fo], lw=1.6,
                    label='%s: %s, c = %.3f' % (NOME[fo], bandas_txt(r).replace(' (normal a %.0f°)' % r['ang_normal'], ''), r['c_normal']))
        topo = max(R[(ru, fo)]['perfil'].max() for fo in ('oraculo', 'rede') if (ru, fo) in R)
        ax.set_ylim(0, 1.4 * topo)
        ax.axhline(1, color=EIXO, lw=0.8)
        ax.grid(True, axis='y')
        ax.set_xlabel('distância ao pico, ao longo da normal  (movimento →)')
        ax.set_ylabel(r'$\rho/\rho_0$')
        ax.set_title(RUIDO[ru]); ax.legend(loc='upper left')
    fig.suptitle('Perfil médio de uma banda (alinhado no pico de cada banda, todos os quadros da produção)',
                 x=0.01, ha='left', fontsize=11.5)
    fig.savefig(os.path.join(SAIDA, 'perfis.png'), dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(SAIDA, exist_ok=True)
    R = {}
    for arq in sorted(glob.glob(os.path.join(PASTA, '*.npz'))):
        r = analisar(arq)
        R[(r['ruido'], r['fonte'])] = r
        print('ok', r['arq'], flush=True)
    cols = ['arq', 'ruido', 'fonte', 'phi', 'chi', 'binder', 'm', 'n', 'lam', 'ang_normal', 'dir_media',
            'nbandas_moda', 'nbandas_media', 'nbandas_hist', 'c_normal', 'coerencia', 'rho_pico', 'rho_fundo',
            'meia_frente', 'meia_cauda']
    with open(os.path.join(SAIDA, 'resumo.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for k in ORDEM:
            if k in R:
                f.write('\t'.join(('%.4g' % R[k][c]) if isinstance(R[k][c], float) else str(R[k][c])
                                  for c in cols) + '\n')
    print('\n%-9s %-8s %6s %4s %4s %6s %6s %9s %6s %6s %6s %6s %6s' % (
        'ruido', 'fonte', 'Phi', 'm', 'n', 'lam', 'ang', 'nb(moda)', 'c', 'coer', 'pico', 'fundo', 'fr/ca'))
    for k in ORDEM:
        if k in R:
            r = R[k]
            print('%-9s %-8s %6.3f %4d %4d %6.1f %6.1f %5d(%.2f) %6.3f %6.3f %6.2f %6.2f %4.0f/%.0f' % (
                r['ruido'], r['fonte'], r['phi'], r['m'], r['n'], r['lam'], r['ang_normal'], r['nbandas_moda'],
                r['nbandas_media'], r['c_normal'], r['coerencia'], r['rho_pico'], r['rho_fundo'],
                r['meia_frente'], r['meia_cauda']))
    for fn in (fig_configuracoes, fig_densidade, fig_kimografos, fig_phi, fig_perfis):
        fn(R)
        print('figura:', fn.__name__, flush=True)


if __name__ == '__main__':
    main()
