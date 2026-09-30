# -*- coding: utf-8 -*-
"""Vídeo lado a lado: Vicsek exato × rede M4 treinada só com gás e líquido (nunca viu bandas).
Mesma condição inicial (bando alinhado), ruído vetorial η = 0,55, ρ = 0,5, caixa 192 × 48.

  python3 gerar_video.py sim      -> video_quadros.npz
  python3 gerar_video.py render   -> video/bandas_rede_x_exato.mp4 e img/video-poster.webp
"""
import math, os, subprocess, sys, time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
LX, LY, RHO, ETA, V0, ROBS = 192.0, 48.0, 0.5, 0.55, 0.5, 1.2
T, CADA, SEM = 3600, 6, 20260930
MODELO = os.path.join(BASE, '..', 'runs', 'vetorial_m4_hist_GL_e60')
NPZ = os.path.join(BASE, 'video_quadros.npz')


def simular():
    from m4np import M4, passo_rede, passo_oraculo, phi
    N = int(round(RHO * LX * LY))
    rng0 = np.random.default_rng(SEM)
    x0, y0 = rng0.random(N) * LX, rng0.random(N) * LY
    th0 = np.zeros(N)
    model = M4(MODELO)
    res = {}
    for fonte in ('oraculo', 'rede'):
        rng = np.random.default_rng(SEM + (1 if fonte == 'rede' else 2))
        x, y, th = x0.copy(), y0.copy(), th0.copy()
        qs, phis = [], []
        t0 = time.time()
        for t in range(1, T + 1):
            if fonte == 'rede':
                x, y, th = passo_rede(model, x, y, th, LX, LY, ROBS, V0, rng)
            else:
                x, y, th = passo_oraculo(x, y, th, LX, LY, ETA, V0, rng)
            phis.append(phi(th))
            if t % CADA == 0:
                qs.append(np.stack([x, y, th]).astype(np.float32))
            if t % 600 == 0:
                print(f'{fonte}: passo {t}/{T}  Φ={phis[-1]:.3f}  ({(time.time() - t0) / t * 1000:.1f} ms/passo)', flush=True)
        res[fonte] = (np.stack(qs), np.array(phis, np.float32))
    np.savez_compressed(NPZ, q_or=res['oraculo'][0], phi_or=res['oraculo'][1], q_rede=res['rede'][0], phi_rede=res['rede'][1],
                        params=np.array([LX, LY, RHO, ETA, V0, ROBS, T, CADA, SEM, N]))
    print('salvo', NPZ)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from PIL import Image
    for f in ('IBMPlexSans-Regular', 'IBMPlexSans-SemiBold', 'IBMPlexMono-Regular', 'IBMPlexMono-Medium'):
        arq = os.path.join(BASE, 'ttf', f + '.ttf')          # opcional: sem as fontes, usa DejaVu
        if os.path.exists(arq):
            font_manager.fontManager.addfont(arq)
    SANS, MONO = ['IBM Plex Sans', 'DejaVu Sans'], ['IBM Plex Mono', 'DejaVu Sans Mono']
    z = np.load(NPZ)
    q_or, q_rd, ph_or, ph_rd = z['q_or'], z['q_rede'], z['phi_or'], z['phi_rede']
    F = len(q_or)
    W, H, DPI = 1280, 800, 100
    FUNDO, TXT, TXT2, GRADE = '#0b0f14', '#e6ebf2', '#8b95a5', '#222a35'
    COR = {'or': (0.36, 0.64, 0.98), 'rd': (0.20, 0.82, 0.60)}         # azul e verde-água, claros para fundo escuro
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=FUNDO)
    PH = 270; PW = int(PH * LX / LY); X0 = (W - PW) // 2               # painéis 4:1, centralizados
    fx = lambda px: px / W
    ret = lambda topo, alt: [fx(X0), (H - topo - alt) / H, PW / W, alt / H]
    ax_or = fig.add_axes(ret(56, PH)); ax_rd = fig.add_axes(ret(382, PH)); ax_ph = fig.add_axes(ret(682, 68))
    for ax in (ax_or, ax_rd):
        ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.set_facecolor('#0f141b')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRADE)
    ax_ph.set_facecolor(FUNDO); ax_ph.set_xlim(0, len(ph_or)); ax_ph.set_ylim(0, 1)
    for sp in ('top', 'right'): ax_ph.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax_ph.spines[sp].set_color(GRADE)
    ax_ph.tick_params(colors=TXT2, labelsize=9.5, length=3)
    ax_ph.set_yticks([0, 0.5, 1]); ax_ph.set_yticklabels(['0', '0,5', '1'], fontfamily=MONO)
    xt = [0, 1000, 2000, 3000]; ax_ph.set_xticks(xt); ax_ph.set_xticklabels(['0', '1.000', '2.000', '3.000'], fontfamily=MONO)
    ty = lambda px: (H - px) / H
    fig.text(fx(X0), ty(670), 'Φ(t)', color=TXT2, fontsize=10.5, fontfamily=MONO)
    fig.text(fx(X0), ty(30), 'MODELO EXATO (VICSEK)', color=TXT, fontsize=14, fontfamily=MONO, fontweight='medium')
    fig.text(fx(X0), ty(48), 'regra verdadeira', color=TXT2, fontsize=11, fontfamily=SANS)
    fig.text(fx(X0), ty(356), 'REDE NEURAL QUE NUNCA VIU UMA BANDA', color=TXT, fontsize=14, fontfamily=MONO, fontweight='medium')
    fig.text(fx(X0), ty(374), 'M4 treinado só com gás e líquido: regra aprendida de exemplos, rodando sozinha', color=TXT2, fontsize=11, fontfamily=SANS)
    t_or = fig.text(fx(X0 + PW), ty(30), '', color=TXT, fontsize=14, fontfamily=MONO, ha='right')
    t_rd = fig.text(fx(X0 + PW), ty(356), '', color=TXT, fontsize=14, fontfamily=MONO, ha='right')
    t_passo = fig.text(fx(X0 + PW), ty(48), '', color=TXT2, fontsize=11, fontfamily=MONO, ha='right')
    fig.text(fx(X0), ty(788), f'Mesma condição inicial (bando alinhado) · {int(RHO * LX * LY):,} partículas · ruído vetorial η = 0,55 · ρ = 0,5 · caixa 192 × 48'.replace(',', '.').replace('η = 0.55', 'η = 0,55').replace('ρ = 0.5', 'ρ = 0,5'),
             color=TXT2, fontsize=10, fontfamily=SANS)
    lc_or = LineCollection([], linewidths=1.3, capstyle='round'); ax_or.add_collection(lc_or)
    lc_rd = LineCollection([], linewidths=1.3, capstyle='round'); ax_rd.add_collection(lc_rd)
    l_or, = ax_ph.plot([], [], color=COR['or'], lw=1.4); l_rd, = ax_ph.plot([], [], color=COR['rd'], lw=1.4)
    marca = ax_ph.axvline(0, color=TXT2, lw=0.8)
    meio = 0.45

    def segs(q, cor):
        x, y, th = q
        Th = math.atan2(np.sin(th).mean(), np.cos(th).mean())
        a = (np.cos(th - Th) + 1) / 2                                   # 1 = alinhado com o bando
        dx, dy = np.cos(th) * meio, np.sin(th) * meio
        s = np.stack([np.stack([x - dx, y - dy], 1), np.stack([x + dx, y + dy], 1)], 1)
        cores = np.zeros((len(x), 4)); cores[:, :3] = cor
        cores[:, 3] = 0.18 + 0.82 * a ** 2
        return s, cores

    fmt = lambda v: f'{v:.2f}'.replace('.', ',')
    if os.environ.get('QUADRO'):                                       # QUADRO=599: só um quadro, para conferir
        k = int(os.environ['QUADRO'])
        s, c = segs(q_or[k], COR['or']); lc_or.set_segments(s); lc_or.set_color(c)
        s, c = segs(q_rd[k], COR['rd']); lc_rd.set_segments(s); lc_rd.set_color(c)
        n = (k + 1) * CADA
        l_or.set_data(np.arange(n), ph_or[:n]); l_rd.set_data(np.arange(n), ph_rd[:n]); marca.set_xdata([n, n])
        t_or.set_text(f'Φ = {fmt(ph_or[n - 1])}'); t_rd.set_text(f'Φ = {fmt(ph_rd[n - 1])}'); t_passo.set_text(f'passo {n:,}'.replace(',', '.'))
        fig.savefig(os.path.join(BASE, 'teste_quadro.png'), dpi=DPI, facecolor=FUNDO); print('quadro de teste salvo'); return
    out = os.path.join(BASE, 'video'); os.makedirs(out, exist_ok=True)
    mp4 = os.path.join(out, 'bandas_rede_x_exato.mp4')
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-preset', 'slow', '-tune', 'animation',
                           '-movflags', '+faststart', mp4], stdin=subprocess.PIPE)
    fmt = lambda v: f'{v:.2f}'.replace('.', ',')
    t0 = time.time()
    for k in range(F):
        s, c = segs(q_or[k], COR['or']); lc_or.set_segments(s); lc_or.set_color(c)
        s, c = segs(q_rd[k], COR['rd']); lc_rd.set_segments(s); lc_rd.set_color(c)
        n = (k + 1) * CADA
        l_or.set_data(np.arange(n), ph_or[:n]); l_rd.set_data(np.arange(n), ph_rd[:n]); marca.set_xdata([n, n])
        t_or.set_text(f'Φ = {fmt(ph_or[n - 1])}'); t_rd.set_text(f'Φ = {fmt(ph_rd[n - 1])}')
        t_passo.set_text(f'passo {n:,}'.replace(',', '.'))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        ff.stdin.write(buf.tobytes())
        if k == F - 1:
            Image.fromarray(buf).save(os.path.join(out, 'video-poster.webp'), 'WEBP', quality=88)
        if k % 100 == 0:
            print(f'quadro {k}/{F}  ({(time.time() - t0) / (k + 1):.2f} s/quadro)', flush=True)
    ff.stdin.close(); ff.wait()
    print('vídeo:', mp4, os.path.getsize(mp4) // 1024, 'KB')


if __name__ == '__main__':
    {'sim': simular, 'render': render}[sys.argv[1]]()
