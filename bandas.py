#!/usr/bin/env python3
"""
Bloco 1: le os campos coarse-grained (cg_*.bin) e mostra onde ha bandas.

  cd ~/vicsek-gnn/cluster
  python3 bandas.py out/bloco1                  (aceita varios diretorios)

Gera:
  bandas_tabela.txt  uma linha por simulacao
  phi_rho.png        Phi x rho, um ponto por semente (circulo = inicio aleatorio, triangulo = alinhado)
  mapas_<ruido>.png  densidade no ultimo quadro (caixa 1024x128 inteira), uma faixa por rho
  kimo_<ruido>.png   perfil de densidade em x ao longo do tempo (bandas = listras inclinadas)

desv_x = desvio padrao relativo do perfil de densidade ao longo de x (media em y),
ref    = o mesmo para particulas espalhadas ao acaso (Poisson).
Gas e liquido homogeneos: desv_x perto de ref. Bandas: desv_x muitas vezes maior.
O numero so ordena; o veredito vem dos mapas e kimografos.
Compativel com Python 3.6 / numpy 1.16 / matplotlib 2.0 (Turing).
"""
import sys, os, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BURN = 0.5          # descarta a primeira metade dos quadros de producao
NOME = {'0': 'angular (n0)', '1G': 'vetorial (-G 1)', '1o': 'vetorial (-G 0)'}


def ler_cg(f):
    with open(f, 'rb') as fh:
        nx, ny = np.fromfile(fh, dtype='<i4', count=2)
        d = np.fromfile(fh, dtype='<f4')
    nq = d.size // (nx * ny)
    return d[:nq * nx * ny].reshape(nq, ny, nx)


runs = []
for d in (sys.argv[1:] or ['out/bloco1']):
    arqs = sorted(glob.glob(os.path.join(d, 'cg_den_*.bin')))
    print('%s: %d simulacoes com campos CG' % (d, len(arqs)))
    for fden in arqs:
        tag = os.path.basename(fden)[len('cg_den_'):-len('.bin')]
        try:
            with open(os.path.join(d, 'meta_' + tag + '.json')) as fh:
                m = json.load(fh)
        except (IOError, ValueError):
            print('  sem meta valido, pulando', tag)
            continue
        if m.get('status') != 'done':
            print('  incompleta (status=%s), pulando %s' % (m.get('status'), tag))
            continue
        den = ler_cg(fden)
        sop = ler_cg(os.path.join(d, 'cg_sop_' + tag + '.bin'))
        vop = ler_cg(os.path.join(d, 'cg_vop_' + tag + '.bin'))
        nq = min(len(den), len(sop), len(vop))
        den, sop, vop = den[:nq], sop[:nq], vop[:nq]
        ny, nx = den.shape[1:]
        s = m['lx'] // nx
        rho = m['rho']
        # Phi global reconstruido das caixas: |sum n_caixa * ordem_local * direcao_local| / N
        w = den * (s * s) * sop
        phi_t = np.hypot((w * np.cos(vop)).sum(axis=(1, 2)),
                         (w * np.sin(vop)).sum(axis=(1, 2))) / m['N']
        i0 = int(nq * BURN)
        px = den[i0:].mean(axis=1)                  # perfil em x, um por quadro
        py = den[i0:].mean(axis=2)                  # perfil em y
        runs.append(dict(
            ruido=tag.split('_')[0][1:], rho=rho, eta=m['eta'], init=m['init'], seed=m['seed'],
            phi=phi_t[i0:].mean(),
            dx=(px.std(axis=1) / px.mean(axis=1)).mean(),
            dy=(py.std(axis=1) / py.mean(axis=1)).mean(),
            ref=1.0 / np.sqrt(rho * s * m['ly']),
            last=den[-1] / rho, kimo=den.mean(axis=1) / rho))

if not runs:
    sys.exit('nenhuma simulacao lida')

runs.sort(key=lambda r: (r['ruido'], r['rho'], r['init'], r['seed']))
ruidos = sorted(set(r['ruido'] for r in runs))

# ---- tabela
with open('bandas_tabela.txt', 'w') as fh:
    fh.write('# ruido rho init semente Phi desv_x desv_y ref_x   (burn=%.2f)\n' % BURN)
    for r in runs:
        fh.write('%s %.4f %d %d %.4f %.4f %.4f %.4f\n' % (
            r['ruido'], r['rho'], r['init'], r['seed'], r['phi'], r['dx'], r['dy'], r['ref']))

print('\nresumo (media das sementes; entre parenteses min-max de Phi)')
print('%-5s %6s %4s  %-22s %7s %7s %7s' % ('ruido', 'rho', 'ini', 'Phi', 'desv_x', 'desv_y', 'ref_x'))
chaves = sorted(set((r['ruido'], r['rho'], r['init']) for r in runs))
for k in chaves:
    g = [r for r in runs if (r['ruido'], r['rho'], r['init']) == k]
    p = [r['phi'] for r in g]
    print('%-5s %6.3f %4d  %.3f (%.3f-%.3f)    %7.3f %7.3f %7.3f' % (
        k[0], k[1], k[2], np.mean(p), min(p), max(p),
        np.mean([r['dx'] for r in g]), np.mean([r['dy'] for r in g]), g[0]['ref']))

# ---- Phi x rho, um ponto por semente
fig, axs = plt.subplots(1, len(ruidos), figsize=(5.5 * len(ruidos), 4), squeeze=False)
for ax, ru in zip(axs[0], ruidos):
    for ini, mk, fac in ((0, 'o', 0.96), (1, '^', 1.04)):
        g = [r for r in runs if r['ruido'] == ru and r['init'] == ini]
        if not g:
            continue
        ax.plot([r['rho'] * fac for r in g], [r['phi'] for r in g], mk,
                mfc='none' if ini == 0 else 'C1', mec='C0' if ini == 0 else 'C1',
                label='inicio %s' % ('aleatorio' if ini == 0 else 'alinhado'))
    rhos = sorted(set(r['rho'] for r in runs if r['ruido'] == ru))
    ax.set_xscale('log')
    ax.set_xticks(rhos)
    ax.set_xticklabels(['%g' % x for x in rhos], rotation=60, fontsize=8)
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel('rho')
    ax.set_ylabel('<Phi> (2a metade)')
    eta = [r['eta'] for r in runs if r['ruido'] == ru][0]
    ax.set_title('%s, eta=%.2f' % (NOME.get(ru, ru), eta))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig('phi_rho.png', dpi=120)
plt.close(fig)


# ---- mapas e kimografos: uma simulacao (a de menor semente) por (rho, init)
def painel(ru, chave, nome, alto, nota):
    esc = {}
    for r in runs:
        if r['ruido'] == ru:
            esc.setdefault((r['rho'], r['init']), r)
    rhos = sorted(set(k[0] for k in esc))
    inits = sorted(set(k[1] for k in esc))
    fig, axs = plt.subplots(len(rhos), len(inits), squeeze=False,
                            figsize=(7 * len(inits), alto * len(rhos) + 0.9))
    for i, rho in enumerate(rhos):
        for j, ini in enumerate(inits):
            ax = axs[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = esc.get((rho, ini))
            if r is None:
                continue
            ax.imshow(r[chave], origin='lower', aspect='auto', vmin=0, vmax=3,
                      cmap='viridis', interpolation='nearest')
            ax.text(0.005, 0.8, 'Phi=%.2f  s=%d' % (r['phi'], r['seed']), transform=ax.transAxes,
                    color='w', fontsize=7)
            if j == 0:
                ax.set_ylabel('rho=%g' % rho, rotation=0, ha='right', va='center', fontsize=9)
            if i == 0:
                ax.set_title('inicio %s' % ('aleatorio (-i 0)' if ini == 0 else 'alinhado (-i 1)'))
    fig.suptitle('%s -- %s' % (NOME.get(ru, ru), nota), y=1.0 + 0.3 / fig.get_figheight())
    fig.savefig(nome, dpi=100, bbox_inches='tight')
    plt.close(fig)


for ru in ruidos:
    painel(ru, 'last', 'mapas_%s.png' % ru, 0.85,
           'densidade/rho no ultimo quadro (escala 0 a 3)')
    painel(ru, 'kimo', 'kimo_%s.png' % ru, 1.3,
           'perfil em x ao longo do tempo (tempo para cima, 1 linha = 1000 passos)')

print('\nsalvos: bandas_tabela.txt, phi_rho.png, ' +
      ', '.join('mapas_%s.png, kimo_%s.png' % (r, r) for r in ruidos))
