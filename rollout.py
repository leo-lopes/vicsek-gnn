#!/usr/bin/env python3
"""
Malha fechada: a politica aprendida roda sozinha na GPU -- ou o Vicsek exato no mesmo codigo (controle).

  python3 rollout.py --modelo runs/angular_m4_hist_GBL_e60 --L 64 --rho 2
  python3 rollout.py --oraculo angular --eta 0.40 --L 64 --rho 2          (mesmas condicoes, regra exata)

Passo (igual ao FastVicsek): no estado t, cada particula ve os vizinhos com d < R_obs no seu
referencial e sorteia dtheta da distribuicao prevista; theta(t+1) = theta(t) + dtheta;
x(t+1) = x(t) + v0 (cos, sin) theta(t+1), com contorno periodico. Atualizacao sincrona.
Phi e medido depois de cada passo.

Saidas: rollouts/<nome>.npz (serie de Phi do transiente e da producao, parametros e, com
--quadros K, posicoes/angulos a cada K passos para animacao) e uma linha em rollouts/rollouts.tsv
com <Phi>, erro por blocos, chi = N(<Phi^2> - <Phi>^2) e Binder = 1 - <Phi^4>/(3<Phi^2>^2).

Compare sempre rede x oraculo NA MESMA caixa, densidade e numero de passos.
Use no mesmo diretorio do treino.py. Precisa de numpy e torch. Modelos suportados: m3, m4, m5.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import treino as T

TWO_PI = 2 * math.pi


def wrap(a):
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


# ------------------------------------------------------------------ vizinhos (listas de celulas)
_OX = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
_OY = [-1, -1, -1, 0, 0, 0, 1, 1, 1]


def vizinhos(x, y, Lx, Ly, R, orcamento=30_000_000):
    """Pares i != j com distancia periodica < R. Retorna i (crescente), j, dx, dy (dx = x_j - x_i)."""
    dev, N = x.device, len(x)
    ncx, ncy = max(3, int(Lx // R)), max(3, int(Ly // R))
    cx = torch.clamp((x * (ncx / Lx)).long(), 0, ncx - 1)
    cy = torch.clamp((y * (ncy / Ly)).long(), 0, ncy - 1)
    cell = cx + ncx * cy
    counts = torch.bincount(cell, minlength=ncx * ncy)
    order = torch.argsort(cell)
    cs = cell[order]
    rank = torch.arange(N, device=dev) - (torch.cumsum(counts, 0) - counts)[cs]
    mmax = int(counts.max())
    table = torch.full((ncx * ncy, mmax), -1, dtype=torch.long, device=dev)
    table[cs, rank] = order
    ox = torch.tensor(_OX, device=dev)
    oy = torch.tensor(_OY, device=dev)
    chunk = max(256, orcamento // (9 * mmax))
    I, J, DX, DY = [], [], [], []
    for a in range(0, N, chunk):
        idx = torch.arange(a, min(a + chunk, N), device=dev)
        nc = torch.remainder(cx[idx, None] + ox, ncx) + ncx * torch.remainder(cy[idx, None] + oy, ncy)
        cand = table[nc].reshape(len(idx), -1)
        ii = idx[:, None].expand_as(cand)
        v = (cand >= 0) & (cand != ii)
        ii, jj = ii[v], cand[v]
        dx = x[jj] - x[ii]
        dx = dx - Lx * torch.round(dx / Lx)
        dy = y[jj] - y[ii]
        dy = dy - Ly * torch.round(dy / Ly)
        m = dx * dx + dy * dy < R * R
        I.append(ii[m]); J.append(jj[m]); DX.append(dx[m]); DY.append(dy[m])
    return torch.cat(I), torch.cat(J), torch.cat(DX), torch.cat(DY)


# ------------------------------------------------------------------ politica neural
class Estrela:
    """Mesmo formato CSR do dataset (laco i->i primeiro), para reusar model.forward."""
    arestas = T.Dados.arestas

    def __init__(self, th, i, j, dx, dy, N):
        c, s = torch.cos(th[i]), torch.sin(th[i])
        dth = th[j] - th[i]
        nb = torch.stack([c * dx + s * dy, -s * dx + c * dy, torch.hypot(dx, dy),
                          torch.cos(dth), torch.sin(dth)], 1)
        k = torch.bincount(i, minlength=N)
        self.ptr = torch.zeros(N + 1, dtype=torch.long, device=th.device)
        self.ptr[1:] = torch.cumsum(k + 1, 0)
        self.feat = torch.zeros(int(self.ptr[-1]), 5, device=th.device)
        self.feat[self.ptr[:-1], 3] = 1.0
        kstart = torch.cumsum(k, 0) - k
        pos = self.ptr[i] + 1 + (torch.arange(len(i), device=th.device) - kstart[i])
        self.feat[pos] = nb


def amostrar_linear(p, w):
    """Amostra da densidade periodica linear por partes com valores p (N, K) nos centros dos bins.
    Retorna o angulo em [-pi, pi)."""
    K = p.shape[1]
    a, b = p, torch.roll(p, -1, dims=1)                      # segmento k: do centro k ao centro k+1
    k = torch.multinomial(a + b, 1)[:, 0]
    ak = a.gather(1, k[:, None])[:, 0]
    bk = b.gather(1, k[:, None])[:, 0]
    u = torch.rand(len(k), device=p.device)
    dif = bk - ak
    quase = dif.abs() < 1e-6 * (ak + bk + 1e-30)
    t = torch.where(quase, u, (-ak + torch.sqrt(torch.clamp(ak * ak + dif * u * (ak + bk), min=0))) /
                    torch.where(quase, torch.ones_like(dif), dif))
    return wrap(-math.pi + (k.float() + 0.5 + t.clamp(0, 1)) * w)


def amostrar(cab, out):
    if isinstance(cab, T.CabecaHist):
        h = out[:, 2:] + cab.prior
        p = torch.softmax(torch.cat([h.flip(1), h], 1), 1)
        return cab.mu(out) + amostrar_linear(p, cab.w)
    if isinstance(cab, T.CabecaHistLivre):
        return amostrar_linear(torch.softmax(out, 1), cab.w)
    k = F.softplus(out[:, 2]) + 1e-3
    return torch.distributions.VonMises(cab.mu(out), k).sample()


# ------------------------------------------------------------------ estatistica
def estat(phi, N, nblocos=10):
    phi = np.asarray(phi, dtype=np.float64)
    m, m2, m4 = phi.mean(), (phi ** 2).mean(), (phi ** 4).mean()
    blocos = np.array_split(phi, nblocos)
    err = np.std([b.mean() for b in blocos], ddof=1) / math.sqrt(nblocos)
    return dict(phi=float(m), erro=float(err), chi=float(N * (m2 - m * m)),
                binder=float(1 - m4 / (3 * m2 * m2)))


def main():
    ap = argparse.ArgumentParser(description='rollout em malha fechada (rede ou Vicsek exato)')
    fonte = ap.add_mutually_exclusive_group(required=True)
    fonte.add_argument('--modelo', help='pasta runs/<nome> de um m3, m4 ou m5')
    fonte.add_argument('--oraculo', choices=['angular', 'vetorial'], help='regra exata do Vicsek')
    ap.add_argument('--eta', type=float, default=None, help='so para o oraculo (o da rede e o do treino)')
    ap.add_argument('--L', type=int, default=64)
    ap.add_argument('--Y', type=int, default=None, help='lado em y (padrao: = L)')
    ap.add_argument('--rho', type=float, default=2.0)
    ap.add_argument('--v0', type=float, default=0.5)
    ap.add_argument('--init', choices=['aleatorio', 'alinhado'], default='aleatorio')
    ap.add_argument('--transiente', type=int, default=10000)
    ap.add_argument('--passos', type=int, default=40000, help='passos de producao')
    ap.add_argument('--semente', type=int, default=1)
    ap.add_argument('--quadros', type=int, default=0, help='guarda posicoes a cada K passos de producao')
    ap.add_argument('--max-quadro', type=int, default=20000, help='particulas guardadas por quadro')
    ap.add_argument('--robs', type=float, default=None, help='padrao: o do dataset do treino (2.0)')
    ap.add_argument('--nome', default=None)
    ap.add_argument('--saida', default='rollouts')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.semente)
    Lx, Ly = float(a.L), float(a.Y or a.L)
    N = int(round(a.rho * Lx * Ly))

    if a.modelo:
        with open(os.path.join(a.modelo, 'resultado.json')) as f:
            args = json.load(f)['args']
        if args['modelo'] not in ('m3', 'm4', 'm5'):
            raise SystemExit('rollout suporta m3, m4 e m5 (este e %s)' % args['modelo'])
        ruido, eta, robs = args['ruido'], None, 2.0
        try:
            meta = T.listar(args['dados'], ruido)[0][1]
            eta, robs = meta['eta'], meta['robs']
        except (SystemExit, OSError, KeyError, IndexError):
            print('aviso: dataset do treino nao encontrado; usando R_obs = 2.0')
        robs = a.robs or robs
        cab = {'hist': lambda: T.CabecaHist(args['bins']), 'hlivre': lambda: T.CabecaHistLivre(args['bins']),
               'vm': T.CabecaVM}[args['cabeca']]().to(dev)
        model = T.MODELOS[args['modelo']](args['oculto'], cab.nout).to(dev)
        model.load_state_dict(torch.load(os.path.join(a.modelo, 'modelo.pt'), map_location=dev))
        model.eval()
        origem = os.path.basename(os.path.normpath(a.modelo))
    else:
        if a.eta is None:
            raise SystemExit('--oraculo precisa de --eta')
        ruido, eta, robs = a.oraculo, a.eta, 1.0
        origem = 'oraculo_%s_e%.2f' % (ruido, eta)
    nome = a.nome or '%s_L%dx%d_r%g_%s_s%d' % (origem, Lx, Ly, a.rho, a.init, a.semente)
    os.makedirs(a.saida, exist_ok=True)
    print('%s  N=%d  (%s)' % (nome, N, dev))

    x = torch.rand(N, device=dev) * Lx
    y = torch.rand(N, device=dev) * Ly
    th = (torch.rand(N, device=dev) * TWO_PI - math.pi) if a.init == 'aleatorio' else torch.zeros(N, device=dev)

    @torch.no_grad()
    def passo(x, y, th):
        i, j, dx, dy = vizinhos(x, y, Lx, Ly, robs)
        if a.modelo:
            E = Estrela(th, i, j, dx, dy, N)
            dth = torch.empty(N, device=dev)
            lote = 16384
            for s in range(0, N, lote):
                idx = torch.arange(s, min(s + lote, N), device=dev)
                out, _ = model(E, idx)
                dth[idx] = amostrar(cab, out)
            novo = th + dth
        else:
            c, s = torch.cos(th), torch.sin(th)
            sx = c.clone().index_add_(0, i, c[j])               # a propria particula conta
            sy = s.clone().index_add_(0, i, s[j])
            if ruido == 'angular':
                novo = torch.atan2(sy, sx) + eta * math.pi * (2 * torch.rand(N, device=dev) - 1)
            else:                                               # vetorial, -G 1: amplitude eta * n
                n = 1.0 + torch.bincount(i, minlength=N).float()
                r = TWO_PI * torch.rand(N, device=dev)
                novo = torch.atan2(sy + eta * n * torch.sin(r), sx + eta * n * torch.cos(r))
        novo = wrap(novo)
        x = torch.remainder(x + a.v0 * torch.cos(novo), Lx)
        y = torch.remainder(y + a.v0 * torch.sin(novo), Ly)
        return x, y, novo, torch.hypot(torch.cos(novo).mean(), torch.sin(novo).mean())

    total = a.transiente + a.passos
    phis = torch.empty(total, device=dev)
    quadros = []
    sub = torch.randperm(N, device=dev)[:a.max_quadro] if a.quadros else None
    t0 = time.time()
    for t in range(total):
        x, y, th, phis[t] = passo(x, y, th)
        tp = t - a.transiente + 1
        if a.quadros and tp > 0 and tp % a.quadros == 0:
            quadros.append(torch.stack([x[sub], y[sub], th[sub]]).cpu().numpy().astype(np.float32))
        if (t + 1) % max(1, total // 10) == 0:
            el = time.time() - t0
            print('  passo %7d/%d  Phi=%.4f  (%.1f ms/passo, faltam ~%.0f s)' % (
                t + 1, total, float(phis[t]), 1000 * el / (t + 1), el / (t + 1) * (total - t - 1)))
    ms = 1000 * (time.time() - t0) / total
    phis = phis.cpu().numpy()
    st = estat(phis[a.transiente:], N)
    print('\n<Phi> = %.4f +- %.4f   chi = %.3f   Binder = %.4f   (%.1f ms/passo)' % (
        st['phi'], st['erro'], st['chi'], st['binder'], ms))

    extra = dict(quadros=np.stack(quadros)) if quadros else {}
    np.savez(os.path.join(a.saida, nome + '.npz'), phi_transiente=phis[:a.transiente],
             phi=phis[a.transiente:], params=np.array(json.dumps(dict(
                 vars(a), N=N, ruido=ruido, eta=eta, robs=robs, origem=origem))), **extra)
    tsv = os.path.join(a.saida, 'rollouts.tsv')
    novo = not os.path.exists(tsv)
    with open(tsv, 'a') as f:
        if novo:
            f.write('data\tnome\torigem\truido\teta\tLx\tLy\trho\tN\tinit\ttransiente\tpassos\tsemente'
                    '\tphi\terro\tchi\tbinder\tms_passo\n')
        f.write('\t'.join(str(v) for v in (
            time.strftime('%Y-%m-%d %H:%M'), nome, origem, ruido, eta, int(Lx), int(Ly), a.rho, N, a.init,
            a.transiente, a.passos, a.semente, '%.5f' % st['phi'], '%.5f' % st['erro'],
            '%.4f' % st['chi'], '%.5f' % st['binder'], '%.2f' % ms)) + '\n')
    print('salvo em %s/%s.npz e %s' % (a.saida, nome, tsv))


if __name__ == '__main__':
    main()
