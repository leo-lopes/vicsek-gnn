#!/usr/bin/env python3
"""
Treino e avaliacao das baselines (M0-M2) e das GNNs (M3-M5) no dataset do bloco 2A.

  python3 treino.py --dados DIR --ruido angular --modelo m3
  python3 treino.py --dados DIR --ruido vetorial --modelo m2 --estados GL   # treina so em G+L

Modelos (--modelo)
  m0   oraculo exato: regra do Vicsek + distribuicao exata do ruido. Piso da NLL, sem treino.
  m0h  oraculo + cabeca: direcao media exata (thbar) e forma do ruido aprendida a partir de C e n.
       Mede quanto a propria cabeca custa em relacao ao piso.
  m1   so a propria particula: nenhum vizinho, uma distribuicao fixa de dtheta ("seguir reto").
  m2   MLP com os 8 vizinhos mais proximos (ordenados por distancia; faltando, zeros + mascara).
  m3   GNN: mensagem por aresta, media uniforme sobre todos os vizinhos com d < R_obs (e o laco i->i).
  m4   GNN radial: media ponderada por um peso w(d) aprendido (so da distancia). A rede acha o degrau em d = R?
  m5   GNN com atencao: pesos softmax de g(aresta) - lambda*d, livres (mapas de atencao).

Cabecas (--cabeca)
  hist   direcao mu + histograma SIMETRICO do residuo wrap(dtheta - mu), K bins, interpolado
         linearmente (densidade continua, gradiente chega em mu).                     [padrao]
  hlivre histograma livre de dtheta, sem mu nem simetria (controle: nenhuma hipotese sobre o ruido).
  vm     von Mises (mu, kappa), a da proposta original.

Treina num tipo de ruido por vez, com os estados de --estados; valida e testa SEMPRE em G, B e L,
mas escolhe a melhor epoca so pelos estados de treino (treinando em G+L, o B fica de fora de verdade).

Saidas: runs/<nome>/modelo.pt e resultado.json, e uma linha por (conjunto, estado) em runs/resultados.tsv.
Precisa de numpy e torch.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ESTADOS = 'GBL'
TWO_PI = 2 * math.pi


# ======================================================================== dados
def wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def nll_oraculo(meta, y, tb, C, n):
    """-log p exata de dtheta dado o que o C calculou (thbar, C, n)."""
    psi = wrap_np(y.astype(np.float64) - tb.astype(np.float64))
    eta = meta['eta']
    if meta['noise'] == 0:
        ok = np.abs(psi) <= np.pi * eta + 1e-5
        return np.where(ok, np.log(2 * np.pi * eta), np.inf)
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


def vizinhos_knn(feat, ptr, k):
    """Os k vizinhos mais proximos de cada amostra (sem o laco), 5 features + mascara."""
    S = len(ptr) - 1
    sid = np.repeat(np.arange(S), np.diff(ptr))
    e = np.nonzero(np.arange(len(sid)) != ptr[sid])[0]           # tira o laco i->i
    order = e[np.lexsort((feat[e, 2], sid[e]))]
    so = sid[order]
    r = np.arange(len(order)) - np.searchsorted(so, np.arange(S))[so]
    keep = r < k
    out = np.zeros((S, k, 6), np.float32)
    out[so[keep], r[keep], :5] = feat[order[keep]]
    out[so[keep], r[keep], 5] = 1.0
    return out.reshape(S, 6 * k)


def listar(dirpath, ruido):
    arqs = []
    for f in sorted(glob.glob(os.path.join(dirpath, '*.npz'))):
        with np.load(f) as z:
            meta = json.loads(str(z['meta']))
        if meta['ruido'] == ruido:
            arqs.append((f, meta))
    if not arqs:
        raise SystemExit('nenhum .npz de ruido %s em %s' % (ruido, dirpath))
    return arqs


class Dados:
    """Um conjunto (treino, validacao ou teste) inteiro na GPU."""

    def __init__(self, arqs, conjunto, estados, dev, precisa, frac=1.0, rng=None):
        sel = [(f, m) for f, m in arqs if m['conjunto'] == conjunto and m['estado'] in estados]
        if not sel:
            raise SystemExit('nenhum arquivo para conjunto=%s estados=%s' % (conjunto, estados))
        cols = {k: [] for k in ('y', 'tb', 'C', 'n', 'est', 'orac', 'knn', 'feat', 'ptr')}
        base = 0
        for f, m in sel:
            with np.load(f) as z:
                y, tb, C, n = z['y'], z['thbar_rel'], z['C'], z['n']
                S = len(y)
                keep = np.arange(S)
                if frac < 1.0:
                    keep = np.sort(rng.choice(S, max(1, int(frac * S)), replace=False))
                cols['y'].append(y[keep])
                cols['tb'].append(tb[keep])
                cols['C'].append(C[keep])
                cols['n'].append(n[keep])
                cols['est'].append(np.full(len(keep), ESTADOS.index(m['estado']), np.int64))
                cols['orac'].append(nll_oraculo(m, y[keep], tb[keep], C[keep], n[keep]))
                if 'knn' in precisa:
                    cols['knn'].append(vizinhos_knn(z['feat'], z['ptr'], 8)[keep])
                if 'arestas' in precisa:
                    feat, ptr = z['feat'], z['ptr']
                    if frac < 1.0:                       # recorta as arestas das amostras mantidas
                        cnt = np.diff(ptr)[keep]
                        eid = np.repeat(ptr[keep], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
                        feat, ptr = feat[eid], np.concatenate([[0], np.cumsum(cnt)])
                    cols['feat'].append(feat)
                    cols['ptr'].append(ptr[:-1] + base)
                    base += ptr[-1]
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
        self.y = t(np.concatenate(cols['y']))
        self.tb = t(np.concatenate(cols['tb']))
        self.C = t(np.concatenate(cols['C']))
        self.n = t(np.concatenate(cols['n']))
        self.est = t(np.concatenate(cols['est']), torch.long)
        self.orac = np.concatenate(cols['orac'])
        self.S = len(self.orac)
        self.knn = t(np.concatenate(cols['knn'])) if cols['knn'] else None
        if cols['feat']:
            self.feat = t(np.concatenate(cols['feat']))
            self.ptr = t(np.append(np.concatenate(cols['ptr']), base), torch.long)
        self.arquivos = len(sel)

    def arestas(self, idx):
        start = self.ptr[idx]
        cnt = self.ptr[idx + 1] - start
        seg = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), cnt)
        off = torch.cumsum(cnt, 0) - cnt
        eid = start[seg] + torch.arange(seg.numel(), device=idx.device) - off[seg]
        return self.feat[eid], seg, cnt


# ======================================================================== cabecas
class CabecaHist(nn.Module):
    """mu + histograma SIMETRICO do residuo psi = wrap(dtheta - mu), interpolado linearmente.

    Simetrico porque, dado o que a particula ve, o ruido do Vicsek (angular e vetorial) e simetrico
    em torno da direcao media. Isso tambem torna mu identificavel: um histograma livre absorveria
    qualquer rotacao de mu. Um vies fixo (gaussiana de 1 rad em torno de 0) evita a solucao
    trocada por pi no inicio do treino; a rede pode cancela-lo.
    """

    def __init__(self, K):
        super().__init__()
        assert K % 2 == 0
        self.K, self.w = K, TWO_PI / K
        self.nout = 2 + K // 2
        c = (torch.arange(K // 2) + 0.5) * self.w
        self.register_buffer('prior', -c ** 2 / 2)

    def mu(self, out):
        return torch.atan2(out[:, 1], out[:, 0])

    def logp(self, out, y, mu=None):
        mu = self.mu(out) if mu is None else mu
        h = out[:, 2:] + self.prior
        logits = torch.cat([h.flip(1), h], 1)                         # centros -pi..pi, simetricos
        p = torch.softmax(logits, dim=1) / self.w
        return _interp_log(p, torch.remainder(y - mu + math.pi, TWO_PI) - math.pi, self.w, self.K)


class CabecaHistLivre(nn.Module):
    """Histograma livre de dtheta, sem mu e sem simetria (nenhuma hipotese sobre o ruido).
    O erro_mu usa a direcao media circular da distribuicao prevista."""

    def __init__(self, K):
        super().__init__()
        self.K, self.w = K, TWO_PI / K
        self.nout = K
        self.register_buffer('c', -math.pi + (torch.arange(K) + 0.5) * self.w)

    def mu(self, out):
        p = torch.softmax(out, dim=1)
        return torch.atan2((p * torch.sin(self.c)).sum(1), (p * torch.cos(self.c)).sum(1))

    def logp(self, out, y, mu=None):
        p = torch.softmax(out, dim=1) / self.w
        center = 0.0 if mu is None else mu
        return _interp_log(p, torch.remainder(y - center + math.pi, TWO_PI) - math.pi, self.w, self.K)


def _interp_log(p, psi, w, K):
    """log da densidade periodica linear por partes com valores p nos centros dos bins."""
    u = (psi + math.pi) / w - 0.5
    k0f = torch.floor(u)
    t = u - k0f
    k0 = torch.remainder(k0f.long(), K)
    k1 = torch.remainder(k0 + 1, K)
    f = (1 - t) * p.gather(1, k0[:, None])[:, 0] + t * p.gather(1, k1[:, None])[:, 0]
    return torch.log(f + 1e-30)


class CabecaVM(nn.Module):
    nout = 3

    def mu(self, out):
        return torch.atan2(out[:, 1], out[:, 0])

    def logp(self, out, y, mu=None):
        mu = self.mu(out) if mu is None else mu
        k = F.softplus(out[:, 2]) + 1e-3
        return k * (torch.cos(y - mu) - 1) - math.log(TWO_PI) - torch.log(torch.special.i0e(k))


# ======================================================================== modelos
def mlp(nin, h, nout, camadas=2):
    mods, d = [], nin
    for _ in range(camadas):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, nout))
    return nn.Sequential(*mods)


class M0h(nn.Module):              # direcao exata + forma aprendida de (C, 1/n)
    precisa = ()

    def __init__(self, h, nout):
        super().__init__()
        self.net = mlp(2, h, nout)

    def forward(self, D, idx):
        x = torch.stack([D.C[idx], 1.0 / D.n[idx]], 1)
        return self.net(x), D.tb[idx]


class M1(nn.Module):               # nenhuma entrada
    precisa = ()

    def __init__(self, h, nout):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(nout))
        with torch.no_grad():
            self.p[0] = 1.0

    def forward(self, D, idx):
        return self.p.expand(len(idx), -1), None


class M2(nn.Module):               # MLP nos 8 vizinhos mais proximos
    precisa = ('knn',)

    def __init__(self, h, nout):
        super().__init__()
        self.net = mlp(48, h, nout, camadas=3)

    def forward(self, D, idx):
        return self.net(D.knn[idx]), None


class M3(nn.Module):               # GNN, media uniforme das mensagens
    precisa = ('arestas',)

    def __init__(self, h, nout):
        super().__init__()
        self.phi = mlp(5, h, h)
        self.rho = mlp(h, h, nout)

    def forward(self, D, idx):
        feat, seg, cnt = D.arestas(idx)
        m = self.phi(feat)
        agg = torch.zeros(len(idx), m.shape[1], device=m.device).index_add_(0, seg, m)
        return self.rho(agg / cnt[:, None].to(m.dtype)), None


class M4(nn.Module):               # GNN radial: media ponderada por w(d) aprendido
    precisa = ('arestas',)

    def __init__(self, h, nout):
        super().__init__()
        self.phi = mlp(5, h, h)
        self.f = mlp(1, 32, 1)
        self.rho = mlp(h, h, nout)

    def peso(self, d):
        return F.softplus(self.f(d[:, None]))[:, 0] + 1e-6

    def forward(self, D, idx):
        feat, seg, cnt = D.arestas(idx)
        m = self.phi(feat)
        w = self.peso(feat[:, 2])
        num = torch.zeros(len(idx), m.shape[1], device=m.device).index_add_(0, seg, w[:, None] * m)
        den = torch.zeros(len(idx), device=m.device).index_add_(0, seg, w)
        return self.rho(num / den[:, None]), None


class M5(nn.Module):               # GNN com atencao: softmax por amostra de g(aresta) - lambda*d
    precisa = ('arestas',)

    def __init__(self, h, nout):
        super().__init__()
        self.phi = mlp(5, h, h)
        self.g = mlp(5, h, 1)
        self.lam = nn.Parameter(torch.tensor(-4.0))       # lambda = softplus(lam) >= 0
        self.rho = mlp(h, h, nout)

    def atencao(self, feat, seg, B):
        s = self.g(feat)[:, 0] - F.softplus(self.lam) * feat[:, 2]
        smax = torch.full((B,), -1e30, device=s.device).scatter_reduce(
            0, seg, s, reduce='amax', include_self=True).detach()
        e = torch.exp(s - smax[seg])
        z = torch.zeros(B, device=s.device).index_add_(0, seg, e)
        return e / z[seg]

    def forward(self, D, idx):
        feat, seg, cnt = D.arestas(idx)
        a = self.atencao(feat, seg, len(idx))
        m = self.phi(feat)
        agg = torch.zeros(len(idx), m.shape[1], device=m.device).index_add_(0, seg, a[:, None] * m)
        return self.rho(agg), None


MODELOS = {'m0h': M0h, 'm1': M1, 'm2': M2, 'm3': M3, 'm4': M4, 'm5': M5}


# ======================================================================== avaliacao
@torch.no_grad()
def avaliar(model, cab, D, lote):
    """NLL por estado, NLL do oraculo nas mesmas amostras e erro da direcao mu (graus)."""
    soma = np.zeros(3)
    erro = np.zeros(3)
    if model is None:                                   # m0: o proprio oraculo
        nll = np.where(np.isfinite(D.orac), D.orac, np.nan)
        mu_err = np.zeros(D.S)
    else:
        model.eval()
        nll, mu_err = [], []
        for i in range(0, D.S, lote):
            idx = torch.arange(i, min(i + lote, D.S), device=D.y.device)
            out, mu = model(D, idx)
            nll.append(-cab.logp(out, D.y[idx], mu).cpu().numpy())
            m = cab.mu(out) if mu is None else mu
            mu_err.append(np.abs(wrap_np((m - D.tb[idx]).cpu().numpy())))
        nll, mu_err = np.concatenate(nll), np.concatenate(mu_err)
    est = D.est.cpu().numpy()
    res = {}
    for k, e in enumerate(ESTADOS):
        m = est == k
        if not m.any():
            continue
        fin = np.isfinite(D.orac[m])
        v = float(np.nanmean(nll[m]) if model is None else nll[m].mean())
        res[e] = dict(nll=v, oraculo=float(D.orac[m][fin].mean()),
                      gap=v - float(D.orac[m][fin].mean()),
                      erro_mu_graus=float(np.degrees(mu_err[m].mean())), amostras=int(m.sum()))
    return res


def tabela(titulo, res):
    print('  %-10s %6s %8s %8s %8s %8s' % (titulo, 'estado', 'NLL', 'oraculo', 'gap', 'erro_mu'))
    for e, r in res.items():
        print('  %-10s %6s %8.4f %8.4f %8.4f %7.1f°' % ('', e, r['nll'], r['oraculo'], r['gap'],
                                                      r['erro_mu_graus']))


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description='baselines e GNN do Vicsek (bloco 2A)')
    ap.add_argument('--dados', required=True, help='diretorio com os .npz do prep_dataset.py')
    ap.add_argument('--ruido', required=True, choices=['angular', 'vetorial'])
    ap.add_argument('--modelo', required=True, choices=['m0', 'm0h', 'm1', 'm2', 'm3', 'm4', 'm5'])
    ap.add_argument('--estados', default='GBL', help='estados de treino, ex. GBL, GL, B')
    ap.add_argument('--cabeca', default='hist', choices=['hist', 'hlivre', 'vm'])
    ap.add_argument('--bins', type=int, default=100)
    ap.add_argument('--oculto', type=int, default=64)
    ap.add_argument('--epocas', type=int, default=20)
    ap.add_argument('--lote', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=None, help='padrao: 2e-3 (m1: 5e-2, so tem 102 parametros)')
    ap.add_argument('--frac', type=float, default=1.0, help='fracao das amostras de treino')
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--runs', default='runs')
    a = ap.parse_args()

    if a.lr is None:
        a.lr = 5e-2 if a.modelo == 'm1' else 2e-3
    torch.manual_seed(a.semente)
    rng = np.random.RandomState(a.semente)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.estados = ''.join(e for e in ESTADOS if e in a.estados.upper())
    nome = '%s_%s_%s_%s%s' % (a.ruido, a.modelo, a.cabeca, a.estados, ('_' + a.tag) if a.tag else '')
    if a.modelo == 'm0':
        nome = '%s_m0%s' % (a.ruido, ('_' + a.tag) if a.tag else '')
    pasta = os.path.join(a.runs, nome)
    os.makedirs(pasta, exist_ok=True)
    print('%s  (%s)' % (nome, dev))

    arqs = listar(a.dados, a.ruido)
    Cls = MODELOS.get(a.modelo)
    precisa = Cls.precisa if Cls else ()
    t0 = time.time()
    Dva = Dados(arqs, 'validacao', ESTADOS, dev, precisa)
    print('validacao: %d amostras (%d arquivos), %.0f s' % (Dva.S, Dva.arquivos, time.time() - t0))

    resultado = dict(nome=nome, args=vars(a))
    if a.modelo == 'm0':
        model = cab = None
        best_ep, nparams = 0, 0
    else:
        t0 = time.time()
        Dtr = Dados(arqs, 'treino', a.estados, dev, precisa, a.frac, rng)
        print('treino:    %d amostras (%d arquivos), %.0f s' % (Dtr.S, Dtr.arquivos, time.time() - t0))
        cab = {'hist': lambda: CabecaHist(a.bins), 'hlivre': lambda: CabecaHistLivre(a.bins),
               'vm': CabecaVM}[a.cabeca]().to(dev)
        model = Cls(a.oculto, cab.nout).to(dev)
        nparams = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas)
        best, best_ep, hist = float('inf'), -1, []
        print('%d parametros. epoca  NLL_treino  NLL_val(%s)  gap_val  [G B L]' % (nparams, a.estados))
        for ep in range(a.epocas):
            t0 = time.time()
            model.train()
            perm = torch.randperm(Dtr.S, device=dev)
            tot = 0.0
            for i in range(0, Dtr.S, a.lote):
                idx = perm[i:i + a.lote]
                out, mu = model(Dtr, idx)
                loss = -cab.logp(out, Dtr.y[idx], mu).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            rv = avaliar(model, cab, Dva, 4 * a.lote)
            vsel = np.mean([rv[e]['nll'] for e in a.estados])
            gsel = np.mean([rv[e]['gap'] for e in a.estados])
            hist.append(dict(epoca=ep, treino=tot / Dtr.S, val=vsel, val_estados=rv))
            marca = ''
            if vsel < best:
                best, best_ep, marca = vsel, ep, ' *'
                torch.save(model.state_dict(), os.path.join(pasta, 'modelo.pt'))
            print('  %3d  %9.4f  %9.4f  %8.4f   [%s]  %.0f s%s' % (
                ep, tot / Dtr.S, vsel, gsel, ' '.join('%.3f' % rv[e]['gap'] for e in ESTADOS if e in rv),
                time.time() - t0, marca))
        del Dtr
        model.load_state_dict(torch.load(os.path.join(pasta, 'modelo.pt'), map_location=dev))
        resultado['historico'] = hist

    rv = avaliar(model, cab, Dva, 4 * a.lote)
    del Dva
    Dte = Dados(arqs, 'teste', ESTADOS, dev, precisa)
    rt = avaliar(model, cab, Dte, 4 * a.lote)
    print('\nmelhor epoca: %d' % best_ep)
    tabela('validacao', rv)
    tabela('teste', rt)
    resultado.update(melhor_epoca=best_ep, parametros=nparams, validacao=rv, teste=rt)
    with open(os.path.join(pasta, 'resultado.json'), 'w') as f:
        json.dump(resultado, f, indent=1)

    tsv = os.path.join(a.runs, 'resultados.tsv')
    novo = not os.path.exists(tsv)
    with open(tsv, 'a') as f:
        if novo:
            f.write('data\tnome\truido\tmodelo\tcabeca\ttreino_em\tconjunto\testado\tnll\toraculo\tgap'
                    '\terro_mu_graus\tamostras\tmelhor_epoca\tparametros\n')
        for conj, res in (('validacao', rv), ('teste', rt)):
            for e, r in res.items():
                f.write('\t'.join(str(v) for v in (
                    time.strftime('%Y-%m-%d %H:%M'), nome, a.ruido, a.modelo,
                    '-' if a.modelo == 'm0' else a.cabeca, '-' if a.modelo == 'm0' else a.estados,
                    conj, e, '%.5f' % r['nll'], '%.5f' % r['oraculo'], '%.5f' % r['gap'],
                    '%.2f' % r['erro_mu_graus'], r['amostras'], best_ep, nparams)) + '\n')
    print('\nsalvo em %s/ e %s' % (pasta, tsv))


if __name__ == '__main__':
    main()
