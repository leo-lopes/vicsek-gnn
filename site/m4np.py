# -*- coding: utf-8 -*-
"""M4 (GNN com peso radial) e cabeça 'hist' do treino.py reimplementados em NumPy, para rodar a
política treinada sem PyTorch. Mesmas contas do treino.py / rollout.py:
  aresta i<-j no referencial de i: (dx', dy', d, cos(θj-θi), sin(θj-θi)); laço i->i = (0,0,0,1,0)
  m = φ(aresta), w = softplus(f(d)) + 1e-6, agregado = Σ w m / Σ w, saída = ρ(agregado)
  μ = atan2(out1, out0); histograma simétrico de 100 bins com prior -c²/2; Δθ = μ + amostra linear."""
import math
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from ptload import load_pt

TWO_PI = 2 * math.pi


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def wrap(a):
    return np.remainder(a + math.pi, TWO_PI) - math.pi


class M4:
    def __init__(self, pasta, K=100):
        sd = {k: v.astype(np.float32) for k, v in load_pt(pasta + '/modelo.pt').items()}
        self.W = sd
        # w(d) só depende da distância: tabela fina calculada uma vez (float64), depois interpolada
        self.dtab = np.linspace(0.0, 2.0, 20001)
        W64 = {k: v.astype(np.float64) for k, v in sd.items()}
        h = self.dtab[:, None]
        h = silu(h @ W64['f.0.weight'].T + W64['f.0.bias']); h = silu(h @ W64['f.2.weight'].T + W64['f.2.bias'])
        self.wtab = (softplus(h @ W64['f.4.weight'].T + W64['f.4.bias'])[:, 0] + 1e-6)
        self.K, self.w = K, TWO_PI / K
        c = (np.arange(K // 2) + 0.5) * self.w
        self.prior = -c ** 2 / 2

    def _mlp(self, pre, x):
        W = self.W
        h = silu(x @ W[pre + '.0.weight'].T + W[pre + '.0.bias'])
        h = silu(h @ W[pre + '.2.weight'].T + W[pre + '.2.bias'])
        return h @ W[pre + '.4.weight'].T + W[pre + '.4.bias']

    def saida(self, feat, seg, N):
        """feat (E,5) com o laço de cada amostra incluído; seg (E,) = amostra de cada aresta."""
        feat = feat.astype(np.float32, copy=False)
        m = self._mlp('phi', feat)
        w = np.interp(feat[:, 2], self.dtab, self.wtab).astype(np.float32)
        A = sp.csr_matrix((w, (seg, np.arange(len(seg)))), shape=(N, len(seg)))
        agg = ((A @ m) / np.asarray(A.sum(1))).astype(np.float32)
        return self._mlp('rho', agg).astype(np.float64)

    def dist(self, out):
        """μ e as probabilidades dos 100 bins (centros de -π a π)."""
        mu = np.arctan2(out[:, 1], out[:, 0])
        h = out[:, 2:] + self.prior
        logits = np.concatenate([h[:, ::-1], h], 1)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        return mu, p

    def logp(self, out, y):
        mu, p = self.dist(out)
        p = p / self.w
        psi = np.remainder(y - mu + math.pi, TWO_PI) - math.pi
        u = (psi + math.pi) / self.w - 0.5
        k0f = np.floor(u)
        t = u - k0f
        k0 = np.remainder(k0f.astype(np.int64), self.K)
        k1 = np.remainder(k0 + 1, self.K)
        r = np.arange(len(y))
        return np.log((1 - t) * p[r, k0] + t * p[r, k1] + 1e-30)

    def amostra(self, out, rng):
        mu, p = self.dist(out)
        a, b = p, np.roll(p, -1, axis=1)                      # segmento k: do centro k ao k+1
        pesos = a + b
        cum = np.cumsum(pesos, 1)
        alvo = rng.random(len(p))[:, None] * cum[:, -1:]
        k = np.minimum((cum < alvo).sum(1), self.K - 1)
        r = np.arange(len(p))
        ak, bk = a[r, k], b[r, k]
        u = rng.random(len(p))
        dif = bk - ak
        quase = np.abs(dif) < 1e-6 * (ak + bk + 1e-30)
        t = np.where(quase, u, (-ak + np.sqrt(np.maximum(ak * ak + dif * u * (ak + bk), 0))) / np.where(quase, 1.0, dif))
        return mu + wrap(-math.pi + (k + 0.5 + np.clip(t, 0, 1)) * self.w)


def vizinhos(x, y, Lx, Ly, R):
    """Pares dirigidos i != j com distância periódica < R: i, j, dx = x_j - x_i, dy."""
    pts = np.stack([np.remainder(x, Lx), np.remainder(y, Ly)], 1)
    pts[:, 0] = np.minimum(pts[:, 0], np.nextafter(Lx, 0)); pts[:, 1] = np.minimum(pts[:, 1], np.nextafter(Ly, 0))
    tree = cKDTree(pts, boxsize=[Lx, Ly])
    par = tree.query_pairs(R, output_type='ndarray')
    i = np.concatenate([par[:, 0], par[:, 1]]); j = np.concatenate([par[:, 1], par[:, 0]])
    dx = pts[j, 0] - pts[i, 0]; dx -= Lx * np.round(dx / Lx)
    dy = pts[j, 1] - pts[i, 1]; dy -= Ly * np.round(dy / Ly)
    return i, j, dx, dy


def passo_rede(model, x, y, th, Lx, Ly, robs, v0, rng):
    N = len(x)
    i, j, dx, dy = vizinhos(x, y, Lx, Ly, robs)
    c, s = np.cos(th[i]), np.sin(th[i])
    dth = th[j] - th[i]
    nb = np.stack([c * dx + s * dy, -s * dx + c * dy, np.hypot(dx, dy), np.cos(dth), np.sin(dth)], 1)
    laco = np.zeros((N, 5)); laco[:, 3] = 1.0
    feat = np.concatenate([laco, nb]); seg = np.concatenate([np.arange(N), i])
    out = model.saida(feat, seg, N)
    novo = wrap(th + model.amostra(out, rng))
    return np.remainder(x + v0 * np.cos(novo), Lx), np.remainder(y + v0 * np.sin(novo), Ly), novo


def passo_oraculo(x, y, th, Lx, Ly, eta, v0, rng):
    N = len(x)
    i, j, _, _ = vizinhos(x, y, Lx, Ly, 1.0)
    c, s = np.cos(th), np.sin(th)
    sx = c + np.bincount(i, weights=c[j], minlength=N)
    sy = s + np.bincount(i, weights=s[j], minlength=N)
    n = 1.0 + np.bincount(i, minlength=N)
    r = TWO_PI * rng.random(N)
    novo = wrap(np.arctan2(sy + eta * n * np.sin(r), sx + eta * n * np.cos(r)))
    return np.remainder(x + v0 * np.cos(novo), Lx), np.remainder(y + v0 * np.sin(novo), Ly), novo


def phi(th):
    return float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
