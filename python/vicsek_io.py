"""Leitura dos arquivos do FastVicsek e utilidades para montar o dataset da GNN.

Formatos (little-endian), ver FastVicsek.c:
  snap_<tag>.bin : sequencia de registros [cabecalho 128 B | id | x | y | th0 | th1 | thbar | C | n]
  chk_<tag>.bin  : [cabecalho 128 B | id | x | y | vx | vy]
  cg_*_<tag>.bin : int32 lx/s, ly/s e depois quadros float32 (ly/s, lx/s)
  ts_<tag>.dat.gz: colunas t, Phi, direcao media, desvio relativo da ocupacao das caixas

Convencao de tempo do snapshot: header['t'] = t e o tempo do estado de entrada;
th1 e a orientacao em t+1. A posicao em t+1 e x + v0*(cos th1, sin th1) com PBC.
"""
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

_HDR = struct.Struct("<8s6iqQ4d")  # 80 bytes uteis, resto do bloco de 128 e zero
HDR_SIZE = 128
NOISE_NAMES = {0: "angular", 1: "vetorial", 2: "vonmises"}


def _header(buf: bytes, off: int) -> dict:
    magic, ver, N, lx, ly, noise, gc, t, seed, eta, rho, v0, kappa = _HDR.unpack_from(buf, off)
    return dict(magic=magic.decode(), version=ver, N=N, lx=lx, ly=ly, noise=noise,
                gcnorm=gc, t=t, seed=seed, eta=eta, rho=rho, v0=v0, kappa=kappa)


@dataclass
class Snapshot:
    header: dict
    id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    th0: np.ndarray
    th1: np.ndarray
    thbar: np.ndarray
    C: np.ndarray
    n: np.ndarray


def read_snapshots(path: str):
    """Gera os snapshots de um arquivo snap_*.bin, um por vez."""
    with open(path, "rb") as f:
        buf = f.read()
    off = 0
    while off < len(buf):
        h = _header(buf, off)
        assert h["magic"] == "VSKSNAP1", f"cabecalho invalido em {off}"
        off += HDR_SIZE
        N = h["N"]
        arrs = {}
        for name, dt in (("id", "<i4"), ("x", "<f8"), ("y", "<f8"), ("th0", "<f4"),
                         ("th1", "<f4"), ("thbar", "<f4"), ("C", "<f4"), ("n", "<i4")):
            a = np.frombuffer(buf, dtype=dt, count=N, offset=off)
            off += a.nbytes
            arrs[name] = a
        yield Snapshot(header=h, **arrs)


def read_checkpoint(path: str):
    with open(path, "rb") as f:
        buf = f.read()
    h = _header(buf, 0)
    assert h["magic"] == "VSKCHKP1"
    N, off, out = h["N"], HDR_SIZE, {}
    for name, dt in (("id", "<i4"), ("x", "<f8"), ("y", "<f8"), ("vx", "<f8"), ("vy", "<f8")):
        a = np.frombuffer(buf, dtype=dt, count=N, offset=off)
        off += a.nbytes
        out[name] = a
    return h, out


def read_cg(path: str) -> np.ndarray:
    """Campos coarse-grained -> array (quadros, ly/s, lx/s)."""
    raw = np.fromfile(path, dtype="<f4", offset=8)
    nx, ny = np.fromfile(path, dtype="<i4", count=2)
    return raw.reshape(-1, ny, nx)


def read_timeseries(path: str) -> np.ndarray:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return np.loadtxt(f, comments="#")


# ----------------------------------------------------------------- vizinhanca
def neighbor_pairs(x, y, lx, ly, R=1.0):
    """Pares (i, j), i != j, com distancia periodica < R, nos dois sentidos."""
    pts = np.column_stack([np.mod(x, lx), np.mod(y, ly)])
    tree = cKDTree(pts, boxsize=[lx, ly])
    pairs = tree.query_pairs(R, output_type="ndarray")
    d = pts[pairs[:, 1]] - pts[pairs[:, 0]]
    d[:, 0] -= lx * np.round(d[:, 0] / lx)
    d[:, 1] -= ly * np.round(d[:, 1] / ly)
    keep = (d ** 2).sum(1) < R * R            # C usa desigualdade estrita
    pairs = pairs[keep]
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return i, j


def vicsek_mean(snap: Snapshot, R=1.0):
    """Recalcula thbar, C e n a partir de (x, y, th0): testa a pipeline de vizinhanca."""
    h = snap.header
    i, j = neighbor_pairs(snap.x, snap.y, h["lx"], h["ly"], R)
    c, s = np.cos(snap.th0.astype(np.float64)), np.sin(snap.th0.astype(np.float64))
    sx = c + np.bincount(i, weights=c[j], minlength=h["N"])
    sy = s + np.bincount(i, weights=s[j], minlength=h["N"])
    n = 1 + np.bincount(i, minlength=h["N"])
    return np.arctan2(sy, sx), np.hypot(sx, sy) / n, n


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- ruido exato
def noise_ratio(C, n, eta, gcnorm):
    """Ruido vetorial: razao r = amplitude do ruido / |soma| (depende so de C e n)."""
    amp = eta * (n if gcnorm else n + 1)
    return amp / (C * n)


def logpdf_vectorial(psi, r):
    """log p(psi) do angulo de 1 + r e^{i phi}, phi ~ U(0, 2pi); psi = th1 - thbar.

    r < 1: p = cos(psi) / (pi sqrt(r^2 - sin^2 psi)) em |psi| < arcsin r, e 0 fora.
    r > 1: p = [1 + cos(psi)/sqrt(r^2 - sin^2 psi)] / (2 pi).
    """
    psi, r = np.broadcast_arrays(np.asarray(psi, float), np.asarray(r, float))
    s2 = np.sin(psi) ** 2
    # th1 e thbar sao float32: pontos a ~1e-7 rad da borda do suporte (r < 1) sao
    # tratados como estando na borda, onde a densidade diverge de forma integravel
    root = np.sqrt(np.maximum(r ** 2 - s2, 1e-12 * r ** 2))
    out = np.full(psi.shape, -np.inf)
    big = r > 1
    out[big] = np.log1p(np.cos(psi[big]) / root[big]) - np.log(2 * np.pi)
    small = (~big) & (s2 < r ** 2 * (1 + 1e-5)) & (np.cos(psi) > 0)
    out[small] = np.log(np.cos(psi[small]) / (np.pi * root[small]))
    return out


def logpdf_angular(psi, eta):
    psi = np.asarray(psi, float)
    return np.where(np.abs(psi) <= np.pi * eta, -np.log(2 * np.pi * eta), -np.inf)
