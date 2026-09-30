"""Testes do FastVicsek: vizinhanca, regra de alinhamento e distribuicao exata do ruido.

Uso: python validate.py <diretorio com snap_*.bin>
Para cada arquivo de snapshot:
  1. recalcula thbar, C e n a partir de (x, y, th0) com cKDTree periodica e compara com o C;
  2. testa se psi = th1 - thbar segue a distribuicao exata do ruido (PIT + Kolmogorov-Smirnov).
"""
import glob
import sys

import numpy as np
from scipy import stats
from scipy.special import i0e, i1e

import vicsek_io as vio


def cdf_vectorial(psi, r):
    """CDF exata de psi para o ruido vetorial (ver vicsek_io.logpdf_vectorial)."""
    psi, r = np.broadcast_arrays(np.asarray(psi, float), np.asarray(r, float))
    out = np.empty(psi.shape)
    big = r > 1
    out[big] = (psi[big] + np.pi) / (2 * np.pi) + np.arcsin(np.sin(psi[big]) / r[big]) / (2 * np.pi)
    sm = ~big
    s = np.clip(np.sin(psi[sm]) / r[sm], -1, 1)
    out[sm] = (np.arcsin(s) + np.pi / 2) / np.pi
    return np.clip(out, 0, 1)


def main(d):
    files = sorted(glob.glob(f"{d}/snap_*.bin"))
    assert files, "nenhum snap_*.bin encontrado"
    for f in files:
        snaps = list(vio.read_snapshots(f))
        h = snaps[0].header
        dth, dC, nbad, ntot, pits, psis = [], [], 0, 0, [], []
        for s in snaps:
            thbar, C, n = vio.vicsek_mean(s)
            dth.append(np.abs(vio.wrap_angle(thbar - s.thbar)).max())
            dC.append(np.abs(C - s.C).max())
            nbad += int((n != s.n).sum())
            ntot += n.size
            psi = vio.wrap_angle(s.th1.astype(float) - s.thbar.astype(float))
            psis.append(psi)
            if h["noise"] == 0:
                pits.append((psi + np.pi * h["eta"]) / (2 * np.pi * h["eta"]))
            elif h["noise"] == 1:
                r = vio.noise_ratio(s.C.astype(float), s.n.astype(float), h["eta"], h["gcnorm"])
                pits.append(cdf_vectorial(psi, r))
            else:
                pits.append(stats.vonmises.cdf(psi, h["kappa"]))
        pit = np.concatenate(pits)
        psi = np.concatenate(psis)
        ks = stats.kstest(pit[:: max(1, pit.size // 200000)], "uniform")
        name = f.split("snap_")[1][:-4]
        print(f"{name}: {len(snaps)} snapshots, N={h['N']}")
        print(f"   |thbar_py - thbar_C| max = {max(dth):.2e} rad | |C_py - C_C| max = {max(dC):.2e} |"
              f" n diferente em {nbad}/{ntot}")
        print(f"   ruido: <cos psi> = {np.cos(psi).mean():.4f} | PIT KS p-valor = {ks.pvalue:.3f}"
              f" (espera-se p >> 0.01 se a distribuicao exata estiver certa)")
        if h["noise"] == 0:
            e = h["eta"]
            print(f"   |psi| max = {np.abs(psi).max():.4f}  (limite eta*pi = {np.pi * e:.4f});"
                  f" <cos> teorico = {np.sin(np.pi * e) / (np.pi * e):.4f}")
        if h["noise"] == 2:
            k = h["kappa"]
            A = i1e(k) / i0e(k)
            print(f"   kappa = {k:.4f}; <cos> teorico = {A:.4f}; alvo sin(pi eta)/(pi eta) = "
                  f"{np.sin(np.pi * h['eta']) / (np.pi * h['eta']):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
