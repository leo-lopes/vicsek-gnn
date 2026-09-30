"""Referencias teoricas para a incerteza da politica aprendida.

Para cada tipo de ruido calcula, como funcao dos parametros:
  - H   : NLL do oraculo (entropia da distribuicao exata de psi = th_{t+1} - thbar), em nats;
  - m1  : <cos psi> (comprimento resultante medio);
  - k*  : kappa da von Mises que minimiza a NLL (casa m1, pois A(k*) = m1);
  - gap : NLL da melhor von Mises menos H = KL(exata || von Mises*), o custo de usar von Mises.
Ruido angular (uniforme em [-eta pi, eta pi]): nada depende da vizinhanca.
Ruido vetorial: psi = arg(1 + r e^{i phi}), r = amplitude/|soma|; com -G 1, r = eta / C.
"""
import numpy as np
from scipy import integrate, optimize
from scipy.special import ellipe, ellipk, i0e, i1e

LOG2PI = np.log(2 * np.pi)


def A(k):
    return i1e(k) / i0e(k)


def kappa_from_m1(m1):
    if m1 <= 1e-9:
        return 0.0
    return optimize.brentq(lambda k: A(k) - m1, 1e-12, 1e7)


def vm_ce(k, m1):
    """Entropia cruzada de uma von Mises(0, k) sob uma distribuicao com <cos psi> = m1."""
    return -k * m1 + LOG2PI + np.log(i0e(k)) + k


# ---------------------------------------------------------------- ruido angular
def angular(eta):
    H = np.log(2 * np.pi * eta)
    m1 = np.sin(np.pi * eta) / (np.pi * eta)
    k = kappa_from_m1(m1)
    return dict(H=H, m1=m1, kappa=k, gap=vm_ce(k, m1) - H)


# ---------------------------------------------------------------- ruido vetorial
def m1_vectorial(r):
    if r < 1:
        return 2 / np.pi * ellipe(r * r)
    m = 1 / (r * r)
    return 2 / np.pi * r * (ellipe(m) - (1 - m) * ellipk(m))


def pdf_vectorial(psi, r):
    s2 = np.sin(psi) ** 2
    if r > 1:
        return (1 + np.cos(psi) / np.sqrt(r * r - s2)) / (2 * np.pi)
    ok = (s2 < r * r) & (np.cos(psi) > 0)
    out = np.zeros_like(psi)
    out[ok] = np.cos(psi[ok]) / (np.pi * np.sqrt(r * r - s2[ok]))
    return out


def entropy_vectorial(r):
    if r < 1:
        # substituicao sin(psi) = r sin(phi) remove a singularidade da borda
        f = lambda ph: -np.log(np.sqrt(1 - (r * np.sin(ph)) ** 2) / (np.pi * r * np.cos(ph)))
        # p(psi) dpsi = dphi / pi  -> H = E_phi[-log p] com phi ~ U(-pi/2, pi/2)
        val, _ = integrate.quad(f, -np.pi / 2, np.pi / 2, limit=400)
        return val / np.pi
    f = lambda p: -pdf_vectorial(np.array([p]), r)[0] * np.log(max(pdf_vectorial(np.array([p]), r)[0], 1e-300))
    val, _ = integrate.quad(f, -np.pi, np.pi, limit=800, points=[-np.pi / 2, 0.0, np.pi / 2])
    return val


def vectorial(r):
    m1 = m1_vectorial(r)
    k = kappa_from_m1(m1)
    H = entropy_vectorial(r)
    return dict(H=H, m1=m1, kappa=k, gap=vm_ce(k, m1) - H)


if __name__ == "__main__":
    print("RUIDO ANGULAR (independe da vizinhanca)")
    print(" eta    H_oraculo  <cos>   kappa*  gap_VM(nats)")
    for e in (0.2, 0.3, 0.4, 0.478, 0.5, 0.6):
        a = angular(e)
        print(f" {e:5.3f}  {a['H']:8.4f}  {a['m1']:6.4f}  {a['kappa']:7.3f}  {a['gap']:.4f}")
    print(f" (sem informacao: log 2pi = {LOG2PI:.4f})")
    print("\nRUIDO VETORIAL (-G 1): r = eta / C")
    print(" r      H_oraculo  <cos>   kappa*  gap_VM")
    for r in (0.2, 0.4, 0.6, 0.8, 0.95, 1.05, 1.2, 1.5, 2, 3, 5):
        v = vectorial(r)
        print(f" {r:5.2f}  {v['H']:8.4f}  {v['m1']:6.4f}  {v['kappa']:7.3f}  {v['gap']:.4f}")
    eta = 0.6144
    print(f"\nkappa*(C) para ruido vetorial -G 1 com eta = {eta} (transicao em rho = 2):")
    for C in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        v = vectorial(eta / C)
        print(f" C={C:.1f}  r={eta / C:5.2f}  kappa*={v['kappa']:7.3f}  H={v['H']:.3f}")
