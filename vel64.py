"""Velocidade das bandas em caixa pequena: fase do modo dominante entre quadros consecutivos com banda.

  python3 vel64.py <pasta> [<pasta> ...]      (os mesmos .npz do bandas64.py; so 0.35 <= rho <= 0.5)

v_banda = deslocamento da fase ao longo da normal, no sentido do movimento, por passo (valido enquanto a
banda anda menos de meio comprimento de onda por quadro); v_part = v0 * <cos(theta - normal)> das particulas.
"""
import glob, json, math, sys
import numpy as np
import bandas64 as B

arqs = sorted(f for pa in (sys.argv[1:] or ['varredura64']) for f in glob.glob(pa + '/*.npz'))
print('%-9s %4s %3s %6s %7s %7s %6s' % ('fonte', 'rho', 'sem', 'pares', 'v_banda', 'dp', 'v_part'))
for arq in arqs:
    d = np.load(arq)
    p = json.loads(str(d['params']))
    rho = float(p['rho'])
    if rho < 0.35 or rho > 0.5:
        continue
    q = d['quadros'].astype(np.float64)
    Lx = float(p['L']); Ly = float(p['Y'] or p['L']); dt = int(p['quadros'])
    x, y, th = q[:, 0], q[:, 1], q[:, 2]
    Th = np.arctan2(np.sin(th).mean(1), np.cos(th).mean(1))
    Z = np.stack([np.exp(1j * 2 * np.pi * (m * x / Lx + n * y / Ly)).mean(1) for m, n in B.MODOS], 1)
    A = np.abs(Z); jd = A.argmax(1); Ad = A[np.arange(len(A)), jd]
    kang = np.array([math.atan2(B.MODOS[j][1] / Ly, B.MODOS[j][0] / Lx) for j in jd])
    cosk = np.cos(kang - Th)
    banda = (Ad > B.A_MIN) & (np.abs(cosk) > B.COS_MIN)
    vs, vp = [], []
    for f in range(len(A) - 1):
        if banda[f] and banda[f + 1] and jd[f] == jd[f + 1]:
            m, n = B.MODOS[jd[f]]
            lam = 1.0 / math.hypot(m / Lx, n / Ly)
            dph = np.angle(Z[f + 1, jd[f]] / Z[f, jd[f]])            # em (-pi, pi]
            vs.append(dph * lam / (2 * np.pi * dt) * np.sign(cosk[f]))
            vp.append(float(p.get('v0', 0.5)) * np.mean(np.cos(th[f] - kang[f])) * np.sign(cosk[f]))   # v0 = 0.5 projetada na normal
    fo = B.fonte_de(p['origem'])
    if vs:
        print('%-9s %4g %3d %6d %7.3f %7.3f %6.3f' % (fo, rho, p['semente'], len(vs), np.mean(vs), np.std(vs) / math.sqrt(len(vs)), np.mean(vp)))
