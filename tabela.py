#!/usr/bin/env python3
"""Tabela do conjunto de teste a partir de runs/resultados.tsv (a ultima rodada de cada nome vale).

  python3 tabela.py              todas as rodadas
  python3 tabela.py m3 m4 m5     so os nomes que contem algum desses textos

gap   = NLL do modelo - NLL do oraculo (nats)
acima = gap - gap do m0h do mesmo ruido: o que sobra alem do piso da cabeca (use este no vetorial)
mu    = |mu da rede - thbar| medio, em graus
"""
import csv
import sys

filtros = sys.argv[1:]
linhas, nomes = {}, []
with open('runs/resultados.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['conjunto'] != 'teste':
            continue
        linhas[(r['nome'], r['estado'])] = r
        if r['nome'] not in nomes:
            nomes.append(r['nome'])

piso = {}
for (n, e), r in linhas.items():
    if r['modelo'] == 'm0h' and r['cabeca'] == 'hist':
        piso[(r['ruido'], e)] = float(r['gap'])

print('TESTE   gap = NLL - oraculo (nats) | acima = gap - piso do m0h | mu = erro de direcao (graus)')
print('%-32s %21s   %21s   %17s' % ('', 'gap  G / B / L', 'acima  G / B / L', 'mu  G / B / L'))
for n in sorted(nomes, key=lambda s: (s.split('_')[0], s)):
    if filtros and not any(f in n for f in filtros):
        continue
    g, ac, mu = [], [], []
    for e in 'GBL':
        r = linhas.get((n, e))
        if r is None:
            g.append('-'); ac.append('-'); mu.append('-')
            continue
        g.append('%.4f' % float(r['gap']))
        p = piso.get((r['ruido'], e))
        ac.append('%.4f' % (float(r['gap']) - p) if p is not None and r['modelo'] not in ('m0', 'm0h') else '-')
        mu.append('%.1f' % float(r['erro_mu_graus']))
    print('%-32s %21s   %21s   %17s' % (n, ' '.join(g), ' '.join(ac), ' '.join(mu)))
