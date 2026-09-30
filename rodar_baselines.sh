#!/bin/bash
# Baselines M0-M2 e a primeira GNN (M3) nos dois tipos de ruido, treinando em G+B+L.
# Coloque treino.py na mesma pasta. ~30-40 min na GTX 1650.
#
#   bash rodar_baselines.sh /mnt/d/vicsek-dados/dataset_bloco2
#   bash rodar_baselines.sh <dataset> --epocas 2      (teste rapido; argumentos extras vao para o treino.py)
#
# Resultados: runs/<nome>/ e runs/resultados.tsv; no fim imprime a tabela do conjunto de teste.

set -euo pipefail
cd "$(dirname "$0")"
DADOS=${1:?uso: bash rodar_baselines.sh <diretorio do dataset> [argumentos extras do treino.py]}
shift
EXTRA=("$@")

for r in angular vetorial; do
  python3 treino.py --dados "$DADOS" --ruido "$r" --modelo m0
  for m in m0h m1 m2 m3; do
    python3 treino.py --dados "$DADOS" --ruido "$r" --modelo "$m" "${EXTRA[@]}"
  done
done

python3 - <<'EOF'
import csv
linhas = {}
with open('runs/resultados.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['conjunto'] == 'teste':
            linhas[(r['nome'], r['estado'])] = r          # a ultima rodada de cada nome vale
nomes = []
for (n, _e) in linhas:
    if n not in nomes:
        nomes.append(n)
print('\nTESTE -- gap = NLL do modelo - NLL do oraculo (nats); erro_mu = |mu - thbar| medio')
print('%-30s %8s %8s %8s   %7s %7s %7s' % ('modelo', 'gap G', 'gap B', 'gap L', 'mu G', 'mu B', 'mu L'))
for n in nomes:
    g = [linhas.get((n, e)) for e in 'GBL']
    print('%-30s %8s %8s %8s   %7s %7s %7s' % ((n,) + tuple('%.4f' % float(x['gap']) if x else '-' for x in g)
                                              + tuple('%.1f' % float(x['erro_mu_graus']) if x else '-' for x in g)))
EOF
