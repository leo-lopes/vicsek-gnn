#!/bin/bash
# GNNs M3, M4 e M5 com treino mais longo (60 epocas), nos dois ruidos, treinando em G+B+L,
# e depois a analise do experimento I (concentracao x C; pesos x distancia).
# Coloque treino.py, analise.py e tabela.py na mesma pasta. Estimativa: ~30 min na GTX 1650.
#
#   bash rodar_gnns.sh /mnt/d/vicsek-dados/dataset_bloco2
#   bash rodar_gnns.sh <dataset> --epocas 2       (teste rapido; argumentos extras vao para o treino.py)

set -euo pipefail
cd "$(dirname "$0")"
DADOS=${1:?uso: bash rodar_gnns.sh <diretorio do dataset> [argumentos extras do treino.py]}
shift
EXTRA=("$@")

for r in angular vetorial; do
  for m in m3 m4 m5; do
    python3 treino.py --dados "$DADOS" --ruido "$r" --modelo "$m" --epocas 60 --tag e60 "${EXTRA[@]}"
  done
done

python3 analise.py runs/{angular,vetorial}_m{3,4,5}_hist_GBL_e60
echo
python3 tabela.py
