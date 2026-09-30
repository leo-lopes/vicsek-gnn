#!/bin/bash
# Gera as grades de parametros.  ./make_grids.sh [bloco]
#
# BLOCO 0 -- varredura grossa para localizar a transicao.
#   eta 0,05..1,00 passo 0,05 (20 pontos) x L {32,64,128} x 2 sementes x 2 ruidos
#   = 240 jobs.  O pico de chi(eta) marca eta_c; o refino vem no bloco 1.
#
# Duracao escalonada por L: o tempo de relaxacao cresce com o sistema, entao
# nao faz sentido gastar 250k passos num sistema de 2048 particulas.
#
# Semente = indice da linha (1..240). Como a sentinela de retomada usa a
# semente e nao o numero da linha, regerar a grade nao invalida o que ja rodou
# -- desde que a ordem das linhas nao mude.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

RHO=2
V0=0.5
ETA_N=20          # numero de pontos em eta
ETA_PASSO=0.05
SEMENTES=2        # replicas por configuracao
LS="128 64 32"    # L decrescente: os jobs longos entram na fila primeiro
RUIDOS="0 1"      # 0 = angular uniforme, 1 = vetorial (-G 0, variante original)

duracao() {       # transiente e producao por tamanho
  case "$1" in
    32)  echo "-t 10000 -T 50000"  ;;
    64)  echo "-t 20000 -T 100000" ;;
    128) echo "-t 50000 -T 200000" ;;
    *)   echo "-t 20000 -T 100000" ;;
  esac
}

bloco0() {
  local i=0 L n rep k eta
  for L in $LS; do
    for n in $RUIDOS; do
      for rep in $(seq 1 $SEMENTES); do
        for k in $(seq 1 $ETA_N); do
          eta=$(awk -v k="$k" -v p="$ETA_PASSO" 'BEGIN{printf "%.4f", k*p}')
          i=$((i + 1))
          echo "-L $L -r $RHO -v $V0 -n $n -e $eta $(duracao "$L") -s $i"
        done
      done
    done
  done
}

BLOCO=${1:-0}
case "$BLOCO" in
  0)
    bloco0 > grid_bloco0.txt
    echo "grid_bloco0.txt: $(wc -l < grid_bloco0.txt) linhas"
    awk '{for(i=1;i<NF;i++) if($i=="-L") print $(i+1)}' grid_bloco0.txt \
      | sort -n | uniq -c | awk '{printf "  L=%-4s %3d jobs\n", $2, $1}'
    ;;
  1|2)
    echo "bloco $BLOCO ainda nao definido -- depende do eta_c que sai do bloco 0." >&2
    exit 1
    ;;
  *)
    echo "uso: ./make_grids.sh [0]" >&2; exit 1 ;;
esac
