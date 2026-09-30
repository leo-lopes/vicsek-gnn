#!/bin/bash
# Progresso da campanha.  ./status.sh 0   |   ./status.sh 0 watch
set -uo pipefail
cd "$(dirname "$0")" || exit 1

BLOCO=${1:?uso: status.sh <bloco> [watch]}
MODO=${2:-}
GRID="grid_bloco${BLOCO}.txt"
OUT="out/bloco${BLOCO}"

TOTAL=$(grep -cve '^[[:space:]]*$' "$GRID" 2>/dev/null || echo 0)
[ "$TOTAL" -gt 0 ] || { echo "grade $GRID nao existe ou esta vazia"; exit 1; }

while true; do
  FEITOS=$(ls -1 "$OUT/.done" 2>/dev/null | wc -l)
  DADOS=$(ls -1 "$OUT"/ts_*.dat.gz 2>/dev/null | wc -l)
  RODANDO=$(squeue -h -u "$USER" -t RUNNING 2>/dev/null | wc -l)
  ESPERA=$(squeue -h -u "$USER" -t PENDING 2>/dev/null | wc -l)
  PCT=$((FEITOS * 100 / TOTAL))
  N=$((PCT * 30 / 100))
  BAR=""
  [ "$N" -gt 0 ] && BAR=$(printf '#%.0s' $(seq 1 "$N"))
  [ "$N" -lt 30 ] && BAR="$BAR$(printf '.%.0s' $(seq 1 $((30 - N))))"

  echo "== $(date '+%Y-%m-%d %H:%M:%S') =="
  printf "bloco %s  %4d/%-4d %5.1f%%  [%s]\n" "$BLOCO" "$FEITOS" "$TOTAL" "$PCT" "$BAR"
  echo "arquivos ts_*.dat.gz: $DADOS"
  echo "fila: $RODANDO rodando, $ESPERA esperando"

  [ "$MODO" = watch ] || break
  [ "$FEITOS" -ge "$TOTAL" ] && { echo; echo "TERMINOU."; break; }
  sleep 15
  echo
done
