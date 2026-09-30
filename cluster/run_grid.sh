#!/bin/bash
# Job array: cada tarefa roda uma linha da grade.
#   sbatch --array=1-N%150 run_grid.sh grid_bloco0.txt out/bloco0
# Use ./submit.sh <bloco> -- ele cria logs/ antes (o SLURM exige que exista).
#
# Retomavel: cada job concluido deixa sentinela em <OUT>/.done/s<semente>.
#SBATCH --job-name=vicsek
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail

GRID_ARG=${1:?uso: run_grid.sh <grid.txt> <dir_saida>}
OUT_ARG=${2:?uso: run_grid.sh <grid.txt> <dir_saida>}
TASK=${SLURM_ARRAY_TASK_ID:-${3:?fora do SLURM, passe o numero da linha como 3o arg}}

# caminhos ABSOLUTOS: o $0 do SLURM vive em /var/tmp/slurmd.spoold, nao no projeto
GRID=$(cd "$(dirname "$GRID_ARG")" && pwd)/$(basename "$GRID_ARG")
mkdir -p "$OUT_ARG"
OUT=$(cd "$OUT_ARG" && pwd)          # <- diretorio inteiro, nao o dirname
mkdir -p "$OUT/.done"

[ -f "$GRID" ] || { echo "grid nao existe: $GRID" >&2; exit 1; }

ARGS=$(sed -n "${TASK}p" "$GRID")
[ -n "$ARGS" ] || { echo "linha ${TASK} vazia em ${GRID}" >&2; exit 1; }

SEED=$(echo "$ARGS" | awk '{for(i=1;i<=NF;i++) if($i=="-s") print $(i+1)}')
[ -n "$SEED" ] || { echo "linha ${TASK} sem -s <semente>: $ARGS" >&2; exit 1; }
SENT="$OUT/.done/s${SEED}"

if [ -f "$SENT" ]; then
  echo "tarefa ${TASK} (semente ${SEED}) ja concluida, pulando"
  exit 0
fi

FV="$HOME/vicsek-gnn/sim/FV"
[ -x "$FV" ] || { echo "FV nao encontrado ou sem permissao: $FV" >&2; exit 1; }

echo "tarefa ${TASK} em $(hostname): $FV $ARGS -o $OUT"
T0=$(date +%s)

"$FV" $ARGS -o "$OUT" || { echo "FV falhou (codigo $?) na tarefa ${TASK}" >&2; exit 1; }

T1=$(date +%s)
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SENT"
echo "tarefa ${TASK} (semente ${SEED}) ok em $((T1 - T0))s"
