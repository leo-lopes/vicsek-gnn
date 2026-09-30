#!/bin/bash
# Submete o array de um bloco, com checagens antes.
#   ./submit.sh 0
set -uo pipefail
cd "$(dirname "$0")" || exit 1

BLOCO=${1:?uso: submit.sh <bloco>}
GRID="grid_bloco${BLOCO}.txt"
OUT="out/bloco${BLOCO}"
FV="$HOME/vicsek-gnn/sim/FV"

echo "== checagens =="
sinfo -h >/dev/null 2>&1 || { echo "  SLURM        FORA DO AR (controlador nao responde)"; exit 1; }
echo "  SLURM        ok ($(sinfo -h -o %D | paste -sd+ | bc 2>/dev/null || echo '?') nos)"
[ -x "$FV" ] || { echo "  FV           NAO ENCONTRADO: $FV"; echo "  -> cd ~/vicsek-gnn/sim && make"; exit 1; }
echo "  FV           ok ($FV)"
[ -f "$GRID" ] || { echo "  $GRID  NAO EXISTE -> rode ./make_grids.sh"; exit 1; }
N=$(grep -cve '^[[:space:]]*$' "$GRID")
echo "  $GRID  ok ($N linhas)"

# CRITICO: o SLURM abre logs/%x_%A_%a.out ANTES de rodar o script.
# Se logs/ nao existir, todo job morre com exit 1 e nenhuma mensagem.
mkdir -p logs "$OUT/.done"
FEITOS=$(ls -1 "$OUT/.done" 2>/dev/null | wc -l)
echo "  ja concluidos $FEITOS de $N"
[ "$FEITOS" -ge "$N" ] && { echo "  nada a fazer"; exit 0; }

echo
echo "== submissao =="
echo "  sbatch --array=1-${N}%150 run_grid.sh $GRID $OUT"
sbatch --array=1-${N}%150 run_grid.sh "$GRID" "$OUT"
echo
echo "Acompanhe com:  ./status.sh $BLOCO"
