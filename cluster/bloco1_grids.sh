#!/bin/bash
# Gera grids de refinamento Bloco 1
# Angular: η ∈ [0.35, 0.65], Vetorial: η ∈ [0.45, 0.65]
# Uso: bash bloco1_grids.sh

set -euo pipefail

seed=1

# ============================================================================
# BLOCO 1A: Angular (n=0) — η ∈ [0.35, 0.65] step 0.01
# ============================================================================
echo "Gerando grid_bloco1a.txt (Angular refinement)..."
> grid_bloco1a.txt

for L in 128 64 32; do  # Ordem decrescente: jobs longos entram na fila primeiro
  case "$L" in
    32)  t=10000;   T=50000 ;;
    64)  t=20000;   T=100000 ;;
    128) t=50000;   T=200000 ;;
  esac

  for eta_int in {35..65}; do
    eta=$(echo "scale=2; $eta_int / 100" | bc -l)
    # 2 replicas por config
    for _ in 1 2; do
      echo "-n 0 -L $L -t $t -T $T -e $eta -s $seed" >> grid_bloco1a.txt
      ((seed++))
    done
  done
done

n_bloco1a=$(wc -l < grid_bloco1a.txt)
echo "✓ grid_bloco1a.txt: $n_bloco1a jobs (Angular refinement, $seed-1 sementes usadas)"

# ============================================================================
# BLOCO 1B: Vetorial (n=1) — η ∈ [0.45, 0.65] step 0.01
# ============================================================================
echo "Gerando grid_bloco1b.txt (Vectorial refinement)..."
> grid_bloco1b.txt

for L in 128 64 32; do
  case "$L" in
    32)  t=10000;   T=50000 ;;
    64)  t=20000;   T=100000 ;;
    128) t=50000;   T=200000 ;;
  esac

  for eta_int in {45..65}; do
    eta=$(echo "scale=2; $eta_int / 100" | bc -l)
    # 2 replicas por config
    for _ in 1 2; do
      echo "-n 1 -L $L -t $t -T $T -e $eta -s $seed" >> grid_bloco1b.txt
      ((seed++))
    done
  done
done

n_bloco1b=$(wc -l < grid_bloco1b.txt)
echo "✓ grid_bloco1b.txt: $n_bloco1b jobs (Vectorial refinement, $seed-1 sementes usadas)"

# ============================================================================
# Resumo
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BLOCO 1 — Grids gerados com sucesso"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Bloco 1A (Angular):    $n_bloco1a jobs   |   η ∈ [0.35, 0.65], step 0.01"
echo "Bloco 1B (Vetorial):   $n_bloco1b jobs   |   η ∈ [0.45, 0.65], step 0.01"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Para submeter:"
echo "  ./submit.sh grid_bloco1a.txt out/bloco1a"
echo "  ./submit.sh grid_bloco1b.txt out/bloco1b"
echo ""
