#!/bin/bash
# Sementes extras do teste 64x64 (vetorial, eta = 0.55) na CPU, enquanto a GPU roda a caixa grande.
# Prioridade: GL (nunca viu bandas) na borda, depois oraculo e GBL. Uma corrida por vez, 4 threads, sem GPU.
# Pula corridas cujo .npz ja existe: rodar de novo retoma de onde parou. ~3.5 h no total.
# Uso (WSL, na pasta D:\vicsek-gnn):   nohup bash sementes64.sh > sementes64.log 2>&1 &
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
X=${DRY:+echo}                                   # DRY=1 so mostra os comandos
S=varredura64
P="--L 64 --init aleatorio --quadros 50 --transiente 5000 --passos 35000 --saida $S"
GL="--modelo runs/vetorial_m4_hist_GL_e60 --robs 1.2"
GBL="--modelo runs/vetorial_m4_hist_GBL_e60 --robs 1.2"
OR="--oraculo vetorial --eta 0.55"

roda() {   # roda <prefixo do arquivo> <rho> <semente> <argumentos da origem...>
  local pre=$1 r=$2 s=$3; shift 3
  local arq="$S/${pre}_L64x64_r${r}_aleatorio_s${s}.npz"
  if [ -f "$arq" ]; then echo "ja existe: $arq"; return; fi
  echo "$(date '+%F %H:%M') ${pre} rho=${r} semente=${s}"
  $X python3 -u rollout.py "$@" $P --rho "$r" --semente "$s" | tail -n 3
}

for s in 2 3; do for r in 0.35 0.4; do roda vetorial_m4_hist_GL_e60 $r $s $GL; done; done
roda oraculo_vetorial_e0.55 0.35 3 $OR
for s in 2 3; do roda oraculo_vetorial_e0.55 0.4 $s $OR; done
for s in 2 3; do roda vetorial_m4_hist_GBL_e60 0.35 $s $GBL; done
for s in 2 3; do roda vetorial_m4_hist_GBL_e60 0.4 $s $GBL; done
for s in 2 3; do roda vetorial_m4_hist_GL_e60 0.3 $s $GL; roda vetorial_m4_hist_GBL_e60 0.3 $s $GBL; done
roda oraculo_vetorial_e0.55 0.3 3 $OR
echo "$(date '+%F %H:%M') FIM"
