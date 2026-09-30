#!/bin/bash
# M4 treinado so em G+L (nunca viu bandas): o experimento das bandas fora da distribuicao.
#   1) confere w(d) e so segue se --robs 1.2 der a mesma politica
#   2) lote da rede 65536 -> 16384 no rollout.py (evita os ciclos de OOM do inicio)
#   3) previa 64x64 perto da transicao (mesmo protocolo do teste rapido do GBL)   ~40 min na GTX 1650
#   4) rodada longa 1024x128, rho = 1 (mesmo protocolo do oraculo e do GBL)        ~7 h
# Uso, na pasta D:\vicsek-gnn (WSL):   nohup bash rodar_GL.sh > rodar_GL.log 2>&1 &
# Acompanhar:                          tail -f rodar_GL.log
set -eo pipefail
cd "$(dirname "$0")"
M=${M:-runs/vetorial_m4_hist_GL_e60}
X=${DRY:+echo}                                   # DRY=1 so mostra os comandos

echo "$(date '+%F %H:%M') conferindo w(d) de $M"
python3 checar_wd.py "$M" | tee checar_wd_GL.txt
grep -A1 "^$(basename "$M") " checar_wd_GL.txt | grep -q "OK, mesma politica" \
  || { echo "w(d) de $M nao permite --robs 1.2; parei (rode sem --robs ou me mande o checar_wd_GL.txt)"; exit 1; }

grep -q "lote = 16384" rollout.py || $X sed -i 's/lote = 65536/lote = 16384/' rollout.py

for r in 0.4 0.5 0.35 0.3; do
  echo "$(date '+%F %H:%M') 64x64 rho = $r"
  $X python3 -u rollout.py --modelo "$M" --robs 1.2 --L 64 --init aleatorio --quadros 50 --rho $r \
    --transiente 5000 --passos 35000 --semente 1 --saida varredura64_GL
done

echo "$(date '+%F %H:%M') 1024x128 rho = 1"
$X python3 -u rollout.py --modelo "$M" --robs 1.2 --L 1024 --Y 128 --rho 1 --init alinhado \
  --transiente 100000 --passos 30000 --quadros 100 --saida bandas_GL
echo "$(date '+%F %H:%M') FIM"
