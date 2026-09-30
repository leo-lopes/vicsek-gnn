#!/bin/bash
# Bloco 2A -- dados de treino da GNN, desenho A: estados G / B / L no mesmo eta.
# Densidades fixadas pelos mapas do bloco 1 (2026-09-25):
#   angular  eta=0.40: gas <= 0.625, bandas 0.75-3, liquido 4
#   vetorial eta=0.55: bandas 0.25-2, liquido 3-4, gas abaixo de 0.25 (sai do bloco 1d)
# (O plano antigo punha L em rho=3: no angular isso ainda e banda.)
#
#   cd ~/vicsek-gnn/cluster
#   bash make_bloco2.sh <rho_G_vetorial>        ex.:  bash make_bloco2.sh 0.125
#   ./submit.sh 2
#
# 10 sementes por estado: rep 1-8 treino, 9 validacao, 10 teste (em bloco2_indice.txt).
# Semente = 200000 + linha. Nao colide com blocos anteriores.
# Desenho B (rho=2, eta fora da janela de bandas) fica para um bloco 2B separado.

set -euo pipefail
cd "$(dirname "$0")"

RHO_G_VEC=${1:?uso: bash make_bloco2.sh <rho_G_vetorial>   (escolhido pelos mapas do bloco 1d)}
S_EVERY=5000        # snapshot a cada 5000 passos -> 20 por job.  1000 -> 100 por job e 5x o volume
T_TRANS=100000
T_PROD=100000
V0=0.5
GRID=grid_bloco2.txt
IDX=bloco2_indice.txt

for f in "$GRID" "$IDX"; do
  [ -e "$f" ] && { echo "$f ja existe -- renomeie ou apague antes de regerar"; exit 1; }
done
if [ -n "$(ls -A out/bloco2/.done 2>/dev/null || true)" ]; then
  echo "AVISO: out/bloco2/.done tem sentinelas antigas -- elas vao pular jobs. Mova out/bloco2 antes de submeter."
fi

# ruido | eta | estado | caixa | rho | init      (L e B comecam alinhados, G aleatorio)
cat > /tmp/bloco2_estados.$$ <<EOF
-n 0|0.40|G|-L 256|0.5|0
-n 0|0.40|B|-L 1024 -Y 128|1.0|1
-n 0|0.40|L|-L 256|4.0|1
-n 1 -G 1|0.55|G|-L 256|$RHO_G_VEC|0
-n 1 -G 1|0.55|B|-L 1024 -Y 128|1.0|1
-n 1 -G 1|0.55|L|-L 256|4.0|1
EOF

while IFS='|' read -r nz eta st box rho init; do
  for rep in $(seq 1 10); do
    echo "$st|$rep|$box -r $rho -v $V0 $nz -e $eta -i $init -t $T_TRANS -T $T_PROD -S $S_EVERY"
  done
done < /tmp/bloco2_estados.$$ |
awk -F'|' -v grid="$GRID" -v idx="$IDX" '
  BEGIN { print "# semente ruido estado rep conjunto" > idx }
  {
    seed = 200000 + NR
    printf "%s -s %d\n", $3, seed > grid
    ruido = ($3 ~ /-n 1/) ? "vetorial" : "angular"
    conj  = ($2 <= 8) ? "treino" : (($2 == 9) ? "validacao" : "teste")
    printf "%d %s %s %d %s\n", seed, ruido, $1, $2, conj > idx
  }'
rm -f /tmp/bloco2_estados.$$

# volume (snapshot = 128 B de cabecalho + 40 B por particula) e tempo (0.49 us/particula/passo,
# x1.27 de rho=2 para rho=4, medido; aproximado como linear em rho)
awk -v se="$S_EVERY" -v tp="$T_PROD" -v tt="$T_TRANS" '
  {
    for (i = 1; i < NF; i++) {
      if ($i == "-L") lx = ly = $(i+1)
      if ($i == "-Y") ly = $(i+1)
      if ($i == "-r") rho = $(i+1)
    }
    n = rho * lx * ly; snaps = int(tp / se)
    gb += (128 + 40 * n) * snaps / 1e9
    h = n * (tt + tp) * 0.49e-6 * (1 + 0.135 * (rho - 2)) / 3600
    cpu += h; if (h > hmax) hmax = h
    ly = 0
  }
  END {
    printf "%d jobs, %d snapshots por job\n", NR, int(tp / se)
    printf "volume: ~%.1f GB de snapshots\n", gb
    printf "tempo: job mais longo ~%.0f h, total ~%.0f CPU.h\n", hmax, cpu
  }' "$GRID"

echo
echo "confira:"
head -1 "$GRID"; tail -1 "$GRID"
echo
echo "submeter:  ./submit.sh 2"
