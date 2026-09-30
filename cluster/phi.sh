#!/bin/bash
# Parametro de ordem medio e suscetibilidade, direto dos ts_*.dat.gz (zcat+awk).
#   ./phi.sh out/bloco0                > phi_bloco0.dat
#   ./phi.sh out/bloco0 0.5            descarta os primeiros 50% (padrao)
#   ./phi.sh out/bloco0 0.5 --por-semente
#
# Saida: ruido L rho eta <Phi> erro chi n_sementes
#   chi = N (<Phi^2> - <Phi>^2), N = rho*lx*ly. O pico de chi marca a transicao.
set -uo pipefail

DIR=${1:?uso: phi.sh <dir com ts_*.dat.gz> [burn] [--por-semente]}
BURN=${2:-0.5}
MODO=${3:-}

ls "$DIR"/ts_*.dat.gz >/dev/null 2>&1 || { echo "nenhum ts_*.dat.gz em $DIR" >&2; exit 1; }

# ruido lx ly rho eta semente <Phi> <Phi^2> n_amostras
resumo() {
  for f in "$DIR"/ts_*.dat.gz; do
    meta=$(basename "$f" .dat.gz); meta=${meta#ts_}
    zcat "$f" 2>/dev/null | awk -v meta="$meta" -v burn="$BURN" '
      BEGIN {
        split(meta, F, "_")
        noise = substr(F[1], 2); rho = substr(F[3], 2); eta = substr(F[4], 2)
        seed  = substr(F[6], 2)
        split(substr(F[5], 2), D, "x"); lx = D[1]; ly = (D[2] == "" ? D[1] : D[2])
      }
      /^[ \t]*#/ { next }
      NF >= 2 { t[++m] = $2 }
      END {
        if (m == 0) exit
        i0 = int(m * burn) + 1
        n = 0; s = 0; s2 = 0
        for (i = i0; i <= m; i++) { n++; s += t[i]; s2 += t[i]*t[i] }
        if (n == 0) exit
        printf "%s %s %s %s %s %s %.8f %.8f %d\n", noise, lx, ly, rho, eta, seed, s/n, s2/n, n
      }'
  done
}

if [ "$MODO" = --por-semente ]; then
  echo "# ruido lx ly rho eta semente <Phi> <Phi^2> n_amostras"
  resumo
  exit 0
fi

echo "# ruido L rho eta <Phi> erro chi n_sementes   (burn=$BURN, dir=$DIR)"
resumo | awk '
  {
    key = $1 " " $2 " " $4 " " $5      # ruido lx rho eta
    N[key] = $4 * $2 * $3              # rho * lx * ly
    n[key]++; s[key] += $7; s2[key] += $7*$7; m2[key] += $8
  }
  END {
    for (k in n) {
      mean = s[k] / n[k]
      err = 0
      if (n[k] > 1) {                  # sem isso, n=1 divide por zero
        var = (s2[k]/n[k] - mean*mean) * n[k]/(n[k]-1)
        if (var > 0) err = sqrt(var/n[k])
      }
      chi = N[k] * (m2[k]/n[k] - mean*mean)
      printf "%s %.6f %.6f %.4f %d\n", k, mean, err, chi, n[k]
    }
  }' | sort -k1,1 -k2,2n -k4,4n
