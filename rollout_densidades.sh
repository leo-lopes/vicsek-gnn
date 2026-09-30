#!/bin/bash
# Primeiro teste em malha fechada: a rede (M4) contra o Vicsek exato, na MESMA caixa, em varias densidades
# (gas, borda, bandas, rho = 2, liquido), para os dois ruidos. Coloque treino.py e rollout.py na mesma pasta.
#
#   bash rollout_densidades.sh                          (L = 64, 10k + 20k passos, semente 1)
#   L=128 PASSOS=50000 bash rollout_densidades.sh       (variaveis de ambiente mudam o padrao)
#   MODELO=m5 bash rollout_densidades.sh                (usa runs/<ruido>_m5_hist_GBL_e60)
#
# No fim imprime a tabela rede x oraculo.

set -euo pipefail
cd "$(dirname "$0")"
L=${L:-64}
TRANS=${TRANS:-10000}
PASSOS=${PASSOS:-20000}
SEM=${SEM:-1}
MODELO=${MODELO:-m4}
INIT=${INIT:-aleatorio}

roda() {   # roda <ruido> <eta> <densidades...>
  local r=$1 e=$2; shift 2
  for rho in "$@"; do
    echo "== $r  rho=$rho  oraculo"
    python3 -u rollout.py --oraculo "$r" --eta "$e" --L "$L" --rho "$rho" --init "$INIT" \
      --transiente "$TRANS" --passos "$PASSOS" --semente "$SEM" | grep --line-buffered -E "passo +[0-9]+/[0-9]+ .*faltam|^<Phi>" | sed -n '1p;$p'
    echo "== $r  rho=$rho  rede ($MODELO)"
    python3 -u rollout.py --modelo "runs/${r}_${MODELO}_hist_GBL_e60" --L "$L" --rho "$rho" --init "$INIT" \
      --transiente "$TRANS" --passos "$PASSOS" --semente "$SEM" | grep --line-buffered -E "passo +[0-9]+/[0-9]+ .*faltam|^<Phi>" | sed -n '1p;$p'
  done
}

roda angular 0.40 0.5 0.75 1.0 2.0 4.0
roda vetorial 0.55 0.125 0.25 1.0 2.0 4.0

python3 - "$L" "$PASSOS" "$SEM" "$MODELO" "$INIT" <<'EOF'
import csv, sys
L, passos, sem, modelo, init = sys.argv[1:]
lin = {}
with open('rollouts/rollouts.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['Lx'] == L and r['passos'] == passos and r['semente'] == sem and r['init'] == init:
            fonte = 'oraculo' if r['origem'].startswith('oraculo') else r['origem']
            lin[(r['ruido'], float(r['rho']), fonte)] = r
print('\nMALHA FECHADA  L=%s, %s passos de producao, init %s   (rede = %s)' % (L, passos, init, modelo))
print('%-9s %6s | %-17s %-17s | %8s %8s | %7s %7s' % ('ruido', 'rho', '<Phi> oraculo', '<Phi> rede',
                                                     'chi or.', 'chi rede', 'U4 or.', 'U4 rede'))
for (ru, rho, fonte), o in sorted(lin.items()):
    if fonte != 'oraculo':
        continue
    rede = [v for (r2, rh2, f2), v in lin.items() if r2 == ru and rh2 == rho and modelo in f2]
    if not rede:
        continue
    n = rede[-1]
    print('%-9s %6g | %.4f +- %.4f   %.4f +- %.4f   | %8.3f %8.3f | %7.4f %7.4f' % (
        ru, rho, float(o['phi']), float(o['erro']), float(n['phi']), float(n['erro']),
        float(o['chi']), float(n['chi']), float(o['binder']), float(n['binder'])))
EOF
