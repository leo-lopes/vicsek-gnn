#!/bin/bash
# Situacao da rodada longa: processo vivo?, ultimo progresso, previsao de termino e GPU.
# Uso (WSL, na pasta D:\vicsek-gnn):   bash status_rodada.sh [arquivo de log]    (padrao: bandas_GL.log)
cd "$(dirname "$0")"
LOG=${1:-bandas_GL.log}

echo "== processo"
pids=$(pgrep -f "^python3 -u rollout.py")
if [ -z "$pids" ]; then
  echo "  nenhum rollout.py rodando (terminou ou parou: veja o fim do log abaixo)"
fi
for p in $pids; do
  ps -o pid=,etime=,%cpu=,rss= -p "$p" | awk '{printf "  pid %s · rodando ha %s · cpu %s%% · ram %.1f GB\n", $1, $2, $3, $4/1048576}'
done
[ "$(echo $pids | wc -w)" -gt 1 ] && echo "  ATENCAO: mais de um rollout.py rodando"

echo "== log ($LOG)"
if [ ! -f "$LOG" ]; then echo "  $LOG nao existe"; exit 0; fi
grep -E "N=|passo|<Phi>|salvo|Error|error|Traceback" "$LOG" | tail -n 3 | sed 's/^ */  /'
ult=$(grep "faltam" "$LOG" | tail -n 1)
if [ -z "$ult" ]; then
  echo "  (ainda sem linha de progresso: a primeira sai com 10 % dos passos)"
elif [ -n "$pids" ]; then
  p=$(echo $pids | awk '{print $1}')
  ms=$(echo "$ult" | sed -E 's/.*\(([0-9.]+) ms\/passo.*/\1/')
  tot=$(echo "$ult" | sed -E 's/.*passo +[0-9]+\/([0-9]+).*/\1/')
  ini=$(date -d "$(ps -o lstart= -p "$p")" +%s)
  fim=$(awk -v i="$ini" -v t="$tot" -v m="$ms" 'BEGIN {printf "%d", i + t * m / 1000}')
  echo "  previsao de termino: $(date -d "@$fim" '+%d/%m %H:%M') (media de $ms ms/passo ate a ultima linha)"
fi

echo "== GPU"
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader \
    | awk -F', ' '{print "  uso " $1 " · memoria " $2 " de " $3 " · " $4 " C"}'
else
  echo "  nvidia-smi indisponivel"
fi
