#!/usr/bin/env bash
# Lançador dos blocos da grade das RQ1/RQ2 em tmux, com o ambiente certo.
#
# POR QUE ESTE SCRIPT EXISTE: sem `OMP_NUM_THREADS`, cada job agarra ~6,8 cores
# em spin-wait e rende 3,6 rodadas/min; com OMP=4 usa 0,9 core e rende ~14
# rod/min (medido em 2026-08-23, dl-16, sob contenção idêntica). O treino roda na
# GPU, então o número de threads NÃO altera resultado — só vazão. Esquecer a
# variável custa ~4× de relógio.
#
# Uso:  scripts/rqs/lanca.sh <bloco> <seed> <gpus>
#   bloco: centralizado | federado | pretrain | finetune | buscalr
#   gpus : lista para --gpus; repita o índice para rodar N jobs na mesma GPU
#          (cada job ocupa ~330 MiB), ex.: "3,3,3,4,4,4"
#
# Ex.:  scripts/rqs/lanca.sh buscalr      0 3,3,3,3,3,3,4,4,4,4,4,4
#       scripts/rqs/lanca.sh centralizado 0 6,6,6,6
set -euo pipefail
cd "$(dirname "$0")/../.."

BLOCO=${1:?bloco}; SEED=${2:?seed}; GPUS=${3:?gpus}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

case "$BLOCO" in
  buscalr)      CMD="scripts/rqs/run_busca_lr.py --seed $SEED" ;;
  centralizado) CMD="scripts/rqs/run_rq1_centralizado.py --seed $SEED" ;;
  federado)     CMD="scripts/rqs/run_rq1_federado.py --seed $SEED --so-lr-decidida" ;;
  pretrain)     CMD="scripts/rqs/run_rq2.py --fase pretrain --seed $SEED" ;;
  finetune)     CMD="scripts/rqs/run_rq2.py --fase finetune --seed $SEED --so-lr-decidida" ;;
  *) echo "bloco desconhecido: $BLOCO" >&2; exit 2 ;;
esac

SESSAO="rq-$BLOCO-s$SEED"
LOG="logs/$SESSAO.log"
mkdir -p logs
tmux kill-session -t "$SESSAO" 2>/dev/null || true
tmux new-session -d -s "$SESSAO"
tmux send-keys -t "$SESSAO" \
  "cd $(pwd) && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 poetry run python $CMD --gpus $GPUS 2>&1 | tee $LOG" Enter

echo "sessão: $SESSAO"
echo "log   : $LOG"
echo "  tmux attach -t $SESSAO   |   tail -f $LOG   |   tmux kill-session -t $SESSAO"
