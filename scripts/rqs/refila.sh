#!/usr/bin/env bash
# Supervisor: reenfileira um bloco à medida que suas dependências ficam prontas.
#
# O bloco `federado` depende da LR (busca S1) e o `finetune` depende da LR E do
# backbone do pré-treino. Ambos aceitam `--so-lr-decidida`, que pula célula sem
# LR fechada. Este laço roda o driver, espera, e roda de novo — pegando o que
# ficou pronto no intervalo. É SEQUENCIAL de propósito: duas instâncias do mesmo
# driver disputariam o mesmo job (o skip é por parcial completo, não por job em
# voo).
#
# Uso: scripts/rqs/refila.sh <federado|finetune> <seed> <gpus> [intervalo_s]
set -uo pipefail
cd "$(dirname "$0")/../.."

BLOCO=${1:?bloco}; SEED=${2:?seed}; GPUS=${3:?gpus}; INTERVALO=${4:-600}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

case "$BLOCO" in
  federado) DRIVER="scripts/rqs/run_rq1_federado.py --seed $SEED --so-lr-decidida" ;;
  finetune) DRIVER="scripts/rqs/run_rq2.py --fase finetune --seed $SEED --so-lr-decidida" ;;
  *) echo "bloco desconhecido: $BLOCO" >&2; exit 2 ;;
esac

# A tabela `lr_escolhida.csv` NAO se regenera sozinha, e `decididas()` le ela --
# nao os parciais. Sem este passo o supervisor gira para sempre sem notar que a
# busca S1 fechou: foi exatamente o que deixou o bloco federado 14h ocioso em
# 2026-08-23. `decidir()` so fecha uma celula com os 12 parciais (6 LRs x 2
# regimes) presentes, entao regenerar no meio da busca e seguro.
while true; do
  echo "=== $(date +%H:%M:%S) rodada do supervisor ($BLOCO seed $SEED) ==="
  poetry run python scripts/rqs/lr_escolhida.py --seed "$SEED"
  poetry run python $DRIVER --gpus "$GPUS"
  echo "=== $(date +%H:%M:%S) aguardando ${INTERVALO}s por novas dependências ==="
  sleep "$INTERVALO"
done
