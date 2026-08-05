#!/usr/bin/env bash
# Exp. 2 — eixo de VOLUME de pré-treino (`@full`).
#
# O Exp. 2 fechado em 2026-08-05 mede `device − single` a budget fixo (192
# janelas/cliente) = custo da federação com dado constante. Falta a outra metade
# da conta: `single@192 − single@full` = efeito de ter caído do dado cheio para
# 1.920 janelas. Sem ela não dá para dizer se o federado rende menos por causa da
# loss batch-hungry ou por causa do orçamento de dado.
#
# Tamanhos medidos (2026-08-05):
#   single:RealWorld_thigh:10   1.920 -> 10.338 janelas  (5,38x)
#   single:MotionSense:10       1.920 ->  2.127 janelas  (1,11x)
# O eixo de volume só existe de verdade no RealWorld_thigh. O MotionSense entra
# como CONTROLE: lá o budget 192 já pega ~90% do que os 10 usuários têm, então a
# previsão é Δ≈0 — se ele se mover junto com o RW, o efeito não é volume.
#
# Ordem: TF-C inteiro (Fases 1 e 2) antes do LFR, a pedido do Miguel, para os
# números do TF-C saírem cedo. O LFR é o item caro (4,6x/rodada) e vai atrás.
#
# Uso:  scripts/federated/lanca_exp2_full.sh
set -u
cd /home/miguel.barreto/fed-har || exit 1

GPUS=0,1,2,4,5,6          # 3 e 7 são de outro usuário (conferido 2026-08-05 09:4x)
PY=.venv/bin/python
GRID=scripts/federated/run_grid_fedssl.py

fase1 () {   # <método> <spec-de-pré-treino-com-@full> <spec-de-finetuning>
  echo "=== [$(date +%H:%M)] FASE 1 $1 $2 ==="
  $PY $GRID --phase 1 --method "$1" --pretrain-spec "$2" --spec "$3" --gpus $GPUS
}

fase2 () {   # idem — CPU é o gargalo da Fase 2, daí o OMP_NUM_THREADS
  echo "=== [$(date +%H:%M)] FASE 2 $1 $2 ==="
  OMP_NUM_THREADS=4 $PY $GRID --phase 2 --method "$1" --pretrain-spec "$2" \
    --spec "$3" --gpus $GPUS
}

RW=single:RealWorld_thigh:10@full;  RW_FT=device:RealWorld_thigh:10
MS=single:MotionSense:10@full;      MS_FT=device:MotionSense:10

for m in tfc lfr; do
  fase1 "$m" "$RW" "$RW_FT"
  fase1 "$m" "$MS" "$MS_FT"
  fase2 "$m" "$RW" "$RW_FT"
  fase2 "$m" "$MS" "$MS_FT"
  $PY $GRID --consolidate
  echo "=== [$(date +%H:%M)] $m CONCLUÍDO ==="
done

echo "=== [$(date +%H:%M)] Exp.2 @full: fim ==="
