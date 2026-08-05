#!/usr/bin/env bash
# uso: lanca_p1.sh <gpu> <method> <encoder> <spec> <seed>
# Lança UM pré-treino federado (Fase 1) fixado numa GPU, desacoplado de quem chamou.
# É enviado para dentro do tmux, que passa a ser dono do processo — foi assim que a
# sessão de 29/07 conseguiu runs de 6 h. Sem lógica de fila: a fila é o refill_p1.py.
cd /home/miguel.barreto/fed-har || exit 1
g=$1; m=$2; e=$3; sp=$4; sd=$5
if [ "$m" = lfr ]; then le=30; else le=5; fi          # PRE_LOCAL_EPOCHS do driver
lbl="pre_${m}_${e}_${sp//:/-}_seed${sd}"
mkdir -p logs/fedssl_cross_device
CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python scripts/ssl/pretrain_fed.py \
  --method "$m" --encoder "$e" --partition "$sp" --rounds 100 \
  --local-epochs "$le" --seed "$sd" \
  >> "logs/fedssl_cross_device/${lbl}.log" 2>&1 &
echo "LAUNCH gpu=$g $lbl pid=$!" | tee -a logs/fedssl_p1_manual.log
