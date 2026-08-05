#!/usr/bin/env bash
# uso: lanca_p2.sh <gpu> <omp_threads> <logfile> <comando completo...>
# Lança UM fine-tuning federado da Fase 2 fixado numa GPU, desacoplado de quem
# chamou. É enviado para dentro do tmux, que passa a ser dono do processo — mesma
# lição do lanca_p1.sh: filho de chamada de ferramenta (ou de cron) morre quando a
# chamada termina. Sem lógica de fila: a fila é o refill_p2.py.
cd /home/miguel.barreto/fed-har || exit 1
g=$1; omp=$2; log=$3; shift 3
mkdir -p "$(dirname "$log")"
CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp \
  nohup "$@" >> "$log" 2>&1 &
echo "LAUNCH gpu=$g omp=$omp pid=$! $(basename "$log")" | tee -a logs/fedssl_p2_manual.log
