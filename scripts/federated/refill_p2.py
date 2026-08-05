#!/usr/bin/env python3
"""Ronda de monitoramento/refill da Fase 2 do Fed-SSL (fine-tuning federado, R=150).

Irmão do `ssl/refill_p1.py`, para a ladder SSL: 2 métodos x 4 encoders x 4 specs x
5 degraus x 4 seeds = 640 runs. A ladder do baseline (`method=none`, 256 runs) já
está fechada em `results/fedssl_cross_device_parts/`.

Por que refill em ondas, e não `run_grid_fedssl.py --phase 2` de uma vez: o driver
fixa o pool no conjunto de GPUs passado na largada e segura o terminal até o fim.
Aqui a Fase 1 ainda ocupa 1,2,6,7 e vai liberá-las aos poucos, e a máquina é
compartilhada — a fila precisa reavaliar as GPUs a cada 15 min.

**A lista de jobs vem do próprio driver** (`finetune_jobs`), então o comando, o
caminho do parcial e o `--init-ckpt` são byte-a-byte os mesmos da grade; este
script só decide ONDE e QUANDO cada job roda. `finetune_jobs` já pula parcial
completo e célula sem `backbone.ckpt` da Fase 1.

O gargalo medido em 2026-07-31 **não é a GPU**: um job usa 326 MiB e 6% de GPU,
mas ~10 cores. Daí as duas travas: `OMP_THREADS` por job e a reserva de cores
ociosos — é a CPU que dita quantos cabem, não a memória de vídeo.

Uso:  python3 scripts/federated/refill_p2.py            # relatório + lança o que couber
      python3 scripts/federated/refill_p2.py --dry-run  # só relatório
      python3 scripts/federated/refill_p2.py --max-launch 1
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from common import SEEDS
from eval_transfer import BUILD_MODEL
from federated.run_grid_fedssl import (FINETUNE_ROUNDS, SPECS, finetune_jobs,
                                       consolidate)

ROOT = Path("/home/miguel.barreto/fed-har")
LANCA = ROOT / "scripts" / "federated" / "lanca_p2.sh"

# sessão SÓ desta fila: se o pane estiver ocupado por um comando em foreground, o
# `send-keys` vira stdin dele e o job some sem log (aconteceu em 31/07 09:56)
TMUX_SESSION = "fedssl-p2-manual"

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]   # todas; quem filtra é a memória livre e o teto por GPU
MAX_PER_GPU = 5                   # contando os jobs da Fase 1 que ainda ocupam 1,2,6,7
MAX_P2_TOTAL = 12                 # teto de jobs de Fase 2 simultâneos (revisar se sobrar CPU)
MEM_PER_RUN = 1200                # medido: 326 MiB (resnetse5); folga para tstcc
MIN_HEADROOM_MIB = 1500           # nunca encostar no teto da GPU
OMP_THREADS = 4                   # sem isto um job come ~10 cores em spin de OMP
CORES_PER_JOB = OMP_THREADS       # cores estimados do job que vai entrar
RESERVE_CORES = 8                 # sempre deixar isto livre para os outros usuários
# processo alheio abaixo disto é contexto parado (ex.: notebook segurando 162 MiB),
# não trabalho — não é motivo para abrir mão da GPU
FOREIGN_MIN_MIB = 1000
# ⚠️ o guarda NÃO pode ser `load average`: com 20 pré-treinos rodando ele marca 79
# numa máquina de 72 cores, mas o `vmstat` mostra 34% de CPU ociosa — o que infla o
# load é `sys` (~50%, ~1M context switches/s), spin de OMP, não trabalho útil.
# Medimos ociosidade real em /proc/stat.

METHODS = ["lfr", "tfc"]


def _sh(args, timeout=45):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"[aviso] falhou/timeout em: {' '.join(args[:2])}", flush=True)
        return ""


def out_path(argv) -> str:
    """O `--out` identifica o job de forma única (é o parcial que ele grava)."""
    return argv[argv.index("--out") + 1]


def running_outs() -> dict[str, int]:
    """{caminho do --out: pid} dos fine-tunings vivos."""
    out = {}
    for line in _sh(["ps", "-eo", "pid,args"]).splitlines():
        if "run_cross_device.py" not in line or " grep " in line:
            continue
        parts = line.split()
        if "--out" not in parts:
            continue
        out.setdefault(parts[parts.index("--out") + 1], int(parts[0]))
    return out


def my_gpu_load() -> tuple[dict[int, int], dict[int, int], set[int]]:
    """(meus jobs por GPU, memória livre por GPU, GPUs com processo de outro usuário).

    A máquina é compartilhada e a decisão de 31/07 é **não entrar em GPU ocupada
    por outra pessoa**, mesmo cabendo memória: um job nosso pesa 326 MiB, mas
    disputa CPU e barramento com o dono da GPU. As GPUs que vão liberando são as
    nossas (1,2,6,7, à medida que a Fase 1 fecha), não as dos outros.
    """
    uuid2idx = {}
    for line in _sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        uuid2idx[uuid] = int(idx)
    free = {}
    for line in _sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]).splitlines():
        idx, used, total = [int(x.strip()) for x in line.split(",")]
        free[idx] = total - used
    mine = {g: 0 for g in free}
    me = os.environ.get("USER") or "miguel.barreto"
    my_pids = {int(l.split()[0]) for l in _sh(["ps", "-u", me, "-o", "pid,args"]).splitlines()
               if "pretrain_fed.py" in l or "run_cross_device.py" in l}
    foreign = set()
    for line in _sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                     "--format=csv,noheader,nounits"]).splitlines():
        uuid, pid, used = [x.strip() for x in line.split(",")]
        if uuid not in uuid2idx:
            continue
        g = uuid2idx[uuid]
        if int(pid) in my_pids:
            mine[g] = mine.get(g, 0) + 1
        elif (int(used) >= FOREIGN_MIN_MIB
              and _sh(["ps", "-o", "user=", "-p", pid]).strip() not in ("", me)):
            foreign.add(g)
    return mine, free, foreign


def idle_cores(window_s: float = 2.0) -> float:
    """Cores realmente ociosos, de /proc/stat em duas leituras."""
    def snap():
        f = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        return sum(f), f[3] + f[4]        # total, idle+iowait
    t0, i0 = snap()
    time.sleep(window_s)
    t1, i1 = snap()
    dt = max(t1 - t0, 1)
    return os.cpu_count() * (i1 - i0) / dt


def ensure_tmux() -> None:
    alive = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not alive:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", str(ROOT)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        print(f"[aviso] sessão tmux {TMUX_SESSION} não existia — recriada", flush=True)


def launch(gpu: int, job) -> str:
    cmd = " ".join(shlex.quote(str(a)) for a in
                   [LANCA, gpu, OMP_THREADS, job.log, *job.argv])
    subprocess.run(["tmux", "send-keys", "-t", TMUX_SESSION, cmd, "Enter"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    return f"LAUNCH gpu={gpu} {job.label}"


def one_round(args, gpus) -> int:
    """Uma decisão de refill. Devolve quantos jobs ainda faltam (0 = grade cheia)."""
    print(f"\n===== ronda P2 {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"{'(dry-run)' if args.dry_run else ''} =====", flush=True)

    jobs, skipped = finetune_jobs(METHODS, sorted(BUILD_MODEL), SPECS, SEEDS,
                                  None, FINETUNE_ROUNDS, 192, False)
    # alterna os dois métodos dentro da mesma célula: se a grade parar no meio,
    # o Δ(LFR − TF-C) existe para as células fechadas em vez de só um braço
    jobs.sort(key=lambda j: j.label.split("_", 1)[1])

    run = running_outs()
    active = [j for j in jobs if out_path(j.argv) in run]
    pending = [j for j in jobs if out_path(j.argv) not in run]

    mine, free, foreign = my_gpu_load()
    idle = idle_cores()

    print(f"FASE 2 SSL: {skipped}/640 DONE | {len(active)} RODANDO | "
          f"{len(pending)} PENDENTES (sem contar células sem backbone)", flush=True)
    print(f"CPU: {idle:.1f} cores ociosos de {os.cpu_count()} "
          f"(reserva {RESERVE_CORES}, {CORES_PER_JOB}/job) | load1={os.getloadavg()[0]:.1f}")
    print("GPUs: " + "  ".join(
        f"g{g}={mine.get(g, 0)}job/{free[g]}MiB livre{'/OUTRO USUÁRIO' if g in foreign else ''}"
        for g in sorted(free)))

    if not args.dry_run and pending:
        ensure_tmux()

    launched = []
    for job in pending:
        if len(launched) >= args.max_launch:
            break
        if len(active) + len(launched) >= MAX_P2_TOTAL:
            print("[trava] teto de jobs de Fase 2 simultâneos atingido.")
            break
        if idle - CORES_PER_JOB * (len(launched) + 1) < RESERVE_CORES:
            print(f"[trava] CPU sem folga ({idle:.1f} cores ociosos, "
                  f"reserva {RESERVE_CORES}).")
            break
        cands = [g for g in gpus
                 if g not in foreign
                 and mine.get(g, 0) < MAX_PER_GPU
                 and free.get(g, 0) >= MEM_PER_RUN + MIN_HEADROOM_MIB]
        if not cands:
            print("[trava] nenhuma GPU com vaga/memória.")
            break
        g = min(cands, key=lambda x: (mine.get(x, 0), -free[x]))
        launched.append(f"(dry) gpu={g} {job.label}" if args.dry_run else launch(g, job))
        mine[g] = mine.get(g, 0) + 1
        free[g] -= MEM_PER_RUN

    print(f"\nLANÇADOS nesta ronda: {len(launched)}")
    for l in launched:
        print("  " + l)

    print("\nRODANDO:")
    for job in active:
        print(f"  {job.label}  pid={run[out_path(job.argv)]}")

    print("\nPRÓXIMOS DA FILA:")
    for job in pending[len(launched):len(launched) + 6]:
        print(f"  {job.label}")
    return len(pending) - len(launched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-launch", type=int, default=99)
    ap.add_argument("--gpus", type=str, default=None, help="Restringe a lista (ex.: '3,4').")
    ap.add_argument("--consolidate", action="store_true")
    # os jobs duram ~3-15 min e a fila esvazia entre duas rondas de cron; o laço
    # interno mantém as GPUs alimentadas dentro da janela de 15 min
    ap.add_argument("--loop", type=int, default=1, help="Nº de rondas, espaçadas de --every.")
    ap.add_argument("--every", type=int, default=90, help="Segundos entre rondas do laço.")
    args = ap.parse_args()

    if args.consolidate:
        consolidate()
        return

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else GPUS

    for i in range(max(1, args.loop)):
        if i:
            time.sleep(args.every)
        if one_round(args, gpus) == 0:
            print("[fila] nada pendente — encerrando o laço.", flush=True)
            break


if __name__ == "__main__":
    main()
