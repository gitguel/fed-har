"""Pool simples de execução de subprocessos fixados por GPU (nó único).

Reutiliza o padrão do driver federado (`scripts/federated/run_all.py`): mantém
até `max_parallel` slots (um por GPU), dispara cada job como subprocesso isolado
com `CUDA_VISIBLE_DEVICES=<id>` (dentro do processo a GPU vira `cuda:0`), e segue
mesmo que um job falhe. Compartilhado pelos drivers de grade supervisionada
(`supervised/run_all_shots.py`) e SSL (`ssl/run_all.py`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import List, Sequence


def detect_gpus() -> List[int]:
    """Índices de GPU via `nvidia-smi -L` (sem inicializar CUDA no orquestrador)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
        n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        return list(range(n))
    except Exception:
        return []


@dataclass
class Job:
    label: str            # identificador legível (ex.: "resnetse5_UCI_seed0_shots1")
    argv: Sequence[str]   # comando completo (sys.executable já incluído pelo chamador)
    log: Path             # arquivo de log (stdout+stderr)


def run_pool(
    jobs: List[Job],
    gpus: Sequence[int],
    max_parallel: int,
) -> List[str]:
    """Executa `jobs` em paralelo, um por GPU. Retorna os labels que falharam."""
    if not jobs:
        print("[POOL] nada a rodar.", flush=True)
        return []
    gpus = list(gpus) or [0]
    max_parallel = max(1, min(max_parallel, len(gpus)))
    free_gpus: deque[int] = deque(gpus[:max_parallel])
    pending: deque[Job] = deque(jobs)
    running: dict[subprocess.Popen, tuple[int, Job, object]] = {}
    failures: List[str] = []
    launched = 0
    total = len(jobs)

    print(f"[POOL] {total} jobs | GPUs={gpus} max_parallel={max_parallel}", flush=True)
    while pending or running:
        while pending and free_gpus:
            job = pending.popleft()
            gpu = free_gpus.popleft()
            launched += 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            job.log.parent.mkdir(parents=True, exist_ok=True)
            logf = open(job.log, "w")
            proc = subprocess.Popen(
                list(job.argv), env=env, stdout=logf, stderr=subprocess.STDOUT
            )
            running[proc] = (gpu, job, logf)
            print(f"[POOL {launched}/{total}] LAUNCH gpu={gpu} {job.label} -> {job.log.name}",
                  flush=True)

        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            gpu, job, logf = running.pop(proc)
            logf.close()
            free_gpus.append(gpu)
            if rc == 0:
                print(f"[POOL] DONE gpu={gpu} {job.label}", flush=True)
            else:
                failures.append(job.label)
                print(f"[POOL] FALHOU (rc={rc}) gpu={gpu} {job.label} — ver {job.log}",
                      flush=True)

        if running and not (pending and free_gpus):
            time.sleep(2)

    if failures:
        print(f"[POOL] concluído com {len(failures)} falha(s): {failures}", flush=True)
    else:
        print("[POOL] concluído sem falhas.", flush=True)
    return failures
