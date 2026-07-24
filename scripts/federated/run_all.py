"""Driver da grade federada: roda `run_federated.py` para cada combinação.

⚠️ DEPRECATED (2026-07-21) — a grade federada CROSS-SILO (cenários 1–8) foi
abandonada como desenho/controle. Mantida só como resultado PRELIMINAR/motivação.
Eixo ativo: CROSS-DEVICE (clientes = usuários) — ver `partition_users.py`,
`docs/plano_fedssl_simulado.md` e `docs/analise_domain_shift.md`.

Itera sobre (encoder × cenário × seed). Cada combinação roda como um
**subprocesso isolado** — assim o Ray é reinicializado por run, sem conflito de
estado global entre simulações.

**Paralelismo em N GPUs (nó único).** Mantém um *pool* de até `--max-parallel`
slots (um por GPU). Cada slot dispara um `run_federated.py` fixado numa GPU via
`CUDA_VISIBLE_DEVICES=<id>` (dentro do subprocesso a GPU vira `cuda:0`, e o Ray,
iniciado ali, enxerga só essa GPU). Ganho ~linear no número de GPUs e robusto: se
um run falha, os outros seguem.

**Sem corrida de escrita.** Cada run grava um CSV **parcial próprio** em
`results/federated_parts/<encoder>_<scenario>_<seed>.csv` (via `--out`). Ao final,
o driver concatena todos os parciais (mais o que já houver em
`results/federated_eval.csv`) no cache consolidado `results/federated_eval.csv`.

**Resume.** Uma combinação é considerada pronta se a rodada final (`--rounds`) já
está no seu parcial. Para reprocessar, use `--force`. Subconjuntos: `--scenario 1 2`,
`--encoder resnetse5`, `--seed 0`.

Os 3 casos do projeto mapeiam para os cenários de `partitions.py`:
    caso 1 "todos datasets em todos clientes"       -> cenário 2    (IID global)
    caso 2 "um dataset por cliente"                 -> cenário 1    (non-IID por domínio)
    caso 3 "um dataset em todos os clientes (x6)"   -> cenários 3..8 (IID intra-domínio)

Ordem padrão dos cenários: [1, 2, 3..8] = caso 2 (oficial) -> caso 1 -> caso 3,
para que os resultados mais informativos saiam primeiro.

Uso:
    python scripts/federated/run_all.py                  # grade completa (96 runs), 1 run/GPU
    python scripts/federated/run_all.py --max-parallel 4 # limita a 4 GPUs simultâneas
    python scripts/federated/run_all.py --scenario 1 2   # só os casos oficiais (1 e 2)
    python scripts/federated/run_all.py --encoder resnetse5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import PROJECT_ROOT, SEEDS
from federated.client import BUILD_MODEL

RUN_FEDERATED = Path(__file__).resolve().parent / "run_federated.py"
CACHE = PROJECT_ROOT / "results" / "federated_eval.csv"
PART_DIR = PROJECT_ROOT / "results" / "federated_parts"
RUN_LOG_DIR = PROJECT_ROOT / "logs" / "fed_runs"
ALL_SCENARIOS = [1, 2, 3, 4, 5, 6, 7, 8]
ALL_ENCODERS = list(BUILD_MODEL)

# Espelha run_federated.COLS/KEY (não importamos de lá para manter este driver
# leve — sem puxar torch/flwr/ray para o processo orquestrador).
COLS = [
    "encoder", "scenario", "seed", "round", "target",
    "test_acc", "test_f1_macro", "uplink_bytes", "downlink_bytes",
]
KEY = ["encoder", "scenario", "seed", "round", "target"]


def detect_gpus() -> list[int]:
    """Índices de GPU disponíveis, via `nvidia-smi -L` (sem inicializar CUDA aqui)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
        n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        return list(range(n)) if n > 0 else []
    except Exception:
        return []


def partial_path(encoder: str, scenario: int, seed: int) -> Path:
    return PART_DIR / f"{encoder}_{scenario}_{seed}.csv"


def is_done(encoder: str, scenario: int, seed: int, rounds: int) -> bool:
    """True se a rodada final (`rounds`) dessa combinação já está no seu parcial."""
    p = partial_path(encoder, scenario, seed)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
    except Exception:
        return False
    if df.empty or "round" not in df.columns:
        return False
    return bool((df["round"] == rounds).any())


def consolidate() -> None:
    """Concatena todos os parciais (+ cache atual) em results/federated_eval.csv."""
    frames: list[pd.DataFrame] = []
    if CACHE.exists():
        frames.append(pd.read_csv(CACHE))  # baseline: preserva o que já existir
    for p in sorted(PART_DIR.glob("*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print(f"[CONCAT] aviso: falha lendo {p}: {exc}", flush=True)
    if not frames:
        print("[CONCAT] nada para consolidar.", flush=True)
        return

    full = pd.concat(frames, ignore_index=True)
    for c in COLS:
        if c not in full.columns:
            full[c] = pd.NA
    # parciais vêm depois do baseline -> keep="last" faz o parcial prevalecer.
    full = (
        full[COLS]
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    n_combos = full.groupby(KEY[:3]).ngroups
    print(
        f"[CONCAT] {len(full)} linhas -> {CACHE} ({n_combos} combinações)",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ALL_ENCODERS, default=ALL_ENCODERS)
    ap.add_argument("--scenario", nargs="+", type=int, default=ALL_SCENARIOS)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Lista de GPUs (ex.: '0,1,2,3'). Padrão: auto-detecta via nvidia-smi.",
    )
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Máx. de runs simultâneos. Padrão: número de GPUs.",
    )
    ap.add_argument("--force", action="store_true", help="Reroda mesmo o que já está no cache.")
    args = ap.parse_args()

    gpus = (
        [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        if args.gpus
        else detect_gpus()
    )
    if not gpus:
        print("[GRID] Nenhuma GPU detectada; rodando 1 run por vez em CPU.", flush=True)
        gpus = [0]  # placeholder; CUDA_VISIBLE_DEVICES=0 sem CUDA -> cai pra CPU no run
    max_parallel = args.max_parallel or len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))

    PART_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Ordem: cenário externo, depois encoder, depois seed (mais informativo primeiro).
    combos = [
        (encoder, scenario, seed)
        for scenario in args.scenario
        for encoder in args.encoder
        for seed in args.seed
    ]
    total = len(combos)

    # Resume: filtra combinações já completas (a menos de --force).
    pending: deque[tuple[str, int, int]] = deque()
    skipped = 0
    for encoder, scenario, seed in combos:
        if not args.force and is_done(encoder, scenario, seed, args.rounds):
            skipped += 1
            print(
                f"[GRID] SKIP (cache) encoder={encoder} scenario={scenario} seed={seed}",
                flush=True,
            )
        else:
            pending.append((encoder, scenario, seed))

    print(
        f"[GRID] {total} combinações | a rodar={len(pending)} skip={skipped} | "
        f"encoders={args.encoder} scenarios={args.scenario} seeds={args.seed} "
        f"rounds={args.rounds} | GPUs={gpus} max_parallel={max_parallel}",
        flush=True,
    )

    n_to_run = len(pending)
    free_gpus = deque(gpus[:max_parallel])
    running: dict[subprocess.Popen, tuple[int, tuple[str, int, int], object]] = {}
    launched = 0
    failures: list[tuple[str, int, int]] = []

    while pending or running:
        # Lança enquanto houver GPU livre e combinação pendente.
        while pending and free_gpus:
            encoder, scenario, seed = pending.popleft()
            gpu = free_gpus.popleft()
            launched += 1
            out = partial_path(encoder, scenario, seed)
            logp = RUN_LOG_DIR / f"{encoder}_{scenario}_{seed}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable, str(RUN_FEDERATED),
                "--encoder", encoder,
                "--scenario", str(scenario),
                "--seed", str(seed),
                "--rounds", str(args.rounds),
                "--local-epochs", str(args.local_epochs),
                "--out", str(out),
            ]
            logf = open(logp, "w")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[proc] = (gpu, (encoder, scenario, seed), logf)
            print(
                f"[GRID {launched}/{n_to_run}] LAUNCH gpu={gpu} "
                f"encoder={encoder} scenario={scenario} seed={seed} -> {logp.name}",
                flush=True,
            )

        # Coleta os que terminaram.
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            gpu, (encoder, scenario, seed), logf = running.pop(proc)
            logf.close()
            free_gpus.append(gpu)
            if rc == 0:
                print(
                    f"[GRID] DONE gpu={gpu} encoder={encoder} scenario={scenario} seed={seed}",
                    flush=True,
                )
            else:
                failures.append((encoder, scenario, seed))
                print(
                    f"[GRID] FALHOU (rc={rc}) gpu={gpu} encoder={encoder} "
                    f"scenario={scenario} seed={seed} — ver logs/fed_runs/"
                    f"{encoder}_{scenario}_{seed}.log",
                    flush=True,
                )

        if running and not (pending and free_gpus):
            time.sleep(2)

    print("[GRID] pool drenado; consolidando parciais...", flush=True)
    consolidate()

    if failures:
        print(f"[GRID] concluído com {len(failures)} falha(s): {failures}", flush=True)
    else:
        print("[GRID] concluído sem falhas.", flush=True)


if __name__ == "__main__":
    main()
