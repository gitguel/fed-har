"""Driver da grade federada: roda `run_federated.py` para cada combinação.

Itera sobre (encoder × cenário × seed), pulando combinações já completas no
cache `results/federated_eval.csv` (resume-friendly). Cada combinação roda como
um **subprocesso isolado** — assim o Ray é reinicializado por run, sem conflito
de estado global entre simulações.

Os 3 casos do projeto mapeiam para os cenários de `partitions.py`:
    caso 1 "todos datasets em todos clientes"       -> cenário 2    (IID global)
    caso 2 "um dataset por cliente"                 -> cenário 1    (non-IID por domínio)
    caso 3 "um dataset em todos os clientes (x6)"   -> cenários 3..8 (IID intra-domínio)

Ordem padrão dos cenários: [1, 2, 3..8] = caso 2 (oficial) -> caso 1 -> caso 3,
para que os resultados mais informativos saiam primeiro.

Uso:
    python scripts/federated/run_all.py                  # grade completa (96 runs)
    python scripts/federated/run_all.py --scenario 1 2   # só os casos 1 e 2
    python scripts/federated/run_all.py --encoder resnetse5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import PROJECT_ROOT, SEEDS
from federated.client import BUILD_MODEL

RUN_FEDERATED = Path(__file__).resolve().parent / "run_federated.py"
CACHE = PROJECT_ROOT / "results" / "federated_eval.csv"
ALL_SCENARIOS = [1, 2, 3, 4, 5, 6, 7, 8]
ALL_ENCODERS = list(BUILD_MODEL)


def is_done(cache: pd.DataFrame | None, encoder: str, scenario: int, seed: int, rounds: int) -> bool:
    """True se a rodada final (`rounds`) dessa combinação já está no cache."""
    if cache is None or cache.empty:
        return False
    mask = (
        (cache.encoder == encoder)
        & (cache.scenario == scenario)
        & (cache.seed == seed)
        & (cache["round"] == rounds)
    )
    return bool(mask.any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ALL_ENCODERS, default=ALL_ENCODERS)
    ap.add_argument("--scenario", nargs="+", type=int, default=ALL_SCENARIOS)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="Reroda mesmo o que já está no cache.")
    args = ap.parse_args()

    # Ordem: cenário externo, depois encoder, depois seed.
    combos = [
        (encoder, scenario, seed)
        for scenario in args.scenario
        for encoder in args.encoder
        for seed in args.seed
    ]
    total = len(combos)
    print(
        f"[GRID] {total} combinações | encoders={args.encoder} "
        f"scenarios={args.scenario} seeds={args.seed} rounds={args.rounds}",
        flush=True,
    )

    for i, (encoder, scenario, seed) in enumerate(combos, 1):
        cache = pd.read_csv(CACHE) if CACHE.exists() else None
        if not args.force and is_done(cache, encoder, scenario, seed, args.rounds):
            print(
                f"[GRID {i}/{total}] SKIP (cache) encoder={encoder} "
                f"scenario={scenario} seed={seed}",
                flush=True,
            )
            continue

        print(
            f"[GRID {i}/{total}] RUN encoder={encoder} scenario={scenario} seed={seed}",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(RUN_FEDERATED),
            "--encoder", encoder,
            "--scenario", str(scenario),
            "--seed", str(seed),
            "--rounds", str(args.rounds),
            "--local-epochs", str(args.local_epochs),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[GRID {i}/{total}] FALHOU (rc={result.returncode}) encoder={encoder} "
                f"scenario={scenario} seed={seed}",
                flush=True,
            )

    print("[GRID] concluído.", flush=True)


if __name__ == "__main__":
    main()
