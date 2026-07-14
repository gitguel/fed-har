"""Entrypoint da simulação federada FedAvg (Flower) supervisionada.

Sobe uma simulação Flower com `num_clients` clientes (um por shard, conforme o
cenário de `partitions.py`), agrega via FedAvg e avalia centralizadamente nos 6
test sets de domínio do DAGHAR a cada rodada. Grava (incrementalmente) em
`results/federated_eval.csv`.

Uso (spike):
    python scripts/federated/run_federated.py --encoder resnetse5 --scenario 1 \
        --seed 0 --rounds 3 --local-epochs 1

Saída — `results/federated_eval.csv`:
    encoder, scenario, seed, round, target, test_acc, test_f1_macro,
    uplink_bytes, downlink_bytes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common` e os módulos federados.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import flwr as fl
from lightning.pytorch import seed_everything

from common import PROJECT_ROOT, SEEDS
from eval_transfer import DEVICE
from federated.client import BUILD_MODEL, FlowerClient, get_parameters
from federated.partitions import NUM_CLIENTS, make_client_datasets
from federated.server import build_strategy, make_evaluate_fn

CACHE = PROJECT_ROOT / "results" / "federated_eval.csv"
COLS = [
    "encoder",
    "scenario",
    "seed",
    "round",
    "target",
    "test_acc",
    "test_f1_macro",
    "uplink_bytes",
    "downlink_bytes",
]
KEY = ["encoder", "scenario", "seed", "round", "target"]


def load_cache(path: Path = CACHE) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", choices=list(BUILD_MODEL), required=True)
    ap.add_argument(
        "--scenario",
        type=int,
        default=1,
        help="1=non-IID por domínio, 2=IID global, 3..8=IID intra-domínio.",
    )
    ap.add_argument("--seed", type=int, default=0, choices=SEEDS)
    ap.add_argument("--rounds", type=int, default=50, help="Rodadas de FedAvg.")
    ap.add_argument("--local-epochs", type=int, default=1, help="Épocas locais por cliente/rodada.")
    ap.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "CSV de saída. Se omitido, usa o cache compartilhado "
            "results/federated_eval.csv. Para rodar vários runs em paralelo "
            "(ver run_all.py --max-parallel), passe um CSV parcial próprio "
            "(ex.: results/federated_parts/<enc>_<sc>_<seed>.csv) — assim cada "
            "run escreve seu próprio arquivo e não há corrida de escrita."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # F1 da auditoria (2026-07-07): semeia o run inteiro (init do modelo global,
    # shuffle dos loaders), não só o particionamento. Runs de encoders anteriores
    # ao tstcc rodaram sem esta linha (RNG arbitrário; ver notebook federado).
    seed_everything(args.seed, workers=True)

    # Dados por cliente (lista de datasets torch, um por cliente).
    client_datasets = make_client_datasets(args.scenario, args.seed, args.num_clients)
    num_clients = len(client_datasets)

    def client_fn(cid: str):
        dataset = client_datasets[int(cid)]
        return FlowerClient(
            args.encoder, dataset, DEVICE, local_epochs=args.local_epochs
        ).to_client()

    # Coleta das métricas centralizadas, repassadas pela evaluate_fn.
    rows: list[dict] = []

    def on_result(server_round: int, target: str, acc: float, f1: float) -> None:
        rows.append(
            dict(round=server_round, target=target, test_acc=acc, test_f1_macro=f1)
        )

    evaluate_fn = make_evaluate_fn(args.encoder, on_result)
    strategy = build_strategy(args.encoder, num_clients, evaluate_fn)

    # 1 GPU disponível -> 1 cliente por vez (num_gpus=1.0) = execução sequencial.
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 1.0 if DEVICE.type == "cuda" else 0.0,
    }

    # Os workers Ray são processos separados e não herdam o sys.path do driver;
    # sem isso, falham com `ModuleNotFoundError: No module named 'federated'`.
    # Exportamos `scripts/` (para `common`/`federated`) e a raiz (para `minerva`).
    scripts_dir = str(Path(__file__).resolve().parents[1])
    ray_init_args = {
        "runtime_env": {
            "env_vars": {"PYTHONPATH": f"{scripts_dir}:{PROJECT_ROOT}"}
        }
    }

    print(
        f"[FED] encoder={args.encoder} scenario={args.scenario} seed={args.seed} "
        f"clients={num_clients} rounds={args.rounds} local_epochs={args.local_epochs} "
        f"device={DEVICE}",
        flush=True,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    if not rows:
        print("[FED] Nenhum resultado coletado — verifique a simulação.")
        return

    # Custo de comunicação: bytes do conjunto de tensores transmitido (state_dict,
    # inclui buffers). Por rodada >= 1, cada cliente baixa (downlink) e sobe
    # (uplink) o modelo; a rodada 0 é só a avaliação dos pesos iniciais.
    bytes_per_model = int(sum(a.nbytes for a in get_parameters(BUILD_MODEL[args.encoder]())))

    df = pd.DataFrame(rows)
    comm = df["round"].apply(lambda r: bytes_per_model * num_clients if r >= 1 else 0)
    df["uplink_bytes"] = comm
    df["downlink_bytes"] = comm
    df["encoder"] = args.encoder
    df["scenario"] = args.scenario
    df["seed"] = args.seed
    df = df[COLS]

    out_path = Path(args.out).resolve() if args.out else CACHE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        pd.concat([load_cache(out_path), df], ignore_index=True)
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    cache.to_csv(out_path, index=False)
    print(
        f"\n[CACHE] {len(df)} linhas novas/atualizadas -> {out_path} (total {len(cache)})"
    )


if __name__ == "__main__":
    main()
