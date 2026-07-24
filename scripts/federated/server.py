"""Estratégia FedAvg + avaliação centralizada por domínio.

⚠️ DEPRECATED (2026-07-21) — parte da pilha federada CROSS-SILO, abandonada como
desenho/controle (agora só resultado preliminar/motivação). Eixo ativo:
CROSS-DEVICE — ver `partition_users.py` e `docs/plano_fedssl_simulado.md`.

A cada rodada (incluindo a rodada 0, com os pesos iniciais), o servidor injeta
os pesos globais num modelo recém-construído e avalia nos 6 test sets de domínio
do DAGHAR, reusando a função `evaluate` de `scripts/eval_transfer.py`
(acurácia + F1-macro via sklearn). A avaliação por cliente é desligada
(`fraction_evaluate=0.0`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Coloca a pasta `scripts/` no sys.path para importar `common` e `eval_transfer`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flwr as fl
from flwr.server.strategy import FedAvg

from common import DATASETS, make_datamodule
from eval_transfer import DEVICE, evaluate  # evaluate(model, loader) -> (acc, f1)
from federated.client import BUILD_MODEL, get_parameters, set_parameters

# Cache dos test loaders por domínio (um por sub-dataset DAGHAR), no padrão de
# `eval_transfer.test_loader`: lê cada `test.csv` uma única vez.
_test_loaders: dict = {}


def domain_test_loader(dataset: str):
    """Retorna (cacheado) o `test_dataloader` de um domínio DAGHAR."""
    if dataset not in _test_loaders:
        dm = make_datamodule(dataset, seed=0, num_workers=0)
        dm.setup("test")
        _test_loaders[dataset] = dm.test_dataloader()
    return _test_loaders[dataset]


def make_evaluate_fn(
    encoder: str, on_result: Callable[[int, str, float, float], None]
):
    """Cria a `evaluate_fn` centralizada do FedAvg.

    A cada chamada, reconstrói o modelo, injeta os pesos globais e avalia nos 6
    domínios, repassando cada resultado a `on_result(round, target, acc, f1)`.
    Retorna `(loss, metrics)` onde `metrics["mean_acc"]` é a média das acurácias.
    """

    def evaluate_fn(server_round: int, parameters, config):
        model = BUILD_MODEL[encoder]().to(DEVICE)
        set_parameters(model, parameters)
        model.eval()

        accs = []
        for target in DATASETS:
            acc, f1 = evaluate(model, domain_test_loader(target))
            on_result(server_round, target, acc, f1)
            accs.append(acc)
            print(
                f"[EVAL] round={server_round} {target}: acc={acc:.4f} f1={f1:.4f}",
                flush=True,
            )

        mean_acc = float(sum(accs) / len(accs))
        print(f"[EVAL] round={server_round} mean_acc={mean_acc:.4f}", flush=True)
        return 0.0, {"mean_acc": mean_acc}

    return evaluate_fn


def build_strategy(encoder: str, num_clients: int, evaluate_fn) -> FedAvg:
    """FedAvg com todos os clientes por rodada e avaliação centralizada."""
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(BUILD_MODEL[encoder]())
    )
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # avaliação só centralizada
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )
