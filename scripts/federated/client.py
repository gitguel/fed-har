"""Cliente Flower (NumPyClient) para a simulação federada supervisionada.

Cada cliente treina localmente um dos encoders supervisionados (resnetse5,
cnnpff, rnn) no seu shard de dados, por `local_epochs` épocas, e devolve os
pesos para agregação FedAvg no servidor. O laço de treino é um loop torch leve
(Adam + CrossEntropy) — deliberadamente **sem** `L.Trainer`, para evitar o
overhead de instanciar um Trainer a cada rodada.

Reusa os `build_model` de `scripts/supervised/` e `BEST_LR` de `common`.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common` e os scripts de treino.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import flwr as fl

from common import BATCH_SIZE, BEST_LR

# build_model por encoder — os mesmos dos baselines supervisionados (encoder + cabeça MLP).
from supervised.train_resnetse5 import build_model as _build_resnetse5
from supervised.train_cnnpff import build_model as _build_cnnpff
from supervised.train_rnn import build_model as _build_rnn
from supervised.train_tstcc import build_model as _build_tstcc

BUILD_MODEL = {
    "resnetse5": _build_resnetse5,
    "cnnpff": _build_cnnpff,
    "rnn": _build_rnn,
    "tstcc": _build_tstcc,
}


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    """`state_dict` -> lista de ndarrays, na ordem (estável) das chaves.

    Inclui buffers (ex.: estatísticas de BatchNorm) — é exatamente o conjunto de
    tensores transmitido na federação, então também serve de base para o custo
    de comunicação.
    """
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    """Lista de ndarrays -> `state_dict` (mesma ordem de chaves de `get_parameters`)."""
    keys = list(model.state_dict().keys())
    state = OrderedDict((k, torch.as_tensor(v)) for k, v in zip(keys, parameters))
    model.load_state_dict(state, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """Cliente FedAvg: treina o encoder no shard local e devolve os pesos."""

    def __init__(
        self,
        encoder: str,
        train_dataset: Dataset,
        device: torch.device,
        local_epochs: int = 1,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.local_epochs = local_epochs
        self.model = BUILD_MODEL[encoder]().to(device)
        self.num_examples = len(train_dataset)
        self.loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        # F2 da auditoria: com drop_last=True, um shard < batch_size produziria
        # um loader vazio e o cliente devolveria os pesos recebidos sem treinar.
        assert len(self.loader) > 0, (
            f"shard com {self.num_examples} amostras < batch_size={batch_size}"
        )

    def get_parameters(self, config):  # noqa: D102
        return get_parameters(self.model)

    def fit(self, parameters, config):  # noqa: D102
        set_parameters(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=BEST_LR[self.encoder])
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device).long()
                optimizer.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
        return get_parameters(self.model), self.num_examples, {}

    def evaluate(self, parameters, config):  # noqa: D102
        # A avaliação principal é centralizada no servidor (evaluate_fn do FedAvg).
        # Mantemos um retorno trivial para satisfazer a interface NumPyClient.
        return 0.0, self.num_examples, {}
