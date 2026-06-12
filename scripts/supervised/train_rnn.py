"""Treino supervisionado de baseline para o encoder RNN (BiGRU) no DAGHAR.

Replica o pipeline supervisionado do benchmark DAGHAR
(da Luz et al., IEEE Access 2026) para o backbone RNN (BiGRU estilo
Tonekaboni et al.):
- 6 sub-datasets × 4 sementes [0, 1, 2, 3]
- Adam, lr=1e-4, batch_size=64, máx. 100 épocas, parada antecipada
  (patience=50 sobre val_loss)
- Encoder: BiGRU(hidden=100, layers=1, bidirectional=True) -> Linear(200, 320)
- Cabeça de predição: Linear(320 -> 128) -> ReLU -> Linear(128 -> 6)
- 3 checkpoints salvos por execução: primeiro, melhor (val_loss) e último.

Saídas:
    checkpoints/supervised/rnn/{dataset}/seed{N}/{first,best,last}.ckpt
    logs/supervised/rnn/{dataset}/seed{N}/...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common` independentemente
# de onde o comando seja executado.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchmetrics import Accuracy

from common import (
    BEST_LR,
    INPUT_CHANNELS,
    NUM_CLASSES,
    build_prediction_head,
    make_argparser,
    run_grid,
)

from minerva.models.nets.base import SimpleSupervisedModel  # noqa: E402
from minerva.models.nets.tnc import RnnEncoder  # noqa: E402

ENCODER_NAME = "rnn"
RNN_HIDDEN_SIZE = 100
RNN_ENCODING_SIZE = 320  # paper, Tabela 2: dimensão de features do encoder RNN
RNN_NUM_LAYERS = 1


def build_model() -> SimpleSupervisedModel:
    """Constrói o modelo supervisionado BiGRU com a cabeça MLP do paper.

    O `RnnEncoder` do minerva produz um tensor 2D ``(B, encoding_size)``, então
    usamos `SimpleSupervisedModel(flatten=False, ...)`. ``permute=True`` faz o
    encoder aceitar o layout DAGHAR ``(B, C, T)`` produzido pelo DataModule
    (internamente o encoder permuta para ``(T, B, C)``, como exigido pelo
    ``torch.nn.GRU(batch_first=False)``).
    """
    backbone = RnnEncoder(
        hidden_size=RNN_HIDDEN_SIZE,
        in_channel=INPUT_CHANNELS,
        encoding_size=RNN_ENCODING_SIZE,
        cell_type="GRU",
        num_layers=RNN_NUM_LAYERS,
        dropout=0,
        bidirectional=True,
        permute=True,
    )
    head = build_prediction_head(in_features=RNN_ENCODING_SIZE, num_classes=NUM_CLASSES)

    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
    }
    return SimpleSupervisedModel(
        backbone=backbone,
        fc=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        learning_rate=BEST_LR[ENCODER_NAME],
        flatten=False,
        train_metrics=dict(metrics),
        val_metrics=dict(metrics),
        test_metrics=dict(metrics),
    )


def main() -> None:
    args = make_argparser(ENCODER_NAME).parse_args()
    run_grid(
        model_factory=build_model,
        encoder_name=ENCODER_NAME,
        datasets=args.dataset,
        seeds=args.seed,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
