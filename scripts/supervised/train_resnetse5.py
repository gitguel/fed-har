"""Treino supervisionado de baseline para o encoder ResNet-SE-5 no DAGHAR.

Replica o pipeline supervisionado do benchmark DAGHAR
(da Luz et al., IEEE Access 2026) para o backbone ResNet-SE-5:
- 6 sub-datasets × 4 sementes [0, 1, 2, 3]
- Adam, lr=1e-4, batch_size=64, máx. 100 épocas, parada antecipada
  (patience=50 sobre val_loss)
- Cabeça de predição: Linear(64 -> 128) -> ReLU -> Linear(128 -> 6)
- 3 checkpoints salvos por execução: primeiro, melhor (val_loss) e último.

Saídas:
    checkpoints/supervised/resnetse5/{dataset}/seed{N}/{first,best,last}.ckpt
    logs/supervised/resnetse5/{dataset}/seed{N}/...
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
    INPUT_SHAPE,
    NUM_CLASSES,
    build_prediction_head,
    make_argparser,
    normalize_shots,
    run_grid,
)

from minerva.models.nets.time_series.resnet import ResNetSE1D_5  # noqa: E402

ENCODER_NAME = "resnetse5"


def build_model() -> ResNetSE1D_5:
    """Constrói o modelo supervisionado ResNet-SE-5 com a cabeça MLP do paper.

    A classe `ResNetSE1D_5` do minerva encapsula um backbone `_ResNet1D`
    (saída `(B, 64)` após pooling médio global) e um classificador linear
    único. Substituímos `model.fc` pela cabeça MLP padronizada de 2 camadas
    do paper para tornar a arquitetura comparável entre encoders.
    """
    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
    }
    model = ResNetSE1D_5(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=BEST_LR[ENCODER_NAME],
        train_metrics=dict(metrics),
        val_metrics=dict(metrics),
        test_metrics=dict(metrics),
    )
    # Substitui a cabeça padrão `Linear(64, 6)` pela MLP padronizada do paper.
    enc_dim = model.fc_input_features  # 64 para o ResNet-SE-5
    model.fc = build_prediction_head(in_features=enc_dim, num_classes=NUM_CLASSES)
    return model


def main() -> None:
    args = make_argparser(ENCODER_NAME).parse_args()
    run_grid(
        model_factory=build_model,
        encoder_name=ENCODER_NAME,
        datasets=args.dataset,
        seeds=args.seed,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        shot_regimes=normalize_shots(args.shots),
    )


if __name__ == "__main__":
    main()
