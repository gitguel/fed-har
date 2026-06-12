"""Treino supervisionado de baseline para o encoder CNN-PFF no DAGHAR.

Replica o pipeline supervisionado do benchmark DAGHAR
(da Luz et al., IEEE Access 2026) para o backbone CNN-PFF:
- 6 sub-datasets × 4 sementes [0, 1, 2, 3]
- Adam, lr=1e-4, batch_size=64, máx. 100 épocas, parada antecipada
  (patience=50 sobre val_loss)
- Cabeça de predição: Linear(768 -> 128) -> ReLU -> Linear(128 -> 6)
- 3 checkpoints salvos por execução: primeiro, melhor (val_loss) e último.

Saídas:
    checkpoints/supervised/cnnpff/{dataset}/seed{N}/{first,best,last}.ckpt
    logs/supervised/cnnpff/{dataset}/seed{N}/...
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
    INPUT_SHAPE,
    INPUT_TIMESTEPS,
    NUM_CLASSES,
    build_prediction_head,
    make_argparser,
    run_grid,
)

from minerva.models.nets.time_series.cnns import CNN_PFF_2D  # noqa: E402

ENCODER_NAME = "cnnpff"


def build_model() -> CNN_PFF_2D:
    """Constrói o modelo supervisionado CNN-PFF com a cabeça MLP do paper.

    `CNN_PFF_2D` espera um formato de entrada com 3 dimensões ``(C_in, H, W)``.
    A amostra DAGHAR tem formato ``(6, 60)``; o backbone faz `.unsqueeze(1)`
    internamente para inserir a dimensão de canal, então passamos
    ``input_shape=(1, 6, 60)``. Depois da pilha convolucional com
    compartilhamento parcial de pesos, a dimensão achatada de features é 768
    (paper, Tabela 2 / Seção III-B).
    """
    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
    }
    model = CNN_PFF_2D(
        input_shape=(1, INPUT_CHANNELS, INPUT_TIMESTEPS),
        num_classes=NUM_CLASSES,
        learning_rate=BEST_LR[ENCODER_NAME],
        train_metrics=dict(metrics),
        val_metrics=dict(metrics),
        test_metrics=dict(metrics),
    )
    # Substitui o FC mais profundo padrão pela cabeça MLP padronizada do paper.
    enc_dim = model.fc_input_channels  # 768 para o CNN-PFF em (1, 6, 60)
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
    )


if __name__ == "__main__":
    main()
