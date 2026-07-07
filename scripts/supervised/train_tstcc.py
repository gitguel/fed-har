"""Treino supervisionado de baseline para o encoder TS-TCC (HARSCnnEncoder) no DAGHAR.

Replica o pipeline supervisionado do benchmark DAGHAR (da Luz et al., IEEE
Access 2026) para o backbone "TS-TCC Encoder" (o `lfr_default`/`tfc_harcnn`
dos YAMLs oficiais):
- 6 sub-datasets × 4 sementes [0, 1, 2, 3]
- Adam, lr=1e-4, batch_size=64, máx. 100 épocas, parada antecipada
  (patience=50 sobre val_loss)
- Encoder: HARSCnnEncoder — 3 blocos Conv1d(32→64→128) + Linear(1280 → 2304)
- Cabeça de predição: Linear(2304 -> 128) -> ReLU -> Linear(128 -> 6)
- 3 checkpoints salvos por execução: primeiro, melhor (val_loss) e último.

Saídas:
    checkpoints/supervised/tstcc/{dataset}/seed{N}/{first,best,last}.ckpt
    logs/supervised/tstcc/{dataset}/seed{N}/...
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
    normalize_shots,
    run_grid,
)

from minerva.models.nets.base import SimpleSupervisedModel  # noqa: E402
from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder  # noqa: E402

ENCODER_NAME = "tstcc"
TSTCC_ENCODING_SIZE = 2304  # YAML oficial (lfr_default/tfc_harcnn): dim do espaço latente
# Saída achatada dos 3 blocos convolucionais para entrada (B, 6, 60): 128 canais
# × 10 timesteps após os 3 MaxPool1d — ver docstring do HARSCnnEncoder.
TSTCC_INNER_CONV_DIM = 1280


def build_model() -> SimpleSupervisedModel:
    """Constrói o modelo supervisionado HARSCnn com a cabeça MLP do paper.

    O `HARSCnnEncoder` já devolve um tensor 2D ``(B, 2304)`` (conv → flatten →
    ``Linear(1280, 2304)``), então usamos `SimpleSupervisedModel(flatten=False)`.
    O encoder aceita o layout DAGHAR ``(B, C, T)`` diretamente (``permute=False``).
    """
    backbone = HARSCnnEncoder(
        dim=TSTCC_ENCODING_SIZE,
        input_channel=INPUT_CHANNELS,
        inner_conv_output_dim=TSTCC_INNER_CONV_DIM,
    )
    head = build_prediction_head(in_features=TSTCC_ENCODING_SIZE, num_classes=NUM_CLASSES)

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
        shot_regimes=normalize_shots(args.shots),
    )


if __name__ == "__main__":
    main()
