"""Treino supervisionado conjunto — todos os sub-datasets DAGHAR juntos.

Mesma receita dos baselines por dataset (da Luz et al., IEEE Access 2026), mas
treinando cada encoder sobre a **concatenação dos 6 sub-datasets** em vez de um
de cada vez. Serve para medir, nas análises de transfer learning, o ganho de
generalização de um modelo "generalista" frente aos especialistas por dataset.

- 3 encoders (`resnetse5`, `cnnpff`, `rnn`) × 4 sementes [0, 1, 2, 3] = 12 treinos
- Conjunto de treino/validação/teste = união (`ConcatDataset`) dos 6 datasets
- Adam, lr=1e-4, batch_size=64, máx. 100 épocas, parada antecipada
  (patience=50 sobre val_loss) — idêntico aos scripts por encoder
- Reaproveita os `build_model` de cada script de encoder, então a arquitetura
  (backbone + cabeça MLP do paper) é exatamente a mesma dos baselines.

Saídas (o pseudo-dataset se chama "combined", encaixando no layout existente):
    checkpoints/supervised/{encoder}/combined/seed{N}/{first,best,last}.ckpt
    logs/supervised/{encoder}/combined/seed{N}/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common` e os outros
# scripts de treino independentemente de onde o comando seja executado.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    COMBINED_DATASET_NAME,
    MAX_EPOCHS,
    SEEDS,
    run_grid,
)

from supervised.train_resnetse5 import build_model as build_resnetse5
from supervised.train_cnnpff import build_model as build_cnnpff
from supervised.train_rnn import build_model as build_rnn

# Mesma ordem/nomes de encoder dos scripts individuais, para que checkpoints e
# logs do treino conjunto fiquem em `checkpoints/supervised/{encoder}/combined/`.
ENCODER_BUILDERS = {
    "resnetse5": build_resnetse5,
    "cnnpff": build_cnnpff,
    "rnn": build_rnn,
}


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treino supervisionado conjunto (todos os datasets DAGHAR juntos) "
            "para os 3 encoders × 4 sementes."
        )
    )
    parser.add_argument(
        "--encoder",
        choices=list(ENCODER_BUILDERS),
        nargs="+",
        default=list(ENCODER_BUILDERS),
        help="Encoders a treinar (padrão: os três).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Sementes a serem rodadas (padrão: 0 1 2 3).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Workers do DataLoader (padrão: 4).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Máximo de épocas por treino (padrão: {MAX_EPOCHS}).",
    )
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    for encoder in args.encoder:
        # `run_grid` percorre (dataset, seed); aqui o único "dataset" é o conjunto
        # unido "combined", então isso roda len(seeds) treinos por encoder.
        run_grid(
            model_factory=ENCODER_BUILDERS[encoder],
            encoder_name=encoder,
            datasets=[COMBINED_DATASET_NAME],
            seeds=args.seed,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
        )


if __name__ == "__main__":
    main()
