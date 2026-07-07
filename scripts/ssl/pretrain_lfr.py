"""Pré-treino SSL centralizado com LFR (Learn From Randomness).

Estágio A do pipeline SSL: para cada (encoder, fonte, semente) pré-treina o
*backbone* do encoder **sem usar rótulos** (o LFR ignora `y`), usando alvos
gerados por projetores aleatórios congelados. Salva apenas o `state_dict` do
backbone — reaproveitado no downstream (`downstream_eval.py`).

Replica a configuração oficial do benchmark (da Luz et al., IEEE Access 2026;
`benchmarks/base_configs/models/train/lfr_<enc>.yaml` do repo
H-IAAC/benchmarking-encoders-ssl-har):

- Projetores: `LFR_HAR_Projector_List` — 60 candidatos convolucionais sobre a
  série bruta (6, 60), saída com dim = `enc_dim` do backbone (middle_dim=544).
- Preditores: `LFR_HAR_Predictor_List` — 60 candidatos de 1 camada
  `Linear(enc_dim -> enc_dim)` (num_layers=1).
- `num_targets=6`: a seleção DPP escolhe 6 projetores diversos entre os 60.
- Loss BatchWise Barlow Twins (default do minerva); Adam lr=3e-4,
  weight_decay=3e-4, betas=(0.9, 0.99).
- `predictor_training_epochs=5`: alterna 1 época de backbone+preditores com 5
  épocas só de preditores; por isso o Trainer roda 600 épocas = 100 épocas
  *efetivas* de backbone (pipeline oficial `train_for_600_epochs`).
- **Sem early stopping** — mantém o último checkpoint (protocolo do paper).
- Dados: train+val da fonte, **sem rótulos** (equivale ao
  `use_val_with_train: True` do benchmark). Fontes = 6 datasets + `combined`.

Saída:
    checkpoints/ssl/lfr/<encoder>/<source>/seed<N>/backbone.ckpt
    logs/ssl/lfr/<encoder>/<source>/seed<N>/...

Uso:
    # Spike (validação rápida ponta-a-ponta: 12 épocas = 2 efetivas de backbone)
    python scripts/ssl/pretrain_lfr.py --encoder resnetse5 --source KuHar --seed 0 --spike

    # Grade completa (3 encoders × 7 fontes × 4 seeds = 84 backbones)
    python scripts/ssl/pretrain_lfr.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader

# A pasta se chama `ssl`, que colide com o módulo `ssl` da stdlib — por isso
# importamos `encoders` diretamente (via scripts/ssl no path), sem o prefixo `ssl.`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

from common import (  # noqa: E402
    BATCH_SIZE,
    COMBINED_DATASET_NAME,
    DATASETS,
    INPUT_CHANNELS,
    LOGS_ROOT,
    PROJECT_ROOT,
    SEEDS,
    make_datamodule,
)
from encoders import ENCODERS, build_backbone  # noqa: E402

from minerva.models.nets.lfr_har_architectures import (  # noqa: E402
    LFR_HAR_Predictor_List,
    LFR_HAR_Projector_List,
)
from minerva.models.ssl.lfr import LearnFromRandomnessModel  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração (Tabela 4 do paper + YAMLs oficiais train/lfr_<enc>.yaml)
# ---------------------------------------------------------------------------
SSL_CKPT_BASE = PROJECT_ROOT / "checkpoints" / "ssl"
SSL_CKPT_ROOT = SSL_CKPT_BASE / "lfr"
SSL_LOGS_ROOT = LOGS_ROOT.parent / "ssl" / "lfr"  # .../logs/ssl/lfr
SOURCES = [COMBINED_DATASET_NAME] + DATASETS

DEFAULT_NUM_PROJECTORS = 60        # pool de candidatos aleatórios
DEFAULT_NUM_TARGETS = 6            # selecionados via DPP no setup
PROJ_MIDDLE_DIM = 544              # 32 canais × 17 timesteps após as convs, p/ entrada (6, 60)
PRED_MIDDLE_DIM = 128              # ignorado com num_layers=1 (mantido p/ paridade com o YAML)
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 3e-4
DEFAULT_PREDICTOR_TRAINING_EPOCHS = 5
BACKBONE_EPOCHS = 100              # épocas efetivas de backbone desejadas
# O backbone só treina 1 a cada (predictor_training_epochs + 1) épocas do Trainer.
DEFAULT_MAX_EPOCHS = BACKBONE_EPOCHS * (DEFAULT_PREDICTOR_TRAINING_EPOCHS + 1)  # 600
SPIKE_BACKBONE_EPOCHS = 2


class PretrainDataModule(L.LightningDataModule):
    """Serve train+val da fonte como um único train_dataloader (sem validação).

    Replica o `use_val_with_train: True` do benchmark: o pré-treino SSL usa
    todos os dados não rotulados disponíveis e não faz parada antecipada, então
    não há loop de validação.
    """

    def __init__(self, source: str, seed: int, batch_size: int, num_workers: int,
                 drop_last: bool = False) -> None:
        super().__init__()
        self.inner = make_datamodule(source, seed, batch_size=batch_size, num_workers=num_workers)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # A NT-Xent poly do TF-C usa máscara de tamanho fixo 2×batch — o loader
        # de pré-treino TF-C precisa de drop_last=True (o LFR não).
        self.drop_last = drop_last
        self.dataset: Optional[ConcatDataset] = None

    @staticmethod
    def _unwrap(split):
        return split[0] if isinstance(split, tuple) else split

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return
        self.inner.setup("fit")
        self.dataset = ConcatDataset([
            self._unwrap(self.inner.datasets["train"]),
            self._unwrap(self.inner.datasets["validation"]),
        ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )


def build_lfr(
    encoder: str,
    num_projectors: int = DEFAULT_NUM_PROJECTORS,
    num_targets: int = DEFAULT_NUM_TARGETS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    predictor_training_epochs: int = DEFAULT_PREDICTOR_TRAINING_EPOCHS,
) -> LearnFromRandomnessModel:
    """Monta o LFR em torno do backbone do encoder, como nos YAMLs oficiais.

    Projetores convolucionais recebem a série bruta (B, 6, 60); preditores
    lineares recebem as features achatadas do backbone (B, enc_dim) — por isso
    `flatten=False` (os três backbones de `encoders.py` já devolvem 2D).
    """
    backbone, enc_dim = build_backbone(encoder)
    projectors = LFR_HAR_Projector_List(
        size=num_projectors,
        encoding_size=enc_dim,
        input_channel=INPUT_CHANNELS,
        middle_dim=PROJ_MIDDLE_DIM,
    )
    predictors = LFR_HAR_Predictor_List(
        size=num_projectors,
        encoding_size=enc_dim,
        middle_dim=PRED_MIDDLE_DIM,
        num_layers=1,
    )
    return LearnFromRandomnessModel(
        backbone=backbone,
        projectors=projectors,
        predictors=predictors,
        num_targets=num_targets,
        learning_rate=lr,
        weight_decay=weight_decay,
        flatten=False,
        predictor_training_epochs=predictor_training_epochs,
    )


def ssl_ckpt_dir(encoder: str, source: str, seed: int, method: str = "lfr") -> Path:
    return SSL_CKPT_BASE / method / encoder / source / f"seed{seed}"


def pretrain_single(
    encoder: str,
    source: str,
    seed: int,
    num_projectors: int,
    num_targets: int,
    lr: float,
    weight_decay: float,
    predictor_training_epochs: int,
    max_epochs: int,
    num_workers: int,
    force: bool = False,
) -> None:
    """Pré-treina e salva o backbone de uma combinação (encoder, fonte, seed)."""
    out_dir = ssl_ckpt_dir(encoder, source, seed)
    ckpt_path = out_dir / "backbone.ckpt"
    if ckpt_path.exists() and not force:
        print(f"[skip] backbone já existe: {ckpt_path}", flush=True)
        return

    seed_everything(seed, workers=True)
    model = build_lfr(encoder, num_projectors, num_targets, lr, weight_decay,
                      predictor_training_epochs)
    dm = PretrainDataModule(source, seed, batch_size=BATCH_SIZE, num_workers=num_workers)

    log_dir = SSL_LOGS_ROOT / encoder / source / f"seed{seed}"
    logger = CSVLogger(save_dir=str(log_dir.parent), name=log_dir.name)
    # Sem EarlyStopping e sem ModelCheckpoint: o protocolo do paper treina até o
    # fim e mantém o último estado (salvo manualmente abaixo).
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        accelerator="auto",
        devices=1,
        deterministic="warn",
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    print(
        f"\n=== [LFR {encoder}] fonte={source} semente={seed} "
        f"K={num_projectors}->{num_targets} lr={lr} wd={weight_decay} "
        f"max_epochs={max_epochs} (backbone ~{max_epochs // (predictor_training_epochs + 1)}) "
        f"-> {ckpt_path} ===",
        flush=True,
    )
    trainer.fit(model, datamodule=dm)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), ckpt_path)
    print(f"[OK] backbone salvo -> {ckpt_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", choices=ENCODERS, nargs="+", default=ENCODERS)
    parser.add_argument("--source", choices=SOURCES, nargs="+", default=SOURCES)
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--num-projectors", type=int, default=DEFAULT_NUM_PROJECTORS)
    parser.add_argument("--num-targets", type=int, default=DEFAULT_NUM_TARGETS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--predictor-training-epochs", type=int,
                        default=DEFAULT_PREDICTOR_TRAINING_EPOCHS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS,
                        help="Épocas do Trainer; efetivas de backbone = max_epochs / "
                             "(predictor_training_epochs + 1).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-treina mesmo se o backbone existir.")
    parser.add_argument(
        "--spike", action="store_true",
        help=f"Validação rápida: força {SPIKE_BACKBONE_EPOCHS} épocas efetivas de backbone.",
    )
    args = parser.parse_args()

    max_epochs = args.max_epochs
    if args.spike:
        max_epochs = SPIKE_BACKBONE_EPOCHS * (args.predictor_training_epochs + 1)

    encoders: Iterable[str] = args.encoder
    sources: Iterable[str] = args.source
    seeds: List[int] = args.seed

    for encoder in encoders:
        for source in sources:
            for seed in seeds:
                pretrain_single(
                    encoder=encoder,
                    source=source,
                    seed=seed,
                    num_projectors=args.num_projectors,
                    num_targets=args.num_targets,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    predictor_training_epochs=args.predictor_training_epochs,
                    max_epochs=max_epochs,
                    num_workers=args.num_workers,
                    force=args.force,
                )


if __name__ == "__main__":
    main()
