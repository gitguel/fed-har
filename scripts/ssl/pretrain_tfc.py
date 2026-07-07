"""Pré-treino SSL centralizado com TF-C (Time-Frequency Consistency).

Estágio A do pipeline SSL para a 2ª técnica do escopo oficial: para cada
(encoder, fonte, semente) pré-treina um `TFC_Backbone` — duas instâncias
independentes do encoder (ramos tempo e frequência) + projetores para 128-d —
**sem usar rótulos**, com a loss NT-Xent poly do TF-C. Salva o `state_dict`
do `TFC_Backbone` completo, reaproveitado no downstream
(`downstream_eval.py --method tfc`).

Replica a configuração oficial do benchmark (da Luz et al., IEEE Access 2026;
`benchmarks/base_configs/models/train/tfc_<enc>.yaml`):

- `TFC_Backbone(single_encoding_size=128)` com time/frequency encoders gêmeos
  (duas construções independentes do mesmo `build_model` — pesos NÃO
  compartilhados). Projetores dimensionados por sondagem do encoder.
- `TFC_Model(num_classes=None, pred_head=None)` => modo SSL; loss default
  NT-Xent poly (temp 0.2, cosine); Adam lr=3e-4, betas (0.9, 0.99), wd=3e-4
  (defaults da classe = o que o YAML usa).
- **100 épocas, sem early stopping** (pipeline `train_without_early_stopping`
  — TF-C NÃO usa as 600 do LFR), batch 64.
- Dados: train+val da fonte (mesmo desvio documentado do LFR: usamos a
  `standardized_view` em vez da view `rodrigues_2024`).

Respostas do spike de integração (2026-07-07, ver logs/spike-tfc.log):
  1. A FFT é calculada DENTRO do `TFC_Backbone.forward` via `TFC_Transforms`
     (fft.fft(x).abs() sobre o batch (B, 6, 60)) — o dataloader padrão serve.
  2. O `state_dict` do `TFC_Backbone` recarrega `strict=True` numa instância
     recém-construída por `build_tfc_backbone` (212 tensores p/ resnetse5).
  3. Saída de inferência do backbone = concat(z_time, z_freq) de 256-d.
  4. Pior caso (tstcc, 2× HARSCnn 2304-d): 7.3M params, ~3.2 s/época no
     KuHar, VRAM pico ~1.3 GB — folga ampla na TITAN Xp.
  5. A NT-Xent poly usa máscara fixa de 2×batch: **quebra com batch parcial**
     => o train loader usa `drop_last=True` (única diferença vs LFR).
  6. RnnEncoder funciona como frequency_encoder (FFT `.abs()` preserva o
     shape (B, 6, 60); `permute=True` continua correto).

Saída:
    checkpoints/ssl/tfc/<encoder>/<source>/seed<N>/backbone.ckpt
    logs/ssl/tfc/<encoder>/<source>/seed<N>/...

Uso:
    # Spike (validação rápida ponta-a-ponta: 3 épocas)
    python scripts/ssl/pretrain_tfc.py --encoder resnetse5 --source KuHar --seed 0 --spike

    # Grade completa (4 encoders × 7 fontes × 4 seeds = 112 backbones)
    python scripts/ssl/pretrain_tfc.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

from common import (  # noqa: E402
    BATCH_SIZE,
    INPUT_CHANNELS,
    INPUT_TIMESTEPS,
    LOGS_ROOT,
    SEEDS,
)
from encoders import ENCODERS, build_tfc_backbone  # noqa: E402
from pretrain_lfr import PretrainDataModule, SOURCES, ssl_ckpt_dir  # noqa: E402

from minerva.models.ssl.tfc import TFC_Model  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração (YAMLs oficiais train/tfc_<enc>.yaml)
# ---------------------------------------------------------------------------
SSL_LOGS_ROOT = LOGS_ROOT.parent / "ssl" / "tfc"  # .../logs/ssl/tfc

DEFAULT_LR = 3e-4                  # default da classe TFC_Model = o do YAML
DEFAULT_MAX_EPOCHS = 100           # sem early stopping (protocolo do paper)
SPIKE_EPOCHS = 3


def build_tfc(encoder: str, lr: float = DEFAULT_LR) -> TFC_Model:
    """Monta o TF-C em modo SSL em torno do backbone gêmeo do encoder."""
    backbone, _ = build_tfc_backbone(encoder)
    return TFC_Model(
        input_channels=INPUT_CHANNELS,
        TS_length=INPUT_TIMESTEPS,
        num_classes=None,          # None + pred_head=None => modo SSL
        pred_head=None,
        batch_size=BATCH_SIZE,     # a NT-Xent poly exige o batch exato
        learning_rate=lr,
        backbone=backbone,
    )


def pretrain_single(
    encoder: str,
    source: str,
    seed: int,
    lr: float,
    max_epochs: int,
    num_workers: int,
    force: bool = False,
) -> None:
    """Pré-treina e salva o TFC_Backbone de uma combinação (encoder, fonte, seed)."""
    out_dir = ssl_ckpt_dir(encoder, source, seed, method="tfc")
    ckpt_path = out_dir / "backbone.ckpt"
    if ckpt_path.exists() and not force:
        print(f"[skip] backbone já existe: {ckpt_path}", flush=True)
        return

    seed_everything(seed, workers=True)
    model = build_tfc(encoder, lr=lr)
    # drop_last=True: a máscara da NT-Xent poly é fixa em 2×batch (spike, item 5).
    dm = PretrainDataModule(source, seed, batch_size=BATCH_SIZE,
                            num_workers=num_workers, drop_last=True)

    log_dir = SSL_LOGS_ROOT / encoder / source / f"seed{seed}"
    logger = CSVLogger(save_dir=str(log_dir.parent), name=log_dir.name)
    # Sem EarlyStopping e sem ModelCheckpoint: o protocolo do paper treina as
    # 100 épocas até o fim e mantém o último estado (salvo manualmente abaixo).
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
        f"\n=== [TF-C {encoder}] fonte={source} semente={seed} lr={lr} "
        f"max_epochs={max_epochs} -> {ckpt_path} ===",
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
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-treina mesmo se o backbone existir.")
    parser.add_argument(
        "--spike", action="store_true",
        help=f"Validação rápida: força {SPIKE_EPOCHS} épocas.",
    )
    args = parser.parse_args()

    max_epochs = SPIKE_EPOCHS if args.spike else args.max_epochs

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
                    lr=args.lr,
                    max_epochs=max_epochs,
                    num_workers=args.num_workers,
                    force=args.force,
                )


if __name__ == "__main__":
    main()
