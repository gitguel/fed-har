"""
Train LFR (Learn From Randomness) on MotionSense for 400 epochs.

Outputs
-------
  checkpoints/lfr/last.ckpt          — Lightning checkpoint (full LFR model)
  checkpoints/lfr/lfr_backbone.pt    — backbone state-dict only (for downstream tasks)
  logs/lfr/version_*/metrics.csv     — epoch-level train_loss / val_loss
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from minerva.data.readers.csv_reader import CSVReader
from minerva.data.datasets.base import SimpleDataset
from minerva.data.data_modules.base import MinervaDataModule
from minerva.models.nets.time_series.resnet import ResNetSE1D_5
from minerva.models.ssl.lfr import LearnFromRandomnessModel
from minerva.models.nets.lfr_har_architectures import (
    LFR_HAR_Projector_List,
    LFR_HAR_Predictor_List,
)
from minerva.transforms import Reshape, CastTo, TransformPipeline

# ── Config ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "datasets/DAGHAR/standardized_view/MotionSense"
LOG_DIR = ROOT / "logs/lfr"
CKPT_DIR = ROOT / "checkpoints/lfr"

BATCH_SIZE = 64
MAX_EPOCHS = 100
NUM_PROJECTORS = 10
ENCODING_SIZE = 64   # must match ResNetSE1D_5.backbone output dim
MIDDLE_DIM_PROJ = 544  # 32 ch * 17 timesteps for (6, 60) input
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

# ── Data ───────────────────────────────────────────────────────────────────
SENSOR_COLS = ["accel-x-*", "accel-y-*", "accel-z-*", "gyro-x-*", "gyro-y-*", "gyro-z-*"]
DATA_TRANSFORM = TransformPipeline([Reshape((6, 60)), CastTo("float32")])


def make_dataset(csv_path: Path) -> SimpleDataset:
    return SimpleDataset(
        readers=[
            CSVReader(path=csv_path, columns_to_select=SENSOR_COLS),
            CSVReader(path=csv_path, columns_to_select="standard activity code", cast_to="int64"),
        ],
        transforms=[DATA_TRANSFORM, CastTo("int64")],
    )


train_ds = make_dataset(DATA_DIR / "train.csv")
val_ds   = make_dataset(DATA_DIR / "validation.csv")
test_ds  = make_dataset(DATA_DIR / "test.csv")

data_module = MinervaDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    name="MotionSense-LFR",
)

# ── Model ──────────────────────────────────────────────────────────────────
resnet = ResNetSE1D_5(input_shape=(6, 60), num_classes=6)

lfr_model = LearnFromRandomnessModel(
    backbone=resnet.backbone,
    projectors=LFR_HAR_Projector_List(
        size=NUM_PROJECTORS,
        encoding_size=ENCODING_SIZE,
        input_channel=6,
        middle_dim=MIDDLE_DIM_PROJ,
    ),
    predictors=LFR_HAR_Predictor_List(
        size=NUM_PROJECTORS,
        encoding_size=ENCODING_SIZE,
        middle_dim=256,
        num_layers=2,
    ),
    learning_rate=LEARNING_RATE,
)

print(f"LFR model ready — backbone params: {sum(p.numel() for p in lfr_model.backbone.parameters()):,}")

# ── Logger & Callbacks ─────────────────────────────────────────────────────
logger = CSVLogger(save_dir=str(LOG_DIR), name="")

checkpoint_cb = ModelCheckpoint(
    dirpath=str(CKPT_DIR),
    filename="lfr-epoch{epoch:03d}-val_loss{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    verbose=True,
)

# ── Training ───────────────────────────────────────────────────────────────
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    logger=logger,
    callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1,
    accelerator="auto",
)

print(f"\nStarting LFR training for {MAX_EPOCHS} epochs on MotionSense …\n")
trainer.fit(model=lfr_model, datamodule=data_module)

# ── Save backbone state-dict for downstream evaluation ────────────────────
CKPT_DIR.mkdir(parents=True, exist_ok=True)
backbone_path = CKPT_DIR / "lfr_backbone.pt"
torch.save(lfr_model.backbone.state_dict(), backbone_path)

print(f"\n✓ Backbone state-dict → {backbone_path}")
print(f"✓ Best LFR checkpoint → {checkpoint_cb.best_model_path}")
print(f"✓ Metrics CSV         → {LOG_DIR}")
