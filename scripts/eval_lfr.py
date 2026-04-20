"""
Evaluate the LFR-pretrained backbone on MotionSense via full fine-tuning.

Method:
  • lfr_full_finetune — all parameters fine-tuned from LFR pretrained weights

Runs 5 regimes × 4 seeds = 20 runs.
Best model (lowest val_loss) is loaded for testing.

Outputs
-------
  results/lfr_eval_results.csv     — acc, f1, recall, precision per run
  logs/eval/<run_name>/            — full training curves (CSVLogger)
"""

import sys, copy, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, F1Score, Recall, Precision

from minerva.data.readers.csv_reader import CSVReader
from minerva.data.datasets.base import SimpleDataset, Subset
from minerva.data.data_modules.base import MinervaDataModule
from minerva.models.nets.time_series.resnet import ResNetSE1D_5
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.transforms import Reshape, CastTo, TransformPipeline

# ── Config ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "datasets/DAGHAR/standardized_view/MotionSense"
BACKBONE_PATH = ROOT / "checkpoints/lfr/lfr_backbone.pt"
RESULTS_DIR   = ROOT / "results"
EVAL_LOG_DIR  = ROOT / "logs/eval"

NUM_CLASSES  = 6
BACKBONE_DIM = 64
BATCH_SIZE   = 64
EVAL_EPOCHS  = 100
NUM_WORKERS  = 4
SEEDS        = [0, 1, 2, 3]
SPC_REGIMES  = [1, 10, 20, 100]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS    = ["accel-x-*", "accel-y-*", "accel-z-*", "gyro-x-*", "gyro-y-*", "gyro-z-*"]
DATA_TRANSFORM = TransformPipeline([Reshape((6, 60)), CastTo("float32")])

# ── Dataset helpers ────────────────────────────────────────────────────────
def make_dataset(csv_path: Path) -> SimpleDataset:
    return SimpleDataset(
        readers=[
            CSVReader(path=csv_path, columns_to_select=SENSOR_COLS),
            CSVReader(path=csv_path, columns_to_select="standard activity code", cast_to="int64"),
        ],
        transforms=[DATA_TRANSFORM, CastTo("int64")],
    )


def samples_per_class_subset(dataset: SimpleDataset, csv_path: Path, n_per_class: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(csv_path)
    indices = []
    for cls in sorted(df["standard activity code"].unique()):
        cls_idx = df.index[df["standard activity code"] == cls].tolist()
        chosen  = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    return Subset(dataset, sorted(indices))


# ── Backbone / metrics / model ─────────────────────────────────────────────
def load_pretrained_backbone() -> torch.nn.Module:
    if not BACKBONE_PATH.exists():
        raise FileNotFoundError(
            f"Backbone not found at {BACKBONE_PATH}. Run scripts/train_lfr.py first."
        )
    resnet = ResNetSE1D_5(input_shape=(6, 60), num_classes=6)
    resnet.backbone.load_state_dict(torch.load(BACKBONE_PATH, weights_only=True))
    return resnet.backbone


def make_metrics() -> dict:
    return {
        "acc":       Accuracy( task="multiclass", num_classes=NUM_CLASSES),
        "f1":        F1Score(  task="multiclass", num_classes=NUM_CLASSES, average="macro"),
        "recall":    Recall(   task="multiclass", num_classes=NUM_CLASSES, average="macro"),
        "precision": Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
    }


def make_model(backbone: torch.nn.Module, lr: float = 1e-3) -> SimpleSupervisedModel:
    return SimpleSupervisedModel(
        backbone=copy.deepcopy(backbone),
        fc=torch.nn.Linear(BACKBONE_DIM, NUM_CLASSES),
        loss_fn=torch.nn.CrossEntropyLoss(),
        flatten=True,
        freeze_backbone=False,
        learning_rate=lr,
        train_metrics=make_metrics(),
        val_metrics=make_metrics(),
        test_metrics=make_metrics(),
    )


# ── Training helper ────────────────────────────────────────────────────────
def train_and_test(
    model: SimpleSupervisedModel,
    train_dataset,
    val_dataset,
    test_dataset,
    run_name: str,
    max_epochs: int = EVAL_EPOCHS,
) -> dict:
    bs = min(BATCH_SIZE, len(train_dataset))
    dm = MinervaDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=bs,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    logger = CSVLogger(save_dir=str(EVAL_LOG_DIR), name=run_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        )
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=[ckpt_cb],
            log_every_n_steps=1,
            accelerator="auto",
            enable_progress_bar=True,
        )
        trainer.fit(model, dm)

        if ckpt_cb.best_model_path:
            ckpt = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"])

        test_results = trainer.test(model, dm, verbose=False)

    return test_results[0] if test_results else {}


# ── Main ───────────────────────────────────────────────────────────────────
print("Loading datasets …")
full_train_ds = make_dataset(DATA_DIR / "train.csv")
val_ds        = make_dataset(DATA_DIR / "validation.csv")
test_ds       = make_dataset(DATA_DIR / "test.csv")

print(f"Loading pretrained backbone from {BACKBONE_PATH} …")
pretrained_bb = load_pretrained_backbone()

regimes    = [(str(n), n) for n in SPC_REGIMES] + [("all", None)]
total_runs = len(regimes) * len(SEEDS)
run_idx    = 0
records    = []

for regime_label, n_spc in regimes:
    for seed in SEEDS:
        run_idx += 1
        run_name = f"lfr_full_finetune_{regime_label}spc_seed{seed}"
        print(f"\n[{run_idx}/{total_runs}] {run_name}")

        L.seed_everything(seed, workers=True)

        train_ds = (
            full_train_ds if n_spc is None
            else samples_per_class_subset(full_train_ds, DATA_DIR / "train.csv", n_spc, seed)
        )
        print(f"  train={len(train_ds)} | seed={seed}")

        model   = make_model(pretrained_bb)
        results = train_and_test(model, train_ds, val_ds, test_ds, run_name=run_name)

        records.append({
            "method":         "lfr_full_finetune",
            "regime":         regime_label,
            "seed":           seed,
            "train_size":     len(train_ds),
            "test_acc":       round(results.get("test_acc",        float("nan")), 4),
            "test_f1":        round(results.get("test_f1",         float("nan")), 4),
            "test_recall":    round(results.get("test_recall",     float("nan")), 4),
            "test_precision": round(results.get("test_precision",  float("nan")), 4),
        })
        r = records[-1]
        print(f"  acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  "
              f"recall={r['test_recall']:.4f}  precision={r['test_precision']:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
results_df   = pd.DataFrame(records)
results_path = RESULTS_DIR / "lfr_eval_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\n{'='*60}")
print(f"Results saved to {results_path}")
print("\nMean per regime:")
print(results_df.groupby("regime")[["test_acc", "test_f1"]].mean().to_string())
