# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fed-har` is a research project exploring **federated learning for Human Activity Recognition (HAR)**. Work is primarily done in Jupyter notebooks using the `minerva-ml` framework on the DAGHAR dataset.

## Environment Setup

```bash
# Install dependencies
poetry install

# Download the DAGHAR dataset
wget "https://zenodo.org/records/13987073/files/standardized_view.zip?download=1" -O daghar_standardized_view.zip
mkdir -p datasets/DAGHAR
unzip -o daghar_standardized_view.zip -d datasets/DAGHAR/
rm daghar_standardized_view.zip
```

## Running Notebooks

```bash
# Start Jupyter in the notebooks/ directory
poetry run jupyter notebook notebooks/
```

Notebooks add the project root to `sys.path` manually at the top — this is the expected pattern since there is no installable source package.

## Architecture

### Data Pipeline (via `minerva-ml`)

The stack is: `CSVReader` → `SimpleDataset` → `MinervaDataModule` → `L.Trainer`

- **`CSVReader`** (`minerva.data.readers.csv_reader`): reads columns from CSVs using glob patterns (e.g., `"accel-x-*"` selects all accelerometer-x timestep columns).
- **`SimpleDataset`** (`minerva.data.datasets.base`): pairs a list of readers with per-reader transforms.
- **`MinervaDataModule`** (`minerva.data.data_modules.base`): PyTorch Lightning `DataModule` wrapping train/val/test datasets.

### Input Format

Each sample is a window of IMU sensor data reshaped to `(6, 60)` — 6 channels (accel-x/y/z + gyro-x/y/z) × 60 timesteps. The label column is `"standard activity code"` (int64, 0–5).

Activity labels: `{0: Sit, 1: Stand, 2: Walk, 3: Stair-up, 4: Stair-down, 5: Run}`

### Model

`ResNetSE1D_5` (`minerva.models.nets.time_series.resnet`): 1D ResNet with Squeeze-and-Excitation blocks (~127K params).

- `model.backbone` — feature extractor (outputs shape `(B, 64, 1)` after global avg pool; squeeze to `(B, 64)`)
- `model.fc` — final linear classifier
- To extract backbone features: `model.backbone(x).view(x.size(0), -1)` (note: `model.feature_extractor` does **not** exist)

### Dataset

DAGHAR `standardized_view` contains 6 sub-datasets in `datasets/DAGHAR/standardized_view/`:
`UCI`, `MotionSense`, `KuHar`, `WISDM`, `RealWorld_thigh`, `RealWorld_waist`

Each has `train.csv`, `validation.csv`, and `test.csv` with 728 columns (6 channels × ~60 timesteps + metadata + label).
