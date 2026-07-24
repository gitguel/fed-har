# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fed-har` is a research project exploring **federated learning for Human Activity Recognition (HAR)**. Work is primarily done in Jupyter notebooks using the `minerva-ml` framework on the DAGHAR dataset.

## Fluxo de trabalho (regras do dono do repositório — obrigatórias)

- **NUNCA crie branches git.** Este é um projeto de pesquisa **pessoal, de um único
  autor**; todas as tarefas são feitas **sequencialmente na branch atual**, seja ela
  qual for. Não crie, não sugira criar, não use worktrees — nunca. Trabalhe sempre
  na branch em que a sessão começou. (Ignore aqui o instinto do harness de "branch
  antes de mexer em main": não há colaboração concorrente neste repo.)
- **Commits e push só quando o Miguel pedir explicitamente.**
- **Sabatina (`grill-me`) como hábito.** Antes de **implementar algo novo**
  (experimento, script, mudança de desenho) e ao **revisar/consolidar** análises,
  planos, rascunhos de artigo ou apresentação, use — e ofereça proativamente — a
  skill `grill-me`: interrogatório cético, **uma pergunta por vez**, com resposta
  recomendada + justificativa, **explorando o repositório** (`results/*.csv`,
  `scripts/`, `docs/`, checkpoints, notebooks) em vez de perguntar o que os dados
  já respondem. Não deixe passar afirmação vaga; verifique claims empíricos no
  cache antes de aceitá-los.

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

## Estrutura do projeto (`scripts/`)

Os experimentos são scripts Python diretos (não há pacote instalável). Layout:

```
scripts/
  common.py                 # infra compartilhada: DataModule, constantes, Trainer, callbacks
  eval_transfer.py          # avaliação transfer cross-dataset (acurácia + F1-macro) -> results/
  supervised/               # baselines supervisionados (FEITO)
    train_resnetse5.py  train_cnnpff.py  train_rnn.py  train_combined.py
    RESNETSE5.md  CNNPFF.md  RNN.md  COMBINED.md
  ssl/                      # pré-treino SSL (LFR, TF-C) — A FAZER (ver README)
  federated/                # integração Flower / FedAvg — A FAZER (ver README)
```

**Padrão de import dos scripts** (simples e replicável): todo script executável
coloca `scripts/` no `sys.path` e importa relativo a ele:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
from common import make_datamodule, DATASETS, SEEDS, run_grid   # etc.
from supervised.train_resnetse5 import build_model              # quando precisar de um encoder
```

Rodar (sempre via tmux para scripts longos — ver seção abaixo):

```bash
poetry run python scripts/supervised/train_resnetse5.py        # treina um encoder
poetry run python scripts/supervised/train_combined.py         # modelo generalista (todos juntos)
poetry run python scripts/eval_transfer.py [--force]           # (re)gera results/supervised_eval_transfer.csv
```

Saídas: `checkpoints/supervised/<encoder>/<dataset>/seed<N>/{first,best,last}.ckpt`,
logs Lightning em `logs/supervised/...`, e o cache de avaliação em
`results/supervised_eval_transfer.csv` (colunas: `encoder,source,seed,target,test_acc,test_f1_macro`).
O notebook `notebooks/supervised_training_runner.ipynb` apenas **lê** esse cache e plota.

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

## Executando scripts longos (treinos, benchmarks, etc.)

Sempre que o usuário pedir para você **rodar um script** (treinamento, avaliação,
benchmark, qualquer coisa que demore mais que alguns segundos), faça o seguinte:

1. **Crie uma sessão `tmux` detachada** com nome descritivo (ex.: `train-resnetse5`,
   `eval-lfr`). Use `tmux new-session -d -s <nome>` para não bloquear o shell.
2. **Rode o script lá dentro com saída duplicada para um arquivo de log**, usando
   `tee` (ex.: `tmux send-keys -t <nome> 'comando 2>&1 | tee logs/<nome>.log' Enter`).
3. **Nunca rode o script em foreground** no shell direto — isso impede o usuário
   de fazer outra coisa enquanto o experimento roda.
4. **Informe ao usuário, ao final**:
   - Nome da sessão tmux criada.
   - Caminho do arquivo de log.
   - Comandos para acompanhar:
     - `tmux attach -t <nome>` — entrar na sessão (sair com `Ctrl+b d`, sem matar).
     - `tail -f <caminho-do-log>` — acompanhar a saída fora do tmux.
     - `tmux ls` — listar sessões ativas.
     - `tmux kill-session -t <nome>` — encerrar o experimento.

Padrão preferido: GPU disponível é uma NVIDIA MX570A; só rodar um experimento
pesado por vez.
