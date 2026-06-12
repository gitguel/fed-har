# Supervised baseline — **ResNet-SE-5**

Script: [`train_supervised_resnetse5.py`](train_supervised_resnetse5.py)
Encoder: `minerva.models.nets.time_series.resnet.ResNetSE1D_5`

## Protocolo

Replica o pipeline supervisionado da Seção V-D do paper de benchmark DAGHAR
(da Luz et al., IEEE Access 2026). Cada execução do script roda a grade
**6 datasets × 4 seeds = 24 treinos** (configurável via CLI).

| Item                  | Valor                                                                     |
| --------------------- | ------------------------------------------------------------------------- |
| Datasets              | `UCI`, `MotionSense`, `KuHar`, `WISDM`, `RealWorld_thigh`, `RealWorld_waist` |
| Sementes              | `0, 1, 2, 3`                                                                |
| Otimizador            | `Adam` (padrão do `SimpleSupervisedModel`)                                  |
| Taxa de aprendizado   | **`1e-4`** (Tabela 12 — melhor LR para ResNet-SE-5: 70,7 % acc)             |
| Tamanho do batch      | `64`                                                                        |
| Épocas máximas        | `100`                                                                       |
| Parada antecipada     | `val_loss`, modo `min`, paciência `50`                                      |
| Função de perda       | `CrossEntropyLoss`                                                          |
| Formato da entrada    | `(6, 60)` — 6 canais IMU × 60 timesteps (`features_as_channels=True`)       |
| Classes de saída      | `6` (padronizado DAGHAR — datasets com menos classes têm logits não usadas) |
| Determinismo          | `seed_everything(seed, workers=True)` + `Trainer(deterministic="warn")`     |

## Arquitetura

```
Input (B, 6, 60)
   │
   ▼
ResNetSE1D_5.backbone            (_ResNet1D com 5 blocos residuais SE)
   │
   ▼
(B, 64)                          (após global avg pool + squeeze)
   │
   ▼
Linear(64 → 128) → ReLU → Linear(128 → 6)   ← cabeça MLP do paper (Tabela 2)
   │
   ▼
Logits (B, 6)
```

- O `fc` default da `ResNetSE1D_5` é uma única `Linear(64, num_classes)`. Ele é
  **substituído** pelo MLP de 2 camadas com hidden 128 (cabeça padronizada do
  paper, Seção V-C).
- `flatten=True` (default da classe) — não muda nada porque a saída do backbone
  já é `(B, 64)`.

## Checkpoints salvos (por execução)

Diretório: `checkpoints/supervised/resnetse5/{dataset}/seed{N}/`

| Arquivo       | Quando é salvo                                              |
| ------------- | ----------------------------------------------------------- |
| `first.ckpt`  | Ao final da **época 0** (callback `FirstEpochCheckpoint`).  |
| `best.ckpt`   | Sempre que `val_loss` melhora (`ModelCheckpoint`).          |
| `last.ckpt`   | Ao final do treino (`save_last=True`).                      |

## Logs

CSV via `CSVLogger`: `logs/supervised/resnetse5/{dataset}/seed{N}/version_0/metrics.csv`.
Inclui `train_loss`, `val_loss`, `train_acc`, `val_acc`, `test_loss`, `test_acc`.

## Uso

```bash
# Grade completa (24 treinos): 6 datasets × 4 sementes
poetry run python scripts/train_supervised_resnetse5.py

# Subconjunto (ex.: só MotionSense, sementes 0 e 1)
poetry run python scripts/train_supervised_resnetse5.py \
    --dataset MotionSense --seed 0 1 --num-workers 4
```

> **Atenção**: peça ao assistente para rodar via `tmux` (ver `CLAUDE.md`).
