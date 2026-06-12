# Supervised baseline — **CNN-PFF**

Script: [`train_supervised_cnnpff.py`](train_supervised_cnnpff.py)
Encoder: `minerva.models.nets.time_series.cnns.CNN_PFF_2D`

## Protocolo

Replica o pipeline supervisionado da Seção V-D do paper de benchmark DAGHAR
(da Luz et al., IEEE Access 2026). Cada execução do script roda a grade
**6 datasets × 4 seeds = 24 treinos** (configurável via CLI).

| Item                  | Valor                                                                     |
| --------------------- | ------------------------------------------------------------------------- |
| Datasets              | `UCI`, `MotionSense`, `KuHar`, `WISDM`, `RealWorld_thigh`, `RealWorld_waist` |
| Sementes              | `0, 1, 2, 3`                                                                |
| Otimizador            | `Adam` (padrão do `SimpleSupervisedModel`)                                  |
| Taxa de aprendizado   | **`1e-4`** (Tabela 12 — melhor LR para CNN-PFF: 63,4 % acc)                 |
| Tamanho do batch      | `64`                                                                        |
| Épocas máximas        | `100`                                                                       |
| Parada antecipada     | `val_loss`, modo `min`, paciência `50`                                      |
| Função de perda       | `CrossEntropyLoss`                                                          |
| Formato da entrada    | `(6, 60)` — o backbone faz `.unsqueeze(1)` internamente para `(1, 6, 60)`   |
| Classes de saída      | `6` (padronizado DAGHAR)                                                    |
| Determinismo          | `seed_everything(seed, workers=True)` + `Trainer(deterministic="warn")`     |

## Arquitetura

```
Input (B, 6, 60)
   │
   ▼
.unsqueeze(1)  →  (B, 1, 6, 60)              (interno ao backbone)
   │
   ▼
CNN_PFF_Backbone   (upper + middle + lower partial weight-sharing convs)
   │                + shared conv + maxpool + flatten
   ▼
(B, 768)                                      (paper Tabela 2)
   │
   ▼
Linear(768 → 128) → ReLU → Linear(128 → 6)   ← cabeça MLP do paper (Tabela 2)
   │
   ▼
Logits (B, 6)
```

- O `fc` default da `CNN_PFF_2D` é `Linear(in→512) + ReLU + Dropout(0.5) +
  Linear(512→num_classes)`. Ele é **substituído** pelo MLP de 2 camadas
  com hidden 128 do paper.
- A dimensão de entrada da cabeça (768) é calculada automaticamente via
  `_calculate_fc_input_features` e exposta em `model.fc_input_channels`.

## Checkpoints salvos (por execução)

Diretório: `checkpoints/supervised/cnnpff/{dataset}/seed{N}/`

| Arquivo       | Quando é salvo                                              |
| ------------- | ----------------------------------------------------------- |
| `first.ckpt`  | Ao final da **época 0** (callback `FirstEpochCheckpoint`).  |
| `best.ckpt`   | Sempre que `val_loss` melhora (`ModelCheckpoint`).          |
| `last.ckpt`   | Ao final do treino (`save_last=True`).                      |

## Logs

CSV via `CSVLogger`: `logs/supervised/cnnpff/{dataset}/seed{N}/version_0/metrics.csv`.
Inclui `train_loss`, `val_loss`, `train_acc`, `val_acc`, `test_loss`, `test_acc`.

## Uso

```bash
# Grade completa (24 treinos): 6 datasets × 4 sementes
poetry run python scripts/train_supervised_cnnpff.py

# Subconjunto (ex.: só WISDM, sementes 2 e 3)
poetry run python scripts/train_supervised_cnnpff.py \
    --dataset WISDM --seed 2 3 --num-workers 4
```

> **Atenção**: peça ao assistente para rodar via `tmux` (ver `CLAUDE.md`).

## Observação

O CNN-PFF é o encoder vencedor quando combinado com SSL TF-C
(`76,9 ± 14,6 %` no paper). Como *baseline* supervisionado puro, fica em
posição intermediária (`63,4 %` médio). Use estes checkpoints como ponto
de partida para experimentos futuros que combinem CNN-PFF + TF-C federado.
