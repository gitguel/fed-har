# Supervised baseline — **RNN (BiGRU)**

Script: [`train_supervised_rnn.py`](train_supervised_rnn.py)
Encoder: `minerva.models.nets.tnc.RnnEncoder` (BiGRU estilo Tonekaboni et al.)

## Protocolo

Replica o pipeline supervisionado da Seção V-D do paper de benchmark DAGHAR
(da Luz et al., IEEE Access 2026). Cada execução do script roda a grade
**6 datasets × 4 seeds = 24 treinos** (configurável via CLI).

| Item                  | Valor                                                                     |
| --------------------- | ------------------------------------------------------------------------- |
| Datasets              | `UCI`, `MotionSense`, `KuHar`, `WISDM`, `RealWorld_thigh`, `RealWorld_waist` |
| Sementes              | `0, 1, 2, 3`                                                                |
| Otimizador            | `Adam` (padrão do `SimpleSupervisedModel`)                                  |
| Taxa de aprendizado   | **`1e-4`** (Tabela 12 — melhor LR para RNN: 52,7 % acc)                     |
| Tamanho do batch      | `64`                                                                        |
| Épocas máximas        | `100`                                                                       |
| Parada antecipada     | `val_loss`, modo `min`, paciência `50`                                      |
| Função de perda       | `CrossEntropyLoss`                                                          |
| Formato da entrada    | `(6, 60)` — `permute=True` no encoder converte para `(T, B, C)`             |
| Classes de saída      | `6` (padronizado DAGHAR)                                                    |
| Determinismo          | `seed_everything(seed, workers=True)` + `Trainer(deterministic="warn")`     |

## Configuração da BiGRU (Tonekaboni et al., adotada no paper)

| Hiperparâmetro     | Valor             |
| ------------------ | ----------------- |
| `hidden_size`      | `100`             |
| `in_channel`       | `6`               |
| `encoding_size`    | `320`             |
| `cell_type`        | `"GRU"`           |
| `num_layers`       | `1`               |
| `dropout`          | `0`               |
| `bidirectional`    | `True`            |
| `permute`          | `True` (aceita `(B, C, T)` do DataModule) |

Saída da BiGRU: concatenação dos dois sentidos no último timestep → `(B, 200)`
→ projeção linear final do `RnnEncoder` (`Linear(200, 320)`) → `(B, 320)`.

## Arquitetura

```
Input (B, 6, 60)                              (B, C, T)
   │
   ▼
RnnEncoder.permute(2, 0, 1)  →  (60, B, 6)    (T, B, C)
   │
   ▼
BiGRU(hidden=100, layers=1, bidirectional=True)
   │
   ▼
out[-1]  →  (B, 200)                          (hidden × 2 directions)
   │
   ▼
Linear(200 → 320)                             (interno ao RnnEncoder)
   │
   ▼
(B, 320)
   │
   ▼
Linear(320 → 128) → ReLU → Linear(128 → 6)   ← cabeça MLP do paper (Tabela 2)
   │
   ▼
Logits (B, 6)
```

- Como o `RnnEncoder` já produz tensor 2D `(B, 320)`, o wrapper
  `SimpleSupervisedModel` é instanciado com `flatten=False`.
- A cabeça MLP é a mesma usada nos outros dois scripts
  (`Linear(enc → 128) → ReLU → Linear(128 → 6)`), construída pelo helper
  `build_prediction_head` em `_supervised_common.py`.

## Checkpoints salvos (por execução)

Diretório: `checkpoints/supervised/rnn/{dataset}/seed{N}/`

| Arquivo       | Quando é salvo                                              |
| ------------- | ----------------------------------------------------------- |
| `first.ckpt`  | Ao final da **época 0** (callback `FirstEpochCheckpoint`).  |
| `best.ckpt`   | Sempre que `val_loss` melhora (`ModelCheckpoint`).          |
| `last.ckpt`   | Ao final do treino (`save_last=True`).                      |

## Logs

CSV via `CSVLogger`: `logs/supervised/rnn/{dataset}/seed{N}/version_0/metrics.csv`.
Inclui `train_loss`, `val_loss`, `train_acc`, `val_acc`, `test_loss`, `test_acc`.

## Uso

```bash
# Grade completa (24 treinos): 6 datasets × 4 sementes
poetry run python scripts/train_supervised_rnn.py

# Subconjunto (ex.: só RealWorld_thigh, semente 0)
poetry run python scripts/train_supervised_rnn.py \
    --dataset RealWorld_thigh --seed 0 --num-workers 4
```

> **Atenção**: peça ao assistente para rodar via `tmux` (ver `CLAUDE.md`).

## Observação

A BiGRU é o encoder mais fraco do paper na configuração supervisionada
(`52,7 %` médio). É útil principalmente como **ponto de referência inferior**
para experimentos federados — ganhos de FL sobre essa arquitetura serão
proporcionalmente maiores.
