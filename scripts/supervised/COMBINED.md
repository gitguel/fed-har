# Supervised baseline — **Combinado (todos os datasets juntos)**

Script: [`train_supervised_combined.py`](train_supervised_combined.py)
Encoders: os mesmos três baselines (`resnetse5`, `cnnpff`, `rnn`).

## Protocolo

Mesma receita supervisionada do benchmark DAGHAR (da Luz et al., IEEE Access
2026) usada nos scripts por encoder, mas treinando cada encoder sobre a **união
dos 6 sub-datasets** em vez de um de cada vez. Cada execução roda a grade
**3 encoders × 4 seeds = 12 treinos** (configurável via CLI).

| Item                  | Valor                                                                       |
| --------------------- | --------------------------------------------------------------------------- |
| Dataset               | `combined` = `ConcatDataset` de UCI + MotionSense + KuHar + WISDM + RealWorld_thigh + RealWorld_waist |
| Encoders              | `resnetse5`, `cnnpff`, `rnn`                                                  |
| Sementes              | `0, 1, 2, 3`                                                                  |
| Otimizador            | `Adam`                                                                        |
| Taxa de aprendizado   | **`1e-4`** (a mesma de cada encoder — Tabela 12)                              |
| Tamanho do batch      | `64`                                                                          |
| Épocas máximas        | `100`                                                                         |
| Parada antecipada     | `val_loss` (união das 6 validações), modo `min`, paciência `50`              |
| Função de perda       | `CrossEntropyLoss`                                                            |
| Formato da entrada    | `(6, 60)` — 6 canais IMU × 60 timesteps                                       |
| Classes de saída      | `6` (padronizado DAGHAR)                                                      |

> Treino/validação/teste são a concatenação dos splits homônimos dos 6 datasets
> (ex.: ~36,8k janelas de treino). A arquitetura (backbone + cabeça MLP do
> paper) é idêntica à dos baselines por dataset — o script reaproveita os
> `build_model` de cada `train_supervised_<encoder>.py`.

## Checkpoints salvos (por execução)

Diretório: `checkpoints/supervised/{encoder}/combined/seed{N}/`

| Arquivo       | Quando é salvo                                              |
| ------------- | ----------------------------------------------------------- |
| `first.ckpt`  | Ao final da **época 0** (callback `FirstEpochCheckpoint`).  |
| `best.ckpt`   | Sempre que `val_loss` melhora (`ModelCheckpoint`).          |
| `last.ckpt`   | Ao final do treino (`save_last=True`).                      |

## Logs

CSV via `CSVLogger`: `logs/supervised/{encoder}/combined/seed{N}/version_0/metrics.csv`.

## Uso

```bash
# Grade completa (12 treinos): 3 encoders × 4 sementes
poetry run python scripts/train_supervised_combined.py

# Subconjunto (ex.: só ResNet-SE-5, sementes 0 e 1)
poetry run python scripts/train_supervised_combined.py \
    --encoder resnetse5 --seed 0 1 --num-workers 4
```

Equivalentemente, os scripts por encoder aceitam `--dataset combined`:

```bash
poetry run python scripts/train_supervised_resnetse5.py --dataset combined
```

## Análise

O modelo combinado entra no notebook
[`notebooks/supervised_training_runner.ipynb`](../notebooks/supervised_training_runner.ipynb)
como uma **fonte extra** (`source = "combined"`) nas análises de transfer
learning: a Seção 4.3 compara, por dataset alvo, a acurácia do generalista
(treinado em tudo) frente ao especialista in-distribution.

> **Atenção**: peça ao assistente para rodar via `tmux` (ver `CLAUDE.md`).
