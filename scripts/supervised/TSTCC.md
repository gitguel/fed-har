# Supervised baseline — **TS-TCC Encoder (HARSCnn)**

Script: [`train_tstcc.py`](train_tstcc.py)
Encoder: `minerva.models.nets.lfr_har_architectures.HARSCnnEncoder`
(o "TS-TCC Encoder" do paper de benchmark = YAML `lfr_default`/`tfc_harcnn`,
CNN 1D de 3 blocos adaptada de Eldele et al., TS-TCC)

## Protocolo

Replica o pipeline supervisionado da Seção V-D do paper de benchmark DAGHAR
(da Luz et al., IEEE Access 2026). Cada execução do script roda a grade
**6 datasets × 4 seeds = 24 treinos** (configurável via CLI).

| Item                  | Valor                                                                     |
| --------------------- | ------------------------------------------------------------------------- |
| Datasets              | `UCI`, `MotionSense`, `KuHar`, `WISDM`, `RealWorld_thigh`, `RealWorld_waist` |
| Sementes              | `0, 1, 2, 3`                                                                |
| Otimizador            | `Adam` (padrão do `SimpleSupervisedModel`)                                  |
| Taxa de aprendizado   | **`1e-4`** (mesmo lr dos overrides oficiais de todos os encoders)           |
| Tamanho do batch      | `64`                                                                        |
| Épocas máximas        | `100`                                                                       |
| Parada antecipada     | `val_loss`, modo `min`, paciência `50`                                      |
| Função de perda       | `CrossEntropyLoss`                                                          |
| Formato da entrada    | `(6, 60)` — o encoder aceita `(B, C, T)` diretamente (`permute=False`)      |
| Classes de saída      | `6` (padronizado DAGHAR)                                                    |
| Determinismo          | `seed_everything(seed, workers=True)` + `Trainer(deterministic="warn")`     |

## Configuração do HARSCnnEncoder (YAML oficial do benchmark)

| Hiperparâmetro          | Valor   |
| ----------------------- | ------- |
| `dim`                   | `2304`  |
| `input_channel`         | `6`     |
| `inner_conv_output_dim` | `1280`  (128 canais × 10 timesteps após os 3 MaxPool) |
| `permute`               | `False` |

## Arquitetura

```
Input (B, 6, 60)                                (B, C, T)
   │
   ▼
Conv1d(6→32, k=8) → BN → ReLU → MaxPool(2) → Dropout(0.35)
Conv1d(32→64, k=8) → BN → ReLU → MaxPool(2)
Conv1d(64→128, k=8) → BN → ReLU → MaxPool(2)
   │
   ▼
flatten  →  (B, 1280)                           (128 canais × 10 timesteps)
   │
   ▼
Linear(1280 → 2304)                             (interno ao HARSCnnEncoder)
   │
   ▼
(B, 2304)
   │
   ▼
Linear(2304 → 128) → ReLU → Linear(128 → 6)    ← cabeça MLP do paper (Tabela 2)
   │
   ▼
Logits (B, 6)
```

- Como o `HARSCnnEncoder` já produz tensor 2D `(B, 2304)`, o wrapper
  `SimpleSupervisedModel` é instanciado com `flatten=False`.
- A cabeça MLP é a mesma dos outros três scripts (`build_prediction_head`
  em `common.py`).

## Checkpoints salvos (por execução)

Diretório: `checkpoints/supervised/tstcc/{dataset}/seed{N}/`

| Arquivo       | Quando é salvo                                              |
| ------------- | ----------------------------------------------------------- |
| `first.ckpt`  | Ao final da **época 0** (callback `FirstEpochCheckpoint`).  |
| `best.ckpt`   | Sempre que `val_loss` melhora (`ModelCheckpoint`).          |
| `last.ckpt`   | Ao final do treino (`save_last=True`).                      |

## Logs

CSV via `CSVLogger`: `logs/supervised/tstcc/{dataset}/seed{N}/version_0/metrics.csv`.
Inclui `train_loss`, `val_loss`, `train_acc`, `val_acc`, `test_loss`, `test_acc`.

## Uso

```bash
# Grade completa (24 treinos): 6 datasets × 4 sementes
poetry run python scripts/supervised/train_tstcc.py

# Subconjunto (ex.: só KuHar, semente 0)
poetry run python scripts/supervised/train_tstcc.py \
    --dataset KuHar --seed 0 --num-workers 4
```

> **Atenção**: peça ao assistente para rodar via `tmux` (ver `CLAUDE.md`).

## Observação

Escolhido como 4º encoder do projeto (decisão 2026-07-06, Seção 9/10 das notas
internas): é o único backbone além da RNN em que o **LFR** ajuda de forma
consistente no benchmark (+5.0 pp @10-shot / +5.2 pp @100%, full finetuning) e
também ganha com **TF-C** (+8.5 pp @10-shot). Supervisionado puro, o paper
reporta média de 60.0% @10-shot e 76.1% @100% nos 6 datasets. É o encoder com
maior dimensão de representação do projeto (2304-d vs 64/256/320), o que o
torna também o mais caro em comunicação no cenário federado.
