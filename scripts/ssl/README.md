# scripts/ssl/ — Pré-treino auto-supervisionado (SSL)

Espaço reservado para os scripts de **pré-treino SSL centralizado** dos 3 encoders
(`resnetse5`, `cnnpff`, `rnn`), seguindo o mesmo padrão direto dos baselines em
`scripts/supervised/`: um script por método, sem hierarquias profundas.

## Plano

| Arquivo (futuro) | Método | Base na Minerva |
|---|---|---|
| `pretrain_lfr.py` | LFR (método principal) | `minerva/models/ssl/lfr.py` |
| `pretrain_tfc.py` | TF-C | `minerva/models/ssl/tfc.py` + `minerva/transforms/tfc.py` (FFT) |
| `linear_readout.py` | Avaliação: backbone congelado + classificador linear | reutiliza `common.make_datamodule` |

## Convenções a manter

- Reutilizar `scripts/common.py` (DataModule, constantes `DATASETS`/`SEEDS`,
  `CHECKPOINTS_ROOT`/`LOGS_ROOT`).
- Salvar os backbones pré-treinados em `checkpoints/ssl/<metodo>/<encoder>/<dataset>/seed<N>/`.
- Avaliar com **acurácia + F1-macro**, reaproveitando a lógica de `scripts/eval_transfer.py`.
- Bootstrap de import idêntico ao dos outros scripts:
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` seguido de `from common import ...`.
