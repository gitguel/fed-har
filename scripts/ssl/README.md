# scripts/ssl/ — Pré-treino auto-supervisionado (SSL) centralizado

Pré-treino **SSL centralizado** dos 4 encoders (`resnetse5`, `cnnpff`, `rnn`,
`tstcc`) e avaliação downstream na **mesma matriz de transfer 7×6** dos
baselines supervisionados, seguindo o padrão direto de `scripts/supervised/`.

## Arquivos

| Arquivo | Papel | Estado |
|---|---|---|
| `encoders.py` | `build_backbone`/`build_tfc_backbone` `(encoder) -> (backbone, enc_dim)` reusando os `build_model()` supervisionados | ✅ |
| `pretrain_lfr.py` | **Estágio A** — pré-treino LFR do backbone na fonte (sem rótulos); salva só o backbone | ✅ |
| `pretrain_tfc.py` | **Estágio A (TF-C)** — encoders gêmeos tempo/freq + NT-Xent poly; grade completa validada contra o benchmark (gate 7) | ✅ |
| `downstream_eval.py` | **Estágios B+C** — `--method {lfr,tfc}`; treina cabeça (`linear`/`finetune`) × 4 regimes e avalia nos 6 alvos | ✅ |
| `run_comb2target.py` | Grade comb→target (backbone `combined`, finetune só no alvo) | ✅ |
| `sl_comb2target_eval.py` | **Skyline** do comb→target: mesmo pipeline com backbone do SL-`combined` (`docs/resultados.md §3`) | ✅ |
| `pretrain_fed.py` | **Pré-treino federado SIMULADO** (FedAvg manual, sem Flower; partições `silo`/`iid`/`device-<dataset>`) — ver `docs/plano_fedssl.md §4` | ✅ implementado |

## Pipeline (espelha o supervisionado centralizado)

| Estágio | O que faz |
|---|---|
| **A** (`pretrain_lfr.py`) | LFR pré-treina o backbone na fonte, **train+val, sem rótulos**, sem early stopping (mantém o último estado). Fontes = 6 datasets + `combined`. Grade: 4 enc × 7 fontes × 4 seeds = **112 backbones**. |
| **B** (`downstream_eval.py`) | Treina a cabeça MLP (a mesma do SL) na fonte rotulada, em **2 protocolos** (`linear` = backbone congelado; `finetune` = descongelado) × **4 regimes** `n_shots` ∈ {1, 10, 100 amostras-por-classe, `full`}, com ES (paciência 50) + melhor estado por val_loss. |
| **C** (`downstream_eval.py`) | Avalia (backbone + cabeça) nos `test.csv` dos 6 alvos → acurácia + F1-macro. |

## Uso

```bash
# Spike ponta-a-ponta (rápido)
python scripts/ssl/pretrain_lfr.py --encoder resnetse5 --source combined --seed 0 --spike
python scripts/ssl/downstream_eval.py --encoder resnetse5 --source combined --seed 0 \
    --protocol both --shots all --epochs 3

# Grade completa (rodar via tmux — ver CLAUDE.md; grade grande pede o cluster)
python scripts/ssl/pretrain_lfr.py       # 112 backbones -> checkpoints/ssl/lfr/...
python scripts/ssl/downstream_eval.py    # 5376 linhas   -> results/ssl_lfr_eval_transfer.csv
```

Argumentos principais: `--encoder {resnetse5,cnnpff,rnn,tstcc}`, `--source {combined,<dataset>}`,
`--seed`, `--protocol {linear,finetune,both}`, `--shots {1,10,100,full,all}`, `--force`.

Cobertura real de cada cache: `poetry run python scripts/analysis/cache_status.py`.

## Saídas

- Backbones: `checkpoints/ssl/lfr/<encoder>/<source>/seed<N>/backbone.ckpt`
- Logs Lightning: `logs/ssl/lfr/<encoder>/<source>/seed<N>/...`
- Cache de avaliação (incremental): `results/ssl_lfr_eval_transfer.csv`
  (colunas: `encoder, source, seed, protocol, n_shots, target, test_acc, test_f1_macro`).
  Visualização em `notebooks/ssl_lfr_avaliation.ipynb` (SL vs SSL, data-efficiency).

## Notas de implementação

- **Colisão de nome**: a pasta chama-se `ssl`, que colide com o módulo `ssl` da
  stdlib. Por isso os scripts importam `encoders` diretamente (adicionando
  `scripts/ssl` ao `sys.path`), sem o prefixo `ssl.`. A stdlib continua ganhando
  para `import ssl` (é módulo regular), então nada mais quebra.
- **LFR** (replica os YAMLs `train/lfr_<enc>.yaml` do repo oficial
  [H-IAAC/benchmarking-encoders-ssl-har](https://github.com/H-IAAC/benchmarking-encoders-ssl-har)
  e a Tabela 4 do paper):
  - Projetores `LFR_HAR_Projector_List` (60 candidatos **convolucionais** sobre a
    série bruta `(6, 60)`, saída = `enc_dim`; `middle_dim=544`); preditores
    `LFR_HAR_Predictor_List` de **1 camada** `Linear(enc_dim -> enc_dim)`.
  - `num_targets=6`: seleção DPP escolhe 6 projetores diversos dentre os 60.
  - Adam lr `3e-4`, weight_decay `3e-4`, betas `(0.9, 0.99)`; batch 64;
    `flatten=False` (os 3 backbones de `encoders.py` já devolvem `(B, enc_dim)`;
    o do cnnpff recebe `flatten=True` interno para isso).
  - `predictor_training_epochs=5` → Trainer roda **600 épocas** = 100 épocas
    *efetivas* de backbone (pipeline oficial `train_for_600_epochs`), **sem
    early stopping** (mantém o último estado).
- **Downstream** = mesmo protocolo do baseline SL: cabeça
  `common.build_prediction_head` (`enc_dim -> 128 -> 6`), Adam lr `1e-4` nos dois
  protocolos, até 100 épocas com ES (paciência 50 na val_loss da fonte) e
  avaliação no melhor estado. Congelamento do `linear` = só `requires_grad=False`
  (semântica `SimpleSupervisedModel(freeze_backbone=True)` do benchmark).
- O regime de dados (`n_shots`) usa `common.subsampled_train_loader` /
  `common.few_shot_indices` (estratificado por classe, determinístico por seed),
  o **mesmo** helper do baseline supervisionado.
