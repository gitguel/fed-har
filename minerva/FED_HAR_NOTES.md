# `minerva/` — Notas para o projeto `fed-har`

> Documento de referência para a implementação de **aprendizado federado para HAR** sobre esta cópia
> local do framework `minerva-ml`. Para o contexto experimental que motiva nossas escolhas, ver
> `../ssl_benchmark_exemple/FED_HAR_NOTES.md` (paper IEEE Access 2026 do mesmo grupo).

---

## 1. O que é esta pasta

Cópia local do pacote **`minerva-ml`** (versão `0.3.8-beta`, ver `__init__.py:1`). Framework baseado em
**PyTorch Lightning** para pesquisa em ML, com forte ênfase em **HAR** e **Self-Supervised Learning**.

- Origem: <https://github.com/discovery-unicamp/Minerva>
- Mantenedores: Gabriel Gutierrez, Otávio Napoli, Fernando Gubitoso Marques, Edson Borin (UNICAMP)
- Volume: **124 arquivos `.py`** organizados em pacotes modulares.
- Também presentes: `README.md` e `README_ROOT.md` (documentação oficial do framework).

A cópia local permite ler/inspecionar o código sem depender da versão instalada, e serve de base para
extensões específicas do `fed-har` (sem modificar o pacote instalado).

---

## 2. Estrutura de subpacotes

| Subpacote        | Função                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `analysis/`      | Métricas (balanced/pixel accuracy), análise de clusters, análise de complexidade × performance. |
| `callback/`      | Callbacks Lightning: `EmbeddingLoggerCallback`, `SpecificCheckpointCallback`.                    |
| `cli/`           | Wrappers de CLI baseados em `jsonargparse` (`cli/experiment.py`).                                |
| `data/`          | Readers (CSV, tabular, numpy, zarr, mdio, png, tiff…), Datasets, DataModules — núcleo do pipeline de dados. |
| `engines/`       | Engines de inferência (`patch_inferencer_engine`, `slidingwindow_inferencer_engine`).            |
| `losses/`        | NT-Xent, NT-Xent-poly, Barlow Twins (batch-wise), Negative Cosine Sim., Dice, weighted Dice, topological. |
| `models/`        | Arquiteturas (`nets/`) + métodos SSL (`ssl/`) + adapters/loaders.                                |
| `optimizers/`    | LARS, schedulers de LR.                                                                          |
| `pipelines/`     | `Pipeline` base + `SimpleLightningPipeline` (entry-point fit/test/predict/evaluate).             |
| `samplers/`      | `RandomDomainSampler` — sampler balanceado por domínio (multi-dataset).                          |
| `schedulers/`    | `WarmupCosineAnnealingLR`.                                                                       |
| `transforms/`    | Aumentações de dados (aleatórias, contrastivas, TFC, perlin, activity-image, etc.).               |
| `utils/`         | Helpers de tensor, posição-embedding, instantiators, deprecated, string ops, output, upsample, typing. |

---

## 3. Pipeline de dados (foco em HAR)

Cadeia padrão usada no projeto:

```
CSVReader → MultiModalSeriesCSVDataset → MultiModalHARSeriesDataModule → L.Trainer
```

- **`MultiModalSeriesCSVDataset`** (`data/datasets/series_dataset.py:12`): lê CSV com colunas
  prefixadas (`accel-x-0`, `accel-x-1`, …) e produz tensores `(C, T)`. Suporta `map_labels`,
  `cast_to`, transforms encadeados.
- **`SeriesFolderCSVDataset`** (mesmo arquivo, linha `216`): variante para casos em que cada amostra é
  um CSV próprio numa pasta.
- **`MultiModalHARSeriesDataModule`** (`data/data_modules/har.py:298`): wrapper Lightning com
  recursos críticos para nossos experimentos:
  - `data_path` aceita **lista de paths** → múltiplos sub-datasets concatenados (perfeito para simular
    silos/clientes federados).
  - `n_domains_per_sample` → ativa o `RandomDomainSampler` para batches balanceados por domínio.
  - `samples_per_class` e `data_percentage` → controle de fração de dados (few-shot).
  - `seed` → amostragem determinística e cumulativa.
  - `map_labels` → renomear classes entre sub-datasets.

- **`MinervaDataModule`** (`data/data_modules/base.py:11`): DataModule genérico, recebe
  `train_dataset/val_dataset/test_dataset` prontos. Útil para construir clientes federados a partir de
  splits arbitrários.

---

## 4. Modelos disponíveis

### 4.1 `models/nets/time_series/` — encoders 1D para sinais IMU

| Modelo                               | Notas relevantes                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `ResNetSE1D_5`, `ResNetSE1D_8`       | ResNet 1D com blocos Squeeze-and-Excitation; `_5` é o já-usado no projeto (~127 K params).             |
| `ResNet1D_5`, `ResNet1D_8`           | Versões sem SE.                                                                                        |
| `CNN_HaEtAl_1D`, `CNN_HaEtAl_2D`     | CNNs do Ha & Choi (2016).                                                                              |
| `CNN_PF_2D`, `CNN_PFF_2D`            | Partial-weight-sharing CNNs (PFF = PF + middle). **CNN-PFF é o melhor encoder do paper para SSL TF-C.** |
| `IMUTransformer`, `IMUCNN`           | Transformer baseado em Shavit & Klein (2021) para IMU; CNN auxiliar.                                   |
| `InceptionTime`                      | Backbone Inception-style para séries temporais.                                                        |
| `TS2VecClassifier`                   | Encoder dilatado do TS2Vec (forte em SSL).                                                             |

Todos derivam de `SimpleSupervisedModel` (`models/nets/base.py:9`), que padroniza
`backbone → adapter → flatten → fc` com loops Lightning prontos (train/val/test/predict). **Atenção:** o
modelo NÃO expõe `feature_extractor`; para extrair features use `model.backbone(x).view(B, -1)`.

### 4.2 `models/nets/lfr_har_architectures.py` — específico do LFR

- `HARSCnnEncoder` (encoder 1D CNN adaptado do paper LFR para entrada `(6, 60)`).
- `LFR_HAR_Projector`, `LFR_HAR_Predictor`, e as `*_List` (módulos repetidos via `RepeatedModuleList`).

### 4.3 `models/ssl/` — métodos de pré-treino auto-supervisionado

| Arquivo                | Classe principal                       | Característica                                                                |
| ---------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `lfr.py`               | `LearnFromRandomnessModel`             | Learn-From-Randomness (Sui et al. 2023). Já em uso no projeto.                |
| `byol.py`              | `BYOL`                                 | Encoder online + de momento, escalonamento cosseno.                          |
| `simclr.py`            | `SimCLR`                               | Contrastivo com NT-Xent + LARS.                                               |
| `simsiam.py`           | `SimSiam`                              | Stop-gradient + preditor (sem amostras negativas).                            |
| `barlowtwins.py`       | `BarlowTwins`                          | Redução de redundância via matriz de covariância.                             |
| `fastsiam.py`          | (variante de SimSiam)                  | Cabeça MLP modular.                                                           |
| `tfc.py`               | `TFC_Model`                            | Time-Frequency Consistency — **melhor SSL do paper para HAR/DAGHAR.**         |
| `tnc.py`               | `TNC`                                  | Temporal Neighborhood Coding.                                                 |
| `cpc.py`               | `CPC`                                  | Contrastive Predictive Coding (encoder + autoregressor + predictors).         |
| `diet.py`              | `DIET`                                 | Datum IndEx as Target (Balestriero 2023) — cada amostra é sua própria classe. |
| `autoencoder.py`       | `Autoencoder`                          | Reconstrução com MSE.                                                         |
| `topological_autoencoder.py` | (topological AE)                 | AE com perda topológica.                                                      |
| `vitmae.py`            | (Masked Auto-Encoder ViT)              | ViT-MAE.                                                                      |

Todos são `L.LightningModule` autossuficientes (training_step/validation_step/configure_optimizers).

---

## 5. Pipelines, samplers, callbacks úteis para FL

- **`SimpleLightningPipeline`** (`pipelines/lightning_pipeline.py:29`): orquestra
  fit/test/predict/evaluate com métricas, model_analysis, salvamento de status YAML, seed,
  cacheamento. Pode servir como base para um `FederatedPipeline` (round-based).
- **`RandomDomainSampler`** (`samplers/domain_sampler.py:8`): batches balanceados por domínio —
  conceito análogo ao de "clientes" no FL. Útil para *centralized baseline* multi-domínio antes do
  split federado.
- **`EmbeddingLoggerCallback`** (`callback/embedding_logger_callback.py:7`): extrai embeddings via
  `backbone`/`encoder` para CSV, suporta análise pós-treino.
- **`SpecificCheckpointCallback`**: salvar checkpoints em épocas/critérios específicos (útil para
  monitorar a evolução do backbone global ao longo dos rounds).

---

## 6. Como isso habilita o setup federado

1. **Cliente federado = `MinervaDataModule` + modelo `L.LightningModule`** — basta enrolar o `fit` do
   `Pipeline` num loop de rounds e adicionar agregação (FedAvg, FedProx, SCAFFOLD…) sobre
   `model.state_dict()`.
2. **Particionamento por domínio "natural"**: cada um dos 6 sub-datasets do DAGHAR (UCI, MotionSense,
   KuHar, WISDM, RealWorld_thigh, RealWorld_waist) pode virar um cliente — `data_path` já aceita lista
   de paths e `RandomDomainSampler` lida com balanceamento por domínio.
3. **SSL federado**: qualquer modelo de `models/ssl/*` pode ser pré-treinado em cada cliente
   (FedSSL). LFR e TF-C são os candidatos mais relevantes (ver paper, §RQ1/RQ6).
4. **Few-shot pós-FL**: `samples_per_class` + `seed` permitem avaliar o backbone agregado em regimes
   1/5/10/25/50/100/200 SPC, replicando o protocolo do paper.

---

## 7. Próximos passos sugeridos

1. **Decidir a particão dos clientes**: por sub-dataset DAGHAR (6 clientes "naturais" non-IID) é a
   opção que melhor combina com o paper.
2. **Escolher framework FL**: Flower é o mais natural sobre Lightning; alternativa caseira via
   `state_dict` averaging usando `SimpleLightningPipeline` por round.
3. **Reusar o pipeline LFR já existente** (`scripts/train_lfr.py`) como template de cliente —
   adicionar serialização/agregação dos pesos do backbone entre rounds.
4. **Replicar o protocolo do paper** para qualquer experimento federado: mesmos seeds (42, 43, 44),
   mesma cabeça MLP (`input → 128 → 6`), mesmos regimes few-shot, mesma métrica (acurácia balanceada).
5. **Aproveitar `RandomDomainSampler`** para construir um *baseline centralizado multi-domínio* antes
   de partir para o federado real — isso isola o efeito de FL versus mero acesso a múltiplos domínios.

---

## 8. Referências rápidas

| Recurso                          | Caminho                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| DataModule HAR principal         | `data/data_modules/har.py:298` (`MultiModalHARSeriesDataModule`)     |
| Dataset série multimodal         | `data/datasets/series_dataset.py:12` (`MultiModalSeriesCSVDataset`)  |
| Encoder ResNet-SE-5              | `models/nets/time_series/resnet.py:263`                              |
| Encoder CNN-PFF                  | `models/nets/time_series/cnns.py:495`                                |
| Encoder IMU Transformer          | `models/nets/time_series/imu_transformer.py:115`                     |
| LFR model                        | `models/ssl/lfr.py:100`                                              |
| TF-C model                       | `models/ssl/tfc.py:16`                                               |
| DIET model                       | `models/ssl/diet.py:12`                                              |
| SimpleLightningPipeline          | `pipelines/lightning_pipeline.py:29`                                 |
| RandomDomainSampler              | `samplers/domain_sampler.py:8`                                       |
| Paper de referência (FED_HAR)    | `../ssl_benchmark_exemple/FED_HAR_NOTES.md`                          |
