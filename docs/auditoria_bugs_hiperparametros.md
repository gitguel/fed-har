# Auditoria de código e hiperparâmetros vs benchmark (2026-07-07)

*Revisão sistemática feita em 2026-07-07 a pedido do Miguel: (a) caça a bugs em
todo o código experimental; (b) re-auditoria de hiperparâmetros contra o
benchmark oficial (da Luz et al., IEEE Access 2026), cobrindo pré-treino (LFR e
TF-C) e downstream (freeze e fine-tuning). Método: leitura integral de
`scripts/common.py`, `scripts/supervised/*`, `scripts/ssl/*`,
`scripts/federated/*`, `scripts/eval_transfer.py`, `scripts/gpu_pool.py`, das
classes minerva envolvidas (`lfr.py`, `tfc.py` [nets+ssl], `base.py`, `mlp.py`,
`resnet.py`, `cnns.py`, `tnc.py`, `har.py`, `har_rodrigues_24.py`,
`transforms/tfc.py`) e comparação item a item com o repo oficial
`H-IAAC/benchmarking-encoders-ssl-har` (clonado em 2026-07-07: YAMLs de
`base_configs/` + CSVs de overrides de `paper_experiments/`). Checagens
numéricas onde indicado.*

**Veredito geral**: nenhum bug que invalide resultados foi encontrado nos
pipelines SL, LFR, TF-C ou downstream. A configuração replica o benchmark em
todos os itens materiais; os desvios existentes são conhecidos, deliberados e
estão listados na §3 com avaliação de impacto. Os achados acionáveis estão na
§1 (o mais relevante é o F1, reprodutibilidade do federado).

---

## 1. Achados (bugs e riscos), por severidade

### F1 — MÉDIO · `run_federated.py` não semeia o run (só o particionamento)

Não há `seed_everything(args.seed)` no `main()`. O `--seed` controla apenas
`make_client_datasets` (shards) — a inicialização do modelo global
(`build_strategy` → `BUILD_MODEL[...]()`), a inicialização dos modelos dos
clientes e o `shuffle` dos DataLoaders locais ficam com RNG arbitrário do
processo.

- **Impacto científico**: baixo — a grade tem 4 seeds e a aleatoriedade extra é
  estatisticamente inócua (vira parte da variância entre runs). Os resultados
  já medidos continuam válidos.
- **Impacto prático**: um run federado **não é reproduzível bit a bit**, nem
  re-executável para depuração. Além disso dois runs com o mesmo `--seed` e
  cenários diferentes partem de pesos iniciais diferentes (no centralizado,
  `run_single` semeia tudo e o par é controlado).
- **Correção sugerida** (1 linha, antes de `make_client_datasets`):
  `seed_everything(args.seed, workers=True)`. Nota: os clientes rodam em
  workers Ray (processos separados) — para semear treino local por completo
  seria preciso semear dentro de `FlowerClient.__init__`/`fit` (ex.:
  `torch.manual_seed(seed * 1000 + int(cid))`). Se for corrigir, fazer **antes
  da grade da Fase 4**, e não misturar runs com/sem seed no mesmo cache.

### F2 — BAIXO · `client.py`: `drop_last=True` com peso FedAvg pelo shard inteiro

O loader local usa `drop_last=True`, mas `fit()` devolve
`self.num_examples = len(train_dataset)`. Consequências: (a) até 63 amostras
por cliente/época ficam de fora do gradiente mas contam no peso da agregação —
desprezível com shards de milhares; (b) armadilha latente: um shard com <64
amostras produziria um loader **vazio** e o cliente devolveria os pesos
recebidos sem nenhum treino, silenciosamente. Não ocorre com DAGHAR nos
cenários 1–8 atuais; vira risco real se um dia particionar por usuário
(cross-device). Sugestão: `assert len(self.loader) > 0` no `__init__`.

### F3 — BAIXO · `few_shot_indices` aceita classe com menos de `n` amostras

O nosso código usa "todas as disponíveis" quando uma classe tem menos que
`n_per_class`; o DataModule do benchmark levanta `ValueError`. **Verificado
numericamente** (train.csv dos 6 datasets): os splits são balanceados e a menor
classe presente tem 232 amostras (KuHar) ⇒ para shots ∈ {1, 10, 100} os dois
comportamentos são idênticos hoje. Classes **ausentes** (UCI sem "Run", WISDM
sem escadas) são tratadas igual nos dois códigos (agrupa-se só o que existe).
Manter como está é razoável; apenas registrar que a célula shots=100 nunca
disparou o fallback.

### F4 — NOTA · FedAvg agrega buffers de BatchNorm

`get_parameters` serializa `state_dict().values()` — inclui `running_mean/var`
dos BN, que são então **mediados** pelo FedAvg. É a escolha padrão da
literatura (e é exatamente o que o FedBN critica), mas deve ser dito no artigo,
até porque "FedBN-SSL" está na lista de variantes de baixo custo. O custo de
comunicação reportado (`uplink_bytes`) também inclui os buffers —
coerente com o que é de fato transmitido.

### F5 — NOTA · TF-C: transform executa 2× por step e também no eval

`TFC_Model.training_step` chama `self.transform(x)` e depois
`TFC_Backbone.forward` chama `self.transform` de novo internamente (para obter
o ramo de frequência). As operações aleatórias (jitter, máscaras) rodam em CPU
e consomem stream de RNG sem afetar a saída usada. No **eval/downstream** o
forward também roda o transform (a saída usada é determinística: `x` limpo +
`fft(x)`), só custo extra de CPU. Herdado do minerva — o benchmark pagou o
mesmo custo; não tocar (mudaria a paridade).

### F6 — COSMÉTICO · `torch.load` sem `weights_only=True`

`downstream_eval.py:121` e `eval_transfer.py:72` emitem `FutureWarning`. Os
checkpoints são nossos (sem risco real). Trocar para
`torch.load(ckpt, map_location="cpu", weights_only=True)` quando conveniente —
os dois pontos carregam só `state_dict`s puros, deve funcionar direto.

### O que foi procurado e NÃO encontrado

- Vazamento de estado entre células da grade downstream: **não há** — o
  `Probe` é reconstruído por (protocolo × regime), com `seed_everything`
  antes, e o backbone é recarregado do checkpoint a cada vez.
- Vazamento de teste: **não há** — ES monitora `val_loss` da fonte; os test
  sets só entram no `evaluate` final. No pré-treino, train+val sem rótulos
  (protocolo do benchmark), teste nunca.
- Erro de época efetiva do LFR: **não há** — `on_train_epoch_start` congela o
  backbone quando `epoch % 6 != 0` ⇒ 600 épocas de Trainer = 100 de backbone,
  como o override oficial `train_for_600_epochs`.
- Restauração do melhor estado no downstream: correta (deep-copy do
  `state_dict` em CPU no melhor `val_loss`; semântica igual ao `best.ckpt` +
  `evaluate` do benchmark). A parada (`epoch - best_epoch >= patience`)
  equivale à `EarlyStopping(patience=50)` do Lightning.
- Contaminação de cache: chaves corretas (`drop_duplicates(subset=KEY,
  keep="last")`); modo `--pretrain-source` grava em arquivo separado.
- Diferença de cabeça: `build_prediction_head` = `Linear(enc,128) → ReLU →
  Linear(128,6)`; o `MLP([enc,128,6])` do minerva gera **exatamente** a mesma
  sequência (verificado no código da classe).

---

## 2. Auditoria de hiperparâmetros vs benchmark

Fontes oficiais citadas: `base_configs/models/{train,finetune,supervised}/*.yaml`,
`base_configs/pipelines/har/*.yaml`, `base_configs/data_modules/**/config*.yaml`
e `paper_experiments/to_be_validated/<téc>/<run>/configs/overrides/*.csv`.

### 2.1 Supervisionado (baseline)

| Item | Benchmark | Nosso | Veredito |
|---|---|---|---|
| Otimizador | Adam (default `SimpleSupervisedModel`), sem wd, betas default | idem (usamos a própria classe) | ✅ |
| LR | 1e-4 (Tabela 12; overrides `full_finetune_lr4`) | `BEST_LR` = 1e-4 nos 4 encoders | ✅ |
| Batch | 64 (`config_0.yaml`) | 64 | ✅ |
| Épocas / ES | 100, ES paciência 50 em `val_loss`, best ckpt (`pipelines/har/train.yaml`) | idem (`common.build_callbacks`) | ✅ |
| Cabeça | `MLP [enc,128,6]` | `build_prediction_head` (idêntica) | ✅ |
| Backbones | `_ResNet1D(avg_pooling=True)+SE`, `CNN_PF_Backbone(include_middle=True)`, `RnnEncoder(GRU,100,320,bi,permute)`, `HARSCnnEncoder(2304,1280)` | `ResNetSE1D_5` (mesmo `_ResNet1D`), `CNN_PFF_2D` (mesmo backbone), `RnnEncoder` idem, `HARSCnnEncoder` idem | ✅ |
| Few-shot | `samples_per_class` estratificado, semente fixa (42 ou 43 conforme o run!) | estratificado, semente = seed do run | ⚠️ desvio D2 (§3) |
| Métricas | torchmetrics acc + F1-macro (`evaluate.yaml`) | sklearn acc + F1-macro | ✅ equivalente |

### 2.2 Pré-treino LFR

| Item | Benchmark (`train/lfr_*.yaml` + overrides) | Nosso (`pretrain_lfr.py`) | Veredito |
|---|---|---|---|
| Projetores | `LFR_HAR_Projector_List`, size=60, encoding=enc_dim, middle=544 | idem | ✅ |
| Preditores | `LFR_HAR_Predictor_List`, size=60, num_layers=1 (middle 128 ignorado) | idem | ✅ |
| Seleção | `num_targets=6` via DPP no `setup` (128 amostras do train) | idem (mesma classe) | ✅ |
| Loss | `loss_fn: null` ⇒ `BatchWiseBarlowTwinLoss` | default da mesma classe | ✅ |
| Otimizador | Adam lr=3e-4, wd=3e-4, betas=(0.9, 0.99) | idem | ✅ |
| Alternância | `predictor_training_epochs=5` | idem | ✅ |
| Épocas | `train_without_early_stopping` + override `max_epochs=600` (= 100 de backbone), último estado | 600, sem ES/ckpt, salva último backbone | ✅ |
| Batch / drop_last | 64 / False (`for_pretrain/config_6_classes.yaml`) | 64 / False | ✅ |
| Corpus | view `rodrigues_2024` (`journal_backbones/for_pretrain`), não rotulada, `use_val_with_train=True`, **sem `combined`** | `standardized_view` train+val; fontes = 6 datasets + `combined` (extensão nossa) | ⚠️ desvio D1 (§3) |

### 2.3 Pré-treino TF-C

| Item | Benchmark (`train/tfc_*.yaml` + código minerva) | Nosso (`pretrain_tfc.py`) | Veredito |
|---|---|---|---|
| Backbone | `TFC_Backbone(single_encoding_size=128)`, encoders tempo/freq **gêmeos não compartilhados**, projetores sondados | idem (`build_tfc_backbone`) | ✅ |
| `num_classes` | YAML passa 6 com `pred_head: null` | passamos `None` | ✅ equivalente — verificado no código: com `pred_head=null`, `num_classes` só criaria `test_metrics` nunca usadas; a loss e o treino são idênticos |
| Loss | `NTXentLoss_poly(device, batch=64, temp=0.2, cosine=True)`; composição `lam=0.2` (hardcoded na classe) | default da mesma classe | ✅ |
| Otimizador | Adam lr=3e-4 (default da classe), wd=3e-4, betas=(0.9,0.99) hardcoded | idem | ✅ |
| Épocas | `train_without_early_stopping`, `max_epochs=100` (override `full_run`), último estado | 100, sem ES | ✅ |
| Batch / drop_last | 64 / **True** (default do `HARDataModuleCPC`, não sobrescrito no config TF-C) | 64 / True | ✅ — o que tratávamos como "desvio necessário" na verdade **coincide com o benchmark** |
| Corpus | view `rodrigues_2024/no_overlap_daghar_standardized_balanced` (≠ da view do LFR!) | `standardized_view` train+val | ⚠️ desvio D1 (§3) |

### 2.4 Downstream (freeze e full fine-tuning) — LFR e TF-C

| Item | Benchmark (`finetune/*.yaml` + `overrides/models.csv`) | Nosso (`downstream_eval.py`) | Veredito |
|---|---|---|---|
| Cabeça | `MLP [enc,128,6]` (TF-C: `[256,128,6]`) | `build_prediction_head` (idêntica; TF-C enc_dim=256) | ✅ |
| Carga do backbone | `FromPretrained(strict=True)` do ckpt de pré-treino | `load_state_dict(strict=True)` | ✅ |
| LR | **1e-4 nos dois protocolos** (`full_finetune` e `freeze`) p/ resnetse5, cnnpff, rnn e harcnn(=tstcc) | 1e-4 nos dois | ✅ |
| Freeze | `freeze_backbone=True` = só `requires_grad=False`; BN do backbone segue atualizando em modo train | réplica explícita da mesma semântica | ✅ |
| Otimizador | Adam default (sem wd, betas default) | idem | ✅ |
| Épocas / ES / seleção | 100, ES paciência 50 em `val_loss`, avalia o `best.ckpt` | 100, ES 50, restaura melhor estado em memória | ✅ |
| Batch / drop_last | 64 / False | 64 / False (default do DataLoader) | ✅ |
| Regimes | samples_per_class ∈ {1,5,10,25,50,100,200} + `perc_100` | {1,10,100} + `full` | ✅ subconjunto compatível (full ↔ perc_100; nas 3 validações contra o `performance_data.json`, nosso `full` casa com a coluna "1000/100%" com MAE ≤ 2.5 pp) |
| Alvo da avaliação | test do **mesmo** dataset (in-domain) | in-domain + cross-dataset (extensão nossa; o gate usa só in-domain) | ✅ |

### 2.5 Curiosidades encontradas no repo oficial (não nos afetam, mas convém saber)

- `lfr_rnn_run3` usou `learning_rate=1e-3` no downstream (run1/run2 usaram
  1e-4). Todos os runs de ts2vec (fora do nosso escopo) usam 1e-3. Ou seja: o
  próprio benchmark tem variação de lr entre re-runs da mesma célula; qual run
  alimentou o `performance_data.json` não é identificável pelo repo. Nossa
  validação por MAE (2–5 pp) já absorve essa incerteza.
- A semente do subset few-shot muda entre runs oficiais (42 no run do TF-C,
  43 no do LFR) — reforça que fixar a MESMA semente deles não é nem possível
  nem necessário (ver D2).
- As views de pré-treino diferem entre técnicas no próprio benchmark
  (LFR: `journal_backbones/for_pretrain`; TF-C:
  `rodrigues_2024/no_overlap_daghar_standardized_balanced`).

---

## 3. Desvios deliberados vs benchmark (para a seção de método do artigo)

**D1 — Corpus de pré-treino** (único desvio material). Usamos
`standardized_view` train+val da fonte; o paper usa views não balanceadas da
família `rodrigues_2024`. Justificativa: é o mesmo corpus dos nossos baselines
SL e do federado (comparações internas limpas), e o efeito medido é pequeno —
os gates in-domain (LFR 3 encoders; SL tstcc; LFR tstcc em 2026-07-07: viés
+0.4..+2.4 pp, MAE 1.7–2.5 pp) mostram replicação fiel apesar dele. Além
disso, pré-treinamos também no `combined` (o benchmark não tem essa fonte).

**D2 — Semente do subset few-shot**. Benchmark: subset fixo (semente 42/43)
compartilhado por todas as células; nós: subset re-sorteado por seed do run
(0–3). O nosso desenho captura a variância de amostragem do regime few-shot
(4 subsets × 4 inits) — estatisticamente mais informativo; o deles congela o
subset. Consequência: nossos desvios-padrão @1/10-shot tendem a ser maiores
que os do paper — não é ruído de implementação.

**D3 — Seleção do melhor estado em memória** em vez de `best.ckpt` em disco
(downstream). Matematicamente idêntico; só evita I/O.

**D4 — Métricas via sklearn** (`accuracy_score`, `f1_score(average="macro")`)
em vez de torchmetrics. Definições idênticas; o `evaluate.yaml` oficial também
reporta F1-macro.

**D5 — `deterministic="warn"` sem `cudnn.benchmark`** (o pipeline oficial usa
`benchmark: True`). Afeta só velocidade/determinismo, não a estatística.

## 4. Fidelidade "ao benchmark" ≠ fidelidade aos papers originais

Herdamos do minerva (de propósito — é o que torna nossos números comparáveis
ao benchmark) algumas divergências em relação aos métodos originais. **Para o
artigo**: descrever os métodos como "a implementação do benchmark
(minerva-ml)", citando os papers originais como origem das técnicas, sem
afirmar que a implementação segue Zhang et al. / Sui et al. ao pé da letra.

1. **TF-C**: a augmentação de frequência (`DataTransform_FD`, add/remove de
   componentes) é computada e **nunca usada** — o par contrastivo do ramo de
   frequência é `fft(x)` vs `fft(jitter(x))` (FFT da augmentação temporal), e
   não uma augmentação nativa do espectro como no paper original.
2. **TF-C**: o esquema de máscara das augmentações zera ~3/4 das amostras
   aumentadas no ramo temporal (sorteio `randint(0,4)` herdado do código
   original dos autores do TF-C — a peculiaridade vem de lá).
3. **TF-C**: `loss_t`/`loss_f` contrastam `h` (pré-projetor) e as perdas de
   consistência contrastam `z` (pós-projetor), com `lam=0.2` fixo.
4. **LFR**: o DPP seleciona os 6 **projetores**, mas os preditores usados são
   simplesmente os 6 primeiros da lista (não os pareados aos índices do DPP).
   Como preditores são inicializados aleatoriamente e treináveis, o efeito é
   nulo na prática — mas é bom saber que existe.
5. **Freeze**: "congelar" = `requires_grad=False` apenas; BN do backbone segue
   atualizando estatísticas com os dados do downstream (semântica do
   `SimpleSupervisedModel`). Relevante ao interpretar o linear readout.

## 5. Recomendações (mínimas, em ordem de prioridade)

1. **Antes da Fase 4 (grade federada do tstcc)**: adicionar
   `seed_everything(args.seed, workers=True)` no `main()` de
   `run_federated.py` (F1). Os 96 runs antigos permanecem válidos; anotar no
   notebook federado que os runs do tstcc têm semeadura completa (ou re-rodar
   os 96 se quiser homogeneidade total — ~40 min nas 8 GPUs).
2. `assert len(self.loader) > 0` no `FlowerClient.__init__` (F2).
3. `weights_only=True` nos dois `torch.load` (F6), quando mexer nos arquivos.
4. Nenhuma mudança nos pipelines SL/LFR/TF-C/downstream: estão fiéis ao
   benchmark e os três gates quantitativos confirmam. Não "consertar" as
   divergências da §4 — elas são a base da comparabilidade.
