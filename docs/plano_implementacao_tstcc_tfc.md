# Plano de implementação — Encoder TS-TCC + TF-C (fechamento da parte centralizada)

*Escrito em 2026-07-06. Baseado na auditoria dos YAMLs oficiais do
[H-IAAC/benchmarking-encoders-ssl-har](https://github.com/H-IAAC/benchmarking-encoders-ssl-har)
e nas medições das grades já rodadas (LFR v1, comb→target). Complementa a
Seção 10 de `notas_internas_projeto_ssl_federado_har.md`.*

## 0. Como usar este plano

- **Execução estritamente sequencial**: cada fase só começa depois que o
  **critério de aceite (gate)** da anterior passou. Se um gate falhar, o
  problema fica confinado à fase — nada da fase seguinte foi tocado.
- **Commit + push ao final de cada fase** (mensagens sugeridas em cada uma).
  `checkpoints/` e `logs/` estão no `.gitignore` — commits carregam só código,
  `results/*.csv`, notebooks e docs.
- **Grades longas sempre em tmux** com log em `logs/` (padrão do CLAUDE.md).
  Cluster Dl-16: torch 2.5.1+cu118 (TITAN Xp sm_61 — o lock 2.10 não roda).
  GPUs: usar `--gpus` explícito; conferir com `nvidia-smi` antes.
- **Spike antes de grade**: toda fase com grade tem um passo de validação
  ponta-a-ponta barato antes de soltar os jobs. Não pular (foi o que pegou a
  v0 anômala do LFR e validou o comb→target).

## 1. Contexto e estado de partida

Objetivo: fechar a parte centralizada do projeto com:

1. **4º encoder — `tstcc`** (o "TS-TCC Encoder" do paper = `HARSCnnEncoder`,
   dim **2304**, já disponível em `minerva/models/nets/lfr_har_architectures.py`).
   Motivo (Seção 9 das notas): único backbone além da RNN em que o LFR ajuda
   consistentemente (+5.0/+5.2 pp), e também ganha com TF-C (+8.5).
   Precisa entrar em: **SL puro** (full + few-shot + combined), **LFR**,
   **TF-C** e **baseline federado**.
2. **2ª técnica SSL — TF-C** nos 4 encoders (compromisso do plano oficial).
   É a técnica mais forte do benchmark (TF-C+CNN-PFF é a melhor célula do
   paper: 77.0% @10-shot / 86.1% @100%).

Pré-existente e reutilizado sem mudança conceitual: `common.py` (DataModule,
few-shot, cabeça MLP), `eval_transfer.py`, `scripts/ssl/{pretrain_lfr,
downstream_eval,run_all,run_comb2target}.py`, `scripts/gpu_pool.py`,
`scripts/federated/*`, builders de notebook.

## 2. Resumo das fases

| Fase | Entrega | Grade | Custo estimado* |
|---|---|---|---|
| 0 | Pré-condições (push ok, ambiente, GPUs) | — | 15 min |
| 1 | Código SL do `tstcc` + spike | 1 treino curto | ½ dia |
| 2 | Grade SL `tstcc` (112 treinos) + transfer + gate vs paper | 112 jobs | 2–5 h de GPU |
| 3 | LFR no `tstcc` (28 pré-treinos + 28 downstream) + gate vs paper | 56 jobs | 1–3 dias de GPU |
| 4 | `tstcc` no federado (32 runs) + notebook federado | 32 jobs | ~4–8 h de GPU |
| 5 | Spike TF-C (perguntas de integração respondidas) | 1 combo curto | ½–1 dia |
| 6 | Infra TF-C (`pretrain_tfc.py`, `--method` no downstream/run_all) | — | ½–1 dia |
| 7 | Grade TF-C 4 encoders (112 pré-treinos + 112 downstream) + gate | 224 jobs | 1–2 dias de GPU |
| 8 | (Opcional) comb→target para TF-C | 96 jobs | ~3 h de GPU |
| 9 | Notebooks finais, docs, checklist, memória | — | ½ dia |

\* Ver Apêndice C — estimativas ancoradas em medições reais; **medir o 1º job
de cada grade antes de extrapolar**.

---

## Fase 0 — Pré-condições

1. `git status` limpo e `git push` funcionando (commit da grade LFR v1 +
   comb→target já subiu; `.gitignore` com `/checkpoints/` e `/logs/`).
2. `poetry run python -c "import torch; print(torch.cuda.is_available())"`
   → `True` no cluster.
3. `nvidia-smi` → anotar GPUs livres (grades abaixo assumem 4–8).
4. Conferir que os caches atuais existem (o notebook SSL lê todos):
   `results/{supervised,ssl_lfr,ssl_lfr_comb2target}_eval_transfer.csv`.

**Gate 0**: os 4 itens acima OK.

---

## Fase 1 — Encoder `tstcc`: código SL + spike

### 1.1 Criar `scripts/supervised/train_tstcc.py`

Seguir o padrão de `train_rnn.py` (o caso mais parecido: backbone externo +
`SimpleSupervisedModel`). Config exata (espelha o YAML `lfr_default` /
`tfc_harcnn` do benchmark):

```python
HARSCnnEncoder(dim=2304, input_channel=6, inner_conv_output_dim=1280)
# + build_prediction_head(2304)  ->  MLP [2304, 128, 6] (igual ao YAML finetune)
# + SimpleSupervisedModel(..., learning_rate=BEST_LR["tstcc"], flatten=False)
```

⚠️ **Nome do arquivo é contrato**: `run_all_shots.py` monta o comando como
`train_{encoder}.py` — tem que ser exatamente `train_tstcc.py`.

⚠️ Verificar no spike o shape de saída do `HARSCnnEncoder` para `(B, 6, 60)`:
esperado `(B, 2304)` (se vier 3D, ajustar `flatten`).

### 1.2 Registrar o encoder (checklist do Apêndice B, itens SL)

| Arquivo | Mudança |
|---|---|
| `scripts/common.py` | `BEST_LR["tstcc"] = 1e-4` (mesmo lr dos outros 3; overrides oficiais usam 1e-4 p/ tudo — o gate da Fase 2 pega desvio grosseiro) |
| `scripts/eval_transfer.py` | `ENCODERS` + `BUILD_MODEL["tstcc"]` |
| `scripts/supervised/run_all_shots.py` | `ENCODERS` |
| `scripts/supervised/train_combined.py` | import do `build_model` + `ENCODER_BUILDERS["tstcc"]` |
| `scripts/supervised/TSTCC.md` | doc curta no padrão dos outros (RESNETSE5.md etc.) |

### 1.3 Spike

```bash
poetry run python scripts/supervised/train_tstcc.py --dataset KuHar --seed 0 --max-epochs 3
poetry run python scripts/eval_transfer.py --encoder tstcc  # deve achar o ckpt e avaliar
```

**Gate 1**: treino de 3 épocas roda sem erro; `checkpoints/supervised/tstcc/KuHar/seed0/best.ckpt`
existe; `eval_transfer` produz linhas com acc > 40% no próprio KuHar (sanidade,
não performance); `nvidia-smi` durante o treino sem surpresa de VRAM.
Depois do gate: **apagar o ckpt do spike** (3 épocas contaminaria a grade:
o resume por `best.ckpt` pularia o treino real).

**Commit**: `git commit -m "Encoder tstcc (HARSCnnEncoder 2304-d): script SL + registros"`

---

## Fase 2 — Grade SL do `tstcc` + gate contra o paper

### 2.1 Rodar (tmux `sl-tstcc`)

```bash
tmux new-session -d -s sl-tstcc
tmux send-keys -t sl-tstcc 'cd ~/fed-har && poetry run python scripts/supervised/run_all_shots.py \
  --encoder tstcc --shots 1 10 100 full --gpus 0,1,2,3 2>&1 | tee logs/sl-tstcc.log' Enter
```

7 fontes (6 + combined) × 4 seeds × 4 regimes = **112 treinos**; ao final o
próprio runner chama `eval_transfer.py` (cache incremental — só o tstcc é novo).

### 2.2 Gate 2 (validação contra o paper)

O `performance_data.json` do repo oficial tem a linha **Supervised × TS-TCC
Encoder** por dataset/regime. Comparar in-domain (média das seeds, acc) nos
regimes 1/10/100/1000:

- **Aceite**: |viés| ≤ 5 pp e MAE ≤ 8 pp (mesma folga observada nos outros 3
  encoders: viés ±2, MAE 2–5). Referência do paper @10-shot (média 6 datasets):
  **60.0%**; @100%: **76.1%**.
- Se falhar: revisar `flatten`/shape (1.1), lr, e se o head é [2304,128,6].

**Commit**: `git commit -m "Grade SL tstcc (112 treinos) + transfer; results atualizados"`

---

## Fase 3 — LFR no `tstcc`

### 3.1 Registrar em `scripts/ssl/encoders.py`

```python
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
ENC_DIM["tstcc"] = 2304
_BUILDERS["tstcc"] = build_tstcc   # from supervised.train_tstcc import build_model
```

`pretrain_lfr.py` e `downstream_eval.py` são genéricos via `encoders.py` —
**zero mudança neles**.

### 3.2 Spike de VRAM (obrigatório antes da grade)

Os preditores do LFR são `Linear(enc→enc)` × 60 candidatos → para 2304-d são
**~318M parâmetros** (~1.3 GB pesos + ~2.6 GB estados do Adam). Deve caber na
TITAN Xp (12 GB), mas confirmar:

```bash
poetry run python scripts/ssl/pretrain_lfr.py --encoder tstcc --source KuHar --seed 0 --spike
```

**Gate 3a**: spike completa (12 épocas = 2 efetivas), VRAM máx. < 11 GB
(observar `nvidia-smi`), DPP seleciona 6/60 sem erro. Apagar o backbone do
spike antes da grade (mesmo motivo da Fase 1.3). Se estourar VRAM: reduzir
`--num-projectors` **não é opção** (quebra a config oficial) — alternativa é
gradiente/batch menor, e aí documentar o desvio.

### 3.3 Grade (tmux `ssl-tstcc`)

```bash
tmux send-keys -t ssl-tstcc 'cd ~/fed-har && poetry run python scripts/ssl/run_all.py \
  --encoder tstcc --gpus 0,1,2,3,4,5,6,7 2>&1 | tee logs/ssl-tstcc.log' Enter
```

Fase 1 do runner: 7 fontes × 4 seeds = **28 pré-treinos de 600 épocas** (o
item mais caro do plano — medir o 1º job; ver Apêndice C). Fase 2: **28 jobs
de downstream** (~7 min cada). Consolida no cache `ssl_lfr_eval_transfer.csv`.

### 3.4 Gate 3b

Comparar com **LFR × TS-TCC Encoder** do `performance_data.json` (mesmo
protocolo do gate 2). Referências do paper (full_ft, média 6 datasets):
@10-shot **65.0%**, @100% **81.2%**; freeze @10-shot **55.5%**. Aceite: mesma
folga do gate 2. Sanidade adicional: Δ(LFR−SL) do tstcc deve ser positivo
(paper: +5.0 @10 / +5.2 @100%) — é o motivo de ter escolhido esse encoder.

**Commit**: `git commit -m "LFR no tstcc: 28 backbones + downstream; cache SSL atualizado"`

---

## Fase 4 — `tstcc` no baseline federado

### 4.1 Registrar em `scripts/federated/client.py`

Import do `build_model` + entrada em `BUILD_MODEL`. O `run_all.py` federado
herda automaticamente (`ALL_ENCODERS = list(BUILD_MODEL)`).

### 4.2 Rodar (tmux `fed-tstcc`)

```bash
tmux send-keys -t fed-tstcc 'cd ~/fed-har && poetry run python scripts/federated/run_all.py \
  --encoder tstcc --gpus 0,1,2,3 2>&1 | tee logs/fed-tstcc.log' Enter
```

8 cenários × 4 seeds = **32 runs** (R=50), parciais em
`results/federated_parts/`, consolidado em `results/federated_eval.csv`.
De brinde: `uplink_bytes` do encoder 2304-d (relevante p/ discussão de
comunicação — é o encoder mais pesado dos 4).

### 4.3 Atualizar `notebooks/_build_federated_nb.py`

⚠️ Tem **"96 runs" e "3 encoders" hardcoded** (linhas ~39, ~122, ~187–198,
incl. o check `combos == 96`). Atualizar para 128 (8×4×4) e incluir `tstcc`
em `ENCODERS`. Reexecutar o notebook federado.

**Gate 4**: 32/32 runs na rodada 50; notebook federado executa sem erro e o
check de completude passa com 128; curva do tstcc plotada nos 8 cenários.

**Commit**: `git commit -m "tstcc no baseline federado (32 runs) + notebook federado 4 encoders"`

---

## Fase 5 — Spike TF-C

Validar o `TFC_Model`/`TFC_Backbone`/`TFC_Transforms` do minerva ponta-a-ponta
**antes de escrever infra**. Um script descartável (`scratch` ou célula) com
1 combo (resnetse5 × KuHar × seed 0, poucas épocas):

```python
TFC_Model(input_channels=6, TS_length=60, num_classes=None,   # None => modo SSL
          single_encoding_size=128, pred_head=None, batch_size=64,
          backbone=TFC_Backbone(input_channels=6, TS_length=60,
                                single_encoding_size=128,
                                time_encoder=E1, frequency_encoder=E2))
# E1/E2 = duas instâncias independentes do encoder (YAML usa cópias gêmeas)
```

**Perguntas que o spike responde** (anotar as respostas no cabeçalho do
`pretrain_tfc.py` na Fase 6):

1. O `forward`/`training_step` calcula a FFT internamente ou o batch precisa
   vir com `(x_time, x_freq)`? (o `transform=None` instala o default com
   augmentations + FFT — confirmar que funciona com nosso batch `(B, 6, 60)`
   e o dataloader do `make_datamodule`).
2. O que salvar: `state_dict` do `TFC_Backbone` inteiro (encoders + projetores
   internos). Recarrega limpo num `TFC_Backbone` recém-construído com
   `strict=True`?
3. Saída do `TFC_Backbone` no modo inferência: concat tempo+freq de **256-d**
   (2×128)? É o que o downstream usa (YAML: `fc [256,128,6]`, `flatten=True`).
4. VRAM/tempo por época com o pior caso (`tstcc`: 2 × HARSCnn 2304-d).
5. A loss NT-Xent poly com batch 64 é estável (não-NaN) nas primeiras épocas?
6. O RnnEncoder funciona como frequency_encoder (a FFT muda a natureza da
   série; `permute=True` continua correto)?

**Gate 5**: pré-treino curto roda; backbone salvo e recarregado; probe
linear+finetune de 2 épocas sobre o backbone produz acc > acaso no KuHar.

**Commit**: (se o spike gerar aprendizado de código útil) notas no README do ssl.

---

## Fase 6 — Infra TF-C

### 6.1 `scripts/ssl/pretrain_tfc.py`

Espelha `pretrain_lfr.py`. Config oficial (YAMLs `train/tfc_*.yaml` +
pipeline dos experimentos do paper — ver Apêndice A):

- `TFC_Backbone` com **duas instâncias** do encoder (tempo e frequência),
  `single_encoding_size=128`.
- `num_classes=None`, `pred_head=None` (modo SSL), loss default (NT-Xent poly),
  `learning_rate=3e-4` (default da classe = o que o YAML usa), batch **64**.
- **100 épocas, sem early stopping** (pipeline `train_without_early_stopping`
  com `max_epochs=100` — TF-C NÃO usa as 600 do LFR).
- Dados: train+val da fonte via `PretrainDataModule` (mesma classe do LFR).
- Saída: `checkpoints/ssl/tfc/<enc>/<source>/seed<N>/backbone.ckpt`
  (state_dict do `TFC_Backbone`).
- `--spike` como no LFR.

### 6.2 `scripts/ssl/encoders.py`: `build_tfc_backbone(encoder) -> (TFC_Backbone, 256)`

Fábrica das duas instâncias por encoder (mesmos `build_model` de sempre,
2 chamadas independentes → pesos não compartilhados, como no YAML).

### 6.3 `downstream_eval.py`: flag `--method {lfr,tfc}` (default `lfr`)

Muda apenas: builder do backbone (`build_backbone` vs `build_tfc_backbone`,
enc_dim 256), raiz dos ckpts (`checkpoints/ssl/<method>/`) e cache default
(`results/ssl_tfc_eval_transfer.csv`). Cabeça/protocolos/regimes idênticos
(o YAML finetune do TF-C usa exatamente `MLP [256,128,6]` + lr 1e-4 + 100
épocas com ES — nosso protocolo atual).

⚠️ `run_comb2target.py` e `run_all.py` ganham o mesmo `--method` (propagado).

### 6.4 Smoke test da infra

1 combo completo em miniatura: `pretrain_tfc.py --spike` + `downstream_eval.py
--method tfc --shots 1 --protocol linear` → 6 linhas sãs no CSV. Apagar
artefatos do smoke.

**Gate 6**: smoke test passa nos **4 encoders** (é barato e pega problema de
shape por encoder antes da grade).

**Commit**: `git commit -m "Infra TF-C: pretrain_tfc + --method no downstream/runners"`

---

## Fase 7 — Grade TF-C (4 encoders)

### 7.1 Rodar (tmux `tfc-grid`)

```bash
tmux send-keys -t tfc-grid 'cd ~/fed-har && poetry run python scripts/ssl/run_all.py \
  --method tfc --gpus 0,1,2,3,4,5,6,7 2>&1 | tee logs/tfc-grid.log' Enter
```

4 encoders × 7 fontes × 4 seeds = **112 pré-treinos** (100 épocas; ~6× mais
curtos que LFR, mas 2 encoders por modelo) + **112 downstream** →
`results/ssl_tfc_eval_transfer.csv` (5376 linhas).

### 7.2 Gate 7 (o teste que pegaria "outra v0")

Validação contra **TFC × {ResNet-SE-5, CNN-PFF, RNN, TS-TCC Encoder}** do
`performance_data.json`, protocolo idêntico ao da Seção 9 das notas (viés/MAE
por encoder×método, in-domain). Referências fortes do paper (full_ft, média
6 datasets): CNN-PFF @10-shot **77.0%** (vs 55.3 supervisionado!), RNN
**69.5%**, TS-TCC **68.6%**. Aceite: |viés| ≤ 5 pp, MAE ≤ 8 pp. Se o ganho
gigante do CNN-PFF **não** aparecer, é red flag de implementação (augment/FFT),
não de seed.

**Commit**: `git commit -m "Grade TF-C 4 encoders (112+112) + cache ssl_tfc"`

---

## Fase 8 — (Opcional, recomendado) comb→target para TF-C

As células `comb` e `comb→t` são **co-principais** (decisão 2026-07-06).
Com `--method tfc` já propagado (Fase 6):

```bash
tmux send-keys -t tfc-c2t 'cd ~/fed-har && poetry run python scripts/ssl/run_comb2target.py \
  --method tfc --gpus 0,1,2,3 2>&1 | tee logs/tfc-c2t.log' Enter
```

4 encoders × 6 alvos × 4 seeds = **96 jobs** (~7 min/job ⇒ ~3 h em 4 GPUs) →
`results/ssl_tfc_comb2target_eval_transfer.csv`. Gate: 96/96 sem falha;
sanidade = repetir a decomposição da §9.2 para TF-C.

---

## Fase 9 — Notebooks, docs e fechamento

1. `notebooks/_build_ssl_nb.py`: incluir `tstcc` em `ENCODERS`; carregar os
   caches TF-C; adicionar comparação **SL vs LFR vs TF-C** (curvas §4 e barras
   §6 ganham a série TF-C; §9.1/9.2 ganham o método como parâmetro);
   reexecutar.
2. `notebooks/centralized_supervised_avaliation.ipynb`: incluir `tstcc` onde o
   notebook enumera encoders (conferir como define a lista) e reexecutar.
3. Atualizar `docs/notas_internas_projeto_ssl_federado_har.md`: checklist da
   Seção 8 (TF-C ✅, tstcc ✅), Seção 9 ganha os números TF-C, e marcar este
   plano como executado.
4. Atualizar a memória do Claude (grades concluídas, gates, achados).
5. Commit final + push.

**Gate 9**: notebooks executam de ponta a ponta sem erro lendo só caches;
checklist consistente com o que existe no repo.

---

## Apêndice A — Config oficial TF-C (resumo dos YAMLs)

```yaml
# train/tfc_<enc>.yaml (pré-treino)
TFC_Model: input_channels 6, TS_length 60, num_classes 6*, batch_size 64,
           single_encoding_size 128, pred_head null
  backbone: TFC_Backbone(time_encoder: <ENC>, frequency_encoder: <ENC>)
# * no pipeline de pretrain o pred_head null => SSL; num_classes é ignorado
# lr: default da classe (3e-4). Pipeline: train_without_early_stopping, 100 épocas.
# Dados de pretrain do paper: view rodrigues_2024 "no_overlap" (nós: standardized_view
# train+val — MESMO desvio documentado do LFR; manter p/ consistência interna).

# finetune/tfc_<enc>.yaml (downstream)
SimpleSupervisedModel:
  backbone: FromPretrained(TFC_Backbone(...), filter_keys ["backbone"], strict)
  fc: MLP [256, 128, 6]   # 256 = concat tempo(128) + freq(128)
  flatten: True
# overrides: freeze/full_finetune ambos lr 1e-4; pipeline train (ES paciência 50,
# best.ckpt, max 100 épocas); regimes por samples_per_class (batch 64).
```

Encoders por YAML (tempo e frequência = duas cópias da mesma arquitetura):
`_ResNet1D`+SE (resnetse5), `CNN_PF_Backbone(include_middle, flatten)` (cnnpff),
`RnnEncoder` GRU-100 bi (rnn), `HARSCnnEncoder(2304, 6, 1280)` (tstcc/harcnn).

## Apêndice B — Checklist de registro do encoder `tstcc`

| # | Arquivo | O quê | Fase |
|---|---|---|---|
| 1 | `scripts/supervised/train_tstcc.py` | novo (nome é contrato do run_all_shots) | 1 |
| 2 | `scripts/common.py` | `BEST_LR["tstcc"]` | 1 |
| 3 | `scripts/eval_transfer.py` | `ENCODERS` + `BUILD_MODEL` | 1 |
| 4 | `scripts/supervised/run_all_shots.py` | `ENCODERS` | 1 |
| 5 | `scripts/supervised/train_combined.py` | `ENCODER_BUILDERS` | 1 |
| 6 | `scripts/supervised/TSTCC.md` | doc | 1 |
| 7 | `scripts/ssl/encoders.py` | `ENCODERS`, `ENC_DIM=2304`, builder | 3 |
| 8 | `scripts/federated/client.py` | `BUILD_MODEL` (run_all herda) | 4 |
| 9 | `notebooks/_build_federated_nb.py` | `ENCODERS` + contagens 96→128 | 4 |
| 10 | `notebooks/_build_ssl_nb.py` | `ENCODERS` | 9 |
| 11 | `notebooks/centralized_supervised_avaliation.ipynb` | lista de encoders | 9 |

## Apêndice C — Medições de referência e estimativas

| Medição real | Valor |
|---|---|
| Job de downstream (2 proto × 4 regimes × 6 alvos, 1 backbone) | ~7 min (grade comb→target: 72 jobs / 4 GPUs = 2h10) |
| Pré-treino LFR 600 épocas (encoders pequenos) | ~1.4 h/job em média (grade v1: 84+84 em ~17.5 h / 8 GPUs) |
| Grade federada (96 runs, R=50, 3 encoders) | rodada no cluster em ~meio dia (jun/2026) |

| Estimativa (medir no 1º job!) | Base |
|---|---|
| Treino SL few-shot (1 combinação) | 2–10 min ⇒ 112 jobs ≈ 2–5 h / 4 GPUs |
| Pré-treino LFR `tstcc` (preditores 2304²) | 2–3 h/job ⇒ 28 jobs ≈ 7–11 h / 8 GPUs |
| Pré-treino TF-C (100 épocas, 2 encoders) | 15–40 min/job ⇒ 112 jobs ≈ 4–9 h / 8 GPUs |
| VRAM LFR `tstcc` | ~4 GB só de preditores+Adam (gate 3a confirma) |
