# Plano do Fed-SSL — eixo cross-device

*Consolidado em 2026-07-27 a partir de `desenho_cross_device.md` (2026-07-24),
`_arquivo/limite_batch_cliente_fssl.md` (2026-07-24) e da parte viva de
`_arquivo/plano_fedssl_simulado.md` (2026-07-13/14). É o documento que dita o
próximo passo de implementação.*

**Contexto do pivô (2026-07-21, com o orientador):** a federação **cross-silo**
(1 dataset por cliente, cenários 1–8) foi abandonada como desenho e como
controle. Os 6,3 pp medidos lá viram **motivação**, não contribuição
(`resultados.md §4`). O eixo ativo é **cross-device: clientes = usuários**.

**Onde vive o quê:** fatos dos datasets em [`dados_daghar.md`](dados_daghar.md);
resultados centralizados em [`resultados.md`](resultados.md); posicionamento na
literatura e checagem de ineditismo em [`estado_da_arte.md`](estado_da_arte.md).

---

## 1. Os três eixos de heterogeneidade

O cross-device permite estressar o Fed-SSL contra **três formas distintas de
heterogeneidade**, que a literatura de FL trata como eixos separados e que aqui
conseguimos **isolar**:

1. **Domain shift** — clientes de datasets diferentes (posição do sensor: coxa vs
   bolso vs cintura). É o custo que o pivô quer medir sem o confundidor do
   cross-silo.
2. **Feature skew** (covariate shift, P(x|y)) — dentro do mesmo dataset, a mesma
   atividade tem sinal diferente entre pessoas/dispositivos.
3. **Label skew** (P(y)) — clientes com distribuições de classe diferentes.

A partição natural por usuário entrega **muito feature skew e quase nenhum label
skew** (exceto KuHar: TV 0.539). Ou seja, ela **isola feature skew de graça**;
label skew precisa ser induzido. Os números por dataset estão em
`dados_daghar.md §4`.

## 2. Desenho fatorial: cada efeito é um Δ contra um controle pareado

O que isola um efeito **não é a célula, é o Δ contra o controle certo**:

| Efeito isolado | Δ = experimento − controle pareado | O que o Δ cancela |
|---|---|---|
| **Domain shift** | cross-domain(user) − in-domain(user) | feature skew (ambos têm) |
| **Feature skew** | in-domain(**partição por usuário**) − in-domain(**shards IID do mesmo dataset**) | dado, rótulos e domínio idênticos; muda só a estrutura da partição |
| **Label skew** | in-domain(rótulo skewado artificial) − in-domain(**IID pareado em volume**) | volume de dado (ver risco 1) |

- **Cross-domain** carrega feature skew *além* do domain shift ⇒ a célula
  cross-domain **não** é "domain shift puro"; é o Δ contra in-domain que isola.
  Prova-de-conceito: **RW_thigh + MotionSense** (grupo perna, maior Δ(SSL−SL)
  centralizado; ambos com as 6 classes ⇒ sem mismatch de classe).
- **Feature skew** só fica isolado com o controle de **shards-IID** (quebrar a
  estrutura de usuário do mesmo dataset, mantendo dado/rótulos). Sem ele temos o
  número da célula, não o efeito atribuível.

### Riscos de método a respeitar

1. **Label skew artificial confunde-se com volume de dado.** Remover rótulos de um
   cliente reduz o dado dele e o total; a degradação pode vir de "menos dado". O
   controle do braço de label skew tem de ser **pareado em volume** (remover a
   mesma quantidade aleatoriamente, preservando as proporções).

2. **"Label skew" significa coisas diferentes no SSL e no supervisionado.** O
   pré-treino SSL **ignora rótulos**. A mesma operação "remover janelas da classe
   c do cliente k" produz:
   - no **baseline supervisionado federado**: skew de rótulo clássico (P(y)
     enviesado) — o efeito desejado, que morde no treino/finetuning rotulado;
   - no **pré-treino SSL**: **skew de cobertura/feature** (o P(x) não-rotulado
     perde uma região da variedade) **+ encolhimento** do cliente (o piso de
     batch da §3 volta).

   Consequência: não usar uma única partição "com label skew" e alegar que ela
   isola label skew para os dois braços. Label skew é construto do estágio
   **rotulado**; atribuir o efeito a esse estágio.

3. **Ortogonalidade parcial.** Os três eixos não são perfeitamente ortogonais
   (cross-domain traz feature skew; label skew artificial traz cobertura+volume).
   Cada efeito só é limpo como Δ pareado, não como célula isolada.

**Veredito da sabatina (2026-07-24):** com (a) cada efeito medido como Δ contra o
controle pareado — incluindo o shards-IID para feature skew, (b) o braço de label
skew pareado em volume, e (c) label skew tratado como construto do estágio
rotulado, o desenho fatorial fecha os três eixos de forma isolada e é publicável.

## 3. O piso de batch: o cliente mínimo viável em FSSL é um *batch*

*Achado registrado em 2026-07-24. É candidato a contribuição nomeada — o
posicionamento na literatura está em `estado_da_arte.md §6`.*

Ao particionar o KuHar por usuário, **48 dos 57 clientes (84%) têm menos janelas
que um batch** (mediana 10, mínimo 1), e eles concentram 48% das janelas do
dataset. Os outros 5 datasets não têm o problema — o menor usuário de cada um já
supera o batch 64 (`dados_daghar.md §3`).

### 3.1 Por que o SSL quebra onde o supervisionado não quebra

A diferença não é "poucas amostras para aprender", é a **granularidade do
objetivo de perda**:

| | Downstream supervisionado (few-shot) | Pré-treino SSL |
|---|---|---|
| Perda | CrossEntropy — **por amostra** | Barlow Twins / NT-Xent — **definida sobre o batch** |
| Cliente mínimo | ~1–2 amostras (2 pelo BatchNorm) | **1 batch** (o objetivo não existe abaixo dele) |
| 6 janelas (1-shot × 6 classes) | 6 termos de CE → treina | matriz/negativos degeneram ou nem montam |

No downstream, `subsampled_train_loader` (`scripts/common.py`) não passa
`drop_last`, então 6 janelas viram um batch de 6 e o CrossEntropy soma 6 termos —
funciona. Já a perda SSL agrega estatísticas **ao longo do batch**; com um batch
minúsculo ela vira ruído, e com 1 amostra é indefinida (variância 0).

**Consequência para o FedAvg:** um cliente de 10 janelas ainda entra na média
ponderada com `n_k = 10` de 1.392 (**0,7%** da rodada). Mesmo que "funcionasse",
o sinal que ele agrega é desprezível.

### 3.2 Vale para os dois métodos do nosso escopo — e é pior do que parece

- **TF-C (contrastivo).** `NTXentLoss_poly` constrói uma **máscara de tamanho fixo
  `2·batch × 2·batch`** no `__init__` (`minerva/losses/ntxent_loss_poly.py:67-72`).
  Batch parcial não encaixa na máscara → o pré-treino exige `drop_last=True`
  (`scripts/ssl/pretrain_tfc.py`). No cross-device, um cliente com menos de 64
  janelas produz **zero batches → zero passos de gradiente**, e devolve os pesos
  globais intactos: um **no-op silencioso**, não um erro — a mesma família de
  falha do `PYTHONPATH` do Ray que já mordeu o projeto.
- **⚠️ O LFR não usa o Barlow Twins clássico.** Usa a **Batch-wise Barlow Twins
  (BBT)** proposta no próprio paper do LFR (`BatchWiseBarlowTwinLoss`, default do
  minerva — `minerva/models/ssl/lfr.py:81`). No BBT a matriz de similaridade é
  **`m×m` com `m` = tamanho do batch**, não `d×d`. Ou seja, **o BBT reintroduz
  exatamente a dependência de batch que o Barlow Twins clássico havia
  eliminado**: com 10 janelas a matriz é 10×10 (posto ≤ 10); com 1 janela é 1×1 e
  o termo de redundância nem existe. **A robustez a batch pequeno atribuída ao
  "Barlow Twins" NÃO se aplica ao nosso LFR.**
- Objetivos por par positivo (BYOL/SimSiam) e reconstrução mascarada não têm o
  problema — é por isso que boa parte da literatura de F-SSL cross-device os
  escolhe. Trocar de família de objetivo está fora do escopo atual.

### 3.3 Implicações práticas

1. ~~**Instrumentar o simulador com um `assert` alto**~~ — **feito**: um cliente
   que não forma ≥ 1 batch aborta o run com mensagem clara
   (`pretrain_fed.py:148`), em vez de virar no-op silencioso do TF-C.
2. **Construir o pipeline primeiro IGNORANDO o KuHar.** Os 3 experimentos de
   controle (§5) não o incluem. **Mas o problema não fica resolvido — apenas
   adiado**; o KuHar é o caso real que o expõe e é o dataset com mais usuários da
   coleção. **A forma de endereçar fica em aberto — discussão deliberadamente
   adiada.**
3. **A decisão D-K está REVOGADA.** Agrupar usuários do KuHar em 6 super-clientes
   **destrói o non-IID que se queria estudar**: o skew de rótulo cai de **0.539
   para 0.068**. Se o KuHar entrar, deve ser como **estudo do limite de
   viabilidade** (cliente = usuário, com limiar de elegibilidade declarado e
   sensibilidade a batch 16/8), **não** como super-clientes fictícios.
   ⚠️ Documentos anteriores a 2026-07-24 (em `_arquivo/`) propagam a versão antiga
   da D-K; o código nunca chegou a implementá-la (`partitions.py:_user_shards` faz
   1 cliente por usuário, sem agrupamento).
4. **Enquadramento para o artigo**: *critério de elegibilidade de cliente* +
   **taxa de exclusão reportada** (KuHar: 48/57 = 84% dos usuários, 48% das
   janelas).

## 4. Implementação

### 4.1 `scripts/ssl/pretrain_fed.py` — ✅ IMPLEMENTADO (301 linhas)

Pré-treino federado **simulado** (FedAvg manual em loop Python, sem Flower). Fica
em `ssl/` porque produz checkpoints consumidos por `downstream_eval.py`. **O
arquivo existe e está completo** — o que falta para rodar os experimentos do §5 é
a partição do §4.2, não este script.

- `fedavg(states, weights, skip_prefixes=())` — média ponderada de `state_dict`s
  (`n_k/n`). Float: média; inteiros (ex.: `num_batches_tracked`): média
  arredondada. `skip_prefixes` fica disponível para variantes `fedbn`; o default
  agrega tudo, inclusive buffers de BatchNorm.
- `ShardPretrainDataModule` — Subset + o mesmo unwrap de rótulos do
  `PretrainDataModule`; `drop_last=True` para TF-C e **`assert len(loader) > 0`**
  (`pretrain_fed.py:148`): um shard de usuário menor que o batch **aborta o run
  com mensagem clara** em vez de virar no-op silencioso. É o guarda da §3.3 —
  já instrumentado.
- `local_pretrain(...)` / `run_fedssl(...)` — 1 cliente × 1 rodada e o loop
  principal, com resume via `state_last.pt` e milestones.

CLI: `--method {lfr,tfc} --encoder --partition --combo --rounds --local-epochs
--seed --force`.

**Decisões (já codificadas):**
- **Paralelismo no nível do RUN, não do cliente**: cada run ocupa 1 GPU e itera
  clientes sequencialmente (rodada ≈ 1 época sobre a união — mesmo custo do
  centralizado). A grade paraleliza runs no `gpu_pool.py` existente.
- **Seeds**: `seed_everything(seed)` no início; antes de cada fit local,
  `seed_everything(seed*10**6 + round*10**3 + client)`. O otimizador é recriado a
  cada rodada (Adam reseta — igual ao FedAvg padrão; o gate G-EQ2 mede esse custo
  isolado).
- **DPP do LFR**: seleção executada **uma vez**, pelo "servidor", sobre 128
  amostras da união do combo; clientes herdam via cópia do estado. Descrever no
  artigo como hipótese de "amostra pública".
- **BatchNorm**: o default agrega `running_mean/var` junto (idêntico ao baseline
  supervisionado). `fedbn` fica como extensão.
- **Checkpoints**: `checkpoints/ssl_fed/<method>/<encoder>/<partition>/<combo>/seed<N>/`
  com `backbone.ckpt`, `round<R>/backbone.ckpt`, `state_last.pt` (resume) e
  `rounds.csv` (round, n_clients, secs, uplink_mb, downlink_mb).
- **Sem validação no pré-treino**: clientes usam só o `train.csv`, então o CSV de
  rodadas loga tempo e bytes analíticos, **não** `val_loss`.
- **A fazer no downstream**: `results/ssl_fed_eval_transfer.csv` + parciais em
  `results/ssl_fed_parts/`; `downstream_eval.py` precisa ganhar `--ckpt-dir`
  explícito (hoje restringe `--pretrain-source` a `SOURCES`).

### 4.2 Partições — e a lacuna que bloqueia o experimento 3

`make_ssl_client_datasets(partition, combo, seed)`
(`scripts/federated/partitions.py:150`) já implementa:

- `silo` — 1 cliente por dataset do combo;
- `iid` — união do combo em 6 fatias IID (gate G-IID);
- `device-<dataset>` — 1 cliente por usuário do train, **de um único dataset**.

> ⚠️ **Lacuna registrada em 2026-07-27:** `device-<dataset>` exige um dataset só
> (`partitions.py` valida `dataset_name in names` e chama `_user_shards` para ele).
> O **experimento 3 do §5 — RW_thigh + MotionSense federados juntos, clientes =
> usuários dos dois** — **não é expressável pela API atual**. Falta um modo tipo
> `device` (todos os datasets do combo, união dos usuários, id de cliente
> prefixado pelo dataset). É a primeira mudança de código a fazer.
>
> Faltam também os dois controles derivados do §2: **shards-IID intra-dataset**
> (para isolar feature skew — o `iid` atual fatia a união do combo, não um dataset
> só) e o braço de **label skew artificial pareado em volume**.

### 4.3 Gates de validação (herdados, ainda válidos)

| # | O quê | Gate |
|---|---|---|
| S0 | Init idêntico entre fontes (hash dos state_dicts) | **✅ PASS (2026-07-14)** |
| S1 | Sopa all6 tfc×cnnpff×seed0 + 1 downstream | acc > acaso (1/6), registrada |
| S2 | Loop 3 rodadas TF-C | loss cai, sem NaN, CSV de bytes ok |
| S3 | LFR 2 rodadas | backbone muda exatamente 1 época efetiva/rodada; DPP global aplicado |
| G-EQ1 | 1 cliente (= `combined`), R=1 × E=100 vs centralizado | métricas downstream idênticas — valida o simulador inteiro |
| G-EQ2 | 1 cliente, R=100 × E=1 vs centralizado | quantifica o custo do fatiamento (reset do Adam) — número novo do artigo |
| G-IID | `iid` 6 clientes, R=100 vs centralizado `combined` | Δ downstream @full ≥ −3 pp |

Regra de sempre: **medir o 1º job de cada onda antes de extrapolar custo**; tmux +
`tee logs/` (ver `CLAUDE.md`).

## 5. Escopo e ordem de execução

1. **Fechar a lacuna da API de partições** (§4.2) — sem ela o experimento 3 não roda.
2. **Os 3 controles, ignorando o KuHar**: (1) in-domain RW_thigh, (2) in-domain
   MotionSense, (3) cross-domain RW_thigh+MotionSense. Nenhum inclui KuHar, então
   o piso de batch não bloqueia. O controle honesto do custo de domain shift é
   **Δ(3 − média de 1,2)**.
3. **Controles extras** do §2: shards-IID intra-dataset (isola feature skew) e o
   braço de label skew artificial pareado em volume.
4. **Depois**: endereçar o piso de batch (KuHar é o caso que o expõe) — **forma em
   aberto** (§3.3).
5. **Segunda onda** (heterogeneidade por pessoa como eixo próprio): WISDM
   (36 clientes, label skew 0.000 — feature skew puro) e KuHar (label skew 0.539)
   como os dois extremos. O gap cintura↔perna prevê que cross-domain **entre
   grupos de posição** é o teste mais duro de mitigação.

## 6. Proveniência

- Fatos de dados (usuários, janelas, skews, classes): `dados_daghar.md`,
  regenerável por `scripts/analysis/dataset_facts.py`.
- Fatos de código (§3.2): `minerva/models/ssl/lfr.py:81`
  (`BatchWiseBarlowTwinLoss` default), `minerva/losses/ntxent_loss_poly.py:67-72`
  (máscara `2·batch` fixa), `scripts/ssl/pretrain_tfc.py` (`drop_last=True`),
  `scripts/ssl/pretrain_lfr.py` (`drop_last=False`), `scripts/common.py`
  (`BATCH_SIZE=64`).
- Partições: `scripts/federated/partitions.py` (`_user_shards`,
  `make_ssl_client_datasets`), `scripts/federated/partition_users.py`.
- A lousa do orientador que originou o desenho: `lousa_orientador_fedssl.png`.

### Referências de método

1. Y. Sui et al., "Self-supervised Representation Learning From Random Data
   Projectors" (**LFR**, com a **Batch-wise Barlow Twins**), ICLR 2024.
   arXiv:2310.07756. *(Fonte do "m×m com m = batch" do BBT.)*
2. J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny, "Barlow Twins:
   Self-Supervised Learning via Redundancy Reduction", ICML 2021. *(Matriz d×d;
   a robustez a batch pequeno do BT clássico NÃO se aplica ao BBT.)*
3. X. Zhang et al., "Self-Supervised Contrastive Pre-Training for Time Series via
   Time-Frequency Consistency" (**TF-C**), NeurIPS 2022. arXiv:2206.08496.
