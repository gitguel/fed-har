# Desenho experimental — RQ1 e RQ2

*Como cada RQ será medida. É o documento que
[`perguntas_de_pesquisa.md`](perguntas_de_pesquisa.md) promete ("como cada RQ será
medida vem depois, em outro documento"). **A pergunta manda no desenho** — se algo
aqui contradiz as RQs, este documento está errado.*

**Fechado na sabatina de 2026-08-23**, a partir da anotação de caderno de 21/08 do
Miguel. Os números dos dados vêm de
[`dados_daghar.md`](dados_daghar.md) e de `scripts/analysis/client_regimes.py`; os
de custo, da calibração da §9. **A RQ3 não está aqui** — não tem desenho, sai do
DAGHAR e o candidato é o ExtraSensory.

---

## 1. As cinco federações

Clientes = **usuários do split `train`** (val/test são disjuntos por usuário, então
particionar o `train` não vaza). **Configuração máxima de clientes** em todas.

| Federação | Clientes | Classes | Janelas | jan/cliente (mín/med/máx) |
|---|---|---|---|---|
| WISDM | 36 | 4 | 8.748 | 235 / 236 / 476 |
| UCI | 21 | 5 | 2.420 | 99 / 113 / 139 |
| MotionSense | 17 | 6 | 3.558 | 165 / 211 / 244 |
| RealWorld_thigh | 10 | 6 | 10.338 | 957 / 1024 / 1204 |
| RealWorld_waist | 10 | 6 | 10.332 | 958 / 1023 / 1201 |

**O KuHar está fora de todos os experimentos** (D2). `scripts/rqs/config.py`
verifica esta tabela contra os CSVs antes de enfileirar qualquer job — o
pareamento da §3 depende dela.

## 2. Rótulos por cliente: o teto de `k`

| Federação | samples/class por cliente: mín(>0) / mediana / máx | clientes com todas as classes |
|---|---|---|
| UCI | 14 / 23 / 32 | 21/21 |
| MotionSense | 16 / 35 / 49 | 17/17 |
| WISDM | 58 / 59 / 119 | 36/36 |
| RealWorld_thigh / _waist | 140 / 168 / 253 | 10/10 |

`k` máximo com **todo** cliente completo: UCI 14, MotionSense 16, WISDM 58,
RealWorld 140. **A escada `k ∈ {1, 2, 4, Full}` cabe com folga nas cinco**; `k=8`
também caberia e fica como 5º regime opcional.

## 3. O pareamento federado ↔ centralizado

`k` são rótulos por classe **por cliente**. O total no sistema é
`Σ_clientes min(k, n_uc)`, que nas cinco federações completas é exatamente
`k × n_clientes`. Logo:

> **Regime centralizado pareado = `k × n_clientes` samples/class.**
> Ele é **diferente em cada dataset** — é o ponto que a anotação mandava verificar.

| Federação | k=1 | k=2 | k=4 | Full |
|---|---|---|---|---|
| RealWorld_thigh / _waist | 10 | 20 | 40 | todo o `train` |
| MotionSense | 17 | 34 | 68 | idem |
| UCI | 21 | 42 | 84 | idem |
| WISDM | 36 | 72 | 144 | idem |

O `Full` é pareado de graça: os dois braços veem o mesmo corpus, os mesmos
usuários. **É a célula mais importante da RQ1** — o Δ ali é o efeito de federar
sobre exatamente as mesmas janelas.

## 4. Batch

**Fine-tuning supervisionado** (CE é por amostra; `drop_last=False`): o cliente tem
`k × C` janelas, então `batch = min(64, k×C)` — 1 batch por época nos regimes `k`,
64 no `Full`.

**Pré-treino SSL** (perda sobre o batch; `drop_last=True`): **batch 64** (D5).
Nenhum cliente é perdido — o menor das cinco federações (UCI, 99 janelas) fecha
1 batch:

| Federação | batches/época por cliente (mín–máx) | sobra descartada/época |
|---|---|---|
| UCI | **1 – 2** | 37,2% |
| MotionSense | 2 – 3 | 13,7% |
| WISDM | 3 – 7 | 18,4% |
| RealWorld_thigh / _waist | 14 – 18 | 3,4% |

A sobra volta em outras épocas (o loader embaralha). **B=128 estaria fora**: o UCI
daria 0 batches e o assert de `pretrain_fed.py:148` abortaria. **O UCI em 1
batch/época é o caso de borda do piso de batch** (`plano_fedssl §3`) e deve ser
reportado como tal.

## 5. As grades

Fatores comuns: **4 encoders** (`resnetse5`, `cnnpff`, `rnn`, `tstcc`) × **5
federações** × **4 regimes** `k ∈ {1,2,4,Full}` × **seeds**.

| Bloco | O que mede | Runs (3 seeds) |
|---|---|---|
| **RQ1 centralizado** | o teto de referência, in-domain, no regime pareado | 240 |
| **RQ1 federado** | o braço federado, partição natural, R=150 | 240 |
| **RQ2 pré-treino** | 2 métodos (TF-C, LFR) × 4 enc × 5 fed × seeds, R=100 | 120 |
| **RQ2 fine-tuning** | 2 métodos × a grade da RQ1 federada | 480 |
| **Busca S1 de LR** | 6 LRs × 4 enc × 5 fed × `{k=1, Full}`, 1 seed | 240 |

Com 4 seeds a grade principal vai a 1.440 runs. **A 1ª onda roda 3 seeds
(0, 1, 2), uma seed por vez** — a seed 3 é incremento posterior, sem re-rodar nada
(os drivers pulam o que já está completo).

O braço **sem** pré-treino da RQ2 é a própria RQ1 federada — não se repete.

## 6. Protocolo

| | |
|---|---|
| **Rodadas** | fine-tuning `R = 150`; pré-treino `R = 100` |
| **Épocas locais** | `E = 5` **efetivas de backbone** → TF-C `--local-epochs 5`, LFR `--local-epochs 30` (alterna 1 backbone : 5 preditor) |
| **Orçamento por cliente** | **natural** (`--budget 0`) — todo o dado do usuário (D3) |
| **Seleção de modelo** | early stopping + `best.ckpt` por validação, **inalterado** (D4) |
| **Agregação** | FedAvg ponderado por `n_k`; LFR com `skip_prefixes=("projectors",)` |
| **Ordem** | sequencial: todo o pré-treino, depois o fine-tuning |
| **Métricas** | `test_acc`, `test_f1_macro`, `uplink_mb`, `downlink_mb` por rodada |

Comunicação já está instrumentada — a métrica (b) da RQ2 ("MB adicionais") sai dos
mesmos runs, sem trabalho novo.

## 7. ⚠️ O confundidor de orçamento de otimização

Os dois braços da RQ1 **não fazem o mesmo número de passos de gradiente sobre o
modelo global**, e a assimetria **troca de sinal ao longo da curva** que a RQ1
plota (RealWorld_thigh):

| Regime | Centralizado | Federado (global) | razão |
|---|---|---|---|
| `k=1` ↔ 10 shots | ~100 passos | 750 | **7,5× pró-federado** |
| `k=10` ↔ 100 shots | ~1.000 | 750 | 0,75× pró-centralizado |
| `Full` | ~16.200 | 12.000 | 0,74× pró-centralizado |

**O cache da grade antiga confirma o padrão** (RW_thigh, in-domain, 4 seeds):

| encoder | centr. 10 vs fed `k=1` | centr. 100 vs fed `k=10` |
|---|---|---|
| cnnpff | **+2,13 pp** (fed ganha) | −2,18 pp |
| resnetse5 | **+0,56** | −4,51 |
| rnn | **+6,07** | −5,10 |
| tstcc | −0,80 | −4,04 |

Onde o federado tem 7,5× mais otimização ele ganha em 3 dos 4 encoders; onde os
orçamentos empatam, perde nos 4 entre 2 e 5 pp.

**Como reportar (D8).** O orçamento é **parte da intervenção** — "centralizado como
se pratica" × "federado como se pratica" —, **e a tabela leva junto a leitura a
orçamento pareado**, que sai de graça: o braço federado loga toda rodada, então
basta ler a rodada em que o orçamento sequencial iguala o centralizado.

### 7.1 Assimetria de *tuning* entre os braços (declarada em 2026-08-24)

Além do orçamento de otimização, os dois braços **não recebem o mesmo cuidado de
hiperparâmetro**:

| Braço | Tratamento da LR |
|---|---|
| **Federado** | tunada por **encoder × federação × regime**, grade de 6 pontos, na validação deste repo (D9) |
| **Centralizado** | **1e-4 fixo** — Tabela 12 do benchmark, calibrada no cenário *centralizado com rótulo cheio*; nunca revalidada aqui, em nenhum regime |

**Decisão de escopo:** manter assim. Buscar LR também no centralizado abriria a
regressão de hiperparâmetros (weight decay, schedule, …) que D6/D7 fecham de
propósito.

**O viés tem direção conhecida e é preciso dizê-la:** ele **subestima o custo da
federação** — ou seja, empurra na direção que favorece a hipótese do trabalho. Como
calibragem, na busca federada o `1e-4` é a **pior** das quatro LRs originais em
11/20 células (`k=1`) e 12/20 (`Full`). Isso é evidência *federada* e não se
transfere 1:1 para o centralizado — FedAvg com `E=5` muda a dinâmica de LR
efetiva —, mas basta para afirmar que a LR do braço centralizado é **não
validada**, não que seja ótima.

Consequência prática para a leitura: o Δ federado − centralizado é um **limite
inferior** do custo de federar. Um Δ favorável ao federado no `k=1` não separa
"federar ajuda" de "o centralizado está mal tunado ali".

**Ameaça à validade a declarar:** `R = 150` é **nosso** (veio da análise de curva de
2026-07-28), não da literatura — FedEMA e FedST usam `R=100`, Saeed et al. 30–50. Sob
o enquadramento "como se pratica", é o único knob sem padrão externo, e é justamente
o que fabrica a assimetria no `k=1`.

## 8. As decisões e o que as sustenta

| # | Decisão | Por quê |
|---|---|---|
| **D1** | SimCLR **fora** da 1ª onda — RQ2 = TF-C + LFR | implementação nova; fica como extensão |
| **D2** | **KuHar fora de todos os experimentos** | (i) pareamento impossível: só 14/57 clientes têm as 6 classes, e o total por classe varia 14–54 em `k=1`; (ii) piso de batch: o menor cliente dá 0 batches; (iii) os autores do DAGHAR relataram que **o split de validação do KuHar** não representa a distribuição global |
| **D3** | **Partição natural** na RQ1/RQ2 | `B=192` quebra o "exatamente as mesmas janelas" do PICOC e é **incompatível** com clientes máximos (`device:MotionSense:17` falha: só 14 elegíveis a 192 janelas). `B=192` segue sendo o protocolo da **RQ3** |
| **D4** | **Seleção inalterada** (early stopping + `best.ckpt`) | o relato dos autores é específico do KuHar |
| **D5** | **Batch 64** no pré-treino | não perde nenhum cliente e preserva os negativos da NT-Xent (`2×batch`) |
| **D6** | Busca federada **S1**: só a LR, `E=5` fixo | `E=5` vira **premissa declarada** com quatro citações (FedSC, FedEMA, Saeed et al., FedST), não resultado medido |
| **D9** *(2026-08-24)* | LR **por regime**, não por célula: `k=1` e `Full` decididos em separado; `k=2`/`k=4` **herdam a do `k=1`**. Grade estendida a 6 pontos (`1e-4 … 3e-2`) | dos 20 pares (encoder, federação), só **3 concordam** sobre a melhor LR entre os dois regimes; colapsar num valor só custava **1,08 pp** no `Full` — do tamanho do efeito que a RQ1 mede. A extensão da grade rendeu **+0,59 pp** médio (máx +5,13). Detalhe e tabela de regret em `scripts/rqs/lr_escolhida.py` |
| **D7** | **Sem** busca de hiperparâmetros de SSL | é o `✗` da anotação; herdados dos papers e já auditados em [`metodo_e_auditoria.md`](metodo_e_auditoria.md) |
| **D8** | Orçamento de otimização **declarado**, com leitura pareada junto | §7 |

## 9. Custo (calibrado em 2026-08-23)

Modelo ajustado por mínimos quadrados sobre medições na MX570A, replicando os
pontos do TITAN Xp de `plano_fedssl §2.3` (segundos por **cliente-rodada**,
`resnetse5`):

| Fase | intercepto | por batch |
|---|---|---|
| fine-tuning SL (E=5) | 0,043 | 0,037 |
| pré-treino TF-C (E=5) | 0,857 | 0,355 |
| pré-treino LFR (6k=30) | 2,081 | 0,417 |

Os 4 encoders custam ≈ **2×** o `resnetse5` (razão 20,2 : 8,5 : 6,0 : 5,4 min/job
de 28/07). Com **4 seeds**:

| Bloco | GPU-h | 3 GPUs |
|---|---|---|
| busca S1 de LR | 11 | 3,7 h |
| RQ1 centralizado | 8 | 2,7 h |
| RQ1 federado | 16 | 5,4 h |
| RQ2 pré-treino | 146 | 49 h |
| RQ2 fine-tuning | 32 | 11 h |
| **Total** | **213** | **~71 h ≈ 3 dias** |

O pré-treino é **69% do orçamento** e o LFR é ~57% dele. Com 3 seeds, ~75% disso.

⚠️ **A calibração mede job isolado.** A grade de 28/07 rodou a **10,0 min/job**
médios no cluster contra **4,4 min** medidos aqui para o mesmo run — ~4,6× de
**contenção** com 8 jobs por nó. Planeje por throughput, não por tempo de job.

## 10. Como rodar

```bash
# 0. confere clientes e regimes contra os CSVs
poetry run python -c "import sys;sys.path.insert(0,'scripts');from rqs.config import *;verifica_clientes()"

# 1. busca da LR (1 seed) -> results/rqs/busca_lr.csv
poetry run python scripts/rqs/run_busca_lr.py --seed 0

# 2. RQ1, uma seed por vez
poetry run python scripts/rqs/run_rq1_centralizado.py --seed 0
poetry run python scripts/rqs/run_rq1_federado.py     --seed 0

# 3. RQ2: pré-treino primeiro, fine-tuning depois
poetry run python scripts/rqs/run_rq2.py --fase pretrain --seed 0
poetry run python scripts/rqs/run_rq2.py --fase finetune --seed 0
```

Todo driver aceita `--encoder --dataset --k --seed --gpus --max-parallel --force`
e **retoma**: pula job cujo parcial/checkpoint já está completo. Saídas em
`results/rqs/`, checkpoints em `checkpoints/rqs/` (isolados dos artefatos das
grades antigas via `FEDHAR_SUP_CKPT_ROOT` / `FEDHAR_FEDSSL_CKPT_ROOT`).

## 11. Fora do escopo / em aberto

- **RQ3** — sem desenho. Sai do DAGHAR; candidato **ExtraSensory**. O protocolo
  `B=192` de `plano_fedssl §2.1` continua sendo o dela.
- **SimCLR** — 2ª onda.
- **`k=8`** — 5º regime opcional se a curva `1-2-4-Full` ficar rala.
- **Seed 3** — incremento posterior; os drivers não re-rodam o que já existe.
- **KuHar como estudo do limite de viabilidade** — `plano_fedssl §3.3` mantém a
  proposta; não faz parte das RQ1/RQ2.

## 12. Proveniência

Anotação de caderno de 2026-08-21 (foto), transcrita e conferida com o Miguel em
23/08; números dos dados medidos por `scripts/analysis/client_regimes.py` e
`dataset_facts.py`; custo calibrado em 23/08 (`logs/calib-mx570a.log`,
`logs/calib2-mx570a.log`); decisões D1–D8 fechadas na sabatina de 23/08.
