> **📦 ARQUIVADO em 2026-07-27.** Design do pré-treino federado simulado (2026-07-13/14). **§3, §6 e §7 foram absorvidos em `../plano_fedssl.md §4`** e implementados em `scripts/ssl/pretrain_fed.py`. O resto (modo `silo`, onda B1, decisão D-K) foi superado pelo pivô e pela revogação da D-K. Mantido pela discussão dos Modos A/B e pela transcrição da lousa do orientador.
>
> Índice dos documentos vivos: `../README.md`.

---

# Plano — Pré-treino federado SIMULADO (sem Flower) + partição cross-device

> **⚠️ ATUALIZAÇÃO 2026-07-21 — cross-silo abandonado; cross-device é o eixo único.**
> A federação **cross-silo** (1 dataset/cliente; cenários 1–8 do `federated_eval.csv`)
> foi **abandonada como desenho e como controle** (decisão com o orientador). Os
> ~8 pp de domain shift viram **preliminar/motivação**. Consequência para ESTE doc:
> o modo de partição **`silo`** e a **Onda B1 (silo, all6)** também são cross-silo →
> **superseded pelo modo `device-<dataset>`** (cross-device). O controle honesto do
> custo de domain shift passa a ser **Δ(cross-domain − in-domain)**; o baseline
> supervisionado federado = 3 experimentos cross-device (in-domain RW_thigh,
> in-domain MotionSense, cross-domain RW_thigh+MotionSense). Ver
> `docs/analise_domain_shift.md`.

*Escrito em 2026-07-13, a partir da conversa do Miguel com o orientador
(mesma data). Duas ideias novas: (1) cenário **cross-device** particionando
por usuário ("cobaia"); (2) **não usar Flower no pré-treino** — simular o
FedAvg com um loop Python direto sobre o pipeline SSL centralizado já
validado. Este doc é o design de implementação dessa proposta e **supersede o
veículo de execução** de `plano_experimento3_fedssl.md` (Fases 1–5 daquele
plano: `ssl_client.py`, `run_federated_ssl.py`, `run_all_ssl.py` não serão
criados). As **decisões científicas** do plano antigo (§3.1–3.6: o que
agregar, paridade de budget, coerência DPP, avaliação em 3 níveis, riscos de
divergência) são herdadas e referenciadas aqui. A Fase 6 do plano antigo
(finetuning federado via Flower) permanece válida e intocada.*

> **Fonte primária (recebida 2026-07-14)**: foto da lousa em
> `docs/lousa_orientador_fedssl.png`. Transcrição do trecho relevante:
> `def pt_(D̄, nome_base, id_client)`, `F(D) → d` (particionamento) e
> `[Pipeline-PT] → w̄_c1, w̄_c2, …` — "lista de listas" ⇒ **conjunto das
> partes**: as combinações de agregação tomadas sobre os PESOS salvos.
> Confirmado com o Miguel (2026-07-14): a lousa desenha o fluxo **one-shot**
> (sem loop de rodadas) ⇒ o **Modo A (§5) é o núcleo da proposta**; o
> multi-round (Modo B) é a extensão natural. `nome_base` nomeia o corpus do
> treino (ex.: `"UCI"`, `"UCI+MotionSense"`); corpora mistos NÃO são clientes
> federados — são o **referencial centralizado** por combo (§5, taxonomia).
> Respostas às perguntas [LOUSA] registradas em §10.

---

## 1. As duas ideias e o que mudam

### 1.1 Cross-device: clientes = usuários

Hoje os cenários federados são cross-silo (`partitions.py`): 1 dataset por
cliente (cenário 1), IID global (2), IID intra-domínio (3–8). A proposta
adiciona o eixo **cross-device**: dentro de um dataset, cada cliente é um
usuário do `train.csv` (coluna `user`). Ganhos:

- **Realismo**: FL existe por causa de dispositivos pessoais; particionar por
  pessoa é o cenário canônico da literatura (non-IID "natural", não simulado).
- **Análise in-domain de colaboração**: FedSSL entre usuários do MESMO dataset
  vs o especialista centralizado nesse dataset (mesmo budget) — a versão
  non-IID dos cenários 3–8 do baseline supervisionado.

**Números medidos (2026-07-13, `train.csv` da standardized_view):**

| Dataset | janelas train | usuários | min/mediana/max por usuário | usuários < 64 | < 128 |
|---|---|---|---|---|---|
| KuHar | 1.392 | 57 | 1 / 10 / 103 | 48 | 57 |
| MotionSense | 3.558 | 17 | 165 / 211 / 244 | 0 | 0 |
| UCI | 2.420 | 21 | 99 / 113 / 139 | 0 | 18 |
| WISDM | 8.748 | 36 | 235 / 236 / 476 | 0 | 0 |
| RealWorld_thigh | 10.338 | 10 | 957 / 1024 / 1204 | 0 | 0 |
| RealWorld_waist | 10.332 | 10 | 958 / 1023 / 1201 | 0 | 0 |

Verificado também: **train/val/test são disjuntos por usuário nos 6 datasets**
⇒ particionar o train por usuário não toca a avaliação (test.csv continua
sendo exatamente o mesmo eval de sempre, sem vazamento).

**Decisão D-K (KuHar)**: com mediana de 10 janelas/usuário e 48/57 usuários
abaixo de um batch (64), clientes-usuário puros são inviáveis (TF-C usa
`drop_last=True` obrigatório ⇒ zero steps; LFR teria batches minúsculos e BN
instável). Proposta: **agrupar os usuários de KuHar em 6 super-clientes**
(~232 janelas cada; usuários inteiros, nunca divididos entre clientes —
preserva o non-IID por pessoa) e registrar como limitação/achado de realismo:
"no cenário cross-device real, participantes com pouquíssimo dado não
sustentam treino local". Alternativa (se preferirmos pureza): excluir KuHar da
onda cross-device. Decisão nossa, não coberta pela lousa — validar com o
orientador (§10, item 7).

Nota UCI: usuários ≥ 99 janelas ⇒ ok com batch 64 (≥ 1 step/época), mas o
`drop_last` do TF-C descarta até 63 janelas/época/cliente — o assert F2 de
`client.py` deve ser replicado no simulador (falhar alto se um shard < batch).

### 1.2 FedAvg manual: simular a federação, não orquestrá-la

No pré-treino não há seleção de clientes, stragglers nem agregação segura —
full participation, cross-silo/cross-device, agregação FedAvg pura. Toda a
maquinaria do Flower (ray, grpc, `NumPyClient`) é overhead sem conteúdo
científico **nesta fase**. A matemática do FedAvg é: para cada rodada,
(a) copiar o estado global para cada cliente; (b) treinar E épocas locais;
(c) média ponderada por n_k dos estados resultantes. Um loop Python faz isso
exatamente, reusando `pretrain_lfr.py`/`pretrain_tfc.py` — o pipeline que JÁ
passou nos gates de validação contra o benchmark.

O que ganhamos (além de simplicidade):

1. **Reuso total do pipeline validado** — mesmas classes, mesmo Trainer, mesma
   semântica de época; a chance de "outra v0 silenciosa" despenca.
2. **Coerência dos projetores/DPP do LFR de graça** (§3.4 do plano antigo era
   um spike inteiro): o modelo global é construído UMA vez; clientes recebem
   uma cópia do estado completo (projetores e seleção inclusos). Sem filtro de
   chaves para correção — o filtro vira só contabilidade de bytes (§3.4 aqui).
3. **Determinismo e resume baratos**: estado global salvo por rodada; retomar
   = carregar a última rodada. Sem portas, sem ray.
4. **Re-agregação post-hoc entre combinações de domínios** (ideia central do
   orientador): no modo one-shot (§5, Modo A), os checkpoints por
   domínio/cliente já existem e QUALQUER subconjunto custa só a média + um
   downstream.
5. **Ablação R×E barata** (frequência de agregação) — o loop expõe `--rounds`
   e `--local-epochs` sem nenhuma infra extra.

O que perdemos / precisamos dizer com honestidade (no artigo também, §8):
é **simulação exata de FedAvg** — como aliás era a simulação do Flower; a
literatura de FL publica esmagadoramente sobre simulação. O Flower continua
no projeto onde ele já provou valor e onde a comparabilidade exige: o
**finetuning federado** (baseline supervisionado já medido com ele; Exp. 2/3
usam a mesma pilha — plano antigo, Fase 6).

---

## 2. Herança do plano antigo (`plano_experimento3_fedssl.md`)

| Decisão do plano antigo | Status aqui |
|---|---|
| §3.1 modelo local = `build_lfr`/`build_tfc` completos; Trainer por rodada | **herdada** (é literalmente o que o loop faz) |
| §3.2 o que agregar (TF-C: backbone gêmeo inteiro; LFR: backbone + 6 preditores ativos) | **herdada como contabilidade de bytes**; na simulação a média default cobre o estado inteiro (§3.4) |
| §3.3 paridade de budget (TF-C R=100×1; LFR R=100×bloco de 6) | **herdada** integralmente |
| §3.4 projetores/DPP idênticos entre clientes | **resolvida por construção** (cópia de estado); spike vira uma verificação de 5 linhas |
| §3.5 avaliação em 3 níveis (loss por rodada / milestones / downstream final) | **herdada** |
| §3.6 riscos de divergência + mitigações (agregação frequente, fedbn, lr, achado negativo) | **herdada** |
| Fases 1–5 (infra Flower SSL: `ssl_client.py`, `run_federated_ssl.py`, `run_all_ssl.py`) | **superseded — não criar** |
| Fase 6 (finetuning federado: `finetune_client.py`, `run_federated_ft.py`, init none/central/fed) | **mantida no Flower**, inalterada |
| Fase 7 (downstream + notebook) | **herdada** (cache novo, §6) |

---

## 3. Implementação: um arquivo, três funções

**Arquivo novo único de infra: `scripts/ssl/pretrain_fed.py`** (fica em
`ssl/` porque produz checkpoints SSL consumidos por `downstream_eval.py`;
importa partições de `federated/partitions.py`). No espírito do "uma ou duas
funções" do orientador:

```python
def fedavg(states: list[dict], weights: list[float],
           skip_prefixes: tuple[str, ...] = ()) -> dict:
    """Média ponderada de state_dicts (pesos n_k/n). Tensores float: média;
    inteiros (ex.: BatchNorm num_batches_tracked): média arredondada.
    skip_prefixes: chaves mantidas do estado global anterior (variantes
    fedbn/backbone-only) — default agrega TUDO (FedAvg padrão)."""

def local_pretrain(method, encoder, shard_dataset, global_state,
                   local_epochs, seed, device) -> dict:
    """1 cliente × 1 rodada: build_{lfr,tfc}(encoder) →
    load_state_dict(global_state) → ShardPretrainDataModule(shard) →
    L.Trainer(max_epochs=local_epochs, enable_checkpointing=False,
    logger=False) → devolve model.state_dict().
    ShardPretrainDataModule = Subset + o mesmo unwrap de rótulos do
    PretrainDataModule; drop_last=True p/ TF-C; assert len(loader) > 0
    (réplica do F2 da auditoria)."""

def run_fedssl(method, encoder, partition, combo, rounds, local_epochs,
               seed, aggregate="full") -> Path:
    """Loop principal: seed_everything(seed) → constrói o modelo global (LFR:
    DPP global na união do combo, UMA vez) → para r in rounds: clientes
    treinam sequencialmente na GPU do run a partir de cópia do estado global
    → fedavg → log CSV (round, ssl_val_loss, uplink/downlink analíticos) →
    milestone se r ∈ {1,5,10,25,50,100}. Salva
    backbone.ckpt final no layout do downstream."""
```

Decisões de implementação:

- **Paralelismo no nível do RUN, não do cliente**: cada run ocupa 1 GPU e
  itera clientes sequencialmente (rodada = ~1 época sobre a união — mesmo
  custo do centralizado). A grade paraleliza runs no `gpu_pool.py` existente,
  como todos os outros runners. Zero orquestração nova.
- **Seeds**: `seed_everything(seed)` no início (init global; garante init
  comum e reprodutível); antes de cada fit local,
  `seed_everything(seed*10**6 + round*10**3 + client)` (shuffles distintos e
  reprodutíveis). O otimizador é recriado a cada rodada (Adam reseta — igual
  ao Flower/FedAvg padrão; o gate G-EQ2 em §6 mede o efeito disso isolado).
- **DPP do LFR**: seleção executada UMA vez, pelo "servidor", sobre 128
  amostras da união do combo (no cross-device, união dos usuários). Clientes
  herdam via cópia do estado + índices fixados. Documentar no artigo como
  hipótese de "amostra pública" (igual ao plano antigo).
- **BatchNorm buffers**: default FedAvg agrega `running_mean/var` junto (é o
  que o baseline supervisionado Flower já faz — manter idêntico). Variante
  `fedbn` (skip_prefixes de BN) fica como extensão; nota: fedbn exige decidir
  que stats o modelo global usa no downstream (re-estimar numa amostra) — só
  atacar se a divergência do TF-C aparecer (§3.6 do plano antigo).
- **Bytes analíticos**: uplink = downlink = bytes dos tensores da coluna
  "Agregado" do §3.2 antigo (TF-C: `TFC_Backbone` inteiro; LFR: backbone + 6
  preditores ativos ≈ 127 MB/rodada no tstcc — continua sendo o resultado de
  comunicação interessante). Mesmo formato de CSV do federado supervisionado.
- **Checkpoints**:
  `checkpoints/ssl_fed/<method>/<encoder>/<partition>/<combo>/seed<N>/`
  com `backbone.ckpt` (final, formato idêntico ao centralizado — recarrega
  `strict=True` no downstream), `round<R>/backbone.ckpt` (milestones) e
  `state_last.pt` (estado global completo + rodada, para resume).
  `partition ∈ {silo, iid, device-<dataset>}`; `combo` (= o `nome_base` da
  lousa) = datasets ordenados unidos por `+` (ex.: `KuHar+UCI+WISDM`), `all6`
  ou `lodo-<dataset>`; `id_client` (lousa) = o átomo dentro do combo — nome do
  dataset no cross-silo, id de usuário no cross-device.

---

## 4. Partições (extensão de `scripts/federated/partitions.py`)

Nova função `make_ssl_client_datasets(partition, combo, seed)`:

- `silo` — 1 cliente por dataset do combo (cenário 1 restrito ao combo).
- `iid` — união do combo em 6 fatias IID (cenário 2; usado no gate G-IID).
- `device-<dataset>` — 1 cliente por usuário do train (KuHar: 6 grupos de
  usuários, decisão D-K). Implementação: ler a coluna `user` do CSV e mapear
  índices → `Subset`s (mesmo padrão `_iid_shards`).

Pesos de agregação: n_k = len(shard). Reusa `_train_dataset` (mesmo pipeline
de leitura dos baselines). Os cenários 1–8 existentes ficam intocados.

---

## 5. Modos e grades (ondas)

### Modo A — one-shot + conjunto das partes (o NÚCLEO da proposta; lousa 2026-07-14)

Cada cliente atômico treina até o fim de forma independente e agrega-se UMA
vez — FedAvg com R=1. **Regra estrutural (decisão do Miguel, 2026-07-14):
cliente federado nunca mistura datasets** — o átomo é 1 dataset (cross-silo)
ou 1 usuário (cross-device). O "conjunto das partes" da lousa aplica-se **na
agregação, sobre os pesos salvos**: com N pesos de init comum, qualquer
subconjunto S custa só a média + downstream.

**Taxonomia por combo S** (é o que estrutura a Tabela VII do artigo):

| Braço | O que é | Custo |
|---|---|---|
| `central(S)` | pré-treino centralizado na UNIÃO dos dados de S (`nome_base="A+B"`) — o contrafactual "e se pudéssemos juntar os dados" | 1 pré-treino por combo — **só p/ combos selecionados** (63 uniões ≈ 2016 pré-treinos, inviável) |
| `soup(S)` | média one-shot dos pesos dos átomos de S — federado com 1 rodada de comunicação | ~zero (média + downstream) |
| `fedavg_R(S)` | multi-round (Modo B) — interpola entre soup e central | 1 run por combo |

Já existem de graça: `central(singleton)` = especialistas do Exp. 1,
`central(all6)` = `combined`, e os átomos do soup cross-silo SÃO os
especialistas (init comum por seed — `pretrain_single` chama
`seed_everything(seed)` antes do build e a fonte não entra no build;
**spike S0: PASS em 2026-07-14**, hashes de init idênticos por seed em
LFR e TF-C, e distintos entre seeds). Logo o Modo A cross-silo **não exige
nenhum pré-treino novo**.

**Ondas do Modo A:**

- **A1 — scan do conjunto das partes (barato)**: TODOS os 57 combos novos
  (|S| ≥ 2; singletons já avaliados) × 4 encoders × 2 métodos × seed 0, com
  downstream reduzido (`linear` × regime {100-shot, full}) — mapa completo de
  "quais combinações de domínios somam e quais se atrapalham". Medir o 1º job
  antes de extrapolar (estimativa: 456 jobs curtos, ~1 noite em 8 GPUs).
- **A2 — avaliação profunda**: combos selecionados (`all6`, `lodo-<d>`×6 e os
  destaques do scan) × 4 seeds × protocolo completo (2 protocolos × 4
  regimes). O `lodo` é o análogo federado do `comb2target` já medido: sopa
  dos 5 domínios avaliada no 6º — comparação-síntese do artigo.
- **A3 (opcional, gated no A2)** — `central(S)` para combos selecionados
  (ex.: uniões LODO nos pares prioritários TF-C×cnnpff, TF-C×rnn, LFR×rnn):
  ~72 pré-treinos; só rodar se a comparação soup vs central exigir o
  contrafactual exato além do `combined`.

- Expectativa honesta: a média de modelos treinados 100 épocas de forma
  independente costuma degradar (drift de bacia mesmo com init comum). Não é
  problema: o Modo A é o extremo R=1 da curva comunicação×qualidade e o piso
  do Modo B. Se NÃO degradar, é resultado forte por si (pré-treino federado a
  custo de comunicação ~zero). O scan A1 revela isso rápido e barato.

### Modo B — multi-round (FedAvg-SSL de verdade, o Exp. 3)

Budget com paridade ao centralizado (§3.3 antigo): TF-C **R=100 × 1 época
local**; LFR **R=100 × bloco de 6 épocas de Trainer** (preserva a alternância
5:1 preditor:backbone; bloco < 6 é proibido).

- **Onda B1 (silo, all6)**: 2 métodos × 4 encoders × 4 seeds = **32 runs**.
  Custo/run ≈ 1 pré-treino centralizado `combined` + overhead de Trainer
  (~2 s × 6 clientes × 100 rodadas ≈ 20 min). Milestones {10, 25, 50, 100} →
  curva "qualidade × rodada" com downstream nos milestones do melhor combo.
- **Onda B2 (cross-device in-domain)**: `device-<dataset>` para os 6 (KuHar
  agrupado). Priorizar os pares (método, encoder) com maior sinal no
  centralizado — pelos números de 2026-07-13: **TF-C×cnnpff** (+18.6 pp
  in-domain 10-shot), **TF-C×rnn** (+26.9), **LFR×rnn** (+17.2); tstcc por
  último (SSL ≈ neutro nele). 3 pares × 6 datasets × 4 seeds = 72 runs, cada
  um custando ~1 pré-treino do dataset (barato; KuHar/UCI/MotionSense são
  pequenos). Comparação: vs especialista centralizado do mesmo dataset.
- **Onda B3 (ablação R×E, opcional)**: R ∈ {1≡Modo A, 10, 25, 50, 100} com
  E = 100/R, silo all6, 1–2 pares — a curva de frequência de agregação.
- **Onda B4 (combos no multi-round, opcional/caro)**: LODO multi-round exige 1
  run por combo (diferente do Modo A, não há re-agregação post-hoc possível —
  o estado global evolui). Só se B1 der sinal.

### Downstream final (Exp. 2/3 fechando)

Inalterado do plano antigo Fase 6: finetuning federado **no Flower** (cenário
1, R=50, init ∈ {aleatório (baseline já medido), SSL-central `combined`,
SSL-fed deste plano}) — comparabilidade direta com `federated_eval.csv`.

---

## 6. Spikes e gates (ordem de execução)

| # | O quê | Gate | Custo |
|---|---|---|---|
| S0 | Init idêntico entre fontes: construir `build_lfr`/`build_tfc` 2× (mesma seed, fontes distintas), comparar hash dos state_dicts | **✅ PASS (2026-07-14)**: hashes idênticos por seed (LFR×{cnnpff,rnn}, TFC×cnnpff), distintos entre seeds ⇒ Modo A com ckpts do Exp. 1 é válido | minutos |
| S1 | Sopa all6 tfc×cnnpff×seed0 + 1 downstream | acc > acaso (1/6) e registrada (expectativa aberta) | ~30 min |
| S2 | Loop 3 rodadas TF-C silo (1 combo) | loss cai, sem NaN, CSV de bytes ok | ~1 h |
| S3 | LFR 2 rodadas × bloco 6 | backbone muda exatamente 1 época efetiva/rodada (hash por época); DPP global aplicado | ~1–2 h |
| G-EQ1 | 1 cliente (=combined), **R=1 × E=100** vs centralizado, mesma seed | métricas downstream idênticas (ideal: ckpt bit-idêntico) — valida o simulador inteiro | 1 pré-treino |
| G-EQ2 | 1 cliente, **R=100 × E=1** vs centralizado | quantifica isolado o custo do fatiamento (reset do Adam/rodada), SEM agregação — número novo do artigo | 1 pré-treino |
| G-IID | `iid` 6 clientes, R=100 vs centralizado `combined` | Δ downstream @full ≥ −3 pp (tolerância a confirmar no primeiro run) — herda a Fase 4 antiga | 2 runs |

Regra de sempre: **medir o 1º job de cada onda antes de extrapolar custo**;
tmux + `tee logs/` (CLAUDE.md); commit ao final de cada onda.

## 7. Downstream e caches

- `downstream_eval.py`: hoje restringe `--pretrain-source` a `SOURCES`; ganhar
  `--ckpt-dir` explícito (caminho direto do `backbone.ckpt`) — mudança mínima,
  sem tocar o fluxo existente.
- Cache novo: `results/ssl_fed_eval_transfer.csv` com colunas
  `method,encoder,mode,partition,combo,rounds,seed,protocol,n_shots,target,
  test_acc,test_f1_macro` + parciais em `results/ssl_fed_parts/`
  (padrão dos runners atuais).
- Notebook: seção nova no `ssl_lfr_avaliation.ipynb` (ou notebook próprio
  `fedssl_avaliation.ipynb`) lendo só o cache.

## 8. Como descrever no artigo

- III-A ganha a partição cross-device (por usuário; splits já são
  user-disjuntos — sem vazamento) e a nota de que KuHar exige agrupamento.
- III-D (Exp. 3): "pré-treino federado por **simulação exata de FedAvg**
  (full participation): treino local com o pipeline SSL validado (Exp. 1) e
  média ponderada dos parâmetros por rodada; custo de comunicação contado
  analiticamente sobre os tensores efetivamente transmissíveis. O finetuning
  federado (Exp. 2/3) executa no Flower, idêntico ao baseline." Nada a
  esconder: é o padrão da literatura de FL.
- Novos resultados que este plano habilita e o antigo não tinha: curva R×E
  (frequência de agregação), G-EQ2 (custo do fatiamento por si), sopas LODO
  (análogo federado do comb2target), cross-device in-domain.

## 9. Riscos

- **Sopa catastrófica (Modo A)** — provável; virar curva R (Modo B3) e/ou
  achado negativo caracterizado (notas §7: também é entregável).
- **Divergência contrastiva do TF-C multi-round** — mitigação herdada §3.6
  (agregação frequente já é default; fedbn; lr; reportar).
- **LFR federado caro** (600 épocas de Trainer/run no combined-equivalente) —
  medir 1º job; se inviável no dl-16, cortar LFR do B1 para {rnn} apenas
  (é o único encoder onde LFR ganha muito).
- **Disco**: milestones × 32+72 runs; tstcc/TFC são os maiores ckpts — manter
  só {10,25,50,100} e o final.
- **KuHar cross-device** — decisão D-K documentada (agrupamento) e defendida
  como achado de realismo.

## 10. Perguntas [LOUSA] — respostas registradas (2026-07-14) e pendências

Respondidas pelo Miguel após a chegada da foto:

1. O parâmetro borrado é **`nome_base`** — o nome do corpus de treino (ex.:
   `"UCI"`, `"MotionSense"`, `"UCI+MotionSense"`); vira a chave de
   nomeação/salvamento dos pesos. A lousa desenha o fluxo **one-shot** (sem
   loop de rodadas) ⇒ Modo A é o núcleo; multi-round é extensão.
2. **"Lista de listas" = conjunto das partes (todas as combinações)** — mas,
   refinado em discussão: combinação aplicada **na agregação dos pesos**
   (clientes atômicos, 1 dataset/usuário por cliente), não na mistura de
   dados dentro de um cliente. Mistura de dados sobrevive só como o braço
   `central(S)` (referencial centralizado, combos selecionados) — ver
   taxonomia do §5.
3. O conteúdo cortado da foto não pertence a este processo — ignorado.

Pendentes (não cobertas pela lousa; levar ao orientador quando conveniente):

4. Confirmar que o "sem Flower" vale só para o pré-treino — o finetuning
   federado (Exp. 2/3) segue no Flower para comparabilidade com o baseline
   já medido (assunção atual do plano).
5. R/E do multi-round: a lousa não prescreve; mantidos os defaults herdados
   (TF-C 100×1, LFR 100×6) + ablação R×E (onda B3).
6. Cross-device: assumido átomo = usuário (o `u¹` da lousa, não confirmado
   explicitamente); in-domain primeiro (onda B2), global (151 usuários) só
   como extensão.
7. KuHar no cross-device: a decisão D-K (agrupar em 6 super-clientes) é
   nossa, não da lousa — validar com o orientador.

## Apêndice — âncoras de custo (grades reais de jul/2026, 8× TITAN Xp)

- Grade TF-C completa (112 pré-treinos de 100 ep + 112 downstream): **12,4 h**
  em 8 GPUs ⇒ pré-treino TF-C médio ~35–45 min; downstream ~7 min.
- LFR tstcc (28 pré-treinos de 600 ep de Trainer + 28 downstream): **~5 h** em
  8 GPUs ⇒ pré-treino LFR ~1–1,4 h (tstcc é o pior caso de VRAM/params).
- Estimativas das ondas: A1 ≈ 456 jobs de downstream reduzido (~1 noite em 8
  GPUs — medir o 1º); A2 ≈ 26 h·GPU de downstream completo (~3–4 h em 8
  GPUs); B1 ≈ 32 runs × custo do pré-treino `combined`-equivalente + ~20 min
  de overhead de Trainer cada; B2 ≈ 72 runs pequenos (datasets individuais).
  Medir o 1º de cada.
