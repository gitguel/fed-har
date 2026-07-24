# Plano de implementação — Experimento 3: pré-treino SSL federado + finetuning federado (FedAvg-SSL)

> **⚠️ ATUALIZAÇÃO 2026-07-21 — PIVÔ cross-silo → cross-device.** Além do STATUS
> abaixo (2026-07-13, sem Flower no pré-treino), a federação **cross-silo** foi
> **abandonada como desenho e como controle** (decisão 2026-07-21). Os ~8 pp de
> domain shift viram **preliminar/motivação**; o controle honesto passa a ser
> **Δ(cross-domain − in-domain)** no eixo **cross-device**. Ver
> `docs/analise_domain_shift.md` e `docs/plano_fedssl_simulado.md`.

> **STATUS (2026-07-13): parcialmente superseded.** Por decisão do orientador,
> o pré-treino federado NÃO usará Flower — será uma simulação exata de FedAvg
> (loop Python sobre o pipeline SSL centralizado), com o eixo novo de partição
> cross-device por usuário. Ver `docs/plano_fedssl_simulado.md`, que herda as
> decisões científicas deste doc (§3.1–3.6) e supersede as Fases 1–5
> (`ssl_client.py`/`run_federated_ssl.py`/`run_all_ssl.py` não serão criados).
> A **Fase 6** (finetuning federado via Flower) e a Fase 7 (downstream)
> continuam válidas como estão.

*Escrito em 2026-07-07, com base na leitura de `scripts/federated/*` e
`scripts/ssl/*` no estado atual do repo (inclui as respostas do spike TF-C de
2026-07-07 registradas no cabeçalho de `pretrain_tfc.py`). Complementa
`plano_implementacao_tstcc_tfc.md` (Exps. centralizados) e as notas internas
(Seções 2 e 7). Este é o componente de **maior risco técnico** do escopo
oficial; um resultado negativo bem caracterizado também é entregável válido
(notas §7).*

> **Nota de processo**: este documento foi produzido por uma sessão paralela
> somente-de-docs. Nenhum arquivo de `scripts/`, `results/` ou `notebooks/`
> foi tocado; todos os trechos de código abaixo são **propostas** a serem
> implementadas pela sessão dona do pipeline, no padrão dos scripts existentes.

## 0. Como usar este plano

Mesmas regras do plano tstcc+TF-C:

- **Execução sequencial por fases com gates**; falhou o gate, o problema está
  confinado à fase.
- **Spike antes de grade** — aqui isso é ainda mais crítico: FedAvg-SSL tem
  dois modos de falha silenciosa (divergência contrastiva e agregação de
  parâmetros incoerentes) que só aparecem com validação ponta a ponta.
- **Commit + push ao final de cada fase**; grades em tmux com `tee` para
  `logs/` (CLAUDE.md); GPUs explícitas; um job pesado por GPU.
- **Pré-requisito duro**: Fases 6–7 do plano tstcc+TF-C concluídas (infra
  `pretrain_tfc.py` + `--method` no downstream, gate 7 aprovado). O Exp. 3
  reusa essa infra sem fork.

## 1. Contexto, objetivo e o que é reusado

**Objetivo (escopo oficial, Mês 3).** Avaliar a segunda estratégia prometida:
o pré-treino SSL (LFR e TF-C) executado **dentro do processo federado**
(FedAvg sobre os parâmetros do modelo SSL, 6 clientes cross-silo), seguido de
**finetuning federado** do classificador. Comparar contra: (a) finetuning
federado com init aleatória (baseline já medido, `results/federated_eval.csv`)
e (b) finetuning federado com init do SSL **centralizado** (Exp. 2 — a infra
de finetune-a-partir-de-checkpoint é a MESMA e nasce aqui, Fase 6).

**Blocos existentes reusados sem mudança conceitual:**

| Bloco | O que fornece ao Exp. 3 |
|---|---|
| `federated/partitions.py` | shards por cliente (cenário 1 non-IID, 2 IID; nada muda — SSL ignora `y`, mas o Dataset já entrega `(x, y)` e o cliente descarta `y`) |
| `federated/server.py` | padrão de estratégia FedAvg + `evaluate_fn` centralizada (adaptada: no pré-treino não há rótulo → ver §4.4) |
| `federated/run_federated.py` / `run_all.py` | esqueleto de entrypoint + driver multi-GPU com parciais/resume (copiar o padrão, não os arquivos) |
| `ssl/pretrain_lfr.py` | `build_lfr()` (LFR completo em torno do backbone), `PretrainDataModule`, `ssl_ckpt_dir()` |
| `ssl/pretrain_tfc.py` | `build_tfc()` (TFC_Model em modo SSL), respostas do spike (FFT interna; `drop_last=True` obrigatório — máscara NT-Xent fixa em 2×batch) |
| `ssl/encoders.py` | `build_backbone` / `build_tfc_backbone` (fábricas por encoder) |
| `ssl/downstream_eval.py` | avaliação da qualidade da representação (linear/finetune × 4 regimes × 6 alvos) — ganha os métodos `*-fed*` (§4.5) |

**Arquivos novos propostos** (checklist completo no Apêndice B):

```
scripts/federated/ssl_client.py        # FlowerClient SSL (LFR e TF-C) + filtro de chaves
scripts/federated/run_federated_ssl.py # entrypoint de um run de pré-treino FedSSL
scripts/federated/finetune_client.py   # FlowerClient de finetuning a partir de backbone (Exp. 2 e 3)
scripts/federated/run_federated_ft.py  # entrypoint de finetuning federado (init none/central/fed)
scripts/federated/run_all_ssl.py       # driver das grades (padrão run_all.py: parciais + resume)
```

## 2. Resumo das fases

| Fase | Entrega | Grade | Custo estimado* |
|---|---|---|---|
| 0 | Pré-condições (TF-C centralizado fechado, commit limpo) | — | 15 min |
| 1 | Decisões de design congeladas + spike FedSSL-**TF-C** (1 combo, 3 rodadas) | 1 run curto | 1 dia |
| 2 | Spike FedSSL-**LFR** (DPP, alternância, filtro de preditores) | 1 run curto | 1 dia |
| 3 | Infra completa (runner, milestones, variantes de agregação, downstream `*-fed*`) | smoke 4 encoders | 1–2 dias |
| 4 | **Gate de equivalência IID** (cenário 2 ≈ centralizado `combined`) | 2 runs + downstream | ~1 dia de GPU |
| 5 | Grade de pré-treino FedSSL (ondas A→C) | até 64 runs | 3–5 dias de GPU |
| 6 | Finetuning federado (infra + runs Exp. 2 + Exp. 3 + baselines) | ~160 runs R=50 | 1–2 dias de GPU |
| 7 | Downstream centralizado dos backbones FedSSL + notebook + docs | 64 jobs | ½–1 dia |

\* Ancorado no Apêndice C; **medir o 1º job de cada grade antes de extrapolar**.

---

## 3. Decisões de design (o coração do plano)

### 3.1 O que é o "modelo local" de cada cliente

- **TF-C**: o cliente instancia `build_tfc(encoder)` (o `TFC_Model` completo em
  modo SSL). Todos os parâmetros treináveis estão no `TFC_Backbone` (2 encoders
  + 2 projetores); a NT-Xent não tem parâmetros. O loader local usa
  `drop_last=True` (spike TF-C, item 5).
- **LFR**: o cliente instancia `build_lfr(encoder)` (backbone + 60 projetores
  congelados + 60 preditores). Treináveis: backbone + preditores.

**Loop local**: diferente do cliente supervisionado (loop torch manual), as
losses SSL vivem dentro dos LightningModules do minerva (alternância do LFR,
NT-Xent do TF-C). Proposta: o `fit()` do cliente dirige um
`L.Trainer(max_epochs=<bloco local>, enable_checkpointing=False, logger=False,
enable_progress_bar=False)` por rodada — replica exatamente a semântica de
treino do centralizado e evita reimplementar as losses. O overhead de
instanciar um Trainer (~1–2 s) é desprezível frente ao custo da época SSL.
⚠️ Consequência para o LFR: o contador `current_epoch` **reinicia a cada
rodada** — ver §3.3.

### 3.2 O que agregar no FedAvg (por método)

Princípio: agregar **apenas o que é treinado e precisa ser globalmente
coerente**; o resto ou é constante (não transmitir) ou é deliberadamente local
(variantes). Implementação: `get_parameters`/`set_parameters` com **filtro de
chaves** — o servidor e os clientes derivam a mesma lista ordenada de chaves do
`state_dict` por regras de prefixo (função `param_keys(model, method, variant)`
em `ssl_client.py`); a contagem de bytes de comunicação passa a usar **só os
tensores efetivamente transmitidos** (hoje `run_federated.py` conta o
state_dict inteiro — correto para o supervisionado, errado aqui).

| Método | Agregado (default `full`) | Constante (não transmitir) | Variantes (flag `--aggregate`) |
|---|---|---|---|
| TF-C | `TFC_Backbone` inteiro (2 encoders + 2 projetores) | — | `fedbn`: exclui params+buffers de BatchNorm (ficam locais) |
| LFR | backbone + **somente os 6 preditores ativos** (selecionados pelo DPP) | 60 projetores congelados (idênticos por construção, §3.4) e 54 preditores não selecionados | `backbone-only`: preditores ficam locais (estilo FedU — preditor como componente personalizado); `fedbn` |

Por que "somente os 6 ativos" importa: para o `tstcc` (2304-d), a lista
completa de preditores tem 60 × 2304² ≈ **318M parâmetros ≈ 1.27 GB fp32 por
cliente por rodada** — inviável e sem sentido (54 deles nunca recebem
gradiente). Os 6 ativos são ≈ 32M params ≈ 127 MB/rodada — pesado mas viável, e
vira um **resultado de comunicação** interessante do artigo (LFR federado é
caro em uplink; TF-C não). ⚠️ Spike da Fase 2 confirma se o minerva treina só
os selecionados ou todos.

### 3.3 Orçamento de treino: rodadas × épocas locais (paridade com o centralizado)

Fixar o **budget total de épocas sobre os dados** igual ao centralizado, para
que a comparação Exp. 1 vs Exp. 3 seja "mesma computação, topologia diferente":

| Método | Centralizado | Federado (proposta) | Racional |
|---|---|---|---|
| TF-C | 100 épocas na fonte | **R=100 rodadas × 1 época local** | agregação mais frequente = mitigação nº 1 de divergência contrastiva; 6 clientes × 1 época/rodada × 100 rodadas ≡ 100 épocas sobre a união |
| LFR | 600 épocas de Trainer (100 efetivas de backbone; alternância 1:5) | **R=100 rodadas × bloco local de 6 épocas de Trainer** (= 1 ciclo completo de alternância ⇒ 1 época efetiva de backbone/rodada) | como o `current_epoch` reinicia por rodada, o bloco de 6 é a menor unidade que preserva a razão 5:1 preditor:backbone do protocolo oficial |

⚠️ Spike (Fase 2) confirma a fase da alternância: se nas épocas 0–4 o LFR
treina preditores e na 5 o backbone (ou o inverso), o bloco de 6 preserva o
protocolo em qualquer caso; o que NÃO pode é bloco < 6 (backbone nunca treina
ou preditores nunca sincronizam). Expor `--local-epochs`/`--rounds` para a
ablação de frequência de agregação (§3.6).

### 3.4 LFR: projetores e seleção DPP idênticos entre clientes (ponto crítico)

Os alvos do LFR são projeções aleatórias congeladas. Se cada cliente tiver
projetores diferentes (ou selecionar 6/60 diferentes via DPP local), o FedAvg
médias preditores/backbones treinados contra **alvos distintos** — incoerente
por construção. Requisitos:

1. **Pesos dos projetores idênticos**: garantido se o servidor constrói o
   modelo com `seed_everything(seed)` e envia os pesos iniciais
   (`initial_parameters`) — mas como os projetores saem da mensagem (§3.2),
   cada cliente precisa construí-los localmente com a MESMA seed. Proposta:
   `seed_everything(seed)` antes do `build_lfr` tanto no servidor quanto em
   cada `client_fn` (mesma seed do run) ⇒ mesmos pesos por determinismo.
   Verificação no spike: hash dos tensores dos projetores em 2 clientes.
2. **Seleção DPP única e global**: a seleção no minerva acontece no
   `setup()` e **pode depender dos dados locais** (spike confirma). Proposta:
   o **servidor** executa a seleção uma única vez antes da rodada 1 (na
   simulação, sobre uma amostra da união; num deployment real seria uma
   amostra pública — documentar como hipótese) e envia os **índices dos 6
   ativos** aos clientes via `config` do `fit()`. Os clientes pulam a seleção
   local e fixam esses índices. Alternativa mais simples se o spike mostrar
   que a seleção é determinística dada a seed e independente dos dados:
   apenas herdar por seed.

### 3.5 Avaliação durante e após o pré-treino federado

Sem rótulos não há `evaluate_fn` de acurácia por rodada (como no
supervisionado). Proposta em 3 níveis:

1. **Por rodada (barato)**: loss SSL do modelo global numa amostra fixa
   held-out sem rótulos (ex.: 512 janelas do val de cada domínio) — detecta
   divergência/NaN cedo; logado no CSV do run
   (`round, ssl_val_loss, uplink_bytes, downlink_bytes`).
2. **Milestones (médio)**: checkpoint do backbone global nas rodadas
   {10, 25, 50, 100} em
   `checkpoints/ssl_fed/<method>/<encoder>/s<scenario>/seed<N>/round<R>/backbone.ckpt`
   — permite reconstruir a curva "qualidade da representação × rodada"
   post-hoc sem pagar probe por rodada.
3. **Final (caro, post-hoc)**: `downstream_eval.py` sobre o backbone final
   (e milestones selecionados), ver §4.5 — é o número que entra no artigo
   (Tabela VII do esqueleto).

O formato salvo é **idêntico ao centralizado** (state_dict do backbone LFR /
do `TFC_Backbone`), então o downstream recarrega `strict=True` sem conversão.

### 3.6 Riscos de divergência e mitigações (mapa completo no Apêndice D)

- **NT-Xent com negativos só locais (TF-C)**: cada cliente contrasta apenas
  dentro do próprio domínio; após a média, os espaços de embedding podem estar
  desalinhados (problema clássico de FedSSL contrastivo — FedU/FedEMA).
  Mitigações em ordem: (1º) agregação frequente (1 época local — já é o
  default §3.3); (2º) variante `fedbn` (o shift entre domínios é de features —
  exatamente o caso do FedBN); (3º) reduzir lr local (3e-4 → 1e-4) se a
  `ssl_val_loss` oscilar pós-agregação; (4º) reportar como achado negativo
  com a curva de divergência como evidência.
- **"Choque de agregação"**: logar a loss local ANTES e DEPOIS de receber os
  pesos globais (`fit()` mede 1 batch antes de treinar) — a diferença é a
  métrica direta de divergência entre rodadas; barata e vai para o CSV.
- **Preditores LFR desatualizados**: preditores perseguem um backbone que muda
  sob seus pés a cada agregação. O bloco local de 6 épocas (5 de preditor)
  re-sincroniza por construção; a variante `backbone-only` testa a alternativa
  de personalização.
- **Silos desbalanceados**: FedAvg pondera por `num_examples` — os domínios
  grandes (RealWorld) dominam a média, como no supervisionado (comparável por
  design). Com `drop_last=True` no TF-C, garantir `len(shard) >= batch` (vale
  p/ todos os shards do DAGHAR; assert no cliente).
- **BatchNorm sob domain shift**: buffers de BN médios entre domínios podem
  degradar todos os clientes — é a hipótese que a variante `fedbn` isola.
  ⚠️ No `fedbn`, o backbone global salvo fica **sem estatísticas de BN
  globais**; para o downstream centralizado, salvar adicionalmente os buffers
  de um cliente de referência OU recalibrar BN na união antes do probe
  (documentar a escolha; recalibração de 1 época sem gradiente é barata).

---

## 4. Fases

### Fase 0 — Pré-condições

1. Plano tstcc+TF-C concluído até a Fase 7 (gate 7 aprovado: TF-C centralizado
   validado contra o paper) — o Exp. 3 depende do `pretrain_tfc.py`/`--method`
   estáveis e dos backbones centralizados `combined` como referência.
2. `git status` limpo, grades federada/LFR/TF-C commitadas.
3. `results/federated_eval.csv` completo (96 ou 128 combos) — baseline de
   init aleatória.
4. GPUs livres anotadas (`nvidia-smi`); cluster Dl-16 (torch 2.5.1+cu118).

**Gate 0**: itens acima OK.

### Fase 1 — Spike FedSSL-TF-C (começa pelo método mais simples)

TF-C primeiro porque não tem DPP nem alternância — isola os problemas de
integração Flower×Lightning dos problemas específicos do LFR.

1. Implementar o mínimo de `ssl_client.py` (método tfc apenas) +
   `run_federated_ssl.py`: cliente monta `build_tfc(encoder)`, recebe pesos
   filtrados (backbone completo), treina 1 época local com `L.Trainer`,
   devolve pesos + `num_examples` + loss média local (antes/depois, §3.6).
2. Rodar: `run_federated_ssl.py --method tfc --encoder resnetse5 --scenario 1
   --seed 0 --rounds 3 --local-epochs 1` (tmux `spike-fedssl`).

**Perguntas que o spike responde** (anotar no cabeçalho do `ssl_client.py`):

1. `L.Trainer` dentro do ator Ray funciona (device, `accelerator="auto"` com
   `CUDA_VISIBLE_DEVICES` herdado)? Overhead por rodada aceitável (<10% do
   tempo de época)?
2. O filtro de chaves round-trip está correto (`state_dict` → ndarrays →
   `state_dict`, `strict=True` no subconjunto)?
3. A `ssl_val_loss` global decresce em 3 rodadas (sanidade, não performance)?
4. Backbone global salvo recarrega no `downstream_eval` (probe linear 2 épocas
   > acaso no KuHar)?
5. VRAM com 1 cliente por vez na TITAN Xp (pior caso tstcc: ~1.3 GB no
   centralizado — folga esperada; medir).

**Gate 1**: 5 respostas anotadas; loss decresce; probe > acaso; apagar
artefatos do spike.

**Commit**: `Spike FedSSL-TFC: ssl_client + run_federated_ssl mínimos`

### Fase 2 — Spike FedSSL-LFR (os riscos específicos)

1. Estender `ssl_client.py` ao LFR: `build_lfr(encoder)`, bloco local de 6
   épocas, filtro backbone+preditores-ativos, mecanismo de seleção DPP global
   (§3.4).
2. Rodar 1 combo (resnetse5 × cenário 1 × seed 0 × 3 rodadas).

**Perguntas do spike**:

1. A seleção DPP do minerva é data-dependente? Determinística dada a seed?
   (⇒ decide entre "índices via config" e "herança por seed", §3.4.)
2. Hash dos projetores é idêntico em 2 clientes construídos com a mesma seed?
3. Em qual época do ciclo o backbone treina (fase da alternância)? O bloco de
   6 épocas produz exatamente 1 época efetiva de backbone por rodada?
4. Os preditores não-selecionados recebem gradiente? (Se sim, o filtro de 6
   ativos descarta treino real — reavaliar §3.2.)
5. Tamanho real da mensagem por encoder (esp. tstcc: esperado ~127 MB) e tempo
   de serialização Flower/Ray com mensagens dessa ordem.

**Gate 2**: 5 respostas anotadas; 3 rodadas sem erro; loss de preditores
decresce; mensagem tstcc < 200 MB; apagar artefatos.

**Commit**: `Spike FedSSL-LFR: DPP global + filtro de preditores ativos`

### Fase 3 — Infra completa

1. `run_federated_ssl.py` final: flags `--method {lfr,tfc}`, `--scenario`,
   `--seed`, `--rounds`, `--local-epochs`, `--aggregate {full,backbone-only,fedbn}`,
   `--out` (parciais); CSV por run:
   `method, encoder, scenario, seed, aggregate, round, ssl_val_loss,
   agg_shock, uplink_bytes, downlink_bytes`; milestones de checkpoint (§3.5).
2. `run_all_ssl.py`: driver no padrão do `run_all.py` federado (pool de GPUs,
   parciais em `results/federated_ssl_parts/`, resume por rodada final,
   consolidação em `results/federated_ssl_pretrain.csv`).
3. Registro no downstream: `downstream_eval.py` aceita métodos
   `lfr-fed<sc>` / `tfc-fed<sc>` (builder herdado de lfr/tfc; raiz
   `checkpoints/ssl_fed/...`; fonte lógica = `combined`, pois o corpus é a
   união dos silos) → cache `results/ssl_fed_eval_transfer.csv`.
4. Smoke test: 1 rodada × 4 encoders × 2 métodos (8 mini-runs) — pega
   problema de shape/chave por encoder antes da grade (lição do gate 6 do
   plano tstcc+TF-C).

**Gate 3**: smoke 8/8 OK; CSVs com schema estável; resume funciona.

**Commit**: `Infra FedSSL: runner, driver, downstream fed, variantes de agregação`

### Fase 4 — Gate de equivalência IID (o teste que valida o método)

Antes de gastar a grade: **FedAvg-SSL no cenário 2 (IID) com budget pareado
deve ≈ SSL centralizado na fonte `combined`** — se nem no caso IID o federado
alcança o centralizado, há bug de agregação (não "domain shift").

- 2 runs: `{lfr, tfc} × resnetse5 × cenário 2 × seed 0` (budget da §3.3).
- Downstream linear readout dos 2 backbones vs os backbones centralizados
  `combined/seed0` já existentes.

**Gate 4 (aceite)**: Δ(linear readout, média 6 alvos) entre FedSSL-IID e
centralizado-combined ≤ **3 pp** por método. Se falhar: debugar agregação
(filtro de chaves, budget, lr) ANTES de qualquer run no cenário 1 — no
cenário 1 não dá para distinguir bug de domain shift.

**Commit**: `Gate IID FedSSL aprovado: fed-c2 ≈ centralizado combined (Δ≤3pp)`

### Fase 5 — Grade de pré-treino FedSSL (ondas por prioridade)

| Onda | Combos | Runs | Racional |
|---|---|---|---|
| A (núcleo oficial) | cenário 1 × {lfr,tfc} × {resnetse5,cnnpff,rnn} × seeds 0–3 | 24 | é o experimento comprometido; encoders do plano oficial |
| B (controle) | cenário 2 × mesmos | 24 | isola domain shift no eixo SSL (par do cenário-1) |
| C (extensão) | {1,2} × {lfr,tfc} × tstcc × seeds 0–3 | 16 | 4º encoder; entra se A+B fecharem no prazo |
| D (ablação, opcional) | cenário 1 × tfc × resnetse5 × seed 0 × {local-epochs 2,5} × {fedbn, backbone-only(lfr)} | ~6 | frequência de agregação + variantes — só se houver tempo; se o cenário 1 divergir na onda A, D vira prioridade (é a investigação da falha) |

```bash
tmux new-session -d -s fedssl-grid
tmux send-keys -t fedssl-grid 'cd ~/fed-har && poetry run python scripts/federated/run_all_ssl.py \
  --scenario 1 --encoder resnetse5 cnnpff rnn --gpus 0,1,2,3,4,5,6,7 \
  2>&1 | tee logs/fedssl-grid-A.log' Enter
```

**Gate 5**: onda A 24/24 na rodada final sem NaN; `ssl_val_loss` final ≤ loss
da rodada 1 em todos (não-divergência); `agg_shock` logado. **Medir o 1º run
de cada método** e recalibrar o Apêndice C antes de soltar B/C.

**Commit por onda**: `Grade FedSSL onda A (24 runs): pré-treinos cenário 1`

### Fase 6 — Finetuning federado (fecha Exp. 2 E Exp. 3)

1. `finetune_client.py` + `run_federated_ft.py`: cliente = backbone (do
   método) + cabeça MLP (`Probe` do downstream), com:
   - `--init {random, central, fed}` + caminho do ckpt (resolve por
     convenção: central → `checkpoints/ssl/<m>/<enc>/combined/seed<N>/`;
     fed → `checkpoints/ssl_fed/...`);
   - `--protocol {linear, finetune}`: `linear` congela o backbone
     (`requires_grad=False`, semântica do downstream) e o FedAvg agrega **só a
     cabeça** (≈134 KB/rodada p/ TF-C — o finetuning federado linear é
     quase-grátis em comunicação: ponto forte do artigo);
   - avaliação centralizada por rodada reusando `make_evaluate_fn` (com o
     Probe no lugar do modelo SL);
   - saída `results/federated_ssl_eval.csv`:
     `method, encoder, init, protocol, scenario, seed, round, target,
     test_acc, test_f1_macro, uplink_bytes, downlink_bytes`.
2. Runs (todos cenário 1, R=50, seeds 0–3, protocolo `linear` como oficial):

| Braço | Combos | Runs |
|---|---|---|
| Exp. 3: init `fed` (onda A) | {lfr,tfc} × 3 enc | 24 |
| Exp. 2: init `central` (combined) | {lfr,tfc} × 3 enc | 24 |
| Baseline pareado: init `random` + linear | 3 enc (sem método) | 12 |
| Extensão tstcc (se onda C rodou) | +{lfr,tfc,random} × tstcc | +12 |
| Ablação `finetune` (opcional) | subconjunto: seed 0, {lfr,tfc}×3 enc×{fed,central} | +12 |

   (O baseline full-training random já existe em `federated_eval.csv`; o
   braço `random+linear` é necessário porque o protocolo linear muda o que é
   comparável.)

**Gate 6**: 60+ runs na rodada 50; curvas init `central`/`fed` ≥ `random` na
rodada 1 (o pré-treino tem que valer algo no início — senão suspeitar do
carregamento do ckpt); CSV consolidado.

**Commit**: `Finetuning federado: Exp2 (init central) + Exp3 (init fed) + baseline linear`

### Fase 7 — Downstream, análise e fechamento

1. Downstream centralizado dos backbones FedSSL finais (+ milestones da onda
   A): `downstream_eval.py --method {lfr,tfc}-fed1 ...` →
   `results/ssl_fed_eval_transfer.csv` (Tabela VII do esqueleto do artigo).
2. Notebook novo `notebooks/federated_ssl_avaliation.ipynb` (via builder
   `_build_federated_ssl_nb.py`, padrão dos existentes; só lê caches):
   curvas de pré-treino (loss/agg_shock), Fig. 6 do artigo (3 inits × métodos
   × encoders), custo de comunicação ponta a ponta (pré-treino + finetune).
3. Atualizar notas internas (checklist §8: Exp. 2 ✅, Exp. 3 ✅/achado
   negativo; nova seção com números), o esqueleto do artigo (preencher
   [PENDENTE] das seções IV-D/IV-E) e a memória do Claude.
4. Commit final + push.

**Gate 7**: notebook roda ponta a ponta só de caches; todos os [PENDENTE] de
Exp. 2/3 do esqueleto têm número ou viraram "achado negativo" documentado.

---

## Apêndice A — Config de referência por método (federado)

```yaml
# FedSSL-TFC (por run)
clientes: 6 (partitions.py, cenário 1 ou 2)   # shards idênticos ao supervisionado
modelo local: build_tfc(encoder)               # TFC_Model modo SSL, lr 3e-4, batch 64, drop_last
rodadas: 100 × 1 época local                   # ≡ 100 épocas centralizadas
agregação: TFC_Backbone completo (FedAvg ponderado por num_examples)
avaliação: ssl_val_loss por rodada + milestones {10,25,50,100}

# FedSSL-LFR (por run)
modelo local: build_lfr(encoder)               # 60 proj. congelados (seed compartilhada), 6 ativos via DPP global
rodadas: 100 × bloco de 6 épocas de Trainer    # ≡ 600 épocas de Trainer / 100 efetivas de backbone
agregação: backbone + 6 preditores ativos      # projetores e 54 preditores inativos: nunca transmitidos
variantes: --aggregate {full, backbone-only, fedbn}

# Finetuning federado (Exp. 2 e 3)
modelo local: Probe(backbone_do_método, cabeça MLP [enc_dim,128,6])
init: {random, central(combined), fed(cenário do pré-treino)}
protocolo linear: backbone congelado; FedAvg só da cabeça (~134 KB/rodada TF-C)
rodadas: 50 × 1 época local; lr 1e-4 (paridade com o downstream centralizado)
```

## Apêndice B — Checklist de arquivos

| # | Arquivo | O quê | Fase |
|---|---|---|---|
| 1 | `scripts/federated/ssl_client.py` | novo: cliente SSL + `param_keys` (filtro) + agg_shock | 1–2 |
| 2 | `scripts/federated/run_federated_ssl.py` | novo: entrypoint pré-treino FedSSL + milestones | 1, 3 |
| 3 | `scripts/federated/run_all_ssl.py` | novo: driver (padrão run_all.py) | 3 |
| 4 | `scripts/ssl/downstream_eval.py` | métodos `lfr-fed<sc>`/`tfc-fed<sc>` + raiz `checkpoints/ssl_fed/` | 3 |
| 5 | `scripts/federated/finetune_client.py` | novo: cliente de finetuning com init/protocolo | 6 |
| 6 | `scripts/federated/run_federated_ft.py` | novo: entrypoint finetuning federado | 6 |
| 7 | `notebooks/_build_federated_ssl_nb.py` | novo: builder do notebook de análise | 7 |
| 8 | `docs/notas_internas_projeto_ssl_federado_har.md` | checklist + seção de resultados Exp. 2/3 | 7 |
| 9 | `docs/paper/esqueleto_artigo.md` | preencher [PENDENTE] IV-D/IV-E | 7 |

Novos artefatos de dados: `checkpoints/ssl_fed/**` (gitignored),
`results/federated_ssl_pretrain.csv`, `results/federated_ssl_eval.csv`,
`results/ssl_fed_eval_transfer.csv` (+ pastas `*_parts/`).

## Apêndice C — Estimativas de custo (ancoradas no Apêndice C do plano tstcc+TF-C)

Âncoras medidas: run federado supervisionado (R=50, 1 época local) ≈ 45 min
(96 runs ≈ meio dia em 8 GPUs); TF-C centralizado 100 épocas ≈ 15–40 min/fonte
(pior caso tstcc ~3.2 s/época no KuHar); LFR centralizado 600 épocas ≈ 1.4 h
médio/fonte; downstream ≈ 7 min/backbone.

| Item | Estimativa (medir o 1º!) | Base do cálculo |
|---|---|---|
| Run FedSSL-TFC (R=100×1) | 1.5–4 h | 100 épocas sobre a união ≈ TF-C `combined` (~6× fonte média) + overhead Flower/eval por rodada (~2× rodadas do supervisionado) |
| Run FedSSL-LFR (R=100×6) | 5–10 h (tstcc: +serialização ~127 MB×12/rodada) | 600 épocas de Trainer sobre a união ≈ 6× LFR de fonte única (~1.4 h) |
| Onda A (24 runs mistos) | ~2–3 dias em 8 GPUs | 12 TFC (~2.5 h) + 12 LFR (~7 h) |
| Ondas B+C (+40 runs) | ~2–3 dias em 8 GPUs | idem |
| Finetuning federado (R=50, linear) | 20–40 min/run ⇒ ~72 runs ≈ ½ dia em 8 GPUs | mais barato que o supervisionado (só cabeça no backward do linear) |
| Downstream fed (64 backbones + milestones) | ~7 min/job ⇒ ~2–4 h em 4 GPUs | âncora medida |
| **Total Exp. 3 (ondas A+B + finetune + análise)** | **~1 semana de cluster** | — |

Se o cronograma apertar: cortar na ordem C → D → B (a onda A + gate 4 + 
finetune já sustentam as afirmações mínimas do artigo; B perde só o par IID
do eixo SSL — recuperável citando o gate 4 como evidência pontual).

## Apêndice D — Mapa risco × mitigação × evidência

| Risco | Detecção | Mitigação (ordem) | Se persistir |
|---|---|---|---|
| Divergência contrastiva TF-C (negativos locais) | `ssl_val_loss` sobe/oscila; `agg_shock` cresce com as rodadas | agregação frequente (default) → `fedbn` → lr menor | achado negativo com curva de evidência (publicável; notas §7) |
| Preditores LFR incoerentes entre clientes | loss de preditor não decresce; gate 4 falha só no LFR | bloco local de 6 épocas (default) → `backbone-only` | idem |
| Seleção DPP divergente entre clientes | hash dos índices difere (spike F2) | índices globais via config (§3.4) | seleção server-side documentada |
| Bug de agregação mascarado de domain shift | **gate 4 (equivalência IID)** | corrigir antes da grade | bloqueia a Fase 5 |
| Mensagem LFR-tstcc grande demais (Ray/Flower) | spike F2, tempo de serialização | filtro de 6 ativos (default) → fp16 na transmissão (documentar) | excluir tstcc do LFR federado (onda C é extensão) |
| BN global degrada todos os domínios | comparação `full` vs `fedbn` (onda D) | `fedbn` como variante reportada | vira resultado (conexão com FedBN/literatura) |
| Budget federado ≠ centralizado (comparação injusta) | revisão da §3.3 nos spikes | paridade de épocas-sobre-dados fixada por construção | — |
