# Plano do Fed-SSL — eixo cross-device

*Consolidado em 2026-07-27 a partir de `desenho_cross_device.md` (2026-07-24),
`_arquivo/limite_batch_cliente_fssl.md` (2026-07-24) e da parte viva de
`_arquivo/plano_fedssl_simulado.md` (2026-07-13/14); §2.1–2.3 e §4.3 vêm da
sabatina de 2026-07-27/28.*

**Contexto do pivô (2026-07-21, com o orientador):** a federação **cross-silo**
(1 dataset por cliente, cenários 1–8) foi abandonada como desenho e como
controle. Os 6,3 pp medidos lá viram **motivação**, não contribuição
(`resultados.md §4`). O eixo ativo é **cross-device: clientes = usuários**.

> **Nota — virada das RQs (2026-08-19).** Este documento é anterior ao
> fechamento das três RQs em [`perguntas_de_pesquisa.md`](perguntas_de_pesquisa.md).
> Vale como **registro do que foi desenhado e medido até aqui**, não como o
> desenho atual: o mapeamento RQ → braço → comparador está sendo redefinido e vai
> morar num documento próprio de desenho experimental.

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

### 2.1 Protocolo de pareamento dos braços (decidido na sabatina de 2026-07-27)

O Δ(cross-domain − in-domain) com os dados naturais **não** mede domain shift: o
braço cross-domain teria 27 clientes e 13.896 janelas contra 10 clientes e 10.338
do RW_thigh — 2,7× mais clientes e 34% mais dado. Pior, o FedAvg pondera por
`n_k` e um usuário do RW_thigh vale ~5× um do MotionSense, então o "cross-domain"
seria dominado por um domínio só (MotionSense: 63% dos clientes, ~26% do peso).

**Decisão: parear nº de clientes E orçamento por cliente.**

| | |
|---|---|
| **Clientes por braço** | **10** (teto imposto pelo RW_thigh, que só tem 10 usuários) |
| **Orçamento por cliente** | **B = 192 janelas** |
| **Braços** | `10+0` (in-domain RW_thigh), `0+10` (in-domain MotionSense), `5+5` (cross-domain) |
| **Amostragem** | aleatória **estratificada por classe**, determinística por seed, **proporcional** à distribuição natural do usuário |
| **Elegibilidade** | usuário com ≥ B janelas no `train`: RW_thigh **10/10**, MotionSense **14/17** |

Assim os três braços têm o mesmo nº de clientes, o mesmo volume por cliente, o
mesmo volume total (1.920 janelas) e **pesos FedAvg uniformes** — muda só a
composição de domínio. O preço, a declarar no artigo: descartamos ~81% do dado do
RW_thigh, e as acurácias absolutas ficam bem abaixo das centralizadas.

**Por que B = 192:** são exatamente **3 batches de 64** com `drop_last=True` — dá
ao TF-C mais de um passo de gradiente por época local sem chegar perto do piso de
batch (§3).

**Por que não balancear as classes por cliente** (32/32/32…): (i) é **inviável** —
só 4 dos 17 usuários do MotionSense têm 32 janelas em *todas* as classes (mínimo
do dataset: 16), e precisamos de 10; (ii) é **desnecessário** — as duas bases já
são globalmente uniformes (1/6 exato por classe; TV entre elas = **0.000**), o
DAGHAR já balanceou.

**Por que amostragem aleatória e não as primeiras N janelas** (prefixo temporal),
verificado no dado em 2026-07-27:

1. **As janelas não se sobrepõem** (0 amostras compartilhadas entre janelas
   consecutivas da mesma sessão) — não existe "janela cortada no meio" a proteger.
2. **O modelo nunca vê a ordem**: a entrada é uma janela isolada `(6,60)`, sem
   modelagem entre janelas.
3. **O prefixo enviesaria por sessão**: no MotionSense, **todos** os 102 pares
   (usuário, classe) têm 2–3 sessões de gravação distintas (ex.: usuário 2,
   escada-acima = 12 + 15 + 6 janelas em três arquivos). Um prefixo puxa a sessão
   mais antiga e faz a amostra representar *uma condição de gravação* em vez da
   pessoa — contaminando exatamente a variabilidade entre pessoas que o
   experimento mede.
4. **Custo de código**: os dois datasets ordenam por mecanismos diferentes
   (`accel-start-time` no RealWorld, `csv` + `window` no MotionSense) — dois
   caminhos e um lugar novo para bug silencioso.

⚠️ **Armadilha de implementação:** `few_shot_indices` (`scripts/common.py:182`)
**degrada em silêncio** quando uma classe tem menos amostras que o pedido ("usa
todas as disponíveis"). Num desenho pareado isso quebraria o pareamento sem
avisar — o helper de orçamento precisa de `assert` explícito de que o cliente
fechou as B janelas.

**Em aberto (ablação futura):** o cenário **natural**, sem parear — 10 / 17 / 27
clientes com todo o dado. Responde uma pergunta diferente e legítima ("o que
acontece com os dados que você realmente tem"), e não substitui o pareado. Rodar
depois, e nunca escrever a frase do pareado usando o número do natural.

### 2.2 Os braços e o que cada Δ mede

| Braço | `spec` | Clientes | Janelas do alvo (lendo alvo = RW) |
|---|---|---|---|
| `in10-RW` | `device:RealWorld_thigh:10` | 10 | 1.920 |
| `in10-MS` | `device:MotionSense:10` | 10 | 0 |
| `cross10+10` | `device:RealWorld_thigh+MotionSense:10+10` | 20 | 1.920 |
| `cross5+5` | `device:RealWorld_thigh+MotionSense:5+5` | 10 | 960 |
| `iid10-RW` | `iid:RealWorld_thigh:10` | 10 | 1.920 |
| `iid10-MS` | `iid:MotionSense:10` | 10 | — |
| gate | `single:RealWorld_thigh:10` | 1 | 1.920 |

```
Δ_custo-do-shift    = cross5+5   − in10-RW    # orçamento fixo: metade do alvo virou estrangeiro
Δ_valor-estrangeiro = cross10+10 − in10-RW    # alvo constante, estrangeiro acrescentado
Δ_feature-skew      = in10-RW    − iid10-RW   # mesmas janelas, só a partição muda
```

⚠️ **Confundidor a declarar no texto:** `cross10+10` tem 20 clientes e 3.840
janelas contra 10 e 1.920 do `in10-RW`. Não existe federação com 20 clientes só
de RW_thigh (a base tem 10 usuários), então o Δ é *"acrescentar um segundo
domínio"* — e **não** *"dado estrangeiro vale tanto quanto dado próprio"*, que
exigiria um in-domain de 3.840 janelas.

### 2.3 Protocolo de treino local e comunicação (sabatina de 2026-07-28)

**A unidade de trabalho local é a época efetiva de *backbone*, não a época bruta.**
O LFR alterna: em `lfr.py:410-417`,
`freeze_backbone = current_epoch % (predictor_training_epochs+1) != 0`. Com
`predictor_training_epochs=5` a **época 0 treina o backbone** e as épocas 1–5
treinam os preditores — nessa ordem. Como `local_pretrain` cria um Trainer novo a
cada rodada, `current_epoch` reinicia em 0 para todo cliente em toda rodada:

| `local_epochs` do LFR | Efeito por rodada |
|---|---|
| 1 | backbone treina; **preditores nunca treinam** — ficam na inicialização e a tarefa deixa de medir o backbone |
| **6** | 1 época de backbone + 5 de preditor (ciclo completo) |
| 30 | 5 épocas de backbone + 25 de preditor |

Logo `local_epochs` do LFR **tem de ser múltiplo de 6**. Definindo `k` = épocas
efetivas de backbone por rodada: **LFR `local_epochs = 6k`, TF-C `local_epochs = k`**
(o TF-C não alterna — todo parâmetro recebe gradiente em toda época).

**`k = 5` em todos os experimentos.** É o consenso da literatura — `E=5` é
unânime em quatro papers primários:

| Paper | Domínio | Épocas locais | Rodadas |
|---|---|---|---|
| FedSC (ICML'24) | imagem | 5 | 200 |
| FedEMA/FedU (ICLR'22) | imagem | 5 | 100 |
| Saeed et al. (IoT-J'21) | **sensores/HAR** | 5 | 30–50 |
| FedST/FedOST (MM'24) | séries temporais, clientes = sujeitos | 5 | 100 |

**A ablação `k=1` foi cortada em 2026-08-04** (decisão do Miguel: não é
prioridade). O plano original pedia `k ∈ {1, 5}` com o argumento de que `k=1` é
"o extremo de sincronização máxima que ninguém testa"; na prática ela nunca foi
executada de verdade — o único `k=1` que existe no cache é um braço de
calibração do baseline supervisionado (`fed_cross_device.csv`, `local_epochs=1`:
**só `resnetse5`**, R=100, 32 células) e ele **não entra em análise nenhuma**,
justamente por ser outro protocolo. Toda grade viva — Fed-SSL e baseline
federado — roda `E=5`. Ressuscitar a ablação é grade nova, não recorte de cache.

**`R = 100` rodadas**, com corte pela curva medida (avaliamos toda rodada, sem
early stopping). É o valor do FedEMA e do FedST e cobre com folga o regime do
Saeed (30–50), que é o vizinho mais parecido conosco.

**Protocolo sequencial** (todo o pré-treino, depois o finetuning) — é o que os
quatro papers fazem. O Saeed é explícito: *"We pre-train … and use the model as
initialization for learning a downstream task"*.

**Custo local — a medição de 2026-07-28 NÃO se sustentou.** ⚠️

A versão anterior desta seção media `resnetse5` na MX570A em "regime
estacionário" e concluía que a época bruta do LFR custava **0,20×** a do TF-C,
portanto que **"a alternância 6× é quase exatamente cancelada"** e que *"o LFR
treina 6× mais localmente não é custo real de computação"*. **Isso está errado no
hardware em que a grade roda.**

Remedido em 2026-07-29 no **TITAN Xp**, no caminho real (`pretrain_fed.py`,
`device:RealWorld_thigh:10`, 10 clientes, B=192 = 3 batches, GPU ociosa,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`), variando `--local-epochs` para separar custo por
época de custo fixo por rodada:

| método | R=1, le=1 | le=5 | le=6 | le=15 | le=30 | **s/época (10 clientes)** | intercepto |
|---|---|---|---|---|---|---|---|
| TF-C | 3,9 s | 15,8 s | — | 42,7 s | — | **2,77** | 1,1 s |
| LFR | — | — | 14,0 s | — | 72,0 s | **2,42** | ~0 |

A época bruta do LFR custa **0,87×** a do TF-C, não 0,20×. A economia das 5 épocas
"só-preditor" não aparece porque, com orçamento de 192 janelas, **cada época tem 3
batches** e o custo fixo da época (reinício do dataloader, loop de época do
Lightning) domina o que se economiza no backward. O número antigo era um artefato
de hardware e de regime de medição.

**Consequência:** casando épocas **efetivas de backbone** (LFR `6k=30` contra TF-C
`k=5`), o pré-treino LFR custa **4,6×** o do TF-C por rodada — 72,0 s contra
15,8 s. Não é neutro, como esta seção afirmava; é o item dominante do orçamento da
grade (~158 das ~200 GPU-h de pré-treino).

A sabatina de 2026-07-29 **manteve `6k=30` mesmo assim**: o eixo de casamento
correto é épocas efetivas de backbone, porque é o que determina quanto o backbone
aprende por rodada. Casar por wall-clock (LFR `k=1`) economizaria ~265 GPU-h mas
daria ao LFR 1/5 do treino de backbone do TF-C, e o Δ(LFR − TF-C) passaria a
confundir método com orçamento. Custo de compute é o argumento mais fraco que
existe para escolher protocolo.

⚠️ **O custo do LFR está na comunicação, e a maior parte é evitável.**

Medir isto tem uma armadilha: o `build_lfr` cria **60 projetores** (38,25 MiB,
96,2% do state_dict), mas o `build_global_model` roda a **seleção DPP e reduz
para 6** *antes* de qualquer transmissão. O objeto de 38 MiB **nunca trafega** —
medir nele superestima o custo em ~9×. O que de fato vai na rede é o pós-DPP:

| | MiB por cliente por rodada | 10 clientes |
|---|---|---|
| LFR sem skip | 4,26 (projetores **86,3%**, backbone 0,49, preditores 0,10) | 42,65 |
| **LFR com skip** | **0,58** | 5,84 |
| TF-C | 1,36 | 13,62 |

Os projetores são funções aleatórias **congeladas** e **idênticas em todos os
clientes** (DPP feita uma vez pelo servidor; `num_targets=None` nas cópias impede
re-seleção), então agregá-los é um no-op. Com
**`fedavg(skip_prefixes=("projectors",))`** — já aplicado automaticamente ao LFR
em `pretrain_fed.py` — o custo cai **7,3×**, e o LFR passa a ser **2,3× mais
barato que o TF-C** por rodada, não empatado.

Isto **corrige o achado F5 de `estado_da_arte.md`**, que antecipava "LFR federado
é caro em uplink (pelos preditores)". Medido: os preditores custam 0,10 MiB (2,2%);
o caro são os projetores, que não precisam trafegar. Com a exclusão, **o LFR é o
método mais barato dos dois** — o oposto do que o projeto vinha assumindo.

**Variante registrada, fora da 1ª onda:** cronograma **alternado** entre estágios
(blocos de rodadas de pré-treino intercalados com blocos de finetuning) em vez de
sequencial. Nenhum dos quatro papers faz isso; a alternativa que a literatura
estuda é *joint training* (as duas perdas simultâneas — arXiv:2607.13192). Só faz
sentido se o finetuning atualizar o backbone (com linear probe é no-op), e
confunde atribuição. É candidata a **contribuição nomeada** (responde a W1 em
`estado_da_arte.md §7.2`), mais original que `fedbn` — mas depois da 1ª onda.

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

### 4.2 Partições — a API herdada do cross-silo

`make_ssl_client_datasets(partition, combo, seed)`
(`scripts/federated/partitions.py:150`) já implementa:

- `silo` — 1 cliente por dataset do combo;
- `iid` — união do combo em 6 fatias IID (gate G-IID);
- `device-<dataset>` — 1 cliente por usuário do train, **de um único dataset**.

> ✅ **Lacuna fechada em 2026-07-28** por `scripts/federated/cross_device.py`, que
> implementa a função única `make_clients(spec, seed, budget=192)` usada pelos
> **dois** braços (supervisionado e SSL) — ver §4.3. A `make_ssl_client_datasets`
> fica só para o eixo cross-silo herdado.
>
> Continua faltando o braço de **label skew artificial pareado em volume** (§2).

### 4.3 `scripts/federated/cross_device.py` — ✅ IMPLEMENTADO (2026-07-28)

Função única `make_clients(spec, seed, budget=DEFAULT_BUDGET)` → `[(id, Subset)]`,
com `spec = <modo>:<datasets>:<contagens>` (tabela dos braços em §2.2). Modos:
`device` (1 cliente por usuário), `iid` (mesmas janelas reparticionadas) e
`single` (mesmas janelas num cliente só, para o gate).

- **Seleção fixa e aninhada** em `scripts/federated/client_selection.csv`
  (versionado): uma permutação de usuários por dataset, sorteada uma vez com
  `SELECTION_SEED = 20260727`. Os braços tomam prefixos ⇒ os 5 clientes de um
  braço são subconjunto dos 10 do outro. Regerar com `--regenerate-selection`
  **muda a população do experimento** — não fazer sem reportar.
- **Elegibilidade a B=192** (do manifesto): RW_thigh 10/10, MotionSense 14/17,
  WISDM 36/36, RW_waist 10/10, **UCI 0/21 e KuHar 0/57** — os usuários dessas duas
  bases não têm 192 janelas. Irrelevante para a 1ª onda; **bloqueia o UCI** na
  segunda, junto com o KuHar (§3).
- `budget=None` devolve o cenário natural (ablação futura) na mesma função.
- `--describe <spec>` monta o braço e imprime a composição sem treinar nada.

Invariantes verificadas em 2026-07-28: aninhamento; `cross10+10` contém os dois
`in10`; **`iid` e `single` reusam exatamente as mesmas 1.920 janelas do `device`**
(é isso que faz do `iid` um controle pareado); orçamento fechando em 192 por
cliente com distribuição de classes proporcional; determinismo por seed; e seed
trocando as janelas (~20% de sobreposição, o esperado por acaso) sem trocar os
clientes.

### 4.4 Gates de validação (herdados, ainda válidos)

| # | O quê | Gate |
|---|---|---|
| S0 | Init idêntico entre fontes (hash dos state_dicts) | **✅ PASS (2026-07-14)** |
| S1 | Sopa all6 tfc×cnnpff×seed0 + 1 downstream | acc > acaso (1/6), registrada |
| S2 | Loop 3 rodadas TF-C | loss cai, sem NaN, CSV de bytes ok |
| S3 | LFR 2 rodadas | backbone muda exatamente 1 época efetiva/rodada; DPP global aplicado |
| G-EQ1 | 1 cliente (= `combined`), R=1 × E=100 vs centralizado | métricas downstream idênticas — valida o simulador inteiro |
| G-EQ2 | 1 cliente, R=100 × E=1 vs centralizado | quantifica o custo do fatiamento (reset do Adam) — número novo do artigo |
| G-IID | `iid` 6 clientes, R=100 vs centralizado `combined` | Δ downstream @full ≥ −3 pp |

**G-EQ1 executado em 2026-07-28 — ✅ PASS.**
`run_cross_device.py --gate --spec single:RealWorld_thigh:10 --encoder resnetse5
--local-epochs 3`: divergência máxima de peso **0,000e+00** e acc/F1 bit-idênticos
entre o federado de 1 cliente e o centralizado. Valida de uma vez `make_clients`,
o loop local e o `fedavg`.

⚠️ **A primeira execução do gate REPROVOU** (divergência 7,7e-03), e a causa não
era bug de agregação: o **cuDNN usa algoritmos não-determinísticos** no backward
das convoluções, então dois treinos idênticos na GPU divergem ~1e-2 depois de
poucas épocas. O gate agora chama `set_deterministic()` (cudnn determinístico +
`use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`, que **precisa estar no
ambiente antes de o CUDA inicializar**). Fora do gate não é obrigatório — a grade
tem 4 seeds e essa variação entra na barra de erro —, mas **qualquer alegação de
reprodutibilidade bit-a-bit exige esses flags**.

Regra de sempre: **medir o 1º job de cada onda antes de extrapolar custo**; tmux +
`tee logs/` (ver `CLAUDE.md`).

## 5. Escopo e ordem de execução

1. ~~**Fechar a lacuna da API de partições**~~ — **feito** (§4.3,
   `scripts/federated/cross_device.py`, 2026-07-28).
2. **Adaptar os dois runners para consumir `make_clients`**: o baseline
   supervisionado (que sai do Flower — o loop local de `client.py:97` é torch puro
   e o `fedavg()` já existe) e o `pretrain_fed.py` (hoje chama
   `make_ssl_client_datasets`). Aplicar `skip_prefixes=("projectors",)` no LFR.
3. **Rodar o gate** `single:RealWorld_thigh:10` com R=1: tem de reproduzir o
   centralizado com a mesma seed. Só depois abrir a grade.
4. **A grade**: 7 configs (§2.2) × `k ∈ {1,5}` × 4 seeds, começando por 1 encoder
   e R=100 com corte pela curva. Nenhum braço inclui KuHar, então o piso de batch
   não bloqueia.
5. **Controle que falta** do §2: o braço de label skew artificial pareado em volume.
6. **Depois**: endereçar o piso de batch (KuHar é o caso que o expõe) — **forma em
   aberto** (§3.3).
7. **Segunda onda** (heterogeneidade por pessoa como eixo próprio): WISDM
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
- A lousa do orientador que originou o desenho: transcrita em
  `_arquivo/plano_fedssl_simulado.md` (a foto foi removida do repo em 2026-08-21).

### Referências de método

1. Y. Sui et al., "Self-supervised Representation Learning From Random Data
   Projectors" (**LFR**, com a **Batch-wise Barlow Twins**), ICLR 2024.
   arXiv:2310.07756. *(Fonte do "m×m com m = batch" do BBT.)*
2. J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny, "Barlow Twins:
   Self-Supervised Learning via Redundancy Reduction", ICML 2021. *(Matriz d×d;
   a robustez a batch pequeno do BT clássico NÃO se aplica ao BBT.)*
3. X. Zhang et al., "Self-Supervised Contrastive Pre-Training for Time Series via
   Time-Frequency Consistency" (**TF-C**), NeurIPS 2022. arXiv:2206.08496.
