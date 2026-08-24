# Mapa dos experimentos — Fed-SSL cross-device

**O que este documento é:** a referência única de *o que foi rodado*. Quantas
federações, com que hiperparâmetros, avaliadas como, e por que cada escolha.
Escrito em 2026-08-05 lendo o **código**, não a memória — cada número aqui tem a
linha de origem indicada.

**O que ele não é:** não traz resultados (isso é `resultados.md` e os notebooks),
nem justificativa de desenho experimental (isso é `plano_fedssl.md`).

> **Nota — virada das RQs (2026-08-19).** Este documento é anterior ao
> fechamento das três RQs em [`perguntas_de_pesquisa.md`](perguntas_de_pesquisa.md).
> Vale como **registro do que foi desenhado e medido até aqui**, não como o
> desenho atual: o mapeamento RQ → braço → comparador está sendo redefinido e vai
> morar num documento próprio de desenho experimental.

---

## 1. O vocabulário — três palavras que confundem

| termo | o que é |
|---|---|
| **`spec`** | **quem treina.** A composição de clientes da federação. Ex.: `device:RealWorld_thigh+MotionSense:5+5` |
| **`target`** | **onde testa.** O dataset cujo `test.csv` avalia o modelo |
| **`shots`** | **quanto rótulo.** Rótulos por classe, **por cliente**, no fine-tuning |

`spec` e `target` são eixos independentes. Um `spec` que mistura dois datasets é
avaliado nos dois — mesmo modelo, dois conjuntos de teste.

Formato do spec: `<modo>:<datasets>:<contagens>`. Modos em
`scripts/federated/cross_device.py:58`: `device` (1 cliente = 1 usuário),
`iid` (mesmas janelas, fatiadas aleatoriamente — controle de feature skew) e
`single` (todas as janelas num cliente só — gate e Exp. 2).

---

## 2. As 4 federações e as 6 células de avaliação

| federação | clientes | composição | testada em |
|---|---|---|---|
| `in10-RW` (`device:RealWorld_thigh:10`) | 10 | 10 usuários do RealWorld_thigh | RW |
| `in10-MS` (`device:MotionSense:10`) | 10 | 10 usuários do MotionSense | MS |
| `cross5+5` | 10 | **5** RW + **5** MS | RW **e** MS |
| `cross10+10` | 20 | 10 RW + 10 MS | RW **e** MS |

**4 treinos → 6 células.** Verificado em `make_clients()` e nos alvos presentes no
cache.

Um `in10` só é testado no domínio que viu. Pegar o modelo do `in10-RW` e testá-lo
no MotionSense seria **transferência cross-dataset**, que é outro experimento (§7
do notebook, e a matriz centralizada `supervised_eval_transfer.csv`).

**Prova de que as duas linhas de um `cross` são o mesmo modelo:** mesmo `spec`,
`seed` e `round`, e `uplink_mb` idêntico (5,23 MB) — custo de comunicação é
propriedade do treino, não do teste. Muda só o `test_acc`.

### 2.1 Os braços de população desacoplada

As 4 federações acima usam a **mesma** população nas duas fases. Três grades
quebram esse casamento, e cada uma responde a uma pergunta diferente. Todas moram
em `fedssl_crossspec.csv`, com `pretrain_spec` preenchido.

| grade | pré-treino em | fine-tuning em | avaliado em | células |
|---|---|---|---|---|
| **transfer de domínio** | `device:MotionSense:10` | `device:RealWorld_thigh:10` | RW | 160 |
| | `device:RealWorld_thigh:10` | `device:MotionSense:10` | MS | 160 |
| **Exp. 2** (budget 192) | `single:X:10` (1 cliente) | `device:X:10` (10 clientes) | X | 320 |
| **Exp. 2 `@full`** *(rodando)* | `single:X:10@full` (sem teto) | `device:X:10` | X | 320 |

- **Transfer de domínio** responde *"o backbone precisa ter sido pré-treinado no
  domínio do alvo?"* (§7 do notebook). Custou **zero** pré-treino novo: reusa os
  backbones da grade base.
- **Exp. 2** separa **custo da federação** de **custo do orçamento de dado**.
  `device − single` a budget fixo = o preço de federar; `single@192 − single@full`
  = o preço de ter caído para 1.920 janelas.

**O que NÃO existe: avaliar fora do domínio de fine-tuning.** Verificado no cache —
**zero** linhas com `target` fora do spec de fine-tuning, nas duas grades
federadas. Toda federação só é testada em domínio que viu treinando.

Esse experimento existe apenas **centralizado**: `supervised_eval_transfer.csv`
(treina na fonte, avalia nos 6 alvos) e `ssl_{lfr,tfc}_eval_transfer.csv`
(pré-treina **e** treina a cabeça na fonte, avalia nos 6 alvos). É de lá que sai o
`gap` usado como régua independente no §9.3 do notebook.

---

## 3. Avaliamos o modelo **global**, nunca os locais

`scripts/federated/run_cross_device.py:266-269`:

```python
global_state = fedavg(states, weights)      # agrega os clientes
model = _fresh(encoder, global_state, method)
scores = _eval_all(model, targets)          # testa o GLOBAL em cada alvo
```

A cada rodada: os clientes treinam localmente, o servidor faz FedAvg ponderado por
`n_k`, e **o modelo agregado** é avaliado. Não há avaliação personalizada nem
por-cliente. Não há, portanto, resultado de personalização nesta grade.

**Como funciona no caso `cross`:** o mesmo modelo global roda duas vezes, uma no
`test.csv` do RealWorld_thigh e outra no do MotionSense. Dois números por rodada.
Não há média interna — quem decide agregar ou não é a figura.

### Os splits de teste são de usuários **disjuntos** do treino

| dataset | usuários no train | val | test | janelas de teste |
|---|---|---|---|---|
| RealWorld_thigh | 10 | 2 | **3** | 2.898 |
| MotionSense | 17 | 2 | **5** | 1.062 |

Interseção train∩test = **0** nos dois. São os splits padrão do DAGHAR
`standardized_view`. Ou seja: toda acurácia reportada é em **sujeitos que o modelo
nunca viu** — o que é o regime realista para HAR e evita o vazamento mais comum da
área.

### Seleção de rodada

`val_loader()` usa o `validation.csv` do alvo — split de seleção, nunca de reporte.
Com `k=5` a curva do FedAvg tem pico cedo e degrada, então "melhor rodada" é uma
escolha real de modelo; fazê-la no teste inflaria o número.

⚠️ **Os notebooks Fed-SSL não usam `argmax(val)` e sim a média das 20 últimas
rodadas** — o `argmax` infla de forma desigual entre braços (demonstração no §3 do
`fedssl_cross_device_avaliation.ipynb`). O baseline supervisionado usa `argmax(val)`.
As duas regras convivem no repositório; ao comparar, confira qual está em uso.

---

## 4. As duas fases de treino

Cada célula com SSL são **dois treinos federados em sequência**. O baseline pula a
primeira.

| | Fase 1 — pré-treino | Fase 2 — fine-tuning |
|---|---|---|
| objetivo | SSL, sem rótulo | supervisionado, com rótulo |
| rodadas | **R = 100** | **R = 150** |
| épocas locais | TF-C **5**, LFR **30** | **5** |
| parte de | pesos aleatórios | o `backbone.ckpt` da Fase 1 |
| produz | `backbone.ckpt` | o modelo avaliado |
| avalia? | não | sim, toda rodada |

Constantes em `scripts/federated/run_grid_fedssl.py:79-84`.

**Um backbone serve os cinco degraus de rótulo.** O pré-treino não usa rótulo, logo
o orçamento de rótulos não o afeta. No disco:

```
checkpoints/ssl_fed/tfc/resnetse5/device-RealWorld_thigh-10/seed0/backbone.ckpt  ← 1
checkpoints/fedssl_cross_device/tfc/resnetse5/device-RealWorld_thigh-10/
    shots1/ shots2/ shots5/ shots10/ shotsfull/                                  ← 5
```

**Na grade base as duas fases usam a mesma população.** É por isso que a coluna
`pretrain_spec` está **vazia** em `fedssl_cross_device.csv` — vazio significa
"casado com o `spec`". Só nas grades `crossspec` elas foram desacopladas.

---

## 5. Os eixos da grade

| eixo | valores | n |
|---|---|---|
| **encoders** | `resnetse5`, `cnnpff`, `rnn`, `tstcc` | 4 |
| **métodos** | `none` (FedAvg supervisionado), `lfr`, `tfc` | 3 |
| **specs** | `in10-RW`, `in10-MS`, `cross5+5`, `cross10+10` | 4 |
| **degraus de rótulo** | 1, 2, 5, 10, `full` (`none` só até 10) | 5 |
| **seeds** | 0, 1, 2, 3 | 4 |
| **domínios** | RealWorld_thigh, MotionSense | 2 |

### Por que só 2 domínios — e o que a elegibilidade de fato restringe

Elegibilidade a `B=192` (usuários com ≥192 janelas no train), medida em
2026-08-05, batendo com o manifesto de `plano_fedssl.md` §4.3:

| dataset | elegíveis / total |
|---|---|
| WISDM | 36 / 36 |
| MotionSense | 14 / 17 |
| RealWorld_thigh | 10 / 10 |
| RealWorld_waist | 10 / 10 |
| **UCI** | **0 / 21** |
| **KuHar** | **0 / 57** |

Ou seja: a elegibilidade **bloqueia UCI e KuHar**, mas *não* seleciona o par
`RealWorld_thigh + MotionSense` — WISDM e RealWorld_waist também qualificam. A
escolha do par para a **primeira onda** não está justificada por escrito em lugar
nenhum do repositório; `plano_fedssl.md` §4.3 só registra a elegibilidade, e a
segunda onda está prevista com o par cintura↔perna (RealWorld_waist), que tem gap
de domínio maior.

**Tratar como pendência de documentação**, não como fato estabelecido: se a
escolha for para o texto do artigo, ela precisa de uma razão declarada.

### Por que 4 seeds

É o mínimo que dá barra de erro sem multiplicar por 4 um orçamento de GPU que já
está em centenas de horas. **Consequência a respeitar:** com 4 seeds, diferenças
abaixo de ~5 pp raramente se distinguem do ruído (o dp de seed medido no §9 é
4,7 pp no Δ A e 5,8 pp no DiD).

---

## 6. Contagem total — só a fase federada

Duas unidades diferentes, e é preciso não confundi-las:

- **run** = uma federação treinada de ponta a ponta (um FedAvg completo).
- **célula de resultado** = um `(run, alvo)`. Um run rende **1 ou 2** células,
  conforme quantos datasets o spec do fine-tuning contém (§2).

É daí que vem a diferença entre "1.632 fine-tunings" e "2.112 resultados": não são
contagens concorrentes, são **unidades diferentes** da mesma coisa.

### 6.1 Pré-treinos (Fase 1)

| grade | conta | concluídos |
|---|---|---|
| base — populações casadas | 2 métodos × 4 encoders × 4 specs × 4 seeds | **128** |
| Exp. 2 — `single:` budget 192 | 2 × 4 × 2 specs × 4 seeds | **64** |
| transfer de domínio | reusa os backbones da base | 0 |
| | **total concluído** | **192** |
| Exp. 2 — `single:@full` | 2 × 4 × 2 × 4 | *64 em execução* |

### 6.2 Fine-tunings (Fase 2) e os resultados que eles produzem

| grade | runs | alvos por run | **células** |
|---|---|---|---|
| base — `none` (4 degraus) | 256 | 1 ou 2 | 384 |
| base — `lfr` (5 degraus) | 320 | 1 ou 2 | 480 |
| base — `tfc` (5 degraus) | 320 | 1 ou 2 | 480 |
| crossspec — transfer de domínio | 320 | 1 | 320 |
| crossspec — Exp. 2 `single@192` | 320 | 1 | 320 |
| baseline FedAvg supervisionado | 96 | 1 ou 2 | 128 |
| **total concluído** | **1.632** | | **2.112** |
| crossspec — Exp. 2 `@full` | *320 em execução* | 1 | *320* |

**Por que "1 ou 2":** na grade base, metade dos runs roda num spec de um domínio só
(`in10-RW`, `in10-MS` → 1 alvo) e metade num spec misto (`cross5+5`,
`cross10+10` → 2 alvos). Daí 256 runs `none` renderem 384 células
(128×1 + 128×2), e assim por diante. Nas grades `crossspec` o fine-tuning é sempre
num spec de domínio único, então run e célula coincidem.

**O baseline FedAvg supervisionado tem 6 specs**, não 4: inclui os dois controles
`iid:*` (mesmas janelas, fatiadas aleatoriamente em vez de por usuário). Ele
também tem um braço de calibração com `local_epochs=1` — **24 runs, só
`resnetse5`, R=100** — que **não entra em análise nenhuma** por ser outro
protocolo. Sempre filtrar `local_epochs == 5`.

### 6.3 O número para citar

> **2.112 resultados**, de **1.632 federações** treinadas, apoiadas em **192
> pré-treinos federados**.

---

## 7. Hiperparâmetros, e por que cada um

### Comuns às duas fases

| parâmetro | valor | origem / motivo |
|---|---|---|
| **batch size** | **64** | `common.py:69`. Vem do benchmark do da Luz (Seção V-D), para manter compatibilidade célula-a-célula |
| **janelas por cliente** | **192** | `cross_device.py:56`. = **3 batches de 64**. É o mínimo que dá ao TF-C mais de um passo por época; abaixo disso a NT-Xent degenera (o "piso de batch" do `plano_fedssl.md` §3) |
| **seleção de janelas** | estratificada por classe | `budget_indices()`. Todo cliente tem as 6 classes, então os pesos do FedAvg ficam **uniformes** e o Δ entre degraus não carrega mudança de agregação |
| **seleção de usuários** | prefixo fixo, aninhado | `pick_users()`. O prefixo de 5 é subconjunto do de 10 — `cross5+5` usa os mesmos usuários que o `cross10+10`, sem sorteio novo |
| **determinismo** | cuDNN determinístico | `set_deterministic()`. Sem isso dois treinos idênticos divergem ~1e-2 nos pesos (medido 2026-07-28) |

### Fase 2 — fine-tuning federado

| parâmetro | valor | motivo |
|---|---|---|
| **otimizador** | **Adam**, sem weight decay, betas default | é o default do `SimpleSupervisedModel` do benchmark — auditado como ✅ em `metodo_e_auditoria.md` §2.4. O loop local é torch puro, mas equivale à classe |
| **perda** | **CrossEntropy** | default da mesma classe do benchmark. Nenhum paper de FedSSL da nossa lista usa outra coisa no downstream de classificação |
| **learning rate** | **1e-4** (todos os encoders) | `common.py:81` — `BEST_LR`. **Proveniência verificada no paper (Apêndice A, `ssl_benchmark_exemple/*.pdf`, lido em 2026-08-24):** ablação de LR com **5 taxas de 1e-1 a 1e-4**, **4.320 modelos supervisionados**, 3 seeds por configuração. A melhor LR foi escolhida **por encoder**, agregando sobre datasets *e* sobre regimes de amostragem ("the optimal value for each encoder considering the supervised training among the considered data sampling") e adotada para **todos** os experimentos daquele encoder — SL puro e fine-tuning downstream de SSL. 1e-4 venceu em todos menos TS2Vec (1e-3, encoder fora do nosso escopo). ⚠️ **Duas consequências:** (i) não há LR por dataset — numa federação específica ela pode estar longe do ótimo; (ii) **1e-4 é o piso da grade deles**, então o ótimo está na fronteira e nada abaixo foi testado. A nossa busca S1 (`results/rqs/busca_lr.csv`) mostra 1e-4 como a **pior** das 4 LRs originais em 11–12 das 20 células no federado — consistente com (i). Ver `desenho_experimental.md` §7.1 |
| **rodadas** | **150** | casa com o baseline supervisionado, para o eixo de comparação ser o mesmo |
| **épocas locais** | **5** | consenso de 4 papers primários (FedEMA, FedST/FedOST, Saeed IoT-J'21). A ablação `k=1` foi **cortada em 2026-08-04** |
| perda | CrossEntropy | — |
| **shots** | 1, 2, 5, 10, `full` | `few_shot_indices()`: corte estratificado **dentro do shard de cada cliente**, aninhado por seed (1 ⊂ 2 ⊂ 5 ⊂ 10), como no benchmark |

### Fase 1 — pré-treino federado

| parâmetro | TF-C | LFR | motivo |
|---|---|---|---|
| **learning rate** | **3e-4** | **3e-4** | ⚠️ **default, não "melhor"** — ver nota abaixo |
| weight decay | 3e-4 | 3e-4 | idem; `betas=(0.9, 0.99)` hardcoded na classe |
| **épocas locais** | **5** | **30** (= 6×5) | casadas em **épocas efetivas de backbone**: o LFR alterna 5 épocas de preditor para 1 de backbone, então precisa de 6× para o backbone aprender o mesmo tanto |
| **rodadas** | **100** | **100** | valor do FedEMA e do FedST; cobre com folga o regime do Saeed (30–50). Sem seleção — o pré-treino roda até o fim |
| `drop_last` | **True** | False | a máscara NT-Xent do TF-C é fixa em 2×batch; um batch curto quebra a perda |
| específicos | temp 0,2; betas (0,9; 0,99) | 60 projetores → **6** por DPP; projetores **congelados** e não agregados | os projetores do LFR são 96,2% do `state_dict`; agregá-los seria no-op custando 38,6 MB por cliente por rodada |

> ⚠️ **Os hiperparâmetros do pré-treino NÃO são "os melhores reportados".** Há uma
> assimetria que vale conhecer: o lr de **1e-4 do fine-tuning** vem de uma varredura
> do benchmark (Tabela 12, melhor por encoder); já o **3e-4 do pré-treino** é o
> *default da classe do Minerva*, idêntico nos dois métodos e nos quatro encoders,
> hardcoded nos YAMLs oficiais — **não existe varredura de lr de pré-treino
> publicada**, nem no benchmark nem nossa. Auditado como ✅ em
> `metodo_e_auditoria.md` §2.2 e §2.3, mas "✅" ali significa *"idêntico ao
> benchmark"*, não *"ótimo"*.
>
> O próprio benchmark tem variação entre re-runs da mesma célula (`lfr_rnn_run3`
> usou 1e-3 no downstream, contra 1e-4 nos run1/run2) — ver §2.5 do
> `metodo_e_auditoria.md`. Se um revisor perguntar "por que 3e-4?", a resposta
> honesta é *"é o do benchmark que replicamos"*, não *"é o melhor"*.

**Nota de custo:** a coluna `uplink_mb` conta **apenas a Fase 2**. A comunicação do
pré-treino não está nela — o que fica registrado é `pretrain_rounds = 100`. Não
cite `uplink_mb` como "o custo do sistema SSL"; ele é igual ao do baseline por
construção, e o SSL gastou 100 rodadas antes de começar.

---

## 8. Onde cada coisa mora

| cache | o que tem | chave que o distingue |
|---|---|---|
| `results/fed_cross_device.csv` | baseline supervisionado, 6 specs (inclui `iid`) | filtrar `local_epochs == 5` |
| `results/fedssl_cross_device.csv` | grade base Fed-SSL, populações **casadas** | `pretrain_spec` **vazio** |
| `results/fedssl_crossspec.csv` | populações **desacopladas** | `pretrain_spec` preenchido |
| `results/supervised_eval_transfer.csv` | transfer **centralizado** (não federado) | outro protocolo |

Os parciais por job ficam em `results/*_parts/`. O braço cruzado tem um
subdiretório **por população de pré-treino** — de propósito: a chave do
`consolidate()` não conhece `pretrain_spec`, e sem a separação em pastas dois
experimentos diferentes colidiriam em silêncio.

---

## 9. O que esta grade **não** mede

Lista curta para evitar a pergunta recorrente:

- **Personalização.** Só o modelo global é avaliado — mas ver §9.1: dá para medir
  sem rodar grade nenhuma, num dos dois domínios.
- **Seleção de clientes.** Todos os clientes participam de toda rodada.
- **Transferência cross-dataset a partir de uma federação.** Cada federação só é
  testada em domínio que viu no treino.
- **Custo de comunicação do pré-treino.** Não está em `uplink_mb` (§7).
- **Mais de 2 domínios.** Limitação de disponibilidade de usuários (§5).
- **`k = 1`.** Ablação cortada; o único `k=1` no cache é um braço de calibração do
  baseline, com `resnetse5` só, e não entra em análise nenhuma.

### 9.1 Personalização: dá para medir sem re-rodar? (sim, no RealWorld_thigh)

A grade não avaliou modelos locais, mas **não é preciso re-treinar nada** para
medir personalização — por uma folga do desenho que não era intencional.

**O que está salvo:** **1.632** checkpoints de modelo global — 1.536 em
`checkpoints/fedssl_cross_device/` (um por run de fine-tuning) e 96 em
`checkpoints/fed_cross_device/`. Cada um é o `best.ckpt`, a rodada escolhida pela
validação.

**A folga:** o orçamento de `B=192` janelas por cliente consome só uma fração do
que o usuário tem. O resto **nunca foi visto por modelo nenhum**:

| dataset | janelas/cliente (mediana) | usadas | **não usadas** | total ocioso |
|---|---|---|---|---|
| RealWorld_thigh | 1.024 | 192 | **832** | **8.418** |
| MotionSense | 209 | 192 | **17** | 207 |

Então, **para o RealWorld_thigh**, cada cliente tem ~832 janelas próprias,
rotuladas, fora do treino. Isso permite, com custo de minutos:

1. **Linha de base por cliente** — avaliar o `best.ckpt` global nas janelas ociosas
   de cada cliente. Zero treino. Já dá a variância entre clientes, que a média
   sobre o `test.csv` esconde.
2. **Personalização propriamente dita** — partir do `best.ckpt` global, treinar
   localmente com *k* janelas ociosas do cliente, avaliar no restante delas. Sem
   vazamento: nenhuma dessas janelas entrou no treino federado.

**No MotionSense isso não é viável:** sobram ~17 janelas por cliente (≈3 por
classe). O budget de 192 consumiu 92% do que cada usuário tem.

**Uma armadilha a declarar:** o `best.ckpt` foi escolhido pela validação do
domínio de treino. Usar esse ponto de partida para personalizar herda uma seleção
que não foi feita pensando no cliente — não invalida, mas muda o que a comparação
significa.

E o que **continua impossível sem re-rodar**: comparar personalização contra o
global *nas mesmas janelas de treino*, ou personalizar no MotionSense.

---

## 10. Ressalvas que valem para tudo acima

1. **F7 — seleção few-shot na validação cheia.** A escolha de rodada usa o split de
   validação **inteiro** mesmo quando o treino tem 1 rótulo por classe. Mantido por
   fidelidade ao Minerva/da Luz, mas infla os degraus baixos. Ver
   `metodo_e_auditoria.md`.
2. **O degrau `full` vem de duas grades.** O lado SSL sai da grade Fed-SSL; o
   supervisionado, de `fed_cross_device.csv`. Mesmo protocolo (R=150, budget 192,
   seeds 0–3, `local_epochs=5`), mas outro mutirão.
3. **O backbone do TF-C tem 2,2–4,0× os parâmetros do supervisionado.** É
   constitutivo do método (dois encoders gêmeos), não desvio nosso — mas toda
   comparação de nível carrega isso. Ver `metodo_e_auditoria.md` §2.3.
4. **`cross10+10` tem 20 clientes contra 10 dos demais.** Não existe federação de
   20 clientes só de RealWorld_thigh (a base tem 10 usuários). O Δ dele é
   "acrescentar um segundo domínio", não "dobrar o dado".
