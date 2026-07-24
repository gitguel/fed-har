# Análise de domain shift entre os datasets do DAGHAR

*Escrito em 2026-07-21; §4.1 (skyline SL-`combined`) acrescentada em 2026-07-24.
Objetivo: fundamentar o domain shift do DAGHAR em duas
frentes — (1) os **protocolos de coleta** de cada dataset (posição do sensor no
corpo, dispositivo, taxa, população) e (2) a **matriz de transferência
centralizada** já medida (`results/*_eval_transfer.csv`) — e conectar as duas.
Serve de base para justificar o eixo federado **cross-device** e para a
apresentação. Todo número aqui é regenerável dos caches (ver §6); as
características de coleta vêm de fonte primária citada (§7).*

Relacionados: `paper/esqueleto_artigo.md` (III-A, IV-A), `plano_fedssl_simulado.md`
(eixo cross-device), `contribuicoes_forte_fraco_defesa.md`.

---

## 1. Protocolos de coleta (fonte primária: DAGHAR, Tabela 2)

O DAGHAR padroniza 6 datasets de HAR de smartphone (unidades, taxa de
amostragem, gravidade, rótulos e janelas), **removendo vieses triviais e
preservando as diferenças intrínsecas de domínio**. A característica de domínio
que mais importa para transferência é a **posição do sensor no corpo**, relatada
na Tabela 2 do paper:

| Dataset | Posição no corpo | Taxa orig. | Sujeitos | Ativ. (padrão) |
|---|---|---|---|---|
| **KuHar** | **cintura** (waist bag) | 100 Hz | 90 | sit, stand, walk, up, down, run |
| **UCI** | **cintura** (waist bag) | 50 Hz | 30 | sit, stand, walk, up, down |
| **RealWorld_waist** | **cintura** | 50 Hz | 15 | sit, stand, walk, up, down |
| **MotionSense** | **bolso** (pocket) | 50 Hz | 24 | sit, stand, walk, up, down, run |
| **WISDM** | **bolso** (pocket) | 20 Hz | 51 | sit, stand, walk, up, down, run |
| **RealWorld_thigh** | **coxa** (thigh) | 50 Hz | 15 | sit, stand, walk, up, down |

Notas de fonte:
- Posição, taxa e nº de sujeitos: **DAGHAR, Tab. 2** (Napoli et al., *Scientific
  Data*, 2024) e a figura "Sensor placement on the subject's body" do mesmo
  paper. RealWorld coleta em 7 posições; o DAGHAR usa **coxa** (RW-Thigh) e
  **cintura** (RW-Waist).
- **Atenção — a coluna "Sujeitos" acima é o TOTAL original do dataset (Tab. 2),
  NÃO o número de clientes federados.** No cross-device os clientes saem dos
  **usuários do split de treino** da `standardized_view` (`partition_users.py` lê
  `train.csv`; val/test ficam intocados no eval). Contagem verificada
  (2026-07-21), usuários por split `train/val/test` = total:
  KuHar 57/7/15 = **79** (⚠️ paper diz 90 — a padronização do DAGHAR descartou
  11 usuários do KuHar); MotionSense 17/2/5 = 24; UCI 21/3/6 = 30;
  WISDM 36/4/11 = 51; RealWorld_thigh e _waist 10/2/3 = 15 cada. Splits
  **disjuntos por usuário** nos 6 ⇒ particionar o train por usuário não vaza no
  eval. Logo os clientes cross-device potenciais (antes do agrupamento do KuHar,
  decisão D-K) são **57 / 17 / 21 / 36 / 10 / 10**, não os totais do paper.
- A `standardized_view` **reamostra** todos para uma taxa comum e alinha
  janelas/gravidade ⇒ a diferença de **taxa de amostragem original não é um
  confundidor na visão que usamos**; ela importa só como característica de
  realismo (ex.: WISDM 20 Hz é o mais pobre em resolução temporal na origem).
- Os rótulos são projetados nas 6 classes padrão do DAGHAR
  {Sit, Stand, Walk, Stair-up, Stair-down, Run}; UCI e RealWorld não têm "run".

## 2. Agrupamento por posição do sensor

As 6 bases caem em **dois grupos cinemáticos** pela posição:

- **Grupo CINTURA** — `{KuHar, UCI, RealWorld_waist}` (tronco/quadril).
- **Grupo PERNA/BOLSO** — `{MotionSense, WISDM, RealWorld_thigh}` (membro
  inferior; bolso frontal e coxa medem essencialmente o movimento da perna).

Hipótese: **a transferência entre datasets segue a similaridade de posição** —
sensores em posições parecidas veem distribuições de aceleração/rotação
parecidas, então uma representação treinada num transfere melhor para o outro.

## 3. Evidência: o transfer centralizado segue a posição

Todas as medidas: acurácia, **full finetuning**, média sobre 4 encoders × 4
seeds, pares cross-domain (source ≠ target, excluindo `combined`). Fonte:
`supervised_eval_transfer.csv`, `ssl_{lfr,tfc}_eval_transfer.csv`.

**(a) Dentro do grupo vs entre grupos** — o teste direto da hipótese:

| Método | mesmo grupo de posição | entre grupos | gap |
|---|---|---|---|
| SL | 0.526 | 0.420 | **+0.105** |
| LFR | 0.546 | 0.410 | +0.136 |
| TF-C | 0.610 | 0.415 | **+0.195** |

Leitura: o transfer é sistematicamente melhor dentro da mesma posição já no
supervisionado (+10.5 pp). O **SSL amplifica o transfer intra-posição** (TF-C
sobe para 0.610) mas **quase não atravessa posições diferentes** (0.415 ≈ SL
0.420). Conclusão honesta e não-óbvia: **o SSL fortalece a transferência entre
sensores parecidos; ele não "resolve" o gap entre cintura e perna.**

**(b) Melhores pares de transfer natural (SL)** — todos dentro do grupo:

| Par | acc SL | grupo |
|---|---|---|
| RealWorld_waist → UCI | **0.686** | cintura |
| RealWorld_thigh → MotionSense | 0.586 | perna |
| RealWorld_waist → KuHar | 0.582 | cintura |
| MotionSense → RealWorld_thigh | 0.572 | perna |
| WISDM → MotionSense | 0.563 | perna |
| RealWorld_thigh → WISDM | 0.562 | perna |

**(c) Onde o SSL mais recompra performance (Δ acc vs SL)** — o par de maior
ganho é intra-grupo perna:

| Par | SL | LFR | TF-C | Δ(TF-C−SL) |
|---|---|---|---|---|
| RealWorld_thigh → MotionSense | 0.586 | 0.634 | **0.782** | **+0.196** |
| MotionSense → RealWorld_thigh | 0.572 | 0.613 | 0.710 | +0.137 |

## 4. Casos, exceções e o efeito do corpus `combined`

- **RW_waist → UCI (0.686)** é o melhor transfer *natural* (ambos cintura); é o
  caso onde o domain shift já é pequeno sem SSL.
- **RW_thigh ↔ MotionSense** é o par de maior **Δ(SSL−SL)** (perna); é onde o
  SSL mais ajuda — por isso é o par escolhido como **prova-de-conceito** do
  federado (ver §5). ⚠️ *Escolher o par pelo maior Δ é seleção-no-desfecho:
  vale para validar o pipeline, não como evidência da tese geral de mitigação.*
- **KuHar é um outlier fraco do grupo cintura**: pares envolvendo KuHar
  transferem 0.448 (vs 0.470 sem KuHar); dentro do grupo, KuHar → UCI é só 0.341
  (contra RW_waist → UCI 0.686). Provável causa: 100 Hz original, "waist bag"
  frouxo (sensor não rígido ao corpo) e usuários minúsculos (mediana ~10
  janelas/usuário). Registrar como limitação ao usar KuHar como "caso difícil".
- **Corpus `combined` (generalista) degrada como fonte única, menos sob SSL.**
  Treinar/finetunar no `combined` e avaliar por alvo, vs o especialista
  in-domain (média sobre alvos, full-ft): SL −0.053, LFR −0.046, **TF-C −0.026**
  (RW_thigh chega a reverter: +0.040 no TF-C). Ou seja: um único modelo
  generalista custa acurácia, mas o SSL atenua o custo.
- **Mas `combined` como pré-treino + finetune no alvo (comb2target) ganha.**
  `pretrain(combined) → finetune(target)`, full-ft, **vs SL in-domain**: **TF-C
  supera o in-domain em todos os 6 alvos** (+0.03 a +0.13; média 0.833 vs 0.766)
  e **LFR não degrada vs SL in-domain** (KuHar exceção, −0.058) — mas **fica
  abaixo do pré-treino supervisionado no `combined`** (ver o skyline logo
  adiante). Isto é o argumento a favor de juntar dados não-rotulados de vários
  domínios no pré-treino — exatamente a proposta de valor do SSL federado.

## 4.1 O skyline SL-`combined` (ablação do comb2target)

*Medido em 2026-07-23/24 por `scripts/ssl/sl_comb2target_eval.py`; cache
`results/sl_comb2target_eval_transfer.csv`. **Números abaixo são PARCIAIS**
(2832 das 3456 células; falta o encoder `rnn` nas seeds 1–3, em execução) e
serão reprocessados quando a grade fechar.*

Para atribuir o ganho do `comb2target` ao **SSL** e não à etapa
`pré-treino → especialização no alvo`, o comparador é o mesmo pipeline com o
backbone vindo do modelo **supervisionado** treinado no `combined`
(`checkpoints/supervised/<enc>/combined/seed<N>/best.ckpt`), com cabeça,
otimizador, protocolos e regimes idênticos aos do `downstream_eval.py`.

**Este comparador NÃO é um baseline pareado — é um *skyline*.** Orçamento de
supervisão de cada braço:

| Braço | Rótulos no pré-treino | Seleção de checkpoint |
|---|---|---|
| LFR / TF-C | **0** (train+val do `combined` sem rótulos) | época fixa (LFR 600, TF-C 100), sem ES |
| SL-`combined` | **36.788** (train) | `best.ckpt` por `val_loss` **rotulada** (+5.844 janelas) |

Todo o viés aponta a favor do SL-`combined`: mais rótulos **e** seleção de
modelo supervisionada. Além disso, ele é **inviável sob a premissa do eixo
federado** (dado não-rotulado abundante, rótulo escasso) — logo não compete
com o SSL, ele o limita superiormente.

Acurácia média, in-domain (`source==target`), full finetuning, sobre as 2832
células comuns aos três métodos:

| shots | SL-`combined` (skyline) | LFR | TF-C | Δ(LFR) | Δ(TF-C) |
|---|---|---|---|---|---|
| 1 | **0.528** | 0.396 | 0.413 | −0.131 | −0.115 |
| 10 | 0.692 | 0.581 | **0.717** | −0.111 | **+0.024** |
| 100 | 0.765 | 0.714 | **0.833** | −0.051 | **+0.068** |
| full | 0.793 | 0.760 | **0.856** | −0.034 | **+0.063** |

Em cross-domain (`source≠target`) o skyline fica acima dos dois métodos SSL em
todos os regimes (Δ TF-C −0.03 a −0.15).

**Como reportar estes números (regra de redação):**

1. O achado positivo é do **TF-C**: ele **supera um comparador com vantagem de
   supervisão** in-domain a partir de 10 shots (+0.02 a +0.07, 6/6 alvos),
   gastando **zero rótulo** no pré-treino. Vencer um baseline enviesado contra
   você é evidência **mais forte** que vencer o SL in-domain, não mais fraca.
2. Para o LFR, reportar a **medição** ("fica abaixo do skyline supervisionado"),
   **nunca o veredito** ("LFR perde" / "LFR degrada"). A comparação não é
   pareada em rótulos, então ela não sustenta juízo de mérito sobre o método —
   sustenta apenas que o LFR não alcança pré-treino supervisionado com 36.8k
   rótulos, o que é quase esperado.
3. **A ablação é um pacote**: entra inteira (TF-C favorável + LFR desfavorável)
   ou fica inteira fora por escopo. Reportar só a metade favorável seria
   cherry-picking, já que as duas vêm da mesma medição.
4. Rótulo canônico em tabelas/figuras: **"skyline (pré-treino supervisionado,
   36.8k rótulos)"** — não "baseline". O rótulo faz o trabalho argumentativo.

Consequência prática: as tabelas da apresentação que reportam "Δ vs SL" contra
o **SL in-domain** (`docs/apresentacao/tabelas/tab_comb2target`,
`fig_{lfr,tfc}_comb2target`) usam um comparador que agora sabemos ser fraco, e
precisam ser refeitas com o skyline ao lado.

Em aberto: um SL-`combined` **pareado em rótulos** (pré-treino supervisionado
usando só N rótulos do `combined`) seria o baseline de fato justo; custo
não-trivial, só se o `comb2target` for promovido a seção do artigo.

## 5. Implicações para o eixo federado cross-device

- O par **RW_thigh + MotionSense** (grupo perna) é o de maior Δ(SSL−SL) ⇒
  **prova-de-conceito**: máximo sinal para validar que o FedAvg-SSL converge,
  antes de estressar com o caso difícil.
- **Controle honesto do custo de domain shift no cross-device** =
  **Δ(cross-domain − in-domain)**: federar usuários de RW_thigh + MotionSense
  juntos (domain shift + skew inter-pessoa) menos federar usuários de um só
  dataset (só skew inter-pessoa). É o análogo cross-device dos ≈8 pp que o
  baseline cross-silo (agora preliminar) mediu — mas sem confundir domain shift
  com heterogeneidade por pessoa.
- **KuHar como "caso difícil" confunde dificuldade de domínio com escassez por
  cliente** (usuários minúsculos) — separar os dois efeitos na leitura.
- O gap cintura↔perna (§3a) prevê que **cross-domain entre grupos diferentes**
  (ex.: adicionar UCI/RW_waist ao pool de perna) é o teste mais duro de
  mitigação — candidato natural à segunda onda depois da POC.

## 6. Proveniência e reprodução dos números

- Caches: `results/supervised_eval_transfer.csv`,
  `results/ssl_lfr_eval_transfer.csv`, `results/ssl_tfc_eval_transfer.csv`,
  `results/ssl_{lfr,tfc}_comb2target_eval_transfer.csv`,
  `results/sl_comb2target_eval_transfer.csv` (§4.1, skyline).
- Agregação padrão desta análise: **acc, `n_shots="full"`, `protocol="finetune"`,
  média sobre encoders × seeds**; pares cross-domain excluem `source==target` e
  `source=="combined"`. Grupos de posição: cintura `{KuHar, UCI, RealWorld_waist}`,
  perna `{MotionSense, WISDM, RealWorld_thigh}`.
- Os números de §3–§4 foram computados diretamente desses caches (2026-07-21);
  os de §4.1, em 2026-07-24 (grade ainda parcial — ver a nota da seção).
  Um script de regeneração pode ser adicionado a `scripts/analysis/` se estas
  tabelas entrarem na apresentação/artigo.

## 7. Referências

1. O. O. Napoli, D. H. P. Soto, G. P. C. P. da Luz, et al., "A benchmark for
   domain adaptation and generalization in smartphone-based human activity
   recognition", *Scientific Data* 11, 1192 (2024).
   DOI 10.1038/s41597-024-03951-4. Open access: PMC11531562. **(Fonte das
   posições no corpo — Tab. 2 e figura "Sensor placement on the subject's
   body".)** Dados: Zenodo 11992126.
2. N. Sikder, A.-A. Nahid, "KU-HAR: An open dataset for heterogeneous human
   activity recognition", *Pattern Recognition Letters* 146:46–54, 2021.
3. M. Malekzadeh, R. G. Clegg, A. Cavallaro, H. Haddadi, "Protecting Sensory
   Data against Sensitive Inferences" (MotionSense), 2018.
4. J. R. Kwapisz, G. M. Weiss, S. A. Moore, "Activity recognition using cell
   phone accelerometers" (WISDM), *SIGKDD Explorations* 12(2):74–82, 2011.
5. D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz, "A Public Domain
   Dataset for Human Activity Recognition Using Smartphones" (UCI-HAR),
   ESANN 2013.
6. T. Sztyler, H. Stuckenschmidt, "On-body localization of wearable devices: An
   investigation of position-aware activity recognition" (RealWorld),
   PerCom 2016.
