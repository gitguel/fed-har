# Dados — DAGHAR `standardized_view`

*Fonte **única** de fatos sobre os datasets: composição, classes, usuários,
janelas e heterogeneidade. Se um número sobre os dados aparece em outro doc, ele
cita este. Todos os valores foram medidos em 2026-07-27 direto de
`datasets/DAGHAR/standardized_view/*/{train,validation,test}.csv` e são
regeneráveis com `poetry run python scripts/analysis/dataset_facts.py`.*

Características de coleta (posição, taxa, sujeitos) vêm de fonte primária (§5).

---

## 1. Os 6 datasets

| Dataset | Posição no corpo | Taxa orig. | Sujeitos (paper) | **Classes na `standardized_view`** | Usuários no `train` |
|---|---|---|---|---|---|
| KuHar | **cintura** (*waist bag*) | 100 Hz | 90 | 6 | 57 |
| UCI | **cintura** (*waist bag*) | 50 Hz | 30 | **5** (sem Run) | 21 |
| RealWorld_waist | **cintura** | 50 Hz | 15 | 6 | 10 |
| MotionSense | **bolso** (*pocket*) | 50 Hz | 24 | 6 | 17 |
| WISDM | **bolso** (*pocket*) | 20 Hz | 51 | **4** (Sit/Stand/Walk/Run) | 36 |
| RealWorld_thigh | **coxa** (*thigh*) | 50 Hz | 15 | 6 | 10 |

Classes padrão do DAGHAR: {0 Sit, 1 Stand, 2 Walk, 3 Stair-up, 4 Stair-down,
5 Run}.

**Notas de leitura — três armadilhas já pisadas:**

1. **A contagem de classes acima é a da visão padronizada, não a da Tab. 2 do
   paper.** Medido nos 3 splits dos 6 datasets: **UCI tem 5** (sem `Run`) e
   **WISDM tem 4** (sem `Stair-up`/`Stair-down`); os outros 4 têm as 6.
   Em particular, **RealWorld TEM `Run`** na visão padronizada — a afirmação
   anterior de que "UCI e RealWorld não têm run" estava errada para o RealWorld.
   Consequência prática: no cross-domain **RW_thigh + MotionSense ambos têm as 6
   classes ⇒ sem mismatch de conjunto de classes** (ver `plano_fedssl.md`).
2. **A coluna "Sujeitos" é o total original do paper, NÃO o número de clientes
   federados.** No cross-device os clientes saem dos usuários do split de
   **treino** (`partition_users.py` lê `train.csv`; val/test ficam intocados na
   avaliação). Ver §3.
3. **A taxa de amostragem original não é confundidor** nesta visão: a
   `standardized_view` reamostra todos para uma taxa comum e alinha
   janelas/gravidade. Ela importa só como característica de realismo (WISDM a
   20 Hz é o mais pobre em resolução temporal na origem).

## 2. Agrupamento por posição do sensor

As 6 bases caem em **dois grupos cinemáticos**:

- **Grupo CINTURA** — `{KuHar, UCI, RealWorld_waist}` (tronco/quadril).
- **Grupo PERNA/BOLSO** — `{MotionSense, WISDM, RealWorld_thigh}` (membro
  inferior; bolso frontal e coxa medem essencialmente o movimento da perna).

A evidência de que **a transferência segue a posição** (+10,5 pp dentro do grupo
no supervisionado, +19,5 pp no TF-C) está em `resultados.md §1`.

## 3. Partição por usuário (base do eixo cross-device)

Usuários por split e janelas por usuário no `train`:

| Dataset | usuários `train`/`val`/`test` | total | janelas/usuário min / mediana / máx | usuários < batch (64) | janelas no `train` |
|---|---|---|---|---|---|
| KuHar | 57 / 7 / 15 | **79** | **1 / 10 / 103** | **48 (84%)** | 1.392 |
| MotionSense | 17 / 2 / 5 | 24 | 165 / 211 / 244 | 0 | 3.558 |
| UCI | 21 / 3 / 6 | 30 | 99 / 113 / 139 | 0 | 2.420 |
| WISDM | 36 / 4 / 11 | 51 | 235 / 236 / 476 | 0 | 8.748 |
| RealWorld_thigh | 10 / 2 / 3 | 15 | 957 / 1024 / 1204 | 0 | 10.338 |
| RealWorld_waist | 10 / 2 / 3 | 15 | 958 / 1023 / 1201 | 0 | 10.332 |
| **Total** | **151** / 20 / 43 | 214 | — | 48 | **36.788** |

- **Os splits são disjuntos por usuário nos 6 datasets** ⇒ particionar o `train`
  por usuário **não vaza** na avaliação.
- **KuHar tem 79 usuários, não 90**: a padronização do DAGHAR descartou 11.
- As 36.788 janelas de `train` são exatamente o corpus `combined` — é esse o
  número de rótulos que o *skyline* consome no pré-treino (`resultados.md §3`).
- **O KuHar é o único dataset em que a partição por usuário quebra o SSL**: 48 de
  57 clientes têm menos janelas que um batch, e a perda auto-supervisionada é
  definida *sobre o batch*. Análise completa em `plano_fedssl.md §3`.

## 4. Heterogeneidade: os dois skews medidos

| Dataset | skew de rótulo — TV | feature skew — η²(feature\|classe) |
|---|---|---|
| KuHar | **0.539** | 0.172 (ruidoso: usuários ~10 janelas) |
| MotionSense | 0.067 | 0.029 |
| UCI | 0.043 | 0.043 |
| RealWorld_thigh | 0.032 | **0.078** (maior dos controles) |
| RealWorld_waist | 0.032 | 0.040 |
| WISDM | **0.000** | 0.023 |

**skew de rótulo (TV)** — *os clientes têm rótulos diferentes?* (P(y)). Distância
de variação total entre a distribuição de classes de um usuário e a global do
dataset, média sobre usuários: `TV = ½ · Σ_c |p_usuário(c) − p_global(c)|`. Vai de
0 (idêntico ao global) a 1 (disjunto). WISDM 0.000 = todo usuário replica a
proporção global; KuHar 0.539 = usuários com 1–2 das 6 classes.

**feature skew (η² controlado por rótulo)** — *dado o mesmo rótulo, os sinais
diferem?* (P(x|y)). Eta-quadrado (fração de variância explicada): **por classe**,
`η² = SS_entre-usuários / SS_total`, média sobre features e classes, colunas
`(accel|gyro)-[xyz]-\d+`. 0 = mesma atividade idêntica entre usuários; 1 = a
identidade do usuário explica tudo. O controle por classe é essencial — senão a
variância "andar × sentar" dominaria. O 0.172 do KuHar é inflado (média
por-usuário mal estimada com ~10 janelas).

**Leitura:** a partição natural por usuário no DAGHAR entrega **muito feature skew
e quase nenhum label skew** (exceto KuHar). Ou seja, ela **isola feature skew de
graça**; label skew precisa ser induzido artificialmente — o que traz um
confundidor de volume de dado, tratado em `plano_fedssl.md §2`.

## 5. Referências

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
