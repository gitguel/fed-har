# Desenho experimental cross-device (clientes = usuários)

*Escrito em 2026-07-24, consolidando a sabatina do desenho cross-device. Fecha o
desenho dos experimentos do eixo cross-device antes da implementação: o que cada
experimento mede, como isolar cada tipo de heterogeneidade por um Δ contra um
controle pareado, e os riscos de método a evitar. Os números vêm dos caches e do
`train.csv` da `standardized_view` (§6). Relacionados:
`docs/analise_domain_shift.md`, `docs/limite_batch_cliente_fssl.md`,
`docs/plano_fedssl_simulado.md`.*

---

## 1. Pergunta e os três eixos de heterogeneidade

O eixo cross-device (cliente = usuário) permite estressar o Fed-SSL contra **três
formas distintas de heterogeneidade**, que a literatura de FL trata como eixos
separados e que aqui conseguimos **isolar**:

1. **Domain shift** — clientes de datasets diferentes (posição do sensor: coxa
   vs bolso vs cintura). É o custo que o pivô cross-device quer medir sem o
   confundidor do cross-silo.
2. **Feature skew** (covariate shift, P(x|y)) — dentro do mesmo dataset, a mesma
   atividade tem sinal diferente entre pessoas/dispositivos.
3. **Label skew** (P(y)) — clientes com distribuições de classe diferentes.

## 2. Taxonomia: os dois skews que medimos (com definições)

| Dataset | skew de rótulo — TV | feature skew — η²(feature\|classe) |
|---|---|---|
| RealWorld_thigh | 0.032 | **0.078** (maior dos controles) |
| MotionSense | 0.067 | 0.029 |
| RealWorld_waist | 0.032 | 0.040 |
| UCI | 0.043 | 0.043 |
| WISDM | **0.000** | 0.023 |
| KuHar | **0.539** | 0.172 (ruidoso: usuários ~10 janelas) |

**skew de rótulo (TV)** — *os clientes têm rótulos diferentes?* (P(y)). Distância
de variação total entre a distribuição de classes de um usuário e a global do
dataset, média sobre usuários: `TV = ½·Σ_c |p_usuário(c) − p_global(c)|`. Vai de
0 (idêntico ao global) a 1 (disjunto). WISDM 0.000 = todo usuário replica a
proporção global; KuHar 0.539 = usuários com 1–2 das 6 classes.

**feature skew (η² controlado por rótulo)** — *dado o mesmo rótulo, os sinais
diferem?* (P(x|y)). Eta-quadrado (fração de variância explicada): **por classe**,
`η² = SS_entre-usuários / SS_total`, média sobre features e classes. 0 = mesma
atividade idêntica entre usuários; 1 = identidade do usuário explica tudo. O
controle por classe é essencial — senão a variância "andar × sentar" dominaria.
KuHar 0.172 é inflado (média por-usuário mal estimada com ~10 janelas).

**Leitura:** a partição natural por usuário no DAGHAR entrega **muito feature skew
e quase nenhum label skew** (exceto KuHar). Logo ela **isola feature skew** de
graça; label skew precisa ser induzido (§3, §4).

## 3. Decomposição fatorial: cada efeito é um Δ contra um controle pareado

O que isola um efeito **não é a célula, é o Δ contra o controle certo**:

| Efeito isolado | Δ = experimento − controle pareado | O que o Δ cancela |
|---|---|---|
| **Domain shift** | cross-domain(user) − in-domain(user) | feature skew (ambos têm) |
| **Feature skew** | in-domain(**partição por usuário**) − in-domain(**shards IID do mesmo dataset**) | dado, rótulos e domínio idênticos; muda só a estrutura da partição |
| **Label skew** | in-domain(rótulo skewado artificial) − in-domain(**IID pareado em volume**) | volume de dado (ver risco 1) |

- **Cross-domain** carrega feature skew *além* do domain shift → a célula
  cross-domain **não** é "domain shift puro"; é o Δ contra in-domain que isola.
  Prova-de-conceito: **RW_thigh + MotionSense** (grupo perna, maior Δ(SSL−SL)
  centralizado; ambos com as 6 classes balanceadas ⇒ sem mismatch de classe).
- **Feature skew** só fica isolado com o controle de **shards-IID** (quebrar a
  estrutura de usuário do mesmo dataset, mantendo dado/rótulos). Sem ele, temos
  o número da célula, não o efeito atribuível.

## 4. Riscos de método a fechar antes de implementar

1. **Label skew artificial confunde-se com volume de dado.** Remover rótulos de
   um cliente reduz o dado dele e o total; a degradação pode vir de "menos dado".
   O controle do braço de label skew tem de ser **pareado em volume** (remover a
   mesma quantidade aleatoriamente, preservando as proporções). Sem isso o efeito
   não é atribuível a skew.

2. **"Label skew" significa coisas diferentes no SSL e no supervisionado.** O
   pré-treino SSL **ignora rótulos** (LFR/TF-C não leem `standard activity code`).
   A mesma operação "remover janelas da classe c do cliente k" produz:
   - no **baseline supervisionado federado**: skew de rótulo clássico (P(y)
     enviesado) — o efeito desejado, que morde no treino/finetuning rotulado;
   - no **pré-treino SSL**: **skew de cobertura/feature** (o P(x) não-rotulado
     perde uma região da variedade) **+ encolhimento** do cliente (o piso de
     batch de `limite_batch_cliente_fssl.md` volta).
   Consequência: não usar uma única partição "com label skew" e alegar que ela
   isola label skew para os dois braços. Label skew é construto do estágio
   **rotulado** (downstream/supervisionado); atribuir o efeito a esse estágio.

3. **Ortogonalidade parcial.** Os três eixos não são perfeitamente ortogonais
   (cross-domain traz feature skew; label skew artificial traz cobertura+volume).
   Cada efeito só é limpo como Δ pareado (§3), não como célula isolada.

**Veredito da sabatina:** com (a) cada efeito medido como Δ contra o controle
pareado — incluindo o shards-IID para feature skew, (b) o braço de label skew
pareado em volume, e (c) label skew tratado como construto do estágio rotulado
(cobertura no SSL) — o desenho fatorial fecha os três eixos de forma isolada e é
publicável.

## 5. Escopo e ordem de execução

1. **Primeiro, ignorando o KuHar**: construir e validar o pipeline cross-device
   com os 3 controles (in-domain RW_thigh, in-domain MotionSense, cross-domain
   RW_thigh+MotionSense) — nenhum inclui KuHar, então o piso de batch não bloqueia.
2. **Depois**: endereçar o piso de batch (KuHar é o caso que o expõe) — **forma em
   aberto, discussão adiada** (ver `limite_batch_cliente_fssl.md §5`).
3. **Controles extras** derivados de §3: shards-IID do mesmo dataset (isolar
   feature skew) e o braço de label skew artificial pareado em volume.
4. **Segunda onda** (heterogeneidade por pessoa como eixo próprio): WISDM
   (36 clientes, label skew 0.000 — feature skew puro) e KuHar (label skew 0.539)
   como os dois extremos.

## 6. Proveniência e reprodução

- Skews da §2: `datasets/DAGHAR/standardized_view/*/train.csv`, coluna `user`.
  TV de rótulo = ½·Σ|p_user − p_global| média sobre usuários; feature skew =
  η² entre-usuários das médias por-usuário-por-classe / var total intra-classe,
  média sobre features×classes, colunas `(accel|gyro)-[xyz]-\d+`.
- Presença de classes (`standardized_view`, todos os splits, 2026-07-24):
  RW_thigh / RW_waist / MotionSense / KuHar = 6 classes; **UCI = 5 (sem run)**;
  **WISDM = 4 (sit/stand/walk/run, sem stair-up/down)**. ⚠️ Isto corrige a Tab.
  de `analise_domain_shift.md §1` (que, copiando a Tab. 2 do DAGHAR original,
  dizia "RealWorld não tem run" — na visão padronizada, tem).
