# Contribuições do trabalho: pontos fortes, fracos e preparação para a orientação

> **⚠️ ATUALIZAÇÃO 2026-07-21 — PIVÔ cross-silo → cross-device.** A federação
> **cross-silo** (1 dataset/cliente; cenários 1–8) foi **abandonada como desenho e
> como controle**. Impacto direto nesta análise: a força **F1 ("desenho com grupo
> de controle")** e a resposta a **W2 (FedBN)** falavam do controle cross-silo
> (cenário 1 vs 2 vs ablação); esse controle específico é **superseded** — o
> controle honesto agora é **Δ(cross-domain − in-domain)** no eixo **cross-device**
> (clientes = usuários). Os ~8 pp viram **motivação preliminar**, não a
> contribuição. O espírito de F1 (atribuição causal controlada) permanece, mas na
> nova topologia. Ver `docs/analise_domain_shift.md` e
> `docs/plano_fedssl_simulado.md`.

*Escrito em 2026-07-07, consolidando a análise desta sessão (verificação de
ineditismo em 4 grafos, mapeamento do estado da arte de F-SSL, auditoria do
que já está medido). Objetivo: te deixar munido para a conversa com o
orientador — inclusive para as perguntas incômodas. Documento franco de
propósito: a parte de fraquezas é onde está o valor, porque é o que ele vai
atacar. Fatos medidos vs pendentes estão marcados.*

Relacionados: `estado_da_arte_fssl_e_contribuicoes.md` (a base factual),
`paper/esqueleto_artigo.md` (II-D/II-E), `plano_experimento3_fedssl.md`.

---

## 0. O pitch de um parágrafo (decore este)

> "A contribuição não é federar o LFR. É o **primeiro estudo controlado de
> *quando e por que* o pré-treino auto-supervisionado — centralizado ou
> federado, contrastivo (TF-C) ou livre de aumentações (LFR) — mitiga
> heterogeneidade de **domínio real** numa federação cross-silo de HAR, com
> **custo de comunicação na conta** e implementação **validada contra o
> benchmark publicado** do próprio grupo. Nenhum trabalho na literatura tem
> esse desenho: os que federam SSL usam uma técnica só, contrastiva, sob
> heterogeneidade simulada, sem grupo de controle e sem comparar topologias de
> pré-treino."

Cada afirmação desse parágrafo é defensável e falsificável (ver §1). É a
diferença entre "fiz uma engenharia" e "respondi uma pergunta que estava
aberta".

---

## 1. Pontos fortes (ordenados por quanto blindam a defesa)

| # | Força | Por que é forte / evidência | Quão único |
|---|---|---|---|
| F1 | **Desenho com grupo de controle** | Cenário 1 (non-IID por domínio) vs cenário 2 (IID, **mesmo volume**) + ablação 3–8 (intra-domínio) → permite **atribuição causal** do custo do domain shift (≈8 pp, medido). UniHAR/FedCSSL/CDFL não têm controle: medem "federado é pior" sem isolar a causa. | Muito alto |
| F2 | **Matriz comparativa que ninguém tem** | 2 famílias de SSL (contrastiva vs augmentation-free) × 4 encoders × 2 topologias de pré-treino (central/federado) × regimes de dados. A literatura federa **uma** técnica por paper. | Alto |
| F3 | **Reprodutibilidade / validação prévia** | Réplica do benchmark de da Luz et al. célula a célula: viés ±2 pp, MAE 2–5 pp, 14/14 hiperparâmetros idênticos. Nossos números são *plug-compatible* com a tabela publicada — o leitor compara direto. | Alto (raro em FedSSL) |
| F4 | **Ineditismo verificado a sério** | LFR e TF-C nunca federados, confirmado em 4 ângulos (busca web + snowball S2 + snowball OpenAlex + queries Scholar-style). Não é um "acho que é novo". | Alto |
| F5 | **Comunicação como métrica de 1ª classe** | uplink/downlink por rodada, pré-treino + finetuning. Achado antecipável e não-óbvio: LFR federado é **caro** em uplink (preditores), linear-readout federado é **quase grátis**. A área reporta acurácia, não bytes. | Alto |
| F6 | **Heterogeneidade real, não simulada** | 1 dataset real por cliente = domain shift por construção, não Dirichlet label-skew. Mais realista que a maioria da literatura de FL (que é vício de visão com label-skew). | Médio-alto |
| F7 | **Resultado garantido nos dois desfechos** | FedAvg-SSL funcionar = 1ª evidência positiva p/ LFR/TF-C federados; divergir = caracterização quantificada (curvas `agg_shock`, efeito FedBN/freq.) de *por que* SSL de séries temporais diverge. Achado negativo é publicável (cf. FedEMA sobre stop-gradient). | Médio |
| F8 | **Ponte entre as duas linhas do HIAAC** | Você (linha distribuída) + dados/protocolo/baselines (linha DAGHAR). Explica a escolha de FedAvg/Flower como *idioma da sua linha*, não falta de ambição; abre coautoria/revisão interna. | Contexto |
| F9 | **Perguntas de design que são pesquisa, não plumbing** | DPP global vs local (coerência de alvos — específico da família augmentation-free, não existe em SimCLR federado); política de preditores (`full` vs `backbone-only`); alternância × rodada. Cada uma vira ablação com resultado. | Médio |

---

## 2. Pontos fracos (ordenados por probabilidade de o orientador atacar × gravidade)

*Para cada um: a crítica crua, quão válida é, e a resposta/mitigação. Não
esconda nenhum — chegar na reunião já tendo nomeado a fraqueza desarma o
ataque.*

### W1 — "Não há método novo; é aplicar técnicas existentes num cenário novo." **(o ataque principal, o mesmo da provocação)**

- **Validade**: parcial e legítima. Não propomos algoritmo novo de SSL nem de
  agregação. O contra-argumento (o desenho + a análise **são** a contribuição
  de uma dissertação empírica) é verdadeiro, mas um comitê cético pode querer
  "um método com nome".
- **Resposta em 2 tempos**: (1) reafirmar que contribuição empírica com
  desenho controlado, ineditismo verificado e validação é padrão-mestrado
  aceito (muitos papers de FedSSL bem citados são exatamente isso — FedU,
  L-DAWA nasceram como "estudo + variante simples"). (2) **Oferecer um
  componente de método de baixo custo** para calar a crítica (§3): a variante
  `backbone-only`/`fedbn`/`agg_shock` já está no plano — basta **promovê-la de
  ablação a contribuição nomeada**.

### W2 — "Só FedAvg. Cadê FedProx/FedBN/algo mais esperto?"

- **Validade**: média. É conservador. Defensável (baseline padrão,
  reprodutível, isola o efeito do *ponto de partida* mantendo o agregador
  fixo), mas soa tímido se enunciado como "só FedAvg".
- **Resposta**: FedAvg é *deliberado* — para atribuir o efeito à representação
  (não ao agregador) você precisa fixar o agregador. **FedBN já está previsto
  como variante** (`fedbn`) exatamente porque o nosso non-IID é de *features*
  (o caso do FedBN). Enunciar assim inverte a fraqueza em decisão metodológica.

### W3 — "É incremental sobre o benchmark de da Luz et al.: mesmo dado, mesmo protocolo, mesmas técnicas."

- **Validade**: é o risco de percepção **mais perigoso**, porque é
  superficialmente verdadeiro (reusamos dados/protocolo de propósito).
- **Resposta**: a continuidade é *ferramenta*, não a contribuição. O benchmark
  responde "qual encoder/técnica é melhor **no centralizado in-domain**"; nós
  respondemos "o pré-treino mitiga **domain shift federado**, e a que custo" —
  eixo (federado), métrica (comunicação) e avaliação (transfer 7×6 +
  central-vs-fed) que **não existem** no benchmark. Frase: *"reusamos o
  protocolo deles pelo mesmo motivo que se reusa uma régua calibrada — para
  que a medida nova seja comparável, não porque a medida nova seja a mesma".*

### W4 — "Vizinho próximo: FedST/FedOST já fazem tempo-frequência em FL."

- **Validade**: real (achado do snowballing). Um reviewer atento acha.
- **Resposta (já no esqueleto II-D)**: FedST/FedOST são **supervisionados**
  (com rótulo), miram **personalização** (não representação transferível),
  usam tempo-frequência como **regularização** (não o método/protocolo TF-C
  nem pré-treino SSL) e não medem comunicação nem comparam topologias. Citar e
  distinguir **proativamente** — nunca deixar o reviewer descobrir sozinho.

### W5 — "Uma coleção só (DAGHAR): 6 datasets, mas todos smartphone-IMU, mesma modalidade."

- **Validade**: média. Limita a generalização das conclusões.
- **Resposta**: escopo declarado; DAGHAR é *desenhado* para isolar domain
  shift com vieses triviais removidos — é a escolha certa para a **pergunta**
  (efeito do domínio, não da modalidade). Generalização a outras modalidades =
  trabalho futuro explícito. Não prometer o que não se vai medir.

### W6 — "6 clientes cross-silo. E escala? Cross-device?"

- **Validade**: média (reviewer de FL cobra número de clientes).
- **Resposta**: cross-silo com 1 domínio/cliente é a topologia que **cria** o
  domain shift estudado; escalar para 30–100 subclientes está no radar
  (notas §5) como extensão, mas mudaria a pergunta (label/quantity skew intra-
  domínio, não domain shift inter-silo). Manter o corte limpo.

### W7 — "Simulação em GPU; a comunicação são bytes teóricos, não tempo real."

- **Validade**: média. Verdade.
- **Resposta**: uplink/downlink em bytes é a métrica **independente de
  ambiente** de propósito (tempo/latência dependem da rede física e não são
  reproduzíveis) — decisão registrada nas notas §6. É honesto e comparável
  entre cenários; wall-clock ficaria não-replicável.

### W8 — "O componente de maior valor (Exp. 3) ainda não rodou." **(risco de cronograma, não de mérito)**

- **Validade**: alta como *risco*, nula como crítica de mérito. Mês 3 é
  apertado (notas §7).
- **Resposta**: o design doc do Exp. 3 já existe (gates, ondas priorizadas,
  gate de equivalência IID que separa bug de domain shift, ordem de corte
  C→D→B). Ou seja: risco **gerenciado**, com plano de contração. Levar isso à
  reunião como "tenho plano B de escopo", não como "pode não dar tempo".

### W9 — "Achado negativo soa como plano de fuga."

- **Validade**: baixa se bem enquadrado, alta se mal enquadrado.
- **Resposta**: não vender como "se der errado, publico o erro". Vender como
  **pergunta científica cujas duas respostas são informativas**: "métodos SSL
  de séries temporais convergem sob agregação federada com domain shift?" —
  isso tem valor decidido antes de saber o resultado (é o que FedEMA fez com
  stop-gradient). O `agg_shock` e o gate IID são o instrumento que torna o
  "não" rigoroso, não anedótico.

---

## 3. A contribuição em três níveis (leve o do meio para a reunião)

Para negociar escopo com o orientador, tenha os três prontos:

| Nível | O que é | Custo extra | Quando escolher |
|---|---|---|---|
| **Mínimo defensável** | Exps. 0–2 (SL + SSL central + baseline federado + finetuning federado com init central) + a quantificação do domain shift. **Já quase tudo medido.** | ~0 (falta finetune fed) | Se o Mês 3 desabar |
| **Recomendado (alvo)** | + Exp. 3 (FedAvg-SSL de LFR e TF-C) **com uma variante de método promovida a contribuição nomeada** (`backbone-only` OU `fedbn`) + custo de comunicação ponta-a-ponta + a pergunta nomeada "augmentation-free é mais robusto à agregação que contrastivo?" | ~1–2 semanas de cluster | Caso base |
| **Stretch** | + agregação adaptativa via `agg_shock` (mini-L-DAWA para séries temporais) OU FedEMA adaptado ao TF-C + 1 baseline FedSSL reproduzido (SimCLR-TS) | +2–3 semanas | Se estiver adiantado |

**Recomendação**: proponha o **Recomendado** e pergunte ao orientador qual
variante de método ele prefere ver promovida (é a decisão dele, §5). Isso já
responde à provocação: não é "federar uma técnica", é "federar duas famílias +
caracterizar + uma variante de método própria".

---

## 4. Opções concretas de "mais método" (custo ~zero, já no plano do Exp. 3)

Detalhadas em `estado_da_arte_fssl_e_contribuicoes.md` §5.3 e no design doc.
Resumo para decidir com o orientador:

1. **`backbone-only` (preditores/projeção pessoais por silo)** — análogo do
   dilema do preditor do FedU/FedEMA, **nunca estudado para preditores de
   alvos aleatórios (LFR)**. É a mais barata e a mais "de pesquisa".
2. **FedBN-SSL para séries temporais** — FedBN nunca avaliado em pré-treino
   SSL de sensores; nosso non-IID é de features (o caso dele). Promover se o
   efeito for grande.
3. **Agregação adaptativa via `agg_shock`** — usar o choque de agregação
   (já logado) para modular frequência/peso. É "método" de verdade, custo de
   dias. Só depois do Exp. 3 base fechado.

Recomendo levar **(1)** como proposta principal e **(2)** como reserva.

---

## 5. Perguntas para levar à reunião (decisões que são do orientador)

1. Para você, a contribuição empírica controlada **basta** para o mestrado, ou
   você quer necessariamente um componente de método nomeado? (Se sim, §4-(1)
   ou §4-(2)?)
2. Escopo: fecho no **Recomendado** (§3) ou miro no **Stretch**?
3. Vale reproduzir **1 baseline FedSSL da literatura** (SimCLR-TS) para
   antecipar o reviewer "por que não comparou com FedSSL existente?" — ou
   citar e delimitar basta?
4. A narrativa "ponte entre as duas linhas do HIAAC" pode ser explícita no
   texto (intro/acknowledgments), ou melhor manter implícita?
5. Alvo de publicação (define formato/página e o nível de novidade exigido):
   conferência de FL? de HAR/ubicomp? venue do grupo?

---

## 6. Resumo de uma linha para cada interlocutor

- **Para você**: você tem uma contribuição empírica sólida e verificada; a
  única fraqueza real e barata de corrigir é "falta um método com nome" —
  promova a variante `backbone-only`.
- **Para o orientador**: "não estou federando uma técnica; estou fazendo o
  primeiro estudo controlado de topologia de pré-treino SSL sob domain shift
  real em HAR federado, com uma variante de agregação própria e comunicação
  medida — e já verifiquei em 4 grafos de citação que LFR/TF-C federados são
  inéditos."
- **Para o reviewer (no paper)**: citar FedST/FedOST e UniHAR proativamente,
  declarar o corte de escopo (1 modalidade, 6 silos, FedAvg fixo, simulação),
  e enquadrar o Exp. 3 como pergunta de dois desfechos informativos.
