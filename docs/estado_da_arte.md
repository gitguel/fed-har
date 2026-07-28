# Estado da arte, ineditismo e posicionamento da contribuição

*Base: `_arquivo/estado_da_arte_fssl_e_contribuicoes.md` (2026-07-07), com
`_arquivo/contribuicoes_forte_fraco_defesa.md` (2026-07-07 → §7) e o
posicionamento do piso de batch (`_arquivo/limite_batch_cliente_fssl.md`,
2026-07-24 → §3.4) absorvidos em 2026-07-27.*

*O levantamento de 2026-07-07 saiu de ~12 buscas web estruturadas. **Limitação
metodológica honesta**: busca com snippets/abstracts, não leitura integral;
cobertura de arXiv/IEEE/ACM/Springer boa, mas não exaustiva (§6.1 lista como
fechar). Nenhuma afirmação de "primeiro" deve ir ao artigo sem essa verificação
final. Três PDFs foram lidos integralmente depois (§3.4).*

> **Efeito do pivô cross-silo → cross-device (2026-07-21)** neste documento: a
> **verificação de ineditismo** de LFR/TF-C federados (§2, §5) permanece válida —
> ela é sobre os métodos, não sobre a topologia. O que muda é o **enunciado da
> contribuição** (§4.2), reescrito para o eixo cross-device, e a fraqueza W6
> (§7.2), que o pivô praticamente responde. Ver `plano_fedssl.md` e
> `resultados.md §4`.

---

## 1. Correção de posicionamento: você é da linha de aprendizado distribuído do HIAAC

Isso **melhora** a narrativa, não a enfraquece. A seção II-E do esqueleto
dizia que o hub tem duas linhas disjuntas (a linha DAGHAR/benchmark de
Napoli–da Luz–Borin–Boccato e a linha de FL de Cerqueira–Villas–Rosário) e que
ninguém as conecta. Com você **dentro** da linha distribuída, a frase certa
passa a ser: *este trabalho é a ponte institucional entre as duas linhas do
próprio hub* — a linha distribuída fornecendo a expertise de FL, a linha de
benchmark fornecendo dados/protocolos/baselines validados. Isso:

- explica a escolha "conservadora" de FedAvg/Flower (é o vocabulário da sua
  linha, não falta de ambição);
- abre coautoria/revisão interna natural dos dois lados;
- e responde parcialmente à provocação do orientador (§5): a contribuição
  não é "federar uma técnica", é **trazer o problema de domain shift real da
  linha de benchmark para dentro da agenda da linha distribuída**, com a
  primeira evidência quantitativa do hub sobre isso.

Sugestão prática: valide com o orientador se a frase de posicionamento do
esqueleto (fim da II-E) pode citar explicitamente as duas linhas do HIAAC —
em texto de artigo isso vira uma frase de "research context" no
acknowledgments ou na introdução, não na related work.

## 2. Verificação do claim C3 ("primeiro FedAvg-SSL com técnicas de SSL de séries temporais")

**Veredito: a formulação atual é forte demais e arriscada. Deve ser
reformulada.** O que as buscas estabeleceram:

### 2.1 O que NÃO foi encontrado (sustenta a parte defensável do claim)

- **Nenhum trabalho federando TF-C**: buscas por "federated" + "time-frequency
  consistency"/"TF-C pretraining" não retornam nada federado — só o paper
  original e derivados centralizados.
- **Nenhum trabalho federando LFR** (nem, mais amplamente, qualquer método
  SSL *augmentation-free* de reconstrução de projeções aleatórias): buscas por
  "federated" + "random data projectors"/"learning from randomness" não
  retornam nada federado.
- **Nenhuma comparação controlada "pré-treino centralizado vs federado do
  MESMO método SSL de séries temporais sob domain shift real com budget
  pareado"** — o desenho do nosso Exp. 1 vs Exp. 3.

### 2.2 O que FOI encontrado (mata a formulação literal do C3)

FedSSL sobre **séries temporais/sensores** existe desde 2021 e está ativo:

| Trabalho | O que federou | Por que não invalida nosso trabalho (mas invalida o "primeiro FedAvg-SSL de séries temporais") |
|---|---|---|
| **Saeed et al., IEEE IoT-J 2021** | Scalogram-signal correspondence (wavelet) sobre EEG/BVP/acelerômetro/WiFi-CSI | Método próprio ad-hoc, 1 técnica, sem domain shift entre silos, sem comparação centralizado-vs-federado sistemática |
| **UniHAR** [Xu et al., MobiCom 2023](https://dl.acm.org/doi/10.1145/3570361.3613299) — **o competidor mais próximo** | Pré-treino contrastivo com aumentações físicas de IMU + FL + treino adversarial, 4 datasets HAR heterogêneos, protótipo mobile | Objetivo é *sistema/deployment* (adoção prática, overhead em celular); SSL próprio baseado em aumentações físicas; não compara técnicas SSL publicadas entre si nem topologias de pré-treino sob budget controlado; não reporta custo de comunicação como métrica de decisão |
| **FedCSSL** [Springer 2026](https://link.springer.com/chapter/10.1007/978-3-032-16995-2_15) | SimCLR federado p/ HAR em smart homes (sensores ambientais, não IMU) | SimCLR, 1 técnica, personalização |
| **CDFL** [arXiv 2407.12287](https://arxiv.org/pdf/2407.12287) | Contrastive + deep clustering p/ HAR federado eficiente | Método próprio, foco eficiência |
| **SSF-HAR** [ICME 2024](https://www.computer.org/csdl/proceedings-article/icme/2024/10687970/20EZTZiSCys) | SSL federado personalizado p/ HAR | Personalização, método próprio |
| **Delta-loss client clustering** [Digital Signal Processing, set/2026] | FedSSL p/ HAR com clusterização de clientes por delta-loss | Eficiência/clustering, não domain shift controlado |
| **FedSC** [arXiv 2405.03949](https://arxiv.org/pdf/2405.03949) | FedSSL com objetivo spectral contrastive + garantias teóricas | Teoria, visão |
| **Fed. time-series foundation models** ([FeDaL](https://arxiv.org/html/2508.04045v1), [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33739/35894), [bi-level](https://arxiv.org/pdf/2604.06727)) | Pré-treino federado de TSFMs (masked reconstruction etc.) sobre séries heterogêneas | Forecasting-oriented, escala FM, não é o regime cross-silo HAR com técnicas do benchmark |

### 2.3 Reformulações recomendadas (da mais segura à mais forte)

1. **Segura (recomendada para a dissertação)**: "Até onde sabemos, somos os
   primeiros a federar **LFR e TF-C** — duas técnicas de SSL de séries
   temporais com protocolo publicado e benchmark centralizado de referência —
   e os primeiros a comparar, sob budget de computação pareado e domain shift
   real cross-silo, o pré-treino **centralizado vs federado** dessas mesmas
   técnicas."
2. **Média**: adicionar "...diferentemente da literatura de FedSSL para
   sensores, que federa métodos contrastivos próprios baseados em aumentações
   (UniHAR, FedCSSL, CDFL), avaliando uma técnica única sem comparação entre
   famílias de SSL nem entre topologias de pré-treino."
3. **A evitar**: qualquer variante de "primeiro FedAvg-SSL com séries
   temporais" (Saeed 2021 e UniHAR falsificam) e "primeiro FedSSL para HAR"
   (meia dúzia de contra-exemplos acima).

A força do nosso claim não está no "primeiro a federar", e sim no **desenho
comparativo controlado** — nenhum dos trabalhos acima tem: 2 famílias de SSL
(uma contrastiva, uma augmentation-free) × 4 encoders × 2 topologias de
pré-treino × cenários non-IID/IID pareados em volume × custo de comunicação ×
validação prévia contra benchmark publicado. Isso é literalmente único no que
foi mapeado, e é falsificável célula a célula (ao contrário de um "primeiro").

## 3. Estado da arte de F-SSL (levantamento)

### 3.1 Linha fundacional (visão computacional, 2020–2023)

O núcleo metodológico do campo, de onde vêm os conceitos que reusamos:

- **FedCA** [Zhang et al., 2020/2023] — primeiro framework de federated
  unsupervised representation learning; identifica *inconsistency of
  representation spaces* entre clientes e alinha com dicionário compartilhado.
- **FedU / FedEMA** [Zhuang et al., ICCV 2021; ICLR 2022,
  [arXiv 2204.04385](https://arxiv.org/abs/2204.04385)] — framework geral
  para FedSSL com redes siamesas (BYOL-like); regras *divergence-aware*:
  agregar só o encoder online, atualização EMA adaptativa. Achados que usamos
  no design do Exp. 3: stop-gradient nem sempre necessário; **reter
  conhecimento local ajuda em non-IID** (inspira nossa variante
  `backbone-only` p/ preditores LFR).
- **Orchestra** [Lubana et al., ICML 2022] — clusterização global-local
  consistente; robusto a non-IID por construção.
- **FedX** [Han et al., ECCV 2022] — destilação cruzada local/global sem
  exigir compartilhamento adicional.
- **L-DAWA** [Rehman et al., ICCV 2023,
  [arXiv 2307.07393](https://arxiv.org/pdf/2307.07393)] — agregação ponderada
  por divergência **camada a camada** (refina FedAvg/FedEMA).
- **SSFL** [He et al., 2021] — personalização + self-supervision p/ escassez
  de rótulo.
- **FedMoCo/FedCo** [2021–2022] — MoCo federado (memória de negativos).
- **FedMAE** [[arXiv 2303.11339](https://arxiv.org/abs/2303.11339)] e a linha
  de **masked modeling federado** — MAE de um bloco em clientes leves,
  cascateado no servidor. Relevante: [Yan et al., arXiv 2205.08576
  ](https://arxiv.org/pdf/2205.08576) (medical imaging) reporta que **masked
  modeling é mais robusto a heterogeneidade em FL que métodos contrastivos** —
  um achado que dialoga diretamente com nossa hipótese LFR (sem negativos,
  sem aumentações) vs TF-C (contrastivo).
- **FedSC** [[arXiv 2405.03949](https://arxiv.org/pdf/2405.03949)] — primeira
  análise com **garantias prováveis** (spectral contrastive) em non-IID.
- **Teoria/fenômenos**: colapso completo/dimensional é **mais grave em FedSSL
  que em FL supervisionado** ([CVPR 2024, Rethinking the Representation
  ](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Rethinking_the_Representation_in_Federated_Unsupervised_Learning_with_Non-IID_Data_CVPR_2024_paper.pdf));
  FedAvg ponderado por tamanho ignora qualidade e induz drift/unfairness.
- **Eficiência**: LW-FedSSL [[arXiv 2401.11647](https://arxiv.org/html/2401.11647v2)]
  (layer-wise, reduz computação/comunicação) — a comunicação como restrição
  já é tema, mas como *técnica*, não como *métrica de comparação entre
  topologias de pré-treino* (nosso ângulo).

### 3.2 Séries temporais e sensores (2021–2026)

- **HAR/vestíveis**: Saeed 2021 (pioneiro), UniHAR (MobiCom 2023, o mais
  forte), FedCSSL, CDFL, SSF-HAR, delta-loss clustering (DSP 2026) — detalhes
  na tabela da §3.2. Padrão comum: **método SSL próprio ou SimCLR adaptado,
  uma técnica só, foco em personalização/eficiência/deployment**, avaliação
  em poucos datasets sem controle de domain shift, sem custo de comunicação
  como métrica de primeira classe.
- **Médico (ECG/EEG)**: FL supervisionado para arritmia
  [[arXiv 2208.10993](https://arxiv.org/pdf/2208.10993)]; FedMCF-xLSTM
  (contrastive multimodal ECG); PCRFed (contrastive p/ segmentação não-IID).
  SSL forte em ECG/EEG centralizado (BENDR etc.), mas a combinação
  FedSSL-para-séries-médicas ainda é rala — mais um indício de lacuna geral.
- **Foundation models de séries temporais federados (2025–2026, quente)**:
  FeDaL, "Federated Foundation Models on Heterogeneous Time Series" (AAAI
  2025), bi-level heterogeneous learning, discrete prototypical memories.
  Direção: pré-treino federado de TSFMs genéricos (masked reconstruction),
  foco em heterogeneidade de *tokens/domínios de aplicação*. É a fronteira
  que mais se aproxima do nosso tema por cima (escala), mas não faz avaliação
  cross-silo de HAR nem compara técnicas SSL discriminativas.
- **Dado empírico útil para nossas hipóteses**: no pré-treino federado de
  BERT clínico [[arXiv 2002.08562](https://arxiv.org/pdf/2002.08562)],
  *pré-treino federado + finetuning centralizado* perde **<5%** vs tudo
  centralizado — sugere que o pré-treino tolera bem a federação; nosso gate 4
  (equivalência IID) e o Exp. 3 testam exatamente isso para LFR/TF-C.

### 3.3 Principais lacunas na literatura (síntese)

1. **Vício de visão**: a metodologia canônica (FedU/FedEMA/Orchestra/L-DAWA)
   foi desenvolvida e avaliada em CIFAR/ImageNet com non-IID **simulado por
   label skew**. Quase nada foi re-testado sob **feature/domain shift real**.
2. **Técnica única, sem comparação entre famílias**: cada paper federa UM
   método (geralmente contrastivo com aumentações). Não há estudo comparando
   contrastivo vs augmentation-free vs masked sob a MESMA federação — apesar
   do indício (Yan et al.) de que a família importa muito em non-IID.
3. **LFR e TF-C seguem inexplorados no contexto federado** (§3.1) — e, por
   extensão, a pergunta "métodos sem aumentações são mais ou menos robustos à
   agregação?" está aberta.
4. **Falta a comparação de topologias**: centralizado vs federado do mesmo
   método com budget pareado praticamente não existe (exceção parcial: BERT
   clínico, NLP). É o coração do nosso Exp. 1 vs 2 vs 3.
5. **Custo de comunicação do PRÉ-TREINO raramente reportado** — a literatura
   reporta acurácia; quando toca em comunicação, é como técnica de redução
   (layer-wise), não como eixo de decisão entre estratégias.
6. **Reprodutibilidade fraca**: métodos próprios sem benchmark centralizado
   de referência contra o qual validar a implementação. Nossa réplica
   validada (viés ±2 pp vs da Luz et al.) é um diferencial metodológico real.

### 3.4 O piso de batch: como a literatura escapa dele (verificado nos PDFs, 2026-07-24)

*Posicionamento do achado descrito em `plano_fedssl.md §3` — o cliente mínimo
viável em FSSL é um **batch**, não uma amostra.*

A comunidade de F-SSL conhece o problema, mas quase sempre o enquadra como
**degradação de qualidade** sob não-IID, não como **piso de viabilidade** (o
cliente não consegue nem um passo de gradiente). As linhas de ataque:

1. **Diagnóstico teórico — o objetivo global do FSSL ≠ soma dos locais.** Com
   objetivo contrastivo, cada amostra só contrasta contra os negativos do próprio
   cliente; sob não-IID isso enviesa o FedAvg. É a motivação explícita de FedSC e
   da família FedU/FedEMA.
2. **Compartilhar estatísticas para recompor negativos entre clientes.** **FedSC**
   (ICML 2024) troca **matrizes de correlação das representações** além dos pesos,
   habilitando *inter-client contrast*. Único método com garantia; aplica DP às
   estatísticas — ao custo de mais comunicação e uma superfície de privacidade nova.
3. **Trocar de objetivo para um que não dependa de negativos/batch.** **FedEMA /
   Divergence-aware FedSSL** (ICLR 2022) constrói sobre **BYOL** exatamente para
   evitar a dependência de negativos.
4. **Reformular como clustering global.** **Orchestra** (ICML 2022), projetado
   para ser robusto a variação de nº de clientes e participação — o regime
   cross-device com muitos clientes pequenos.
5. **HAR especificamente.** **UniHAR** (MobiCom 2023) usa LIMU-BERT (reconstrução
   mascarada, por-amostra); **Saeed et al.** (IoT-J 2021) usam correspondência
   scalograma-sinal (classificação binária). Ambos assumem clientes com dado
   suficiente e **nenhum isola o piso de batch por cliente**.

**O que cada vizinho realmente pré-treina** (lido no PDF):

| Paper | Objetivo SSL | "Batch-hungry"? | Como escapa do piso | Datasets |
|---|---|---|---|---|
| **UniHAR** (MobiCom'23) | LIMU-BERT = **reconstrução mascarada** | Não (perda por-amostra) | Por construção do objetivo | UCI, HHAR, MotionSense, Shoaib |
| **Saeed** (IoT-J'21) | Scalogram-signal = **classificação binária** | Quase não | Objetivo + **shards IID**; batch federado **12** | Sleep-EDF, HHAR, MobiAct, WiFi-CSI, WESAD |
| **FedSC** (ICML'24) | **Spectral Contrastive** | **Sim** | **Batch 512/256** + dado local enorme + matrizes de correlação | SVHN, CIFAR-10/100 |

**A literatura evita o piso de batch por construção, não por acaso:**
- **FedSC** roda **cross-silo com dado local enorme e batch 512** — a prova de
  convergência tem termo de erro que só some com "large batch size B". Resolve o
  skew de rótulo, não o batch pequeno; assume o oposto do KuHar.
- **Saeed** é quase uma confissão do ponto: **não particiona por usuário** — cita
  textualmente *"we randomly divide the training set into multiple subsets…
  due to fewer users in existing datasets. This choice might result in a
  decentralized IID dataset… does not suffer from extreme heterogeneity."*
- **UniHAR** usa reconstrução mascarada (por-amostra); o piso nem se coloca.

**Lacuna (candidata a contribuição, a afirmar com cautela):** a literatura ou
(a) troca para objetivo por-amostra/binário/BYOL, ou (b) foge do cross-device
real com shards IID e batch grande. **Ninguém roda um objetivo contrastivo
batch-hungry (SimCLR/NT-Xent/BBT) sobre partição real por usuário com clientes
minúsculos** — exatamente onde o nosso TF-C + LFR/BBT sobre KuHar-por-usuário cai.
O achado não é "a qualidade degrada" (isso já se sabe), e sim um **piso de
viabilidade**: com objetivo definido sobre o batch, um participante real com
< 1 batch **não produz nenhum gradiente** e falha em silêncio no FedAvg.
Enquadrar como *critério de elegibilidade de cliente* e **reportar a taxa de
exclusão** (KuHar: 48/57 = 84% dos usuários, 48% das janelas).

> **Conferência de ineditismo (2026-07-24).** Os três itens antes bloqueados
> foram lidos: **B1** (da Luz, benchmark IEEE Access'26) é **centralizado** — zero
> conteúdo federado/piso de batch; **B5** (survey de SSL-HAR, **Logacjov**
> IMWUT'24) é survey **centralizado**, "federated" aparece 1× só em referência;
> **C8** (FedST/FedOST, ACM MM'24) é FL **supervisionado** (pFL) de classificação
> de séries temporais, batch 128 fixo, sem discutir piso de batch. **Nenhum dos
> três antecipa o achado.** Ressalva permanente: nenhuma varredura é exaustiva.

## 4. Sobre a provocação do orientador

> "Apenas federar uma técnica (LFR por exemplo) não é contribuição o bastante
> para você defender seu mestrado..."

**Ele tem razão na letra — e o seu projeto já não é isso.** Vale separar as
duas coisas com frieza:

### 4.1 O que seria de fato insuficiente

"Peguei o LFR, coloquei no Flower, rodei, acurácia X." Isso é engenharia de
integração: sem pergunta científica, sem desenho experimental, sem
falsificabilidade. Se a dissertação fosse isso, a crítica procederia.

### 4.2 O que o projeto já é (e como enunciá-lo)

A contribuição de uma dissertação empírica não é o verbo "federar" — é a
**pergunta + o desenho experimental que a responde de forma confiável**:

1. **Pergunta científica clara e aberta** (§3.3, lacunas 1–4): *o pré-treino
   SSL compra de volta a acurácia que a heterogeneidade custa a uma federação
   **cross-device** — e isso sobrevive quando o próprio pré-treino é federado?*
   Nenhum trabalho mapeado responde isso. *(Enunciado atualizado em 2026-07-27:
   a versão de julho dizia "cross-silo".)*
2. **Desenho experimental que ninguém tem**: heterogeneidade real por construção
   (não Dirichlet) **com controle pareado para cada eixo** — domain shift,
   feature skew e label skew isolados cada um por um Δ contra o seu próprio
   controle (`plano_fedssl.md §2`), o que permite atribuição causal, coisa que
   UniHAR/FedCSSL/CDFL não fazem; 2 famílias de SSL × 4 encoders × 2 topologias
   × budget pareado; validação prévia da implementação contra benchmark
   publicado.
3. **As decisões de design do FedSSL-LFR são questões de pesquisa, não
   plumbing** — e o design doc já as formula como ablações: (i) seleção DPP
   global vs local (coerência de alvos entre clientes — problema que NÃO
   existe em SimCLR federado, é específico da família augmentation-free);
   (ii) política de agregação de preditores (`full` vs `backbone-only` — o
   análogo do dilema do preditor no FedU/FedEMA, nunca estudado para
   preditores-de-alvos-aleatórios); (iii) interação alternância×rodada. Cada
   uma com resultado próprio.
4. **Métrica que a área negligencia**: comunicação de ponta a ponta como eixo
   de decisão (com o achado já antecipável de que LFR federado é caro em
   uplink e linear-readout federado é quase grátis).
5. **Resultado garantido nos dois desfechos**: se FedAvg-SSL funcionar, é a
   primeira evidência positiva para LFR/TF-C federados; se divergir, é a
   caracterização quantificada (curvas de `agg_shock`, efeito de FedBN/freq.
   de agregação) de POR QUE métodos SSL de séries temporais divergem sob
   domain shift — igualmente publicável (cf. achados negativos influentes na
   literatura de FedSSL, como os do próprio FedEMA sobre stop-gradient).

**Enunciado-síntese para usar com o orientador** *(atualizado para cross-device
em 2026-07-27)*: "A contribuição não é federar o LFR; é o primeiro estudo
controlado de *quando e por quê* o pré-treino auto-supervisionado — centralizado
ou federado, contrastivo ou livre de aumentações — mitiga heterogeneidade real em
HAR **cross-device (clientes = usuários)**, separando domain shift de skew
inter-pessoa, com custo de comunicação na conta."

### 4.3 Se ainda assim quiser "mais método" (opções de baixo risco, em ordem de custo)

Todas encaixam no design doc do Exp. 3 sem retrabalho:

1. **Nomear e formalizar a variante `backbone-only` do LFR federado** como
   estratégia (preditores personalizados por silo — análogo FedU para
   augmentation-free). Custo: zero além da ablação já planejada.
2. **FedBN-SSL para séries temporais**: FedBN nunca foi avaliado em
   pré-treino SSL de sensores; a variante `fedbn` já está no plano — basta
   promovê-la de ablação a contribuição se o efeito for grande.
3. **Agregação adaptativa via `agg_shock`**: usar a métrica de choque de
   agregação (já logada) para modular a frequência de agregação ou o peso do
   cliente — um "mini-L-DAWA para séries temporais". Custo: dias, não semanas.
   Só atacar depois do Exp. 3 base fechado.
4. (Mais caro, só se sobrar tempo) Comparar com **FedEMA adaptado ao TF-C** —
   posiciona contra a linha divergence-aware.

## 5. Snowballing reverso de TF-C e LFR (executado em 2026-07-07)

### 5.1 O que é e como foi feito

**Snowballing** é a técnica de expansão de bibliografia a partir de um paper-
semente. Tem duas direções: o **backward** (para trás) percorre as
*referências* do semente — o passado dele; o **forward/reverso** (para frente)
percorre todos os papers que *citam* o semente — o futuro dele. Para
**verificar ineditismo**, o reverso é o teste mais forte que existe, porque
vale a premissa: *se alguém tivesse federado o TF-C ou o LFR, esse trabalho
quase certamente citaria o paper original do método*. Então, em vez de torcer
para uma busca por palavra-chave acertar os termos, varremos **todo** o
conjunto "citado por" das duas sementes e filtramos por federação.

Execução (script descartável, não versionado): API de grafo de citações do
**Semantic Scholar**, campos título+abstract+ano+venue, filtro
`federat|decentraliz` sobre título+abstract. Cobertura: TF-C = **499 papers
citantes**; LFR = **19 citantes**. Limitação: o índice do S2 não é exaustivo
(pode faltar workshop/preprint muito recente) e o filtro pega só título+
abstract — por isso a §6.3 mantém a checagem final no Scholar.

### 5.2 Resultado — a contribuição se sustenta (com uma ressalva a citar)

**Nenhum dos 518 papers citantes federa TF-C ou LFR como método de pré-treino
SSL.** Detalhe dos "quase":

- **LFR (19 citantes, 2 com "federated")**: os 2 são o **mesmo** trabalho
  (Towards Active Participant-Centric Vertical FL, arXiv:2410.17648 / KBS
  2026) que apenas *menciona* LFR como uma representação possível em
  **vertical FL** — não federa LFR, não é HAR, não é séries temporais. **LFR
  no contexto federado horizontal está literalmente intocado.**
- **TF-C (499 citantes, 6 com "federated")**, nenhum federa TF-C, mas dois
  são **vizinhos próximos que PRECISAM ser citados e distinguidos**:
  - ⚠️ **FedST** [ACM Multimedia 2024] e sua extensão **FedOST** [IEEE Trans.
    Mobile Computing 2026] — *personalized FL* para classificação de séries
    temporais que **usa visões de tempo E frequência** (mutual information +
    projeção ortogonal para desacoplar features compartilhadas/pessoais entre
    clientes). É o trabalho mais perto do nosso no eixo "tempo-frequência em
    FL". **Diferenças a explicitar**: (i) é *supervisionado* (classificação
    com rótulos), não pré-treino SSL; (ii) o objetivo é personalização
    (pFL), não mitigar domain shift via representação transferível; (iii) usa
    a *ideia* de tempo-frequência como regularização, não o método TF-C nem
    seu protocolo; (iv) sem comparação centralizado-vs-federado nem custo de
    comunicação. **Ação**: virar parágrafo dedicado na related work — é a
    citação que um reviewer atento cobraria.
  - **Time-FFM** [NeurIPS 2024] — foundation model federado para *forecasting*
    via LMs; regime e tarefa diferentes (já coberto pela família TSFM da §4.2).
  - **Rethinking the Starting Point / CoPreFL** [AAAI 2024, arXiv:2402.02225]
    — pré-treino *centralizado* colaborativo para inicializar FL downstream.
    **Diretamente relevante ao enquadramento do nosso Exp. 2** (init
    centralizada para FL); citar como fundamento da estratégia, não como
    concorrente (não é SSL de séries temporais nem HAR).
  - Os outros 3 (seizure detection FFSL, orthogonal reg., medical TS
    transformer) citam TF-C só como baseline/inspiração de séries temporais.

**Conclusão para o claim**: a formulação calibrada da §3.3 (item 1) está
**confirmada** — somos os primeiros a federar LFR e TF-C especificamente, e os
primeiros a comparar pré-treino centralizado vs federado dessas técnicas sob
domain shift real. A única correção é acrescentar FedST/FedOST à related work
e marcar a fronteira ("tempo-frequência já apareceu em pFL supervisionado; o
pré-treino SSL tempo-frequência federado, não").

### 5.3 Verificação cruzada: OpenAlex (2º grafo) + queries estilo Scholar

O Google Scholar **não tem API e bloqueia acesso automatizado** (403/CAPTCHA),
então não dá para raspar a UI dele daqui. Para cumprir o *objetivo* da etapa —
um segundo grafo de citações independente — usei o **OpenAlex** (aberto,
consultável) e rodei as queries no estilo Scholar via busca web. Resultados:

- **OpenAlex, citações de TF-C**: 128 citantes indexados, **0 com
  federat/decentrali**. (Correção honesta ao que eu havia dito antes: para
  *estes* dois papers o OpenAlex indexou **menos** citações que o Semantic
  Scholar — 128 vs 499 — não mais; o valor é ser um grafo *independente* que,
  ainda assim, não revelou nenhum TF-C federado novo.)
- **OpenAlex, citações de LFR**: 3 citantes indexados, **0 federados** (o LFR
  é de 2023/2024 e ainda é pouquíssimo citado — S2 tinha 19, OpenAlex 3;
  ambos sem federação). ⚠️ *Cuidado de implementação registrado*: a busca de
  título do OpenAlex rankeou "BYOL" acima do LFR; foi preciso fixar o work-id
  correto (`W4387634839`). O primeiro passe pegou as ~3.4k citações do BYOL
  por engano.
- **Bycatch útil (as 62 citações federadas do BYOL)**: um bom corpus de FedSSL
  — FedU, FedX, L-DAWA, SSFL, Ensemble Similarity Distillation, "Federated
  Representation Learning Through Clustering", "A Deep Cut Into Split FedSSL",
  "Global representation fine-tuning for FedSSL" (2025) etc. **Todos de visão
  ou imagem médica; nenhum é séries temporais/HAR e nenhum federa TF-C ou
  LFR.** Reforça o mapeamento da §4.
- **Queries estilo Scholar (busca web)**: `"federated" "time-frequency
  consistency"` → nada federado; `"federated" "learning from randomness"/
  "random data projectors"` → aparece **FedRP / FedSLoP / random-projection
  pFL**. ⚠️ **Colisão de terminologia a conhecer** (não invalida nada): esses
  usam projeção aleatória para *compressão de gradiente/eficiência de
  comunicação*, **não** como método SSL à la LFR — não citam nem usam o LFR.
  Vale uma frase preventiva no texto para o reviewer não confundir "federated
  random projection" (comunicação) com "federated LFR" (SSL).

**Veredito final — ineditismo triplamente checado** (busca web + Semantic
Scholar snowball + OpenAlex snowball + queries Scholar-style): nenhum trabalho
federa LFR ou TF-C como pré-treino SSL. O que resta é a checagem manual na UI
do Scholar na semana da submissão (§7.1), formalidade de baixo risco.

## 6. Próximos passos, estruturados

### 6.1 Ações de verificação (fechar antes de submeter) — ordem de execução

1. **Confirmação manual na UI do Google Scholar** (única peça que falta — não
   dá para automatizar; S2 e OpenAlex já feitos na §6): abrir "cited by" de
   TF-C e LFR e usar "search within citing articles" com `federated`; buscar
   `"federated" "time-frequency consistency"` e `"federated" "random data
   projectors"`. É formalidade de baixo risco — o ineditismo já está
   triplamente checado (busca web + S2 + OpenAlex). Fazer na semana da
   submissão (a área publica AGORA).
   ⚠️ **Não confundir** com FedRP/FedSLoP/random-projection-pFL: usam projeção
   aleatória para *eficiência de comunicação*, não como SSL — deixar uma frase
   preventiva no texto.
2. **Alertas do Scholar** (a área publica AGORA — delta-loss é de set/2026):
   "federated self-supervised human activity recognition"; "federated
   contrastive time series"; "federated pretraining wearable". Revalidar na
   semana da submissão.
3. **Ler FedST/FedOST na íntegra** e redigir o parágrafo de distinção (§6.2)
   — é o item de maior risco de reviewer.

### 6.2 Lista de leitura priorizada

Critério de corte: cada item ou (a) o reviewer vai cobrar, ou (b) muda uma
decisão de design/enquadramento nossa. ~22 papers em 3 blocos, em 3 ondas.
**Onda 1** = ler antes de escrever a related work (essencial, ~8);
**Onda 2** = ler antes de submeter (contexto, ~9); **Onda 3** = consultar se
o tempo permitir / para blindar contra reviewer (~5). Muitos já foram lidos em
abstract nesta análise — "ler" aqui = leitura integral com fichamento.

#### Bloco A — Federated Learning (base)

| # | Paper | Onda | Por que |
|---|---|---|---|
| A1 | McMahan et al., *Communication-Efficient Learning… (FedAvg)*, AISTATS 2017 | 1 | agregador que usamos; ler a seção de non-IID original |
| A2 | Li et al., *FedBN*, ICLR 2021 | 1 | feature-shift non-IID = nosso caso exato; base da variante `fedbn` |
| A3 | Li et al., *FedProx*, MLSys 2020 | 2 | baseline de heterogeneidade citado por reflexo; delimitar por que não usamos |
| A4 | *Rethinking the Starting Point / CoPreFL*, AAAI 2024 (arXiv:2402.02225) | 2 | fundamenta o Exp. 2 (pré-treino centralizado → init FL); achado do snowballing |
| A5 | Survey *Foundational Models + FL*, PeerJ CS 2025 | 3 | taxonomia p/ situar o trabalho; varredura de cobertura |

#### Bloco B — Self-Supervised Learning (os métodos e o benchmark)

| # | Paper | Onda | Por que |
|---|---|---|---|
| B1 | da Luz et al., *Benchmarking Encoders and SSL for HAR*, IEEE Access 2026 | 1 | nosso protocolo e baselines saem daqui; ler integralmente é obrigatório |
| B2 | Zhang et al., *TF-C*, NeurIPS 2022 | 1 | uma das 2 técnicas; entender a loss NT-Xent e o ramo de frequência |
| B3 | Sui et al., *LFR*, ICLR 2024 | 1 | a outra técnica; DPP, projetores/preditores aleatórios |
| B4 | Napoli et al., *DAGHAR*, Scientific Data 2024 | 1 | o dataset; splits, padronização, baselines de domínio |
| B5 | **Logacjov**, *SSL for Accelerometer-based HAR: A Survey*, IMWUT 8(4):149, 2024 | 2 | mapa da SSL-HAR; taxonomia p/ related work. ⚠️ autoria corrigida em 2026-07-24 (**não** é Haresamudram) |
| B6 | Rodrigues da Silva et al., *Impact of Pre-training Datasets (CPC)*, BRACIS 2024 | 2 | trabalho do grupo; origem do desenho comb→target |
| B7 | Eldele et al., *TS-TCC*, IJCAI 2021 | 3 | encoder tstcc vem daqui; citar corretamente |
| B8 | Yue et al., *TS2Vec*, AAAI 2022 | 3 | citado no benchmark; melhor teto @100% — contexto |

#### Bloco C — Federated Self-Supervised Learning (o coração)

*C-i. Fundamentos metodológicos (visão) — de onde vêm os conceitos de design:*

| # | Paper | Onda | Por que |
|---|---|---|---|
| C1 | Zhuang et al., *Divergence-aware FedSSL (FedU/FedEMA)*, ICLR 2022 | 1 | framework siamês federado; dilema do preditor = nosso `backbone-only`; achado "reter local ajuda" |
| C2 | Rehman et al., *L-DAWA*, ICCV 2023 | 2 | agregação por divergência camada-a-camada; inspira `agg_shock` adaptativo |
| C3 | Lubana et al., *Orchestra*, ICML 2022 | 2 | clustering global-local; alternativa de agregação robusta a non-IID |
| C4 | Liao et al., *Rethinking the Representation in Federated Unsup. Learning*, CVPR 2024 | 2 | colapso é pior em FedSSL; risco central do Exp. 3 |
| C5 | Han et al., *FedX*, ECCV 2022 | 3 | destilação cruzada; alternativa de design |
| C6 | *FedSC (spectral contrastive, provable)*, ICML 2024 (arXiv:2405.03949) | 3 | única com garantia teórica em non-IID; blindar discussão |

*C-ii. Séries temporais / sensores / HAR federado — os vizinhos diretos:*

| # | Paper | Onda | Por que |
|---|---|---|---|
| C7 | Xu et al., *UniHAR*, MobiCom 2023 | 1 | **competidor mais próximo**; SSL+FL em HAR; reviewer VAI cobrar |
| C8 | *FedST*, ACM MM 2024 + *FedOST*, IEEE TMC 2026 | 1 | **tempo-frequência em FL** (achado do snowballing); distinguir com cuidado |
| C9 | Saeed et al., *FedSSL of Multisensor Representations*, IEEE IoT-J 2021 | 2 | pioneiro FedSSL-sensores; falsifica "primeiro"; citar como marco |
| C10 | Yan et al., *Label-Efficient SSFL (masked vs contrastive)*, IEEE TMI 2023 | 2 | evidência de que família importa em FL — nossa hipótese LFR vs TF-C |
| C11 | *FedCSSL* (smart homes, SimCLR), Springer 2026 | 3 | vizinho HAR; delimitar (sensores ambientais, 1 técnica) |
| C12 | *CDFL* (contrastive+clustering), arXiv 2407.12287 | 3 | vizinho HAR eficiência; delimitar |

*C-iii. Fronteira por cima (TSFM federado) — 1 leitura de contexto:*

| # | Paper | Onda | Por que |
|---|---|---|---|
| C13 | *Federated Foundation Models on Heterogeneous Time Series*, AAAI 2025 (ou Time-FFM, NeurIPS 2024) | 3 | delimita escopo (forecasting/FM ≠ nosso regime); mostra que você conhece a direção |

**Resumo de esforço**: Onda 1 = 8 papers (A1–A2, B1–B4, C1, C7–C8) — o mínimo
para escrever a related work com autoridade. Onda 2 = +9. Onda 3 = +5
(consulta/blindagem). Se o tempo for curtíssimo, os **inegociáveis** são
B1–B4 (seu próprio protocolo/dados), C1 (design do Exp. 3), C7 e C8 (os dois
que o reviewer vai cobrar).

### 6.3 Para fortalecer a pesquisa (além da leitura)

- **Nomear a hipótese "augmentation-free (LFR) é mais/menos robusto à
  agregação que contrastivo (TF-C)"** como pergunta de pesquisa explícita —
  dialoga direto com Yan et al. (C10) e, confirmada em qualquer direção, é
  manchete de paper.
- **Reproduzir 1 baseline FedSSL da literatura** (SimCLR de séries temporais,
  já na minerva) — antecipa o "por que não comparou com FedSSL existente?".
- **Redigir o parágrafo FedST/FedOST** cedo (item de risco de reviewer).
- Levar à reunião de orientação: qual item da §5.3 vira "capítulo de método"
  (recomendo `backbone-only` ou `fedbn`, custo ~zero) — é a resposta concreta
  à provocação sobre suficiência de contribuição.

---

## 7. Forças e fraquezas da contribuição (munição para a defesa)

*De `contribuicoes_forte_fraco_defesa.md` (2026-07-07), absorvido em 2026-07-27.
Documento franco de propósito: a parte de fraquezas é onde está o valor, porque é
o que o orientador/banca vai atacar. Chegar já tendo nomeado a fraqueza desarma o
ataque.*

### 7.1 Pontos fortes (ordenados por quanto blindam a defesa)

| # | Força | Evidência | Quão único |
|---|---|---|---|
| F1 | **Desenho com controle pareado** | Cada eixo de heterogeneidade isolado por um Δ contra o seu próprio controle (`plano_fedssl.md §2`) → atribuição causal. UniHAR/FedCSSL/CDFL medem "federado é pior" sem isolar a causa. ⚠️ *A versão cross-silo desta força (cenário 1 vs 2) foi **superseded** pelo pivô; o espírito — atribuição causal controlada — permanece, na topologia nova.* | Muito alto |
| F2 | **Matriz comparativa que ninguém tem** | 2 famílias de SSL (contrastiva vs augmentation-free) × 4 encoders × 2 topologias de pré-treino × regimes de dados. A literatura federa **uma** técnica por paper. | Alto |
| F3 | **Reprodutibilidade / validação prévia** | Réplica do benchmark de da Luz et al. célula a célula: viés ±2 pp, MAE 2–5 pp, 14/14 hiperparâmetros idênticos (`metodo_e_auditoria.md`). Nossos números são *plug-compatible* com a tabela publicada. | Alto (raro em FSSL) |
| F4 | **Ineditismo verificado a sério** | LFR e TF-C nunca federados, confirmado em 4 ângulos (busca web + snowball S2 + snowball OpenAlex + queries Scholar-style). Não é "acho que é novo". | Alto |
| F5 | **Comunicação como métrica de 1ª classe** | uplink/downlink por rodada, pré-treino + finetuning. Achado não-óbvio, **medido em 2026-07-28**: o LFR trafega 4,26 MiB/cliente/rodada contra 1,36 do TF-C, mas **86,3% são projetores aleatórios congelados e idênticos entre clientes** — excluídos da agregação, o LFR cai para 0,58 MiB e fica **2,3× mais barato que o TF-C**. *(A versão anterior desta força dizia "LFR é caro em uplink pelos preditores"; os preditores são 2,2% do payload. Ver `plano_fedssl.md §2.3`.)* | Alto |
| F6 | **Heterogeneidade real, não simulada** | Domain shift por construção (datasets reais) e skew inter-pessoa pela partição natural por usuário — não Dirichlet. | Médio-alto |
| F7 | **Resultado garantido nos dois desfechos** | Funcionar = 1ª evidência positiva p/ LFR/TF-C federados; divergir = caracterização quantificada de *por que* SSL de séries temporais diverge. Achado negativo é publicável (cf. FedEMA sobre stop-gradient). | Médio |
| F8 | **Ponte entre as duas linhas do HIAAC** | Linha distribuída (você) + dados/protocolo/baselines (linha DAGHAR). Explica FedAvg/Flower como *idioma da linha*, não falta de ambição. | Contexto |
| F9 | **Perguntas de design que são pesquisa, não plumbing** | DPP global vs local (específico da família augmentation-free); política de preditores (`full` vs `backbone-only`); alternância × rodada. Cada uma vira ablação com resultado. | Médio |
| F10 | **O piso de batch como achado próprio** | Critério de elegibilidade de cliente em FSSL, com taxa de exclusão medida — lacuna real na literatura (§3.4). | Alto, se bem enquadrado |

### 7.2 Pontos fracos (ordenados por probabilidade de ataque × gravidade)

- **W1 — "Não há método novo; é aplicar técnicas existentes num cenário novo."**
  *(o ataque principal)* Validade parcial e legítima. Resposta em 2 tempos:
  (1) contribuição empírica com desenho controlado, ineditismo verificado e
  validação é padrão-mestrado aceito (FedU, L-DAWA nasceram como "estudo +
  variante simples"); (2) **oferecer um componente de método de baixo custo** —
  `backbone-only`/`fedbn`/`agg_shock` já estão no plano, basta **promover de
  ablação a contribuição nomeada**.
- **W2 — "Só FedAvg. Cadê FedProx/FedBN?"** Validade média. FedAvg é *deliberado*:
  para atribuir o efeito à representação (não ao agregador) é preciso fixar o
  agregador. FedBN já está previsto como variante, e o nosso non-IID é de
  *features* — o caso dele. Enunciar assim inverte a fraqueza em decisão.
- **W3 — "É incremental sobre o benchmark de da Luz et al."** O risco de percepção
  **mais perigoso**, porque é superficialmente verdadeiro. Resposta: a
  continuidade é *ferramenta*, não contribuição. *"Reusamos o protocolo deles pelo
  mesmo motivo que se reusa uma régua calibrada — para que a medida nova seja
  comparável, não porque a medida nova seja a mesma."*
- **W4 — "FedST/FedOST já fazem tempo-frequência em FL."** Real (achado do
  snowballing). São **supervisionados**, miram personalização, usam
  tempo-frequência como regularização, não medem comunicação nem comparam
  topologias. Citar e distinguir **proativamente**.
- **W5 — "Uma coleção só (DAGHAR), toda smartphone-IMU."** Escopo declarado; o
  DAGHAR é desenhado para isolar domain shift com vieses triviais removidos.
  Generalização a outras modalidades = trabalho futuro explícito.
- **W6 — "6 clientes cross-silo. E escala? Cross-device?"** ✅ **Praticamente
  respondida pelo pivô**: o eixo agora é cross-device, com até 151 usuários reais
  (`dados_daghar.md §3`). Era a fraqueza mais fácil de atacar e virou o desenho.
- **W7 — "Comunicação são bytes teóricos, não tempo real."** Verdade. Bytes é a
  métrica **independente de ambiente** de propósito; wall-clock não seria
  replicável.
- **W8 — "O componente de maior valor ainda não rodou."** Risco de cronograma, não
  de mérito. Existe plano com gates e ordem de corte — levar como "tenho plano B
  de escopo".
- **W9 — "Achado negativo soa como plano de fuga."** Baixa se bem enquadrado: é
  **pergunta científica cujas duas respostas são informativas**, decidida antes de
  saber o resultado. O `agg_shock` e o gate IID tornam o "não" rigoroso.

### 7.3 A contribuição em três níveis (para negociar escopo)

| Nível | O que é | Custo extra | Quando escolher |
|---|---|---|---|
| **Mínimo defensável** | SL + SSL central + baseline federado + finetuning federado com init central + a quantificação da heterogeneidade. Quase tudo medido. | ~0 | Se o cronograma desabar |
| **Recomendado (alvo)** | + FedAvg-SSL de LFR e TF-C **com uma variante de método promovida a contribuição nomeada** (`backbone-only` OU `fedbn`) + comunicação ponta-a-ponta + a pergunta "augmentation-free é mais robusto à agregação que contrastivo?" | ~1–2 semanas de cluster | Caso base |
| **Stretch** | + agregação adaptativa via `agg_shock` (mini-L-DAWA para séries temporais) OU FedEMA adaptado ao TF-C + 1 baseline FedSSL reproduzido | +2–3 semanas | Se estiver adiantado |

**Recomendação**: propor o **Recomendado** e perguntar ao orientador qual variante
de método ele prefere ver promovida — é decisão dele. Isso já responde à
provocação: não é "federar uma técnica", é "federar duas famílias + caracterizar
+ uma variante de método própria".

---

*Bibliografia local (PDFs baixados): `papers/README.md`. Desenho experimental:
`plano_fedssl.md`. Resultados medidos: `resultados.md`.*
