# Esqueleto do artigo — SSL Federado para Mitigação de Heterogeneidade de Domínio em HAR

*Esqueleto de artigo de conferência (formato IEEE, duas colunas, ~8 páginas).
Escrito em português para redação interna; tradução para inglês na fase de
submissão. Placeholders de resultados apontam para os CSVs de `results/` que
cada tabela/figura consome — nenhum número é inventado: ou já está medido, ou
está marcado como `[PENDENTE]`.*

*Criado em 2026-07-07. Convenções: `[PENDENTE: ...]` = resultado ainda não
medido; `[REDIGIR]` = texto a escrever quando os números fecharem;
`(→ arquivo)` = fonte de dados da tabela/figura.*

---

## Título provisório

**Federated Self-Supervised Pre-Training for Mitigating Domain Heterogeneity
in Cross-Silo Human Activity Recognition**

Alternativas:
- *On the Role of Self-Supervised Pre-Training in Domain-Heterogeneous Federated HAR*
- *Centralized vs. Federated Self-Supervised Pre-Training for Cross-Silo HAR under Domain Shift*

Título interno (HIAAC): "Aprendizado Auto-Supervisionado Federado para
Mitigação de Heterogeneidade de Domínio em HAR".

---

## Abstract (provisório)

> O aprendizado federado (FL) permite treinar modelos de Reconhecimento de
> Atividades Humanas (HAR) sem centralizar dados sensíveis de sensores
> vestíveis, mas federações cross-silo realistas sofrem de heterogeneidade de
> domínio: cada silo coleta dados com dispositivos, posicionamentos e
> protocolos distintos. Neste trabalho quantificamos esse efeito numa federação
> cross-silo construída sobre o benchmark DAGHAR (6 datasets, 1 por cliente) e
> investigamos se o pré-treino auto-supervisionado (SSL) mitiga a degradação
> causada pelo *domain shift*. Comparamos duas técnicas de SSL de famílias
> distintas — LFR (reconstrução de projeções aleatórias, livre de aumentações)
> e TF-C (contrastiva tempo-frequência) — sobre quatro encoders (ResNet-SE-5,
> CNN-PFF, BiGRU e TS-TCC/HARCNN), em duas estratégias: (i) pré-treino SSL
> centralizado seguido de finetuning federado e (ii) pré-treino SSL executado
> no próprio processo federado (FedAvg-SSL), seguido de finetuning federado.
> Nossa avaliação cobre [PENDENTE: N total] treinos, com acurácia, F1-macro e
> custo de comunicação. Os resultados mostram que a heterogeneidade de domínio
> custa ≈8 pontos percentuais de acurácia em relação a uma federação IID com o
> mesmo volume de dados; que o benefício do SSL é fortemente dependente do
> encoder ([PENDENTE: síntese LFR vs TF-C]); e que [PENDENTE: achado principal
> do Exp. 3 — o pré-treino federado preserva/degrada X% do benefício do
> pré-treino centralizado, a um custo de comunicação Y]. Código e resultados:
> [PENDENTE: URL do repositório].

**Palavras-chave**: federated learning, self-supervised learning, human
activity recognition, domain shift, cross-silo, FedAvg, time series.

---

## I. Introdução (~1 página)

**Parágrafo 1 — HAR e o problema de dados.** HAR com sensores inerciais
(acelerômetro + giroscópio) de smartphones/vestíveis; aplicações em saúde,
monitoramento e bem-estar. Rotulagem é cara; dados são pessoais e sensíveis
(padrões de movimento identificam o usuário) — motivação dupla para FL e SSL.

**Parágrafo 2 — FL e a heterogeneidade que a literatura não modela bem.**
FedAvg [McMahan et al., 2017] como protocolo padrão. A maior parte da
literatura de FL não-IID modela *label skew* (partições Dirichlet sobre um
único dataset). Em HAR cross-silo o problema dominante é outro: *feature/domain
shift* — silos diferentes usam dispositivos, posições de sensor, taxas de
amostragem e populações distintas. A federação DAGHAR (1 dataset real por
cliente) cria essa heterogeneidade *por construção*, sem simulação artificial.

**Parágrafo 3 — SSL como candidato a mitigador.** SSL aprende representações
sem rótulos; a hipótese é que representações pré-treinadas sobre dados de
múltiplos domínios (ou agregadas de múltiplos silos) sejam mais transferíveis
e deem ao finetuning federado um ponto de partida melhor que a inicialização
aleatória. Duas famílias testadas: LFR [Sui et al., ICLR 2024] — livre de
aumentações, relevante porque aumentações para séries temporais IMU são mal
definidas — e TF-C [Zhang et al., NeurIPS 2022] — contrastiva
tempo-frequência, proposta exatamente para transferência entre domínios de
séries temporais, e a técnica mais forte do benchmark centralizado de
referência [da Luz et al., IEEE Access 2026].

**Parágrafo 4 — Pergunta de pesquisa e contribuições.** Perguntas: (RQ1)
quanto custa a heterogeneidade de domínio numa federação cross-silo de HAR?
(RQ2) o pré-treino SSL centralizado melhora o finetuning federado sob domain
shift? (RQ3) o pré-treino SSL pode ser feito *no próprio processo federado*
sem perder o benefício — e a que custo de comunicação? Contribuições:

1. Quantificação controlada do custo do domain shift em FedAvg cross-silo
   sobre 6 datasets reais de HAR padronizados (cenário non-IID por domínio vs
   controle IID com volume idêntico), com 4 encoders × 4 seeds.
2. Comparação sistemática de LFR e TF-C sobre 4 encoders em regimes de dados
   rotulados de 1 a 100% (grade centralizada validada contra o benchmark
   publicado), estendida com a matriz de transferência 7 fontes × 6 alvos.
3. [PENDENTE Exp. 2/3] Avaliação das duas estratégias de pré-treino
   (centralizado vs federado) como inicialização do finetuning federado,
   incluindo custo de comunicação de ponta a ponta.
4. Código, checkpoints e caches de avaliação públicos.

---

## II. Trabalhos Relacionados (~1 página)

### II-A. HAR baseado em sensores e domain shift

- Surveys de SSL para HAR com vestíveis: Haresamudram et al. (IMWUT/ACM) —
  ["Self-supervised Learning for Accelerometer-based Human Activity
  Recognition: A Survey"](https://dl.acm.org/doi/10.1145/3699767); tutorial
  de HAR com vestíveis (arXiv 2411.14452).
- **DAGHAR** [Napoli et al., *Scientific Data*, 2024]
  (https://www.nature.com/articles/s41597-024-03951-4; Zenodo 11992126):
  benchmark de adaptação/generalização de domínio em HAR de smartphone; 6
  datasets (KuHar, MotionSense, RealWorld-Thigh, RealWorld-Waist, UCI, WISDM)
  padronizados em unidades, taxa de amostragem, gravidade, rótulos e janelas —
  remove vieses triviais e preserva as diferenças intrínsecas de domínio. É a
  base de dados deste trabalho.

### II-B. SSL para séries temporais e o benchmark de referência

- **TF-C** [Zhang et al., NeurIPS 2022] (arXiv 2206.08496;
  github.com/mims-harvard/TFC-pretraining): consistência tempo-frequência
  como sinal auto-supervisionado; proposto para pré-treino transferível entre
  domínios de séries temporais; ganhos médios de 15.4% F1 em one-to-one.
- **LFR** [Sui et al., ICLR 2024, "Self-supervised Representation Learning
  from Random Data Projectors"] (arXiv 2310.07756; github.com/layer6ai-labs/lfr):
  reconstrução de projeções aleatórias congeladas; dispensa aumentações
  específicas de domínio — atrativo para IMU, onde a invariância a
  transformações é difícil de postular.
- **Benchmark de encoders + SSL para HAR** [da Luz et al., IEEE Access, vol.
  14, pp. 37451–37475, 2026, DOI 10.1109/ACCESS.2026.3669412;
  github.com/H-IAAC/benchmarking-encoders-ssl-har]: avalia LFR, TF-C, TNC e
  DIET sobre 7 backbones no DAGHAR centralizado. Nossos protocolos de
  pré-treino/finetuning replicam os deste benchmark (validação célula a célula
  na Seção IV-B), e nossa contribuição o estende para o eixo **federado**.
- Comparações adicionais de SSL para HAR: arXiv 2404.15331 (comparing SSL
  techniques for wearable HAR); TS2Vec, TS-TCC como métodos correlatos
  (citar brevemente).

### II-C. FL sob heterogeneidade

- **FedAvg** [McMahan et al., AISTATS 2017]. **FedProx** [Li et al., MLSys
  2020] — regularização proximal contra drift de cliente. **FedBN** [Li et
  al., ICLR 2021] (github.com/med-air/FedBN) — estatísticas de BatchNorm
  locais para *feature shift* non-IID; diretamente relevante ao nosso cenário
  (heterogeneidade de features por construção) e usado aqui como mitigação
  candidata no Exp. 3.
- Posicionamento: nosso foco não é propor agregador novo, e sim medir o efeito
  do *ponto de partida* (representação pré-treinada) mantendo o agregador fixo
  (FedAvg), o caso mais padrão e reprodutível.

### II-D. SSL federado

- **Saeed et al., IEEE IoT Journal 2021** (DOI 10.1109/JIOT.2020.3009358) —
  primeiro FedSSL para sensores multimodais (scalogram-signal correspondence);
  demonstra viabilidade de pré-treino auto-supervisionado federado em sinais
  vestíveis.
- **FedU** [Zhuang et al., ICCV 2021] e **FedEMA** [Zhuang et al., ICLR 2022,
  "Divergence-aware Federated Self-Supervised Learning", arXiv 2204.04385] —
  framework geral de FedSSL com redes siamesas; regras de atualização
  conscientes de divergência (agregar encoder online, atualização EMA
  adaptativa do conhecimento global); achados: *stop-gradient* nem sempre é
  necessário e reter conhecimento local ajuda em non-IID.
- **Orchestra** [Lubana et al., ICML 2022] — FedSSL por clusterização
  global-local consistente.
- **FedCA / federated unsupervised representation learning** [Zhang et al.] —
  alinhamento de representações entre clientes com dicionário compartilhado.
- Aplicações a HAR: FedCSSL para smart homes (Springer, 2025/2026);
  SSF-HAR (IEEE, 2024); semi-supervisionado personalizado (arXiv 2104.08094).
- **Lacuna que exploramos**: essas obras usam métodos contrastivos de visão
  (SimCLR/BYOL) e heterogeneidade simulada; nenhuma compara *pré-treino
  centralizado vs federado* das mesmas técnicas de SSL de séries temporais
  (LFR/TF-C) sob *domain shift real* cross-silo, com custo de comunicação
  medido — exatamente o eixo deste artigo.

### II-E. Posicionamento em relação à linha H-IAAC (DAGHAR / benchmark)

*Levantamento de 2026-07-07 da produção do grupo Napoli–da Luz–Soto–Rocha–
Boccato–Borin (H-IAAC/Unicamp), para delimitar explicitamente o que este
trabalho adiciona. Importante para a redação: somos usuários diretos do
dataset e do protocolo deles — a diferenciação precisa ficar inequívoca.*

Produção mapeada do grupo (toda **centralizada**; nenhuma envolve federação):

| Trabalho | O que fez | O que NÃO fez (e nós fazemos) |
|---|---|---|
| **DAGHAR** [Napoli et al., Sci Data 2024] | Recurso: 6 datasets padronizados (unidades, taxa, gravidade, rótulos, splits por usuário, janelas); baselines supervisionados de adaptação/generalização de domínio cross-dataset | Nenhum SSL; nenhuma federação; sem regimes few-shot; sem custo de comunicação |
| **Benchmark encoders+SSL** [da Luz et al., IEEE Access 2026] | 7 backbones × 4 técnicas SSL (LFR, TF-C, TNC, DIET) + supervisionado, no DAGHAR; regimes de 1 a 100% amostras/classe; linear readout + full finetuning; checkpoints publicados (Zenodo 19301058) | Avaliação **só in-domain** (sem matriz de transferência cross-dataset); tudo centralizado; sem eixo federado; sem comunicação; acurácia como métrica principal (sem F1-macro) |
| **CPC × corpus de pré-treino** [Rodrigues da Silva et al., BRACIS 2024] | Efeito do dataset de pré-treino no CPC para HAR (origem da view `rodrigues_2024` usada no benchmark) | Uma técnica só (CPC); centralizado; sem federação |
| **TNC variants** [BRACIS 2024, conferir autores] | Variantes de Temporal Neighborhood Coding no HAR de smartphone | Idem: técnica única, centralizado |
| **Mix-based DG** [ESANN 2025, conferir autores] | Generalização de domínio via métodos mix (MixStyle etc.; melhor célula: TS2Vec+MixStyle) | Mitigação por manipulação de dados, não por pré-treino; centralizado |
| **LIME vs SHAP** [Alves et al., BRACIS 2024] | Explicabilidade em HAR | Ortogonal ao nosso eixo |

Nota: o H-IAAC tem também uma linha de FL (seleção de clientes, eficiência de
comunicação, robustez — Cerqueira, Villas, Rosário et al.), mas **disjunta**
desta: não trata HAR sob domain shift nem SSL. Nenhum trabalho do hub conecta
as duas linhas — é exatamente a junção que este artigo faz.

**Diferenças estruturais deste trabalho:**

1. **O eixo federado é inteiramente novo.** O grupo estuda domain shift e SSL
   no regime centralizado; nós recolocamos o mesmo domain shift como
   heterogeneidade non-IID *por construção* numa federação cross-silo
   (1 dataset = 1 cliente) e medimos o que o FedAvg faz com ele — incluindo o
   controle IID de volume idêntico (cenário 2) e a ablação intra-domínio
   (cenários 3–8), um desenho experimental que não existe na linha deles.
2. **SSL como estratégia de *inicialização* federada, não só como método
   centralizado.** O benchmark responde "qual técnica/encoder é melhor no
   centralizado"; nós respondemos "o pré-treino (centralizado OU federado)
   melhora o finetuning federado sob domain shift, e a que custo?" — as duas
   estratégias (Exp. 2 e 3) não têm análogo em nenhum paper do grupo.
3. **Pré-treino SSL federado (FedAvg-SSL) de LFR/TF-C**: inédito também na
   literatura de FedSSL (que usa SimCLR/BYOL de visão) — e inédito para o
   grupo.
4. **Matriz de transferência cross-dataset para SSL.** O benchmark é só
   in-domain; o DAGHAR original faz cross-dataset mas só supervisionado. Nós
   cobrimos as 7 fontes × 6 alvos zero-shot para SL, LFR e TF-C, mais o
   desenho comb→target (que estende a pergunta do paper de CPC deles para
   LFR/TF-C com avaliação cruzada).
5. **Custo de comunicação como métrica de primeira classe** (uplink/downlink
   por rodada, pré-treino + finetuning) — ausente de toda a produção mapeada.
6. **F1-macro** ao lado da acurácia em todas as grades (o desbalanceamento do
   HAR torna a acurácia otimista; gap médio observado ≈ 8 pp).
7. **Continuidade metodológica como força, não fraqueza**: replicamos o
   protocolo oficial (auditoria 14/14 hiperparâmetros; validação célula a
   célula com viés ±2 pp / MAE 2–5 pp) antes de estender — nossos números são
   *plug-compatible* com o benchmark publicado, o que permite ao leitor
   comparar diretamente as duas tabelas.

**Frase de posicionamento (candidata para o fim do related work):** "While
DAGHAR established *what* domain heterogeneity looks like in smartphone HAR
and the encoder/SSL benchmark established *which* representations help under
label scarcity, both operate in a centralized regime. This work asks what
happens when the same heterogeneity becomes a *federation topology* — and
whether self-supervised pre-training, centralized or federated, buys back the
accuracy that domain-heterogeneous FedAvg loses."

---

## III. Metodologia (~1.5–2 páginas)

*Princípio: descrever o que JÁ ESTÁ MEDIDO ou tem infra pronta; o Exp. 3 segue
o design doc `docs/plano_experimento3_fedssl.md`.*

### III-A. Dados e formulação

- DAGHAR `standardized_view`: 6 datasets, janelas de 60 timesteps × 6 canais
  (accel x/y/z + gyro x/y/z), 6 classes
  {Sit, Stand, Walk, Stair-up, Stair-down, Run}; splits train/val/test por
  usuário, fornecidos pelo benchmark.
- Pseudo-fonte `combined` = união dos 6 train sets (só como fonte de
  treino/pré-treino, nunca como alvo).
- Federação cross-silo: 6 clientes, 1 dataset por cliente (cenário 1);
  controle IID: união dividida em 6 fatias disjuntas de volume idêntico
  (cenário 2) — isola o efeito do domain shift; ablação: cada dataset
  individual dividido IID em 6 clientes (cenários 3–8) — isola o custo da
  federação sem heterogeneidade.
- Desvio documentado vs benchmark: pré-treino SSL usa train+val da
  `standardized_view` (o benchmark usa a view `rodrigues_2024`).

### III-B. Encoders

4 encoders (mesmos `build_model` em todos os experimentos):

| Encoder | Arquitetura | dim features | Params |
|---|---|---|---|
| ResNet-SE-5 | ResNet 1D + Squeeze-Excitation | 64 | ~127K |
| CNN-PFF | CNN parcial-completa fusão | 768 | [PENDENTE: conferir] |
| BiGRU (rnn) | GRU bidirecional 100 unidades | 320 | [PENDENTE] |
| TS-TCC enc. (tstcc) | HARSCnnEncoder | 2304 | [PENDENTE] |

Cabeça de classificação idêntica em tudo: MLP `Linear(dim→128)→ReLU→Linear(128→6)`.

### III-C. Técnicas de SSL

- **LFR**: 60 projetores convolucionais aleatórios congelados sobre a série
  bruta, 6 alvos selecionados via DPP, preditores lineares `dim→dim`, loss
  Barlow Twins batch-wise; 600 épocas de Trainer = 100 efetivas de backbone
  (alternância 1:5 backbone/preditores); Adam 3e-4, wd 3e-4, batch 64.
- **TF-C**: encoders gêmeos tempo/frequência (FFT interna), projeção para
  128-d por ramo, NT-Xent poly (temp 0.2); 100 épocas sem early stopping;
  Adam 3e-4, batch 64 com `drop_last`; downstream usa concat 256-d.
- Hiperparâmetros idênticos aos YAMLs oficiais do benchmark (auditoria 14/14
  itens — citar como garantia de comparabilidade).

### III-D. Protocolo de avaliação

- **Downstream**: *linear readout* (backbone congelado) e *full finetuning*;
  Adam 1e-4, até 100 épocas, early stopping paciência 50 na val da fonte,
  melhor estado restaurado.
- **Regimes de dados rotulados**: 1 / 10 / 100 amostras-por-classe e 100%
  (eixo `n_shots` compartilhado entre SL e SSL).
- **Transferência zero-shot**: todo modelo treinado numa fonte é avaliado nos
  test sets dos 6 alvos (matriz 7×6).
- **Métricas**: acurácia, F1-macro (classes desbalanceadas), e custo de
  comunicação (uplink/downlink em bytes por rodada, contando o conjunto
  completo de tensores transmitidos, buffers incluídos).

### III-E. Desenho experimental

- **Exp. 0 (baseline SL centralizado + transfer)**: 4 encoders × 7 fontes ×
  4 seeds × 4 regimes; avaliação 7×6.
- **Exp. 1 (SSL centralizado)**: LFR e TF-C, 4 encoders × 7 fontes × 4 seeds;
  downstream nos 2 protocolos × 4 regimes; validação célula a célula contra o
  benchmark publicado (gate de sanidade).
- **Baseline federado supervisionado**: FedAvg, R=50 rodadas, 1 época local,
  avaliação centralizada por domínio a cada rodada; 8 cenários × 4 encoders ×
  4 seeds.
- **Exp. 2 (SSL centralizado → finetuning federado)**: backbones do Exp. 1
  (fonte `combined`) como inicialização do FedAvg supervisionado no cenário 1;
  [PENDENTE: grade final — ver design doc Exp. 3, a infra de finetune é comum].
- **Exp. 3 (SSL federado → finetuning federado, FedAvg-SSL)**: pré-treino LFR/
  TF-C dentro do FedAvg (agregação de backbone [+ preditores], cenários 1 e 2),
  seguido do mesmo finetuning federado — design completo em
  `docs/plano_experimento3_fedssl.md`.
- Seeds 0–3 em tudo; médias ± desvio-padrão.

---

## IV. Resultados (~2–2.5 páginas)

*Cada tabela/figura indica o CSV-fonte e o notebook que a gera. Números já
conhecidos estão preenchidos; o resto é [PENDENTE].*

### IV-A. Baseline supervisionado e o custo do domain shift centralizado

- **Tabela I — matriz de transferência supervisionada (acc/F1, 100% dos
  dados)**: diagonal in-domain vs off-diagonal cross-dataset, 4 encoders.
  (→ `results/supervised_eval_transfer.csv`;
  notebook `centralized_supervised_avaliation.ipynb`).
  Já medido (3 encoders): média global acc ≈ 0.539, F1 ≈ 0.455 — o gap
  diagonal/off-diagonal quantifica o domain shift centralizado; o gap acc–F1
  justifica reportar F1-macro. [PENDENTE: linha tstcc.]
- **Fig. 1 — heatmaps de transferência 7×6** por encoder.
  (→ mesmo CSV/notebook.)

### IV-B. SSL centralizado (Exp. 1) e validação contra o benchmark

- **Tabela II — SL vs LFR vs TF-C in-domain por encoder × regime de dados**
  (full finetuning e linear readout).
  (→ `results/ssl_lfr_eval_transfer.csv`, `results/ssl_tfc_eval_transfer.csv`
  [PENDENTE: grade TF-C], `results/supervised_eval_transfer.csv`;
  notebook `ssl_lfr_avaliation.ipynb`.)
  Já medido (LFR, 3 encoders): LFR só ajuda consistentemente a **RNN**
  (+17 pp F1 in-domain @10-shot full-ft; paper reporta +18.1 pp acc);
  ResNet-SE-5 ~neutro; CNN-PFF misto; linear readout do ResNet-SE-5 é fraco
  por natureza do método (−20 pp @10-shot). [PENDENTE: LFR tstcc (em
  execução); TF-C nos 4.]
- **Fig. 2 — curvas de data-efficiency** (acc × n_shots, SL vs LFR vs TF-C).
  (→ mesmos CSVs.)
- **Tabela III (ou parágrafo) — validação vs benchmark publicado**: viés e MAE
  célula a célula vs `performance_data.json` do repo oficial. Já medido (LFR,
  3 encoders): viés por encoder×protocolo entre −1.2 e +1.8 pp, MAE 2.1–4.6 pp.
  [PENDENTE: mesmas estatísticas p/ tstcc e TF-C — gates 2, 3b e 7 do plano
  tstcc+TF-C.] Argumento de solidez: replicamos o benchmark antes de
  estendê-lo ao eixo federado.
- **Tabela IV (opcional/apêndice) — comb→target**: backbone pré-treinado no
  `combined`, refinado no alvo (isola o efeito do corpus de pré-treino).
  (→ `results/ssl_lfr_comb2target_eval_transfer.csv`,
  [PENDENTE: `results/ssl_tfc_comb2target_eval_transfer.csv`].)

### IV-C. Baseline federado: quantificando o domain shift (RQ1)

- **Fig. 3 — acurácia média × rodada de comunicação**, cenário 1 (non-IID por
  domínio) vs cenário 2 (IID global), 4 encoders, R=50.
  (→ `results/federated_eval.csv`; notebook `federated_avaliation.ipynb`.)
  Já medido (3 encoders): cenário 2 ≈ 0.74 acc vs cenário 1 ≈ 0.66 — **o
  domain shift custa ≈8 pp** com volume de dados idêntico. [PENDENTE: tstcc
  (32 runs, fase 4 do plano tstcc+TF-C).]
- **Tabela V — acc/F1 finais (rodada 50) por cenário × encoder**; ablação
  intra-domínio (cenários 3–8, ≈0.45–0.53) mostra que federar um único domínio
  pequeno é pior que federar domínios heterogêneos — o volume também importa.
  (→ mesmo CSV.)
- **Fig. 4 — custo de comunicação**: bytes acumulados × rodada por encoder
  (o tstcc 2304-d é o mais caro; conexão com a discussão do Exp. 3).
  (→ colunas `uplink_bytes`/`downlink_bytes`.)

### IV-D. Exp. 2 — pré-treino centralizado + finetuning federado (RQ2)

[PENDENTE — infra de finetune federado a partir de checkpoint em
implementação; ver design doc Exp. 3, Fase 5.]

- **Fig. 5 — curvas FedAvg cenário 1**: init aleatório vs init LFR-`combined`
  vs init TF-C-`combined`, por encoder. Hipóteses a testar: (i) SSL
  centralizado acelera a convergência federada (menos rodadas para a mesma
  acc ⇒ menos comunicação); (ii) o teto final melhora sobretudo para os
  encoders em que o SSL ajuda no centralizado (RNN p/ LFR; CNN-PFF p/ TF-C).
  (→ [PENDENTE: `results/federated_ssl_eval.csv`, coluna `init`].)
- **Tabela VI — rodadas até atingir X% da acc final** (métrica de economia de
  comunicação). (→ mesmo CSV.)

### IV-E. Exp. 3 — pré-treino SSL federado (RQ3)

[PENDENTE — componente de maior risco; design em
`docs/plano_experimento3_fedssl.md`. Resultado negativo (divergência do
FedAvg-SSL) também é publicável e será reportado como tal se ocorrer.]

- **Tabela VII — qualidade da representação**: linear readout centralizado dos
  backbones FedSSL vs backbones centralizados equivalentes (mesma técnica,
  fonte `combined` ≙ união dos silos) vs baseline SL. Compara: pré-treino
  federado preserva quantos pp do benefício do centralizado?
  (→ [PENDENTE: `results/ssl_fed_eval_transfer.csv`].)
- **Fig. 6 — pipeline completo**: acc final do finetuning federado com as 3
  inicializações (aleatória / SSL centralizado / SSL federado) × 2 técnicas ×
  4 encoders, cenário 1. É a figura-síntese do artigo (responde RQ2+RQ3 lado a
  lado). (→ [PENDENTE: `results/federated_ssl_eval.csv`].)
- **Tabela VIII — custo de comunicação de ponta a ponta** (pré-treino + 
  finetuning): FedSSL paga comunicação no pré-treino (nota: no LFR, só
  backbone + preditores ativos são transmitidos; ver design doc §5) — o
  benefício líquido depende do quanto o finetuning encurta.
  (→ [PENDENTE: colunas de bytes do mesmo CSV].)

---

## V. Discussão (~0.75 página)

*Achados já estabelecidos (redigir em torno deles):*

1. **O domain shift domina o custo da federação (RQ1).** Ceteris paribus
   (mesmo volume, mesmo protocolo), passar de partição IID para 1-domínio-por-
   cliente custa ≈8 pp de acurácia. A ablação intra-domínio mostra que isso
   não é artefato de volume por cliente. Implicação: benchmarks de FL para HAR
   que só usam label-skew Dirichlet subestimam o problema real.
2. **SSL não é panaceia: o benefício é encoder-dependente.** LFR ajuda
   consistentemente apenas a RNN (replicando o benchmark centralizado — nossa
   medição independente confirma o achado publicado); [PENDENTE: TF-C deve
   ajudar mais amplamente — é a técnica mais forte do benchmark (ex.:
   TF-C+CNN-PFF: 77.0% @10-shot vs 55.3% supervisionado). Confirmar na nossa
   grade.] Escolher a técnica de SSL sem considerar o encoder é um erro de
   projeto experimental.
3. **[PENDENTE: RQ2]** — o pré-treino centralizado melhora ponto de partida
   e/ou velocidade de convergência federada?
4. **[PENDENTE: RQ3]** — quanto do benefício sobrevive quando o pré-treino
   também é federado; riscos observados de divergência contrastiva
   (NT-Xent com negativos apenas locais; preditores LFR desalinhados entre
   clientes) e o efeito das mitigações (frequência de agregação, FedBN).
5. **Custo de comunicação como eixo de decisão.** [PENDENTE: comparação
   bytes-para-atingir-X% entre as 3 inicializações; o tamanho do encoder
   (64-d vs 2304-d) muda a conclusão?]

**Limitações**: FedAvg apenas (FedProx/SCAFFOLD como trabalho futuro); 6
clientes cross-silo (não cross-device); simulação em GPU única/cluster (sem
latência real de rede); linear readout + full finetuning apenas; view
`standardized_view` no pré-treino (desvio documentado do benchmark).

---

## VI. Conclusão (~0.25 página)

[REDIGIR ao final: retomar RQ1–RQ3 com os números fechados; mensagem
provável: heterogeneidade de domínio é o custo dominante em HAR federado
cross-silo; SSL mitiga de forma encoder-dependente; pré-treino federado
{preserva|não preserva} o benefício a um custo de comunicação {aceitável|
proibitivo}. Trabalho futuro: FedBN/FedProx, TNC, foundation models de séries
temporais, mais clientes, análise de privacidade.]

---

## Referências (reais, verificadas em 2026-07-07)

1. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas,
   "Communication-Efficient Learning of Deep Networks from Decentralized
   Data", AISTATS 2017.
2. G. P. C. P. da Luz, D. H. P. Soto, O. O. Napoli, A. Rocha, L. Boccato,
   E. Borin, "Benchmarking Encoders and Self-Supervised Learning for
   Smartphone-Based Human Activity Recognition", IEEE Access, vol. 14,
   pp. 37451–37475, 2026. DOI 10.1109/ACCESS.2026.3669412.
   Repo: github.com/H-IAAC/benchmarking-encoders-ssl-har.
3. X. Zhang, Z. Zhao, T. Tsiligkaridis, M. Zitnik, "Self-Supervised
   Contrastive Pre-Training for Time Series via Time-Frequency Consistency",
   NeurIPS 2022. arXiv:2206.08496. Repo: github.com/mims-harvard/TFC-pretraining.
4. Y. Sui et al., "Self-supervised Representation Learning from Random Data
   Projectors", ICLR 2024. arXiv:2310.07756. Repo: github.com/layer6ai-labs/lfr.
5. O. O. Napoli et al., "A benchmark for domain adaptation and generalization
   in smartphone-based human activity recognition", Scientific Data 11, 1192
   (2024). DOI 10.1038/s41597-024-03951-4. Dados: Zenodo 11992126 (DAGHAR).
6. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith,
   "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020.
7. X. Li, M. Jiang, X. Zhang, M. Kamp, Q. Dou, "FedBN: Federated Learning on
   Non-IID Features via Local Batch Normalization", ICLR 2021.
   Repo: github.com/med-air/FedBN.
8. A. Saeed, F. D. Salim, T. Ozcelebi, J. Lukkien, "Federated Self-Supervised
   Learning of Multisensor Representations for Embedded Intelligence",
   IEEE Internet of Things Journal, 8(2):1030–1040, 2021.
   DOI 10.1109/JIOT.2020.3009358.
9. W. Zhuang, X. Gan, Y. Wen, S. Zhang, S. Yi, "Collaborative Unsupervised
   Visual Representation Learning from Decentralized Data" (FedU), ICCV 2021.
10. W. Zhuang, Y. Wen, S. Zhang, "Divergence-aware Federated Self-Supervised
    Learning" (FedEMA), ICLR 2022. arXiv:2204.04385.
11. E. S. Lubana, C. I. Tang, F. Kawsar, R. P. Dick, A. Mathur, "Orchestra:
    Unsupervised Federated Learning via Globally Consistent Clustering",
    ICML 2022.
12. H. Haresamudram, I. Essa, T. Plötz, "Self-supervised Learning for
    Accelerometer-based Human Activity Recognition: A Survey", Proc. ACM
    IMWUT (dl.acm.org/doi/10.1145/3699767).
13. D. J. Beutel et al., "Flower: A Friendly Federated Learning Research
    Framework", arXiv:2007.14390, 2020.
14. [Conferir na redação] F. Zhang et al., "Federated Unsupervised
    Representation Learning" (FedCA), Front. Inf. Technol. Electron. Eng.,
    2023.
15. [Conferir/decidir] FedCSSL — "Federated Contrastive Self-supervised
    Learning for Human Activity Recognition in Smart Homes", Springer
    (10.1007/978-3-032-16995-2_15).

16. B. E. Rodrigues da Silva, O. Napoli, J. Vargas, A. Rocha, L. Boccato,
    E. Borin, "Impact of Pre-training Datasets on Human Activity Recognition
    with Contrastive Predictive Coding", BRACIS 2024 (Springer LNCS,
    DOI 10.1007/978-3-031-79035-5_21).
17. [Conferir autores] "An Evaluation of Temporal Neighborhood Coding
    Variants in Smartphone-Based Human Activity Recognition", BRACIS 2024
    (DOI 10.1007/978-3-031-79035-5_6).
18. [Conferir autores] "On Domain Generalization for Human Activity
    Recognition with Mix-Based Methods", ESANN 2025
    (esann.org/sites/default/files/proceedings/2025/ES2025-135.pdf).

*Ainda por buscar na fase de redação: citação canônica de TS-TCC (Eldele et
al., IJCAI 2021) e TS2Vec (Yue et al., AAAI 2022) se entrarem no related work;
UCI-HAR/WISDM/etc. originais (citados via DAGHAR); autores completos das
refs. 17–18.*

---

## Apêndice interno — mapa resultado→fonte (não vai no artigo)

| Elemento | Fonte de dados | Status |
|---|---|---|
| Tab. I, Fig. 1 | `results/supervised_eval_transfer.csv` | ✅ 3 enc; tstcc ✅ (commit 80cb212) |
| Tab. II, Fig. 2 (LFR) | `results/ssl_lfr_eval_transfer.csv` | ✅ 3 enc; tstcc 🟡 rodando |
| Tab. II, Fig. 2 (TF-C) | `results/ssl_tfc_eval_transfer.csv` | ⬜ fase 7 do plano tstcc+TF-C |
| Tab. III (validação) | `performance_data.json` do repo oficial + caches | ✅ LFR 3 enc (notas §9) |
| Tab. IV | `results/ssl_{lfr,tfc}_comb2target_eval_transfer.csv` | ✅ LFR / ⬜ TF-C |
| Fig. 3–4, Tab. V | `results/federated_eval.csv` | ✅ 96 runs; +32 tstcc ⬜ |
| Fig. 5, Tab. VI (Exp. 2) | `results/federated_ssl_eval.csv` (proposto) | ⬜ design doc Exp. 3 |
| Tab. VII (Exp. 3 repr.) | `results/ssl_fed_eval_transfer.csv` (proposto) | ⬜ design doc Exp. 3 |
| Fig. 6, Tab. VIII | `results/federated_ssl_eval.csv` (proposto) | ⬜ design doc Exp. 3 |

Orçamento de páginas (IEEE 2 colunas): Introdução 1.0 · Related 1.0 ·
Metodologia 1.75 · Resultados 2.5 · Discussão 0.75 · Conclusão 0.25 ·
Referências 0.75 ≈ 8.0.
