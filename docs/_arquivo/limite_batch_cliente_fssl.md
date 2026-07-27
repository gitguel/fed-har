> **📦 ARQUIVADO em 2026-07-27.** Achado do piso de batch (2026-07-24). **Absorvido em `../plano_fedssl.md §3`** (o achado e suas consequências) e **`../estado_da_arte.md §3.4`** (o posicionamento na literatura e a conferência de ineditismo). Mantido como registro da investigação original.
>
> Índice dos documentos vivos: `../README.md`.

---

# O cliente mínimo viável em SSL federado é um *batch*, não uma amostra

*Escrito em 2026-07-24. Registra um achado que surgiu na sabatina do desenho
cross-device (ver `docs/analise_domain_shift.md` §5 e `docs/plano_fedssl_simulado.md`):
ao particionar o KuHar por usuário, **84% dos clientes têm menos janelas que um
batch**, e o objetivo auto-supervisionado — diferentemente do supervisionado —
**não está definido abaixo de um batch**. Aqui: (1) por que isso acontece, (2) se
vale para outras losses SSL (SimCLR, Barlow Twins, NT-Xent, TS-TCC, BYOL), e (3)
como a literatura de F-SSL lida com o problema. Todo número do nosso lado é
regenerável do repo (§5); os fatos de método vêm de fonte primária citada (§6).*

Relacionados: `docs/analise_domain_shift.md`, `docs/plano_fedssl_simulado.md`
(decisão D-K), `docs/estado_da_arte_fssl_e_contribuicoes.md`, `docs/papers/README.md`.

---

## 1. O achado, em números (KuHar por usuário)

`scripts/federated/partition_users.py` particiona o `train.csv` por usuário. No
cross-device, cada cliente é um usuário. Medido na `standardized_view`
(2026-07-24):

| Dataset | usuários | janelas/usuário min/mediana/máx | usuários < batch (64) |
|---|---|---|---|
| KuHar | 57 | **1 / 10 / 103** | **48 (84%)** |
| MotionSense | 17 | 165 / 211 / 244 | 0 |
| UCI | 21 | 99 / 113 / 139 | 0 |
| WISDM | 36 | 235 / 236 / 476 | 0 |
| RealWorld_thigh | 10 | 957 / 1024 / 1204 | 0 |
| RealWorld_waist | 10 | 958 / 1023 / 1201 | 0 |

Detalhe do KuHar por corte de tamanho: **21 usuários (37%) têm ≤ 8 janelas**, 41
(72%) têm ≤ 16, e um usuário tem **1 janela só**. Os 48 usuários abaixo de um
batch concentram **48% das janelas** do dataset.

Os outros 5 datasets não têm o problema: o menor usuário de cada um já supera o
batch 64. **O KuHar é o único caso** — mas é o dataset com **mais usuários** da
coleção, então não dá para simplesmente ignorá-lo no eixo "cross-device" sem
justificar.

## 2. Por que SSL quebra onde o supervisionado não quebra

A diferença não é "poucas amostras para aprender". É a **granularidade do
objetivo de perda**:

| | Downstream supervisionado (few-shot) | Pré-treino SSL |
|---|---|---|
| Perda | CrossEntropy — **por amostra** | Barlow Twins / NT-Xent — **definida sobre o batch** |
| Cliente mínimo | ~1–2 amostras (2 pelo BatchNorm) | **1 batch** (o objetivo não existe abaixo dele) |
| 6 janelas (1-shot × 6 classes) | 6 termos de CE → treina | matriz/negativos degeneram ou nem montam |

No nosso downstream, `subsampled_train_loader` (`scripts/common.py`) **não** passa
`drop_last`, então 6 janelas viram um batch de 6 e o CrossEntropy soma 6 termos —
funciona. Já a perda SSL agrega estatísticas **ao longo do batch** (correlação
entre dimensões, ou pares positivo/negativo); com um batch minúsculo ela vira
ruído, e com 1 amostra ela é indefinida (variância 0 → divisão por zero).

**Consequência para o FedAvg**: um cliente de 10 janelas ainda entra na média
ponderada com `n_k=10` de 1.392 (**0,7%** da rodada). Mesmo que "funcionasse",
o sinal que ele agrega é desprezível — investir engenharia para acomodá-lo compra
pouco.

## 3. Isso vale para qualquer loss SSL? (SimCLR, Barlow Twins, BBT, NT-Xent, TS-TCC, BYOL)

Depende de **como a loss usa o batch**. Há três regimes:

### 3.1 Contrastivas com negativos no batch — **as mais frágeis**
SimCLR / NT-Xent / TS-TCC contrastam cada amostra contra os **negativos do
próprio batch**. Menos amostras = menos negativos = sinal pior; a literatura
documenta que o SimCLR perde ~4 pp já ao cair para batch 256, e é "heavily
dependent on the number of negative samples" (Barlow Twins paper §; survey de
batch em contrastivo). No limite non-IID em que um cliente só tem 1 classe, **os
negativos viram falsos-negativos** — o objetivo continua conceitualmente válido,
mas degrada (FedSC).

- **Nosso TF-C** usa `NTXentLoss_poly` com uma **máscara de tamanho fixo
  `2·batch × 2·batch`** construída no `__init__`
  (`minerva/losses/ntxent_loss_poly.py:67-72`; o `forward` usa `2·batch` em toda
  parte). Um batch parcial **não encaixa na máscara** → por isso o pré-treino
  TF-C exige `drop_last=True` (`scripts/ssl/pretrain_tfc.py`). Efeito no
  cross-device: um cliente com menos de 64 janelas produz **zero batches → zero
  passos de gradiente**, e devolve os pesos globais intactos. É um **no-op
  silencioso**, não um erro — a mesma família de falha do `PYTHONPATH` do Ray que
  já mordeu o projeto (acurácia no acaso, sem exceção).

### 3.2 Não-contrastivas por correlação — **depende da variante**
Barlow Twins clássico calcula a matriz de correlação cruzada **entre as `d`
dimensões do embedding** (matriz `d×d`), normalizada ao longo do batch. Por isso é
**robusto a batch pequeno** — "performance almost unaffected for a batch as small
as 256", ao contrário do SimCLR (Zbontar et al., 2021). *Mas robusto não é
imune*: a normalização ainda estima média/variância **ao longo do batch**, então
batch 1 tem variância 0 → NaN.

- **⚠️ Nosso LFR NÃO usa o Barlow Twins clássico.** Ele usa a **Batch-wise Barlow
  Twins (BBT)**, a variante proposta no próprio paper do LFR
  (`BatchWiseBarlowTwinLoss`, default do minerva —
  `minerva/models/ssl/lfr.py:81`). No BBT a matriz de similaridade é **`m×m` com
  `m` = tamanho do batch**, e não `d×d` (Sui et al., ICLR 2024, §método: *"their
  cosine similarity is an m × m matrix with m the batch size, whereas in Barlow
  Twins it is a d(k) × d(k) matrix"*). Ou seja: **o BBT reintroduz exatamente a
  dependência de batch que o Barlow Twins clássico havia eliminado.** Com 10
  janelas, a matriz do objetivo é 10×10 (posto ≤ 10) — degenerada; com 1 janela, é
  1×1 e o termo off-diagonal (redundância) nem existe. **A robustez a batch
  pequeno que costuma ser atribuída ao "Barlow Twins" NÃO se aplica ao nosso LFR.**

### 3.3 Não-contrastivas por par positivo — **as mais robustas a batch**
BYOL / SimSiam não têm negativos nem estatística de batch na perda: o alvo é a
similaridade de cosseno **por par de vistas de uma mesma amostra**. Em princípio
o objetivo está definido para batch 1 (a única trava restante é o BatchNorm do
encoder, que precisa de >1). É **por isso** que boa parte da literatura de F-SSL
cross-device escolhe BYOL/SimSiam (§4).

**Resumo da §3** — o "cliente mínimo = um batch" vale para os **dois** métodos do
nosso escopo (TF-C contrastivo e LFR/BBT batch-wise), e é *pior* no nosso caso do
que a fama do "Barlow Twins robusto" sugeriria, porque usamos a variante BBT. Só
seria contornado trocando de família de objetivo (BYOL/SimSiam ou reconstrução
mascarada), o que está fora do escopo atual.

## 4. Como a literatura de F-SSL lida com isso

A comunidade de F-SSL conhece o problema, mas quase sempre o enquadra como
**degradação de qualidade** sob não-IID, não como **piso de viabilidade** (o
cliente não consegue nem um passo). As linhas de ataque:

1. **Diagnóstico teórico — o objetivo global do FSSL ≠ soma dos locais.** Com
   objetivo contrastivo, cada amostra só contrasta contra os negativos do próprio
   cliente; sob não-IID isso enviesa o FedAvg. É a motivação explícita de FedSC e
   da família FedU (Zhuang). *(FedSC, ICML 2024; FedEMA/FedU, ICLR 2022.)*

2. **Compartilhar estatísticas para recompor negativos entre clientes.** **FedSC**
   (ICML 2024) faz clientes trocarem **matrizes de correlação das representações**
   além dos pesos, habilitando *inter-client contrast* — ataca diretamente a falta
   de negativos/diversidade local (o nosso problema de batch pequeno). É o único
   método **com garantia** e aplica DP às estatísticas trocadas. *(atenção: mais
   comunicação e uma superfície de privacidade nova.)*

3. **Trocar de objetivo para um que não dependa de negativos/batch.** **FedEMA /
   Divergence-aware FedSSL** (Zhuang et al., ICLR 2022) constrói sobre **BYOL**
   (não-contrastivo) justamente para evitar a dependência de negativos, e mostra
   que o *stop-gradient* nem sempre é necessário no FSSL e que reter conhecimento
   local ajuda no não-IID. Avaliações em imagem médica federada (Yan/C10;
   2303.05556) confirmam que SSL **não-contrastivo / mascarado** é mais robusto que
   contrastivo sob dado limitado e heterogêneo.

4. **Reformular a tarefa como clustering global.** **Orchestra** (Lubana et al.,
   ICML 2022) faz clustering consistente entre clientes, projetado para ser robusto
   a variação de **nº de clientes, taxa de participação e épocas locais** — ou
   seja, o regime cross-device com muitos clientes pequenos.

5. **HAR especificamente.** **UniHAR** (Xu et al., MobiCom 2023 — nosso competidor
   mais próximo) usa **LIMU-BERT (reconstrução mascarada, por-amostra)** +
   augmentation informada pela física do IMU e integra FL; **Saeed et al.**
   (IEEE IoT-J 2021) fazem FedSSL com **scalogram-signal correspondence**
   (classificação binária alinhado/desalinhado). Ambos assumem clientes com dado
   suficiente e **nenhum isola o piso de batch por cliente**.

### 4.1 O que cada método vizinho realmente pré-treina (verificado no PDF, 2026-07-24)

| Paper | Objetivo SSL | "Batch-hungry"? | Como escapa do piso de batch | Datasets |
|---|---|---|---|---|
| **UniHAR** (MobiCom'23) | LIMU-BERT = **reconstrução mascarada** (+ aug. física) | Não (perda por-amostra) | Por construção do objetivo | UCI, HHAR, MotionSense, Shoaib |
| **Saeed** (IoT-J'21) | Scalogram-signal correspondence = **classif. binária** | Quase não (1 pos + negativos de fora do batch) | Objetivo + **shards IID** (ver abaixo); batch federado **12** | Sleep-EDF, **HHAR (9)**, **MobiAct (61)**, WiFi-CSI, WESAD |
| **FedSC** (ICML'24) | **Spectral Contrastive** | **Sim** | **Batch grande (512/256)** + dado local enorme (10k/5k/2.5k) + compartilha matrizes de correlação | SVHN, CIFAR-10, CIFAR-100 (imagem) |

**A literatura de F-SSL evita o piso de batch por construção, não por acaso**:
- **FedSC** roda **cross-silo com dado local enorme e batch 512** — a prova de
  convergência tem termo de erro que só some com "large batch size B". Resolve o
  **skew de rótulo** (contraste inter-cliente via matrizes de correlação), não o
  batch pequeno; assume o oposto do KuHar.
- **Saeed** é quase uma confissão do nosso ponto: ele **não particiona por
  usuário** — cita textualmente *"we randomly divide the training set into
  multiple subsets (representing each client)… due to fewer users in existing
  datasets. This choice might result in a decentralized IID dataset… does not
  suffer from extreme heterogeneity."* Ou seja, foge do cross-device real com
  **shards IID**, e ainda assim só usa batch 12 porque o pretext é binário.
- **UniHAR** usa reconstrução mascarada (por-amostra); o piso nem se coloca.

**Lacuna (candidata a contribuição, a afirmar com cautela):** a literatura ou
(a) troca para objetivo por-amostra/binário/BYOL, ou (b) foge do cross-device
real com shards IID e batch grande. **Ninguém roda um objetivo contrastivo
batch-hungry (SimCLR/NT-Xent/BBT) sobre partição real por usuário com clientes
minúsculos** — exatamente onde o nosso TF-C + LFR/BBT sobre KuHar-por-usuário cai.
O achado não é "a qualidade degrada" (isso a literatura já sabe), e sim um **piso
de viabilidade**: com objetivo definido sobre o batch, um participante real com
< 1 batch **não produz nenhum gradiente** e falha em silêncio no FedAvg.
Enquadrar como *critério de elegibilidade de cliente* e **reportar a taxa de
exclusão** (KuHar: 48/57 = 84% dos usuários, 48% das janelas) é o recorte pouco
explorado.

> **Conferência de ineditismo (2026-07-24).** Os três itens antes bloqueados
> foram lidos: **B1** (da Luz, benchmark IEEE Access'26) é **centralizado** — zero
> conteúdo federado/piso de batch; **B5** (survey de SSL-HAR, **Logacjov** IMWUT'24
> — não Haresamudram, ver `docs/papers/README.md`) é survey **centralizado**,
> "federated" aparece 1× só em referência; **C8** (FedST/FedOST, ACM MM'24) é FL
> **supervisionado** (pFL) de classificação de TS com foco em heterogeneidade de
> *feature*, batch 128 fixo, sem discutir piso de batch. **Nenhum dos três
> antecipa o achado.** Ressalva permanente: nenhuma varredura de literatura é
> exaustiva; reafirmar contra B6/C8-FedOST se surgirem.

## 5. Implicações práticas para o nosso projeto

1. **Instrumentar o simulador FedAvg-SSL com um `assert` alto**: falhar
   explicitamente quando um cliente não formar ≥ 1 batch, trocando o no-op
   silencioso do TF-C por erro visível. Vale o mesmo espírito do assert F2 já
   previsto em `client.py` e da lição do `PYTHONPATH`/Ray.
2. **Plano de execução acordado (2026-07-24): implementar primeiro IGNORANDO o
   KuHar.** Os 3 experimentos de controle (RW_thigh, MotionSense,
   RW_thigh+MotionSense) não o incluem, então o pipeline cross-device pode ser
   construído e validado sem tocar no problema do batch. **Mas o problema NÃO fica
   resolvido — apenas adiado**: depois de o pipeline estar de pé, precisamos
   endereçar o piso de batch de alguma forma (o KuHar é o caso real que o expõe, e
   é o dataset com mais usuários da coleção). **A forma de endereçar fica em aberto
   — discussão deliberadamente adiada, sem proposta neste documento.**
3. **Decisão D-K revista** (ver `analise_domain_shift.md`): agrupar usuários em
   super-clientes **destrói** o non-IID que se queria estudar (o skew de rótulo do
   KuHar cai de 0.539 para 0.068). Se o KuHar entrar, deve ser como **estudo do
   limite de viabilidade** (cliente = usuário, com limiar de elegibilidade
   declarado e sensibilidade a batch 16/8), não como super-clientes fictícios.
4. **Escolha de método sob esta luz**: TF-C (contrastivo) e LFR/BBT (batch-wise)
   são ambos sensíveis; se a agenda futura quiser cross-device com clientes
   pequenos de verdade, um objetivo estilo BYOL/mascarado seria estruturalmente
   mais adequado — anotado como extensão, fora do escopo atual.

## 6. Proveniência e reprodução

- Números da §1: `datasets/DAGHAR/standardized_view/*/train.csv`, coluna `user`;
  reprodutível com o snippet da sabatina (agrupar por `user`, contar janelas e
  classes). `scripts/federated/partition_users.py` gera o manifest equivalente.
- Fatos de código (§3): `minerva/models/ssl/lfr.py:81`
  (`BatchWiseBarlowTwinLoss` default), `minerva/losses/ntxent_loss_poly.py:67-72`
  (máscara `2·batch` fixa), `scripts/ssl/pretrain_tfc.py` (`drop_last=True`),
  `scripts/ssl/pretrain_lfr.py` (`drop_last=False`), `scripts/common.py`
  (`BATCH_SIZE=64`; `subsampled_train_loader` sem `drop_last`).

## 7. Referências

1. Y. Sui et al., "Self-supervised Representation Learning From Random Data
   Projectors" (**LFR**, com a **Batch-wise Barlow Twins**), ICLR 2024.
   arXiv:2310.07756. *(Fonte do "m×m com m=batch" do BBT.)*
2. J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny, "Barlow Twins: Self-Supervised
   Learning via Redundancy Reduction", ICML 2021. *(Matriz d×d; robustez a batch
   pequeno do BT clássico — que NÃO se aplica ao BBT.)*
3. X. Zhang et al., "Self-Supervised Contrastive Pre-Training for Time Series via
   Time-Frequency Consistency" (**TF-C**), NeurIPS 2022. arXiv:2206.08496.
4. S. Jing, A. Yu, S. Zhang, S. Zhang, "FedSC: Provable Federated Self-supervised
   Learning with Spectral Contrastive Objective over Non-i.i.d. Data", ICML 2024.
   arXiv:2405.03949. *(Compartilhar matrizes de correlação p/ contraste
   inter-cliente; global ≠ soma dos locais.)*
5. W. Zhuang, Y. Wen, S. Zhang, "Divergence-aware Federated Self-Supervised
   Learning" (**FedEMA/FedU**, sobre BYOL), ICLR 2022. arXiv:2204.04385.
6. E. S. Lubana et al., "Orchestra: Unsupervised Federated Learning via Globally
   Consistent Clustering", ICML 2022.
7. H. Xu, P. Zhou, R. Tan, M. Li, "Practically Adopting Human Activity
   Recognition" (**UniHAR**), ACM MobiCom 2023. *(SSL = LIMU-BERT, reconstrução
   mascarada; datasets UCI/HHAR/MotionSense/Shoaib.)*
8. A. Saeed, F. D. Salim, T. Ozcelebi, J. Lukkien, "Federated Self-Supervised
   Learning of Multisensor Representations for Embedded Intelligence", IEEE IoT-J
   8(2):1030-1040, 2021. arXiv:2007.13018. *(Scalogram-signal correspondence;
   batch federado 12; **shards IID aleatórios, não por usuário** — assumido no
   próprio texto.)*
9. Yan et al., "Label-Efficient Self-Supervised Federated Learning for Tackling
   Data Heterogeneity in Medical Imaging" (contrastivo vs mascarado em FL),
   arXiv:2205.08576 / IEEE TMI 2023.
10. A. Logacjov, "Self-supervised Learning for Accelerometer-based Human Activity
    Recognition: A Survey" (**B5**), ACM IMWUT 8(4):149, 2024.
    DOI 10.1145/3699767 (CC-BY). *(Survey centralizado; sem seção de FSSL.)*
11. G. P. C. P. da Luz et al., "Benchmarking Encoders and Self-Supervised Learning
    for Smartphone-Based HAR" (**B1**), IEEE Access 2026.
    DOI 10.1109/ACCESS.2026.3669412 (gold OA; H.IAAC/Unicamp). *(Protocolo e
    baselines centralizados.)*
12. C. Wu, H. Wang, X. Zhang et al., "Spatio-temporal Heterogeneous Federated
    Learning for Time Series Classification with Multi-view Orthogonal Training"
    (**C8 / FedST**), ACM MM 2024. DOI 10.1145/3664647.3680733. *(FL supervisionado
    pFL; heterogeneidade de feature; batch 128; não é pré-treino SSL.)*
