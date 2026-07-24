# Biblioteca de leitura — F-SSL para HAR federado

*PDFs da lista de leitura priorizada de
`docs/estado_da_arte_fssl_e_contribuicoes.md` §7.2. Baixados em 2026-07-07
(arXiv + open access). Códigos (A1, B2, C7…) e ondas de leitura são os mesmos
da §7.2. ⚠️ `docs/papers/*.pdf` provavelmente deve entrar no `.gitignore` —
são ~93 MB; confirme antes de commitar (o repo versiona código/CSV/docs, não
binários grandes).*

Ondas: **1** = ler antes de escrever a related work · **2** = antes de
submeter · **3** = blindagem/consulta. Inegociáveis: **B1–B4, C1, C7, C8**.

## Bloco A — Federated Learning (`A_FL/`)

| Código | Onda | Arquivo | Paper |
|---|---|---|---|
| A1 | 1 | `A1_McMahan_FedAvg_2017.pdf` | McMahan+, *Communication-Efficient Learning… (FedAvg)*, AISTATS 2017 |
| A2 | 1 | `A2_Li_FedBN_2021.pdf` | Li+, *FedBN*, ICLR 2021 |
| A3 | 2 | `A3_Li_FedProx_2020.pdf` | Li+, *FedProx*, MLSys 2020 |
| A4 | 2 | `A4_CoPreFL_RethinkingStartingPoint_2024.pdf` | *Rethinking the Starting Point / CoPreFL*, AAAI 2024 |
| A5 | 3 | `A5_Survey_FoundationModels_FL_2025.pdf` | *Foundational Models + FL: Survey*, 2025 |

## Bloco B — Self-Supervised Learning (`B_SSL/`)

| Código | Onda | Arquivo | Paper |
|---|---|---|---|
| B2 | 1 | `B2_Zhang_TFC_2022.pdf` | Zhang+, *TF-C*, NeurIPS 2022 |
| B3 | 1 | `B3_Sui_LFR_2023.pdf` | Sui+, *LFR (Random Data Projectors)*, ICLR 2024 |
| B4 | 1 | `B4_Napoli_DAGHAR_2024.pdf` | Napoli+, *DAGHAR*, Scientific Data 2024 |
| B7 | 3 | `B7_Eldele_TSTCC_2021.pdf` | Eldele+, *TS-TCC*, IJCAI 2021 |
| B8 | 3 | `B8_Yue_TS2Vec_2022.pdf` | Yue+, *TS2Vec*, AAAI 2022 |

## Bloco C — Federated Self-Supervised Learning (`C_FSSL/`)

| Código | Onda | Arquivo | Paper |
|---|---|---|---|
| C1 | 1 | `C1_Zhuang_FedEMA_2022.pdf` | Zhuang+, *FedU/FedEMA (Divergence-aware FedSSL)*, ICLR 2022 |
| C2 | 2 | `C2_Rehman_LDAWA_2023.pdf` | Rehman+, *L-DAWA*, ICCV 2023 |
| C3 | 2 | `C3_Lubana_Orchestra_2022.pdf` | Lubana+, *Orchestra*, ICML 2022 |
| C4 | 2 | `C4_Liao_RethinkingRepresentation_2024.pdf` | Liao+, *Rethinking the Representation in FedU*, CVPR 2024 |
| C5 | 3 | `C5_Han_FedX_2022.pdf` | Han+, *FedX*, ECCV 2022 |
| C6 | 3 | `C6_FedSC_2024.pdf` | *FedSC (spectral contrastive, provable)*, ICML 2024 |
| C7 | 1 | `C7_Xu_UniHAR_2023.pdf` | Xu+, *UniHAR*, MobiCom 2023 — **competidor mais próximo** |
| C9 | 2 | `C9_Saeed_FedSSL_Multisensor_2021.pdf` | Saeed+, *FedSSL of Multisensor Representations*, IEEE IoT-J 2021 |
| C10 | 2 | `C10_Yan_LabelEfficientSSFL_2023.pdf` | Yan+, *Label-Efficient SSFL (masked vs contrastive)*, IEEE TMI 2023 |
| C12 | 3 | `C12_CDFL_2024.pdf` | *CDFL (contrastive + clustering)*, 2024 |
| C13 | 3 | `C13_TimeFFM_2024.pdf` | *Time-FFM (federated TS foundation model)*, NeurIPS 2024 |

## ⚠️ Faltando — exigem acesso institucional (sem versão aberta)

Baixar via VPN/proxy da Unicamp (IEEE Xplore / ACM DL / SpringerLink):

| Código | Onda | Paper | Onde | Nota |
|---|---|---|---|---|
| **B1** | 1 | da Luz+, *Benchmarking Encoders and SSL for HAR*, IEEE Access 2026 | IEEE Xplore, DOI 10.1109/ACCESS.2026.3669412 | **Inegociável** e o mais importante (seu protocolo/baselines). IEEE Access é *gold open access* — o PDF é livre no Xplore, mas o Xplore bloqueia download automatizado. Baixar manual. Checkpoints no [Zenodo 19301058](https://zenodo.org/records/19301058) |
| **B5** | 2 | **Logacjov**, *SSL for Accelerometer-based HAR: A Survey*, ACM IMWUT 2024 | ACM DL, DOI 10.1145/3699767 | Sem arXiv; **CC-BY (OA)**. ⚠️ autor corrigido 2026-07-24: é **Logacjov** (DBLP `journals/imwut/Logacjov24`), não Haresamudram. Baixado → `B5_Logacjov_survey.pdf` |
| **B6** | 2 | Rodrigues da Silva+, *Impact of Pre-training Datasets (CPC)*, BRACIS 2024 | SpringerLink, DOI 10.1007/978-3-031-79035-5_21 | Trabalho do grupo |
| **C8** | 1 | *FedST* (Wu+, ACM MM 2024) + *FedOST* (IEEE TMC 2026) | ACM DL / IEEE Xplore | **Inegociável** — vizinho de tempo-frequência em FL a distinguir. FedST baixado (`C8_FedST.pdf`), DOI 10.1145/3664647.3680733; é **FL supervisionado** (pFL) de classificação de TS, não pré-treino SSL. FedOST (TMC) ainda pendente. |

Se algum desses estiver inacessível mesmo com VPN, peça ao coautor/orientador
(B1, B6 são do próprio grupo H-IAAC).

## Notas de download

- Fonte preferida: arXiv (PDF aberto, resolvido por título via API para evitar
  paper errado). DAGHAR veio do PDF aberto do Springer/Nature.
- Um susto registrado: a busca por título "SSL for HAR" no arXiv devolveu
  **ColloSSL** (paper diferente) em vez do survey do Haresamudram — por isso
  B5 ficou de fora (o survey real é ACM, sem arXiv). Se quiser o ColloSSL como
  leitura extra de SSL-HAR, é [arXiv 2202.00758](https://arxiv.org/abs/2202.00758).
- Script: `scratchpad/dl.py`.
