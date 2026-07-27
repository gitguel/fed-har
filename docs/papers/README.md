# `docs/papers/` — PDFs da bibliografia

**Os PDFs não são versionados** (`.gitignore`: `docs/papers/**/*.pdf`) — são obras
de terceiros, ~93 MB. Esta pasta é local; num clone novo ela vem só com este
README.

A **lista de leitura completa** (blocos A/B/C, com prioridade e justificativa de
cada item) está em [`../estado_da_arte.md §6.2`](../estado_da_arte.md). Aqui fica
só o registro do que foi de fato baixado e lido.

## Presentes nesta máquina (2026-07-27)

| Arquivo | Referência | Estado |
|---|---|---|
| `B1_daLuz_benchmark.pdf` | G. P. C. P. da Luz et al., *Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based HAR*, IEEE Access 2026. DOI 10.1109/ACCESS.2026.3669412 | ✅ lido integralmente (2026-07-24) — **centralizado**, sem conteúdo federado |
| `B5_Logacjov_survey.pdf` | A. Logacjov, *Self-supervised Learning for Accelerometer-based HAR: A Survey*, ACM IMWUT 8(4):149, 2024. DOI 10.1145/3699767 | ✅ lido integralmente (2026-07-24) — survey **centralizado**; ⚠️ autoria corrigida: **não** é Haresamudram |
| `C8_FedST.pdf` | C. Wu, H. Wang, X. Zhang et al., *Spatio-temporal Heterogeneous FL for Time Series Classification with Multi-view Orthogonal Training* (FedST/FedOST), ACM MM 2024. DOI 10.1145/3664647.3680733 | ✅ lido integralmente (2026-07-24) — FL **supervisionado** (pFL), não SSL |

Os três foram lidos para a conferência de ineditismo do piso de batch: **nenhum
antecipa o achado** (`../estado_da_arte.md §3.4`).

## Convenção de nomes

`<bloco><n>_<primeiro autor>_<apelido>.pdf`, com o código vindo da lista de
leitura (`A` = Federated Learning, `B` = Self-Supervised Learning, `C` =
Federated SSL). Ex.: `B3_Sui_LFR.pdf`.
