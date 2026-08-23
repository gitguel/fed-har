# `results/` — o que é das RQs e o que é anterior a elas

A partir de **2026-08-23** os caches se dividem em dois grupos. A regra é simples:

> **Tudo que vai ser reportado nas RQ1/RQ2 vive em [`rqs/`](rqs/).**
> Nada fora de `rqs/` entra em tabela ou figura das RQs.

## `rqs/` — a grade do desenho atual

Produzido pelos drivers de `scripts/rqs/`, conforme
[`docs/desenho_experimental.md`](../docs/desenho_experimental.md). 5 federações
(sem KuHar), partição natural, `k ∈ {1,2,4,Full}` por cliente.

| Arquivo | Bloco | Gerador |
|---|---|---|
| `rq1_centralizado.csv` | RQ1, teto de referência in-domain | `run_rq1_centralizado.py` |
| `rq1_federado.csv` (+ `_parts/`) | RQ1, braço federado | `run_rq1_federado.py` |
| `rq2_finetuning.csv` (+ `_parts/`) | RQ2, fine-tuning com pré-treino SSL | `run_rq2.py --fase finetune` |
| `busca_lr.csv` (+ `_parts/`) | busca S1 da LR do cliente | `run_busca_lr.py` |

O pré-treino da RQ2 não gera CSV próprio: grava `rounds.csv` (tempo e bytes) e o
`backbone.ckpt` em `checkpoints/rqs/ssl_fed/`, consumidos pela fase de fine-tuning.

## Caches anteriores às RQs — **não reportar**

Ficam onde estão porque **11 notebooks os leem**. São de grades com outro
protocolo (orçamento `B=192`, KuHar incluído, outras escadas de `n_shots`) e
**não são comparáveis** com `rqs/` célula a célula.

| Arquivo | O que é | Ainda serve para |
|---|---|---|
| `supervised_eval_transfer.csv` | centralizado, transfer entre os 6 datasets | transfer cross-dataset; a evidência de `desenho_experimental.md §7` |
| `fed_cross_device.csv` | FedAvg supervisionado cross-device, `B=192` | o preliminar de 28/07 |
| `fedssl_cross_device.csv`, `fedssl_crossspec.csv` | Fed-SSL cross-device, `B=192` | idem, e a evidência do §7 |
| `federated_eval.csv` | cross-silo | **motivação apenas** (rebaixado em 2026-07-21) |
| `ssl_{lfr,tfc}_eval_transfer.csv`, `*_comb2target_*`, `sl_comb2target_*` | eixo SSL centralizado e `comb2target` | o eixo centralizado, que está fechado |
| `_arquivo/` | caches legados sem leitor | nada — ver o README de lá |

Estado de cobertura: `poetry run python scripts/analysis/cache_status.py`.
