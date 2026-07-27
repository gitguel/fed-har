# `results/_arquivo/` — caches sem leitor

Rodadas antigas mantidas por registro, **fora do pipeline**. Nenhum script,
notebook ou doc lê daqui. Se algo aqui voltar a ser usado, tire da pasta.

| Item | O que é | Por que saiu |
|---|---|---|
| `ssl_lfr_eval_transfer_v0_mlpproj.csv` (4.032 linhas, 3 encoders) + `ssl_lfr_parts_v0_mlpproj/` (84 parciais) | Rodada do LFR com **cabeça de projeção MLP** (v0), anterior aos projetores convolucionais do paper | Substituída pela versão fiel ao benchmark (`ssl_lfr_eval_transfer.csv`, 5.376 linhas, 4 encoders). Zero referências no repo desde então. Arquivada em 2026-07-27. |

Os checkpoints correspondentes seguem em `checkpoints/ssl/lfr_v0_mlpproj/`
(versionados desde antes da regra de ignore — não movidos para não gerar churn).
