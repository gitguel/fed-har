# `docs/` — índice

Compactado em **2026-07-27**: 13 documentos viraram 6; o mapa de experimentos entrou
em 2026-08-05 e o Wilcoxon por par em 2026-08-06, somando 8. O que foi superado está em
[`_arquivo/`](_arquivo/) com um cabeçalho dizendo por que morreu e quem o
substituiu — nada foi apagado.

## O que ler pra quê

| Documento | Use quando precisar de… |
|---|---|
| [`dados_daghar.md`](dados_daghar.md) | **fatos** dos datasets: posição do sensor, classes por dataset, usuários, janelas por usuário, skews. É a fonte única — os outros docs citam este. |
| [`mapa_experimentos.md`](mapa_experimentos.md) | **o que foi rodado**: as 4 federações e as 6 células de avaliação, as duas fases (pré-treino/fine-tuning), contagem total de runs e avaliações, e todo hiperparâmetro com o motivo da escolha. Comece por aqui quando se perder entre os braços. |
| [`resultados.md`](resultados.md) | o que já foi **medido**: transfer centralizado, `comb2target`, o skyline SL-`combined`, e o preliminar federado cross-silo. |
| [`plano_fedssl.md`](plano_fedssl.md) | o **próximo passo**: desenho fatorial cross-device, o piso de batch, a implementação e a ordem de execução. |
| [`estado_da_arte.md`](estado_da_arte.md) | **posicionamento**: estado da arte de F-SSL, verificação de ineditismo, forças/fraquezas da contribuição, lista de leitura. |
| [`metodo_e_auditoria.md`](metodo_e_auditoria.md) | material da **seção de método**: hiperparâmetros vs benchmark, desvios deliberados, achados da auditoria de código. |
| [`wilcoxon_pares.md`](wilcoxon_pares.md) | **o que dá para afirmar com 4 seeds**: o que é o Wilcoxon pareado, quais testes o benchmark de fato faz (e qual ele não faz), a conferência da réplica par a par, e quais pares SSL×encoder vencem o supervisionado no centralizado, no federado e nas federações mistas. |
| [`papers/README.md`](papers/README.md) | quais PDFs estão baixados nesta máquina (os PDFs não são versionados). |

**Registros de sessão** (`sessao_<AAAA-MM-DD>_<assunto>.md`) não são docs vivos:
valem para os commits que citam e envelhecem. O mais recente é
[`sessao_2026-07-28_grade_cross_device.md`](sessao_2026-07-28_grade_cross_device.md)
— fecha o baseline FedAvg cross-device. Para retomar o trabalho, comece por
[`prompt_proxima_sessao.md`](prompt_proxima_sessao.md).

Apresentações ficam em `apresentacao_<DD_MM>/`, uma pasta por apresentação. **A
pasta de uma apresentação já entregue é registro imutável** — nunca regerar
dentro dela. Assets novos nascem numa pasta nova:

```bash
poetry run python scripts/analysis/build_presentation_assets.py --outdir docs/apresentacao_<DD_MM>
```

## O que já rodou

Não mantemos essa contagem à mão — a versão manual já esteve errada em 6 pontos
simultâneos. Para o estado real:

```bash
poetry run python scripts/analysis/cache_status.py    # cobertura das grades
poetry run python scripts/analysis/dataset_facts.py   # fatos dos datasets
```

Retrato de **2026-07-27** (todas as grades completas, zero NaN):

| Cache | Linhas | Encoders |
|---|---|---|
| `supervised_eval_transfer.csv` | 2.688 | 4 |
| `ssl_lfr_eval_transfer.csv` / `ssl_tfc_eval_transfer.csv` | 5.376 cada | 4 |
| `ssl_{lfr,tfc}_comb2target_eval_transfer.csv` | 4.608 cada | 4 |
| `sl_comb2target_eval_transfer.csv` (skyline) | 4.608 | 4 |
| `federated_eval.csv` (cross-silo, preliminar) | 39.168 | 4 |

Caches legados sem leitor foram para `results/_arquivo/` (ver o README de lá).

## Estado do projeto em uma linha

O eixo centralizado está fechado e medido; o eixo federado **cross-silo** foi
rebaixado a motivação (2026-07-21) e o eixo ativo é **cross-device (clientes =
usuários)**, cujo pré-treino federado (`scripts/ssl/pretrain_fed.py`) já está
implementado — falta a partição multi-dataset por usuário e os experimentos de
controle (`plano_fedssl.md §5`).
