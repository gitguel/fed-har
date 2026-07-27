> **📦 ARQUIVADO em 2026-07-27.** Referência operacional interna (snapshot de 2026-06-30). **O §2 e o checklist §8 estão desatualizados em ~6 números** (falam em 96 runs/3 encoders/504 linhas; hoje são 128 runs/4 encoders/2.688 linhas). Para o estado real use `../README.md` e `scripts/analysis/cache_status.py`. Mantido pelo log de decisões §6 e pela validação LFR v1 vs paper §9.
>
> Índice dos documentos vivos: `../README.md`.

---

# Notas Internas — Projeto SSL Federado para HAR

> **⚠️ ATUALIZAÇÃO 2026-07-21 — PIVÔ cross-silo → cross-device.** O escopo oficial
> abaixo fala em **federação cross-silo com 6 clientes (1 por dataset)**; por decisão
> com o orientador (2026-07-21) esse eixo foi **abandonado como desenho e como
> controle**. Os ~8 pp de custo de domain shift medidos no cross-silo viram
> **resultado PRELIMINAR/motivação**. Eixo ativo: **cross-device** (clientes =
> usuários, `partition_users.py`); controle honesto do custo de domain shift =
> **Δ(cross-domain − in-domain)**; baseline supervisionado federado = 3 experimentos
> cross-device (in-domain RW_thigh, in-domain MotionSense, cross-domain
> RW_thigh+MotionSense). Ver `docs/analise_domain_shift.md` e
> `docs/plano_fedssl_simulado.md`.

*Documento de uso interno.* Referência operacional do projeto: o que está oficialmente prometido, o que já foi implementado, o que pretendemos fazer se houver tempo, e o que fica como ideia para expansões futuras. Atualizar conforme o projeto avança.

---

## 1. Visão geral do projeto

**Escopo oficial (3 meses, comprometido).** O plano de trabalho oficial entregue ao HIAAC propõe investigar o uso de modelos pré-treinados via Aprendizado Auto-Supervisionado (SSL) para mitigar os efeitos da heterogeneidade de domínio em federações cross-silo de Reconhecimento de Atividades Humanas (HAR). Duas estratégias serão avaliadas: pré-treino SSL centralizado seguido de finetuning federado, e pré-treino SSL feito no próprio processo federado, seguido de finetuning federado. As técnicas de SSL comprometidas no documento oficial são LFR e TF-C, com três encoders (ResNet-SE-5, CNN-PFF, BiGRU), avaliação por *linear readout*, agregação via FedAvg, e federação cross-silo com 6 clientes (um por dataset do DAGHAR).

**Escopo estendido (ambição completa, sem compromisso).** A visão mais ampla do projeto inclui: matriz expandida de experimentos de transfer learning (pré-treino dataset-a-dataset com avaliação cruzada), uma terceira técnica de SSL (TNC), avaliação adicional via *full finetuning*, integração de Foundation Models para séries temporais (Chronos, MOMENT, TimesFM ou similares), configurações federadas mais complexas (mais clientes, clientes multi-domínio) e estratégias de agregação alternativas ao FedAvg (FedProx, FedBN, etc.). Esses elementos não entram no compromisso oficial mas orientam decisões de implementação para deixar o código preparado para essas extensões.

---

## 2. Estado atual de implementação

*Atualizado em 2026-06-30, refletindo o estado real do código e dos resultados.*

**Implementado e validado**

- **Pipeline de treinamento supervisionado centralizado** com os três encoders (ResNet-SE-5, CNN-PFF, BiGRU) sobre os 6 datasets do DAGHAR. **84/84 treinos completos**: 72 base (3 encoders × 6 datasets × 4 seeds) + 12 do pseudo-dataset `combined` (3 encoders × 4 seeds, treino na união `ConcatDataset` dos 6 sub-datasets). Checkpoints em `checkpoints/supervised/<encoder>/<dataset>/seed<N>/{first,last,best}.ckpt` (28 `best.ckpt` por encoder = 7 fontes × 4 seeds).
- **Avaliação de transfer learning cross-dataset (zero-shot)**: cada modelo treinado numa fonte é avaliado por inferência direta (modelo + classificador congelados) no `test.csv` dos 6 alvos. Resultado cacheado em `results/supervised_eval_transfer.csv` com **504 linhas** = 3 encoders × 7 fontes (6 datasets + `combined`) × 4 seeds × 6 alvos. **Métricas: acurácia (`test_acc`) e F1-macro (`test_f1_macro`)** — F1-macro adicionado em 2026-06-08. `combined` entra só como fonte extra, nunca como alvo.
- **Infraestrutura de código** (reorganizada em 2026-06-08 em subpastas por etapa): `scripts/common.py` (DataModule, constantes `DATASETS`/`SEEDS`/`NUM_CLASSES`, Trainer/callbacks), `scripts/supervised/train_{resnetse5,cnnpff,rnn,combined}.py` (+ docs `*.md`), e `scripts/eval_transfer.py` (avaliação transfer com acc+F1, incremental, `--force` recalcula tudo). Pastas `scripts/ssl/` e `scripts/federated/` criadas como placeholders (com README do plano) para o trabalho seguinte.
- **Notebook de visualização** `notebooks/supervised_training_runner.ipynb`: agora **apenas lê** o cache `results/supervised_eval_transfer.csv` (a avaliação migrou para `scripts/eval_transfer.py`) e plota tabelas (acurácia + F1-macro), heatmaps de transfer e t-SNE. Roda end-to-end sem erros.
- **Blocos de construção SSL disponíveis (não integrados ainda)**: a lib `minerva` vendorizada já contém implementações de **LFR, TF-C e TNC** (`minerva/models/ssl/{lfr,tfc,tnc}.py`), além de SimCLR, BYOL, Barlow Twins, CPC, etc., e o transform de FFT do TF-C (`minerva/transforms/tfc.py`). Ainda **não há script/notebook do projeto** aplicando esses métodos aos encoders.
- **Federação supervisionada FedAvg — grade completa rodada no cluster (CONCLUÍDA).** `flwr` 1.31.0 + `ray` 2.55.1 (declarados no `pyproject.toml`). Pipeline em `scripts/federated/` (`partitions.py`, `client.py`, `server.py`, `run_federated.py`, `run_all.py`): simulação Flower, 6 clientes cross-silo, avaliação centralizada por domínio (acc + F1-macro) **e custo de comunicação (uplink/downlink em bytes)**. A grade de **96 runs** (8 cenários × 3 encoders × 4 seeds, R=50 rodadas FedAvg) foi paralelizada em N GPUs e executada no cluster (8× TITAN Xp). Saída consolidada em `results/federated_eval.csv` (**29.376 linhas**, 96/96 combos com `round==50`) + parciais versionados em `results/federated_parts/`. Visualização em `notebooks/federated_avaliation.ipynb`. **Domain shift confirmado**: cenário 2 (IID global) ≈ 0,74 acc > cenário 1 (non-IID por domínio) ≈ 0,66; ablação intra-domínio (cenários 3–8) ≈ 0,45–0,53.

**Pendente para o escopo oficial**

- Implementação/aplicação de **LFR** sobre os três encoders no pipeline do projeto (código-base existe na Minerva).
- Implementação/aplicação de **TF-C** sobre os três encoders (código-base + FFT existem na Minerva; validar pipeline de dados).
- Pipeline de ***linear readout*** sobre encoders pré-treinados via SSL.
- Pipeline de **pré-treino SSL centralizado + finetuning federado** (Experimento 2): juntar o SSL acima com o pipeline federado já pronto.
- Pipeline de **pré-treino SSL federado + finetuning federado** (Experimento 3, FedAvg-SSL) — componente de maior risco técnico.

**Resultados já obtidos**

- **Baselines supervisionados + transfer 7×6 (zero-shot), acurácia + F1-macro**: 504 medições consolidadas em `results/supervised_eval_transfer.csv`, com visualização em `notebooks/centralized_supervised_avaliation.ipynb`. Cobre, para cada encoder, a diagonal (in-domain: treino e teste no mesmo dataset) e o off-diagonal (transferência cross-dataset), além da linha do modelo generalista `combined`. Médias globais: acurácia ≈ 0,539, F1-macro ≈ 0,455 (o gap acc–F1 reflete o desbalanceamento de classes do HAR, justificando o F1-macro). Referência centralizada contra a qual os cenários SSL e federado são comparados.
- **Grade federada FedAvg supervisionada (96 runs, baseline federado do Exp. 2)**: `results/federated_eval.csv` (29.376 linhas). Estabelece o baseline federado supervisionado e quantifica o efeito do *domain shift* (cenário 1 vs 2) e o custo da federação sem heterogeneidade (ablação 3–8). É a referência federada contra a qual os cenários SSL serão comparados.

---

## 3. Plano oficial dos 3 meses (resumo)

Espelho compacto do documento oficial, para consulta rápida.

| Mês | Foco | Entregável |
|---|---|---|
| 1 | Revisão de literatura + validação centralizada de LFR e TF-C (Experimento 1). | Relatório técnico interno. |
| 2 | Pré-treino SSL centralizado + finetuning federado, com baseline supervisionado como referência (Experimento 2). | Relatório técnico interno. |
| 3 | Pré-treino SSL federado + finetuning federado (Experimento 3). Análise comparativa + redação e submissão de artigo. | Artigo submetido a conferência. |

**Configurações fixas**: DAGHAR (6 datasets), 6 clientes cross-silo (1 dataset por cliente), FedAvg, *linear readout*, métricas (acurácia, F1-macro, custo de comunicação).

---

## 4. Extensões garantidas se houver tempo (prioridade alta)

Lista priorizada de coisas que entrariam imediatamente se a execução estiver à frente do cronograma:

1. **Full finetuning** como alternativa ao *linear readout* — adiciona uma comparação importante e geralmente é citada como ablação em papers de SSL.
2. **TNC como terceira técnica de SSL** — complementa a comparação entre famílias contrastivas e aumenta a robustez da análise.
3. **Matriz expandida de transfer learning**: rodar pré-treino dataset-a-dataset (cenário original com 7 experimentos: 6 pré-treinos individuais + 1 agregado, com avaliação cruzada nos 6 destinos). Permite mapear quais datasets são bons "doadores" de representação e quais não são.
4. **Mais rodadas de comunicação e variação de hiperparâmetros** (número de épocas locais, learning rate, batch size) — útil para a robustez do artigo final.

---

## 5. Trabalho futuro / ideias de expansão (sem prioridade definida)

Ideias para iterações posteriores do projeto, possíveis capítulos de dissertação ou trabalhos paralelos:

- **Foundation Models para séries temporais**: Chronos, MOMENT, TimesFM, Lag-Llama. Possíveis estratégias de integração: (a) encoder congelado + cabeça de classificação federada, (b) finetuning federado leve via LoRA ou adapters, (c) comparação direta com SSL pré-treinado no DAGHAR. A maioria desses modelos foi treinada com foco em forecasting, então a adaptação para classificação de HAR exige cuidado.
- **Configurações federadas mais complexas**: federação com mais clientes (split de cada dataset em subclientes para chegar a 30–100 clientes); clientes multi-domínio (cada cliente com mistura de 2+ datasets), criando heterogeneidade *intra-cliente* além da *inter-cliente*.
- **Outras estratégias de agregação**: FedProx (mitigação de drift de cliente), FedBN (preservação de estatísticas locais de BatchNorm — particularmente interessante para SSL com encoders contendo BN), SCAFFOLD, agregação personalizada.
- **Outros métodos SSL**: SimCLR adaptado a séries temporais, TS2Vec, TS-TCC, masked autoencoders temporais (TimeMAE, PatchTST self-supervised).
- **Diversificação da avaliação downstream**: além de HAR, considerar outras tarefas de classificação de séries temporais para testar a generalização das representações aprendidas.
- **Análise de privacidade**: medir vazamento de informação via inversão de gradiente nas duas estratégias (centralizada vs. federada), conectando com a justificativa de privacidade do FL.

---

## 6. Decisões metodológicas registradas

Log das principais decisões tomadas durante o planejamento, com justificativa. Útil para defender escolhas em conversas com o Allan e na redação do artigo.

- **Foundation Models fora do escopo oficial.** Risco alto de não conseguir entregar em 3 meses; ficaria como promessa não cumprida. Mantemos o termo guarda-chuva "modelos pré-treinados" na narrativa oficial para preservar a possibilidade de inclusão futura sem comprometimento explícito.
- **TF-C escolhido em vez de TNC.** Implementação mais simples (estrutura próxima de SimCLR com duas visões, sem teste de estacionariedade), codebase oficial bem mantido (`mims-harvard/TFC-pretraining`), avaliado originalmente em HAR e proposto justamente para o problema de transferência entre domínios distintos de séries temporais.
- **LFR como segundo (e principal) método.** Adaptável a diferentes tipos de dados, não depende de aumentações específicas de domínio — característica relevante para séries temporais, em que aumentações são difíceis de definir.
- **FedAvg ingênuo como agregação.** Baseline mais simples e padrão da literatura. Variações (FedProx, FedBN) ficam como extensão.
- **Linear readout como avaliação padrão.** Barato, padrão na literatura SSL, permite rodar muitas variantes rápido. Full finetuning fica como ablação se houver tempo.
- **6 clientes cross-silo, 1 dataset por cliente.** Cria heterogeneidade de domínio "por construção", sem necessidade de modelar artificialmente via Dirichlet (que faria mais sentido para label shift do que para domain shift).
- **Acurácia + F1-macro.** F1-macro é importante porque os datasets de HAR costumam ter classes desbalanceadas.
- **Custo de comunicação restrito a uplink/downlink/total.** Métricas mais fáceis de coletar e mais informativas para a comparação entre cenários. Tempo de treino e latência foram deixados de fora por dependerem do ambiente físico de execução.
- **(2026-07-13, orientador) Pré-treino federado SEM Flower — simulação exata de FedAvg.** Como o pré-treino não usa seleção de clientes/stragglers (full participation), a federação é simulada num loop Python que reusa o pipeline SSL centralizado validado (gates vs benchmark) e faz a média ponderada dos state_dicts por rodada. Ganhos: reuso total do código validado, coerência DPP/projetores do LFR por construção, resume barato, re-agregação post-hoc de combinações de domínios (modo one-shot) e ablação R×E de graça. O Flower permanece no finetuning federado (comparabilidade com o baseline já medido). Design: `docs/plano_fedssl_simulado.md`; supersede as Fases 1–5 de `docs/plano_experimento3_fedssl.md`.
- **(2026-07-13, orientador) Cenário cross-device por usuário no pré-treino.** 1 cliente por usuário (coluna `user` do DAGHAR; splits já são user-disjuntos ⇒ eval intocado). Realismo + análise in-domain de colaboração. Restrição medida: KuHar tem mediana de 10 janelas/usuário (48/57 usuários < batch 64) ⇒ agrupar em 6 super-clientes (decisão D-K do design doc).

---

## 7. Riscos e pontos de atenção

- **Mês 3 apertado.** Concentra Experimento 3 + análise comparativa + redação e submissão de artigo. Recomendação: começar a redigir o esqueleto do artigo no final do Mês 2, em paralelo aos experimentos.
- **Risco de FedAvg-SSL não convergir bem.** O pré-treino SSL federado é o componente mais arriscado tecnicamente — métodos contrastivos podem sofrer com a divergência entre encoders locais entre rodadas. Se isso ocorrer, opções: (a) reduzir frequência de agregação, (b) testar FedBN sobre o encoder, (c) reportar como achado negativo no artigo (também é resultado válido e publicável).
- **Custo computacional da matriz.** Mesmo enxuta (2 SSL × 3 encoders × 2 cenários federados + Exp 1), são várias rodagens. Vale rodar primeiro o pipeline completo com configurações reduzidas (poucas épocas, pouco dado) para garantir que tudo funciona ponta a ponta antes de fazer as rodagens "de verdade".
- **TF-C exige FFT no preprocessamento.** Não é difícil, mas é uma etapa adicional no pipeline de dados — vale validar cedo se a implementação da Minerva já cobre isso ou se precisa ser adicionada.
- **Integração Minerva + Flower.** Não temos certeza se o casamento entre as duas bibliotecas é trivial. Recomendação: fazer um *spike* técnico nos primeiros dias do Mês 1 para validar a integração antes de mergulhar nas implementações dos métodos SSL.

---

**Próximos passos imediatos**

1. *Spike* técnico de integração Minerva + Flower no início do Mês 1 (item de maior risco/incerteza).
2. Estender a avaliação centralizada com **F1-macro** (hoje só temos acurácia) — barato e fecha a métrica oficial do baseline.
3. Aplicar **LFR** aos três encoders reusando `minerva/models/ssl/lfr.py`, seguido de *linear readout*.
4. Validar título oficial do documento entregue ao HIAAC (das três sugestões: "Aprendizado Auto-Supervisionado Federado para Mitigação de Heterogeneidade de Domínio em HAR" é a minha recomendada).

---

## 8. Checklist de andamento

*Snapshot 2026-06-30. Legenda: ✅ feito e validado · 🟡 parcial/em andamento · ⬜ não iniciado.*

### Escopo oficial (3 meses, comprometido)

| Item | Status | Observações |
|---|---|---|
| Pipeline supervisionado centralizado (3 encoders × 6 datasets × 4 seeds) | ✅ | 72 treinos base completos; checkpoints + scripts + docs |
| Pseudo-dataset `combined` (união dos 6, 3 encoders × 4 seeds) | ✅ | 12 treinos; entra como fonte extra no transfer |
| Avaliação transfer cross-dataset zero-shot | ✅ | 504 linhas em `results/supervised_eval_transfer.csv` via `scripts/eval_transfer.py` |
| Métrica acurácia **+ F1-macro** na avaliação | ✅ | F1-macro adicionado em 2026-06-08 (`test_f1_macro`) |
| Notebook de visualização dos resultados | ✅ | `centralized_supervised_avaliation.ipynb` (centralizado) + `federated_avaliation.ipynb` (federado); só leem cache |
| Reorganização do código (`scripts/{supervised,ssl,federated}/`) | ✅ | Subpastas por etapa; placeholders SSL/federado com README |
| Custo de comunicação (uplink/downlink/total) | ✅ | Coletado na grade federada (`uplink_bytes`/`downlink_bytes` por rodada) |
| LFR aplicado aos 3 encoders | ✅ | `scripts/ssl/pretrain_lfr.py` (backbone LFR; 3 enc × 7 fontes × 4 seeds = 84 backbones) |
| TF-C aplicado aos 3 encoders | ⬜ | Código-base + FFT na Minerva; reusa `scripts/ssl/downstream_eval.py` sem mudanças |
| Pipeline de *linear readout* sobre encoders SSL | ✅ | `scripts/ssl/downstream_eval.py` (protocolo `linear`) → `results/ssl_lfr_eval_transfer.csv` |
| Avaliação SSL em regimes de dados (1/10/100/100% samples-per-class) | ✅ | Eixo `n_shots` compartilhado (SL e SSL); `common.few_shot_indices`/`subsampled_train_loader` |
| Baseline supervisionado nos regimes de dados (transfer 7×6) | 🟡 | Código pronto (`--shots all` em `train_*`/`eval_transfer.py`); falta rodar a grade |
| Notebook SSL (comparação SL vs SSL, data-efficiency) | ✅ | `notebooks/ssl_lfr_avaliation.ipynb` (só lê caches) |
| *Spike* técnico Minerva + Flower | ✅ | FedAvg supervisionado em `scripts/federated/` (Flower 1.31 + Ray) |
| Integração Flower — federação cross-silo (6 clientes, FedAvg) | ✅ | Grade de 96 runs (8 cenários × 3 enc. × 4 seeds, R=50) rodada no cluster |
| Pré-treino SSL centralizado + finetuning federado (Exp. 2) | 🟡 | Lado federado pronto; falta a parte SSL |
| Baseline federado supervisionado (Exp. 2) | ✅ | `results/federated_eval.csv` (29.376 linhas); domain shift confirmado |
| Pré-treino SSL federado + finetuning federado (Exp. 3) | ⬜ | Componente de maior risco técnico |
| Análise comparativa + redação/submissão do artigo (Mês 3) | ⬜ | — |

### Extensões garantidas se houver tempo (prioridade alta)

| Item | Status | Observações |
|---|---|---|
| *Full finetuning* como alternativa ao *linear readout* | ✅ | `scripts/ssl/downstream_eval.py` (protocolo `finetune`), avaliado nos 4 regimes de dados |
| TNC como terceira técnica de SSL | ⬜ | Código-base existe em `minerva/models/ssl/tnc.py` |
| Matriz expandida de transfer (pré-treino dataset-a-dataset, 7 exp.) | 🟡 | Transfer supervisionado zero-shot 7×6 já feito; falta versão SSL/pré-treino |
| Mais rodadas de comunicação + varredura de hiperparâmetros | ⬜ | — |

### Trabalho futuro (sem prioridade definida)

| Item | Status |
|---|---|
| Foundation Models p/ séries temporais (Chronos, MOMENT, TimesFM, Lag-Llama) | ⬜ |
| Configurações federadas complexas (30–100 clientes, clientes multi-domínio) | ⬜ |
| Estratégias de agregação alternativas (FedProx, FedBN, SCAFFOLD) | ⬜ |
| Outros métodos SSL (SimCLR-TS, TS2Vec, TS-TCC, TimeMAE, PatchTST) | ⬜ |
| Diversificação da avaliação downstream (além de HAR) | ⬜ |
| Análise de privacidade (inversão de gradiente) | ⬜ |

## 9. Validação da grade LFR v1 contra o paper de benchmark (2026-07-06)

A grade LFR v1 (config oficial; 3 encoders × 7 fontes × 4 seeds × 2 protocolos
× 4 regimes) foi comparada célula a célula com os resultados publicados pelo
paper de referência (da Luz et al., IEEE Access 2026), usando o
`benchmarks/scripts/performance_data.json` do repo oficial
[H-IAAC/benchmarking-encoders-ssl-har](https://github.com/H-IAAC/benchmarking-encoders-ssl-har).
Auditoria de hiperparâmetros: **14/14 itens idênticos** aos YAMLs/overrides
oficiais (projetores/preditores, DPP `num_targets=6`, Adam 3e-4/wd 3e-4/betas
(0.9, 0.99), 600 épocas sem ES, batch 64, backbones, cabeça MLP `[enc,128,6]`,
lr 1e-4 nos dois protocolos, ES paciência 50 + best.ckpt, freeze via
`requires_grad`). Análise completa (5 figuras):
[artifact lfr-v1-analise-completa](https://claude.ai/code/artifact/07996207-a97d-4c02-90c4-f65d6060749f).

### Nossos resultados vs paper (LFR, in-domain)

Acurácia média dos 6 datasets DAGHAR, formato **nós / paper**, por
amostras-por-classe. "Nós" = média de 4 seeds; "paper" = média reportada de
3 seeds (a coluna 100% do paper corresponde ao spc `1000` do JSON).

| Encoder | Protocolo | 1 | 10 | 100 | 100% |
|---|---|---|---|---|---|
| ResNet-SE-5 | full finetuning | 54.6 / 52.4 | 69.4 / 70.8 | 78.4 / 75.8 | 80.9 / 79.8 |
| ResNet-SE-5 | freeze / linear | 38.3 / 33.8 | 39.1 / 37.1 | 60.0 / 60.2 | 71.6 / 70.7 |
| CNN-PFF | full finetuning | 35.2 / 37.9 | 54.7 / 55.5 | 72.5 / 71.5 | 75.7 / 74.0 |
| CNN-PFF | freeze / linear | 35.2 / 36.1 | 49.8 / 48.6 | 67.9 / 72.3 | 72.5 / 73.4 |
| RNN | full finetuning | 41.2 / 41.6 | 58.1 / 58.7 | 69.7 / 69.2 | 73.0 / 71.3 |
| RNN | freeze / linear | 41.8 / 38.7 | 50.4 / 50.2 | 67.9 / 66.1 | 72.6 / 72.1 |

Viés por encoder×protocolo entre −1.2 e +1.8 pp; desvio absoluto médio
(célula a célula, dataset × regime) entre 2.1 e 4.6 pp — sem discrepância
sistemática, e dentro do desvio-padrão que o próprio paper reporta nas células
de few-shot (até ±18 pp entre seeds).

**Conclusões que replicamos do paper**: (i) LFR só ajuda de forma consistente
a **RNN** (paper: +18.1 pp vs supervisionado @10-shot full-ft; nós: +17 pp
in-domain em F1); (ii) ResNet-SE-5 fica ~neutro no full finetuning;
(iii) CNN-PFF é misto; (iv) o linear readout do ResNet-SE-5 é fraco por
natureza do método (paper: −32 pp @10-shot; nós: −20 pp) — não é bug da nossa
implementação.

**Desvios conhecidos do protocolo oficial** (não são hiperparâmetros):
pré-treino na `standardized_view` train+val em vez da view
`rodrigues_2024_datasets` (não balanceada, mais dado sem rótulo) — único
desvio material; 4 seeds (0–3, afetando init + subsample) vs 3 runs com seed
de dados fixa 42; grade de regimes {1, 10, 100, 100%} ⊂ paper
{1, 5, 10, 25, 50, 100, 200, 100%}; nossa avaliação estende para a matriz de
transfer 7×6 + fonte `combined` (paper é só in-domain).

### Panorama do benchmark além do LFR (base p/ escolhas futuras)

Δ(técnica − supervisionado) em pp de acurácia, full finetuning, média dos 6
datasets (do `performance_data.json`; formato @10-shot / @100%):

| Backbone | LFR | TF-C | TNC | DIET |
|---|---|---|---|---|
| ResNet-SE-5 | +1.5 / +0.3 | +0.4 / +4.2 | +0.3 / −1.3 | −2.4 / +1.2 |
| CNN-PFF | +0.1 / −3.7 | **+21.7 / +8.4** | −1.6 / +0.3 | −2.9 / −1.0 |
| RNN | +18.1 / +2.2 | **+28.9 / +14.6** | +1.5 / −1.8 | +3.0 / +1.8 |
| IMU Transformer | +1.1 / +5.8 | −7.4 / +4.9 | −1.5 / −1.1 | −15.3 / −16.7 |
| ResNet-1D | +1.4 / +3.4 | +1.8 / +7.4 | −0.9 / +0.7 | −4.0 / +3.5 |
| TS-TCC Encoder | **+5.0 / +5.2** | +8.5 / +3.7 | +0.7 / −1.4 | −5.2 / +3.1 |
| TS2Vec Encoder | −4.3 / +2.8 | +6.9 / +5.0 | −0.5 / +0.4 | −20.2 / +1.1 |

Leituras principais: **TF-C é a técnica mais forte do benchmark** (positiva em
quase todos os backbones @100%; TF-C + CNN-PFF é a melhor célula absoluta do
paper: 77.0% @10-shot / 86.1% @100%); **TNC é ~neutro** e **DIET é negativo em
few-shot** — nenhum dos dois justifica entrar no escopo; o **TS-TCC Encoder**
é o único backbone além da RNN em que o LFR ajuda de forma consistente; o
**TS2Vec Encoder** tem o melhor teto absoluto @100% (81–86% em todas as
técnicas).

## 10. Plano de implementação — fechamento da parte centralizada (2026-07-06)

Objetivo: adicionar o **encoder TS-TCC** (`tstcc` = `HARSCnnEncoder`, dim 2304,
o `tfc_harcnn`/`lfr_default` do benchmark) como 4º encoder em todo o pipeline
(supervisionado, LFR, federado) e implementar o **TF-C** como 2ª técnica SSL
nos 4 encoders. Configs oficiais já levantadas: TF-C pré-treina **100 épocas
sem ES** (vs 600 do LFR), lr 3e-4, batch 64, `TFC_Backbone` com
time/frequency encoders gêmeos e projeção p/ `single_encoding_size=128`;
downstream = `SimpleSupervisedModel` com cabeça MLP `[256, 128, 6]` sobre o
concat tempo+freq, lr 1e-4, mesmos protocolos freeze/full.

| # | Etapa | Entregável | Grade a rodar |
|---|---|---|---|
| 0 | *Spike* TF-C (minerva `TFC_Model`/`TFC_Backbone`/`TFC_Transforms`) | perguntas de integração respondidas (FFT no forward, formato do ckpt, VRAM) | 1 combo, épocas reduzidas |
| 1 | Encoder `tstcc` no pipeline supervisionado (`train_tstcc.py` + registro em `common.BEST_LR`, `eval_transfer.py`, `run_all_shots.py`, `train_combined.py`) | baseline SL do 4º encoder | 7 fontes × 4 seeds × 4 regimes = 112 treinos + transfer 7×6 |
| 2 | LFR no `tstcc` (reusa `pretrain_lfr.py`/`downstream_eval.py` via `encoders.py`; validar VRAM: 60 preditores 2304² ≈ 318M params) | LFR completo nos 4 encoders | 28 pré-treinos (600 ép.) + 28 downstream |
| 3 | TF-C nos 4 encoders (`pretrain_tfc.py` + `--method tfc` no downstream/run_all; cache `results/ssl_tfc_eval_transfer.csv`) | 2ª técnica SSL do escopo oficial | 112 pré-treinos (100 ép.) + 112 downstream |
| 4 | `tstcc` no baseline federado (registro em `federated/client.py`) | grade federada com 4 encoders | 8 cenários × 4 seeds = 32 runs (R=50) |
| 5 | Notebooks + validação vs paper (o `performance_data.json` cobre TF-C e o backbone TS-TCC → repetir a validação de MAE/viés da Seção 9) | notebooks atualizados, checklist, commit | — |

Ordem: 0 → 1 → {2, 4 em paralelo} → 3 → 5 (o spike 0 pode rodar em paralelo
com 1). Tudo no cluster Dl-16 (torch 2.5.1+cu118), via tmux + `run_all.py`
de cada etapa. Antes de começar: **commitar o estado atual** (grade LFR v1 +
notebook + Seções 9–10, além da grade federada de junho ainda não commitada).

> **Plano detalhado e sequencial (2026-07-06)**: ver
> **`docs/plano_implementacao_tstcc_tfc.md`** — 10 fases com critérios de
> aceite (gates de validação contra o `performance_data.json` do paper),
> comandos prontos, checklist de registro do encoder e estimativas de custo
> ancoradas nas grades já medidas. É a versão executável desta seção.