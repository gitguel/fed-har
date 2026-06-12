# Notas Internas — Projeto SSL Federado para HAR

*Documento de uso interno.* Referência operacional do projeto: o que está oficialmente prometido, o que já foi implementado, o que pretendemos fazer se houver tempo, e o que fica como ideia para expansões futuras. Atualizar conforme o projeto avança.

---

## 1. Visão geral do projeto

**Escopo oficial (3 meses, comprometido).** O plano de trabalho oficial entregue ao HIAAC propõe investigar o uso de modelos pré-treinados via Aprendizado Auto-Supervisionado (SSL) para mitigar os efeitos da heterogeneidade de domínio em federações cross-silo de Reconhecimento de Atividades Humanas (HAR). Duas estratégias serão avaliadas: pré-treino SSL centralizado seguido de finetuning federado, e pré-treino SSL feito no próprio processo federado, seguido de finetuning federado. As técnicas de SSL comprometidas no documento oficial são LFR e TF-C, com três encoders (ResNet-SE-5, CNN-PFF, BiGRU), avaliação por *linear readout*, agregação via FedAvg, e federação cross-silo com 6 clientes (um por dataset do DAGHAR).

**Escopo estendido (ambição completa, sem compromisso).** A visão mais ampla do projeto inclui: matriz expandida de experimentos de transfer learning (pré-treino dataset-a-dataset com avaliação cruzada), uma terceira técnica de SSL (TNC), avaliação adicional via *full finetuning*, integração de Foundation Models para séries temporais (Chronos, MOMENT, TimesFM ou similares), configurações federadas mais complexas (mais clientes, clientes multi-domínio) e estratégias de agregação alternativas ao FedAvg (FedProx, FedBN, etc.). Esses elementos não entram no compromisso oficial mas orientam decisões de implementação para deixar o código preparado para essas extensões.

---

## 2. Estado atual de implementação

*Atualizado em 2026-06-08, refletindo o estado real do código e dos resultados.*

**Implementado e validado**

- **Pipeline de treinamento supervisionado centralizado** com os três encoders (ResNet-SE-5, CNN-PFF, BiGRU) sobre os 6 datasets do DAGHAR. **84/84 treinos completos**: 72 base (3 encoders × 6 datasets × 4 seeds) + 12 do pseudo-dataset `combined` (3 encoders × 4 seeds, treino na união `ConcatDataset` dos 6 sub-datasets). Checkpoints em `checkpoints/supervised/<encoder>/<dataset>/seed<N>/{first,last,best}.ckpt` (28 `best.ckpt` por encoder = 7 fontes × 4 seeds).
- **Avaliação de transfer learning cross-dataset (zero-shot)**: cada modelo treinado numa fonte é avaliado por inferência direta (modelo + classificador congelados) no `test.csv` dos 6 alvos. Resultado cacheado em `results/supervised_eval_transfer.csv` com **504 linhas** = 3 encoders × 7 fontes (6 datasets + `combined`) × 4 seeds × 6 alvos. **Métricas: acurácia (`test_acc`) e F1-macro (`test_f1_macro`)** — F1-macro adicionado em 2026-06-08. `combined` entra só como fonte extra, nunca como alvo.
- **Infraestrutura de código** (reorganizada em 2026-06-08 em subpastas por etapa): `scripts/common.py` (DataModule, constantes `DATASETS`/`SEEDS`/`NUM_CLASSES`, Trainer/callbacks), `scripts/supervised/train_{resnetse5,cnnpff,rnn,combined}.py` (+ docs `*.md`), e `scripts/eval_transfer.py` (avaliação transfer com acc+F1, incremental, `--force` recalcula tudo). Pastas `scripts/ssl/` e `scripts/federated/` criadas como placeholders (com README do plano) para o trabalho seguinte.
- **Notebook de visualização** `notebooks/supervised_training_runner.ipynb`: agora **apenas lê** o cache `results/supervised_eval_transfer.csv` (a avaliação migrou para `scripts/eval_transfer.py`) e plota tabelas (acurácia + F1-macro), heatmaps de transfer e t-SNE. Roda end-to-end sem erros.
- **Blocos de construção SSL disponíveis (não integrados ainda)**: a lib `minerva` vendorizada já contém implementações de **LFR, TF-C e TNC** (`minerva/models/ssl/{lfr,tfc,tnc}.py`), além de SimCLR, BYOL, Barlow Twins, CPC, etc., e o transform de FFT do TF-C (`minerva/transforms/tfc.py`). Ainda **não há script/notebook do projeto** aplicando esses métodos aos encoders.
- **`flwr` 1.31.0 + `ray` 2.55.1 instalados** (declarados no `pyproject.toml`). Integração federada FedAvg supervisionada implementada em `scripts/federated/` (`partitions.py`, `client.py`, `server.py`, `run_federated.py`): simulação Flower, 6 clientes, avaliação centralizada por domínio (acc + F1-macro) e custo de comunicação. Saída em `results/federated_eval.csv`.

**Pendente para o escopo oficial**

- Adicionar **custo de comunicação** às métricas (quando federado). *F1-macro já coletado.*
- Implementação/aplicação de **LFR** sobre os três encoders no pipeline do projeto (código-base existe na Minerva).
- Implementação/aplicação de **TF-C** sobre os três encoders (código-base + FFT existem na Minerva; validar pipeline de dados).
- Pipeline de ***linear readout*** sobre encoders pré-treinados via SSL.
- **Integração com Flower** para federação cross-silo (6 clientes / 6 datasets / FedAvg) — fazer o *spike* técnico Minerva+Flower primeiro.
- Pipeline de **pré-treino SSL federado** (FedAvg-SSL).
- Coleta de **métricas de custo de comunicação** (uplink / downlink / total agregado por rodada).
- **Baseline federado supervisionado** (parte do Experimento 2 do documento oficial). Obs.: o baseline supervisionado *centralizado* já está pronto e serve de referência.

**Resultados já obtidos**

- **Baselines supervisionados + transfer 7×6 (zero-shot), acurácia + F1-macro**: 504 medições consolidadas em `results/supervised_eval_transfer.csv`, com visualização em `notebooks/supervised_training_runner.ipynb`. Cobre, para cada encoder, a diagonal (in-domain: treino e teste no mesmo dataset) e o off-diagonal (transferência cross-dataset), além da linha do modelo generalista `combined`. Médias globais: acurácia ≈ 0,539, F1-macro ≈ 0,455 (o gap acc–F1 reflete o desbalanceamento de classes do HAR, justificando o F1-macro). Esses números são a referência centralizada contra a qual os cenários SSL e federado serão comparados.

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

*Snapshot 2026-06-08. Legenda: ✅ feito e validado · 🟡 parcial/em andamento · ⬜ não iniciado.*

### Escopo oficial (3 meses, comprometido)

| Item | Status | Observações |
|---|---|---|
| Pipeline supervisionado centralizado (3 encoders × 6 datasets × 4 seeds) | ✅ | 72 treinos base completos; checkpoints + scripts + docs |
| Pseudo-dataset `combined` (união dos 6, 3 encoders × 4 seeds) | ✅ | 12 treinos; entra como fonte extra no transfer |
| Avaliação transfer cross-dataset zero-shot | ✅ | 504 linhas em `results/supervised_eval_transfer.csv` via `scripts/eval_transfer.py` |
| Métrica acurácia **+ F1-macro** na avaliação | ✅ | F1-macro adicionado em 2026-06-08 (`test_f1_macro`) |
| Notebook de visualização dos resultados | ✅ | `supervised_training_runner.ipynb` só lê o cache; tabelas acc + F1 |
| Reorganização do código (`scripts/{supervised,ssl,federated}/`) | ✅ | Subpastas por etapa; placeholders SSL/federado com README |
| Custo de comunicação (uplink/downlink/total) | ⬜ | Só faz sentido no cenário federado |
| LFR aplicado aos 3 encoders | ⬜ | Código-base existe em `minerva/models/ssl/lfr.py`; falta integrar |
| TF-C aplicado aos 3 encoders | ⬜ | Código-base + FFT na Minerva; validar pipeline de dados |
| Pipeline de *linear readout* sobre encoders SSL | ⬜ | — |
| *Spike* técnico Minerva + Flower | ✅ | FedAvg supervisionado em `scripts/federated/` (Flower 1.31 + Ray); avaliação por domínio + custo de comunicação |
| Integração Flower — federação cross-silo (6 clientes, FedAvg) | ⬜ | Exp. 2 e 3 |
| Pré-treino SSL centralizado + finetuning federado (Exp. 2) | ⬜ | Depende de SSL + Flower |
| Baseline federado supervisionado (Exp. 2) | ⬜ | Baseline centralizado já pronto como referência |
| Pré-treino SSL federado + finetuning federado (Exp. 3) | ⬜ | Componente de maior risco técnico |
| Análise comparativa + redação/submissão do artigo (Mês 3) | ⬜ | — |

### Extensões garantidas se houver tempo (prioridade alta)

| Item | Status | Observações |
|---|---|---|
| *Full finetuning* como alternativa ao *linear readout* | ⬜ | Ablação clássica em papers de SSL |
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
