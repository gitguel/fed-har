# Pitch — reuniões semanais do grupo

*Pitch curto (~5 min) na rodada de pitches das reuniões semanais do grupo do
prof. Allan. Deck fechado em **2026-08-21**: `pitch_reuniao_semanal.pdf`.*

## O contexto

- Grupo de ~**16 alunos**; cada um faz um pitch breve da própria pesquisa.
- O grupo está montando uma **planilha de nomes × tópicos** para depois formar
  **sub-grupos de 2–3 alunos** com temas relacionados.
- Há outros orientandos em **foundation models para séries temporais / dados
  sequenciais**, e o público tem noção prévia de **SSL** e **HAR** — o pitch não
  traduz o tema nem se defende de estar deslocado.
- Os tópicos que registrei na planilha: `Federated SSL` · `Domain Shift` ·
  `Data Efficiency on the Edge` · `On Device Pretrain`.

## O escopo

**Sem resultados.** Os experimentos rodados até aqui vão ser redesenhados para
responder às RQs de 19/08 ([`../perguntas_de_pesquisa.md`](../perguntas_de_pesquisa.md)).
Apresentar número da grade antiga criaria compromisso com um desenho que está
mudando — o pitch é de **direção**, não de entrega.

**Sem detalhe de dataset.** Nada de listar as 6 bases do DAGHAR nem as 6 classes
de atividade; "quando os clientes vêm de domínios diferentes" carrega o que
precisa ser carregado.

## O deck

Google Slides, 16:9, exportado em **19 páginas** — são **10 slides**, e as páginas
a mais são as animações de build. Quatro seções, cada uma aberta por um divisor
preto de tela cheia.

| Página(s) | Slide | Conteúdo |
|---|---|---|
| 1 | Capa | "Pitch sobre temas" · Grupo de Pesquisa do prof. Allan · Miguel Francisco |
| 2 | *divisor* | **O que faço?** |
| 3 | O que faço? | A tese em uma frase — *"Pré-treino Auto-Supervisionado Federado (**Fed-SSL**) para reconhecimento de atividades humanas (**HAR**) com **sensores de smartphone**, quando os clientes vêm de **domínios diferentes**"* — sobre o diagrama do fluxo: clientes de domínios diferentes → pré-treino SSL federado (FedAvg, sem rótulo) → backbone pré-treinado → fine-tuning com poucos rótulos → modelo HAR, com a barreira tracejada marcando que o dado bruto não sai do cliente |
| 4 | *divisor* | **Por que é importante?** |
| 5 | Por que é importante? | Dois quadros lado a lado. **Federated Learning**: preservação da privacidade; diferença de domínios é a regra em aprendizado distribuído. **Self-Supervised Learning**: rotulagem é rara na borda; dado não rotulado é abundante |
| 6 | *divisor* | **Um Breve Background** |
| 7 | Background | Dados de HAR: as atividades do usuário (sentado, em pé, andando, escada, correndo) passando pelo mesmo smartphone com IMU e virando sinal contínuo de 6 canais; no canto, a nota de que a mesma atividade muda com a posição do sensor (cintura / bolso / coxa) |
| 8–11 | Background | Build em quatro tempos: **Federated Learning** (agregação de modelos locais na nuvem) → **Self-Supervised Learning** (pretext task treina o backbone, transfer para a downstream task) → o "**+**" → o balão **Federated Self-Supervised Learning** |
| 12 | *divisor* | **O que espero atingir?** |
| 13 | Callback | Repete a tese e o diagrama do slide 3 antes de abrir os objetivos |
| 14–18 | O que espero atingir? | Build: o objetivo — *"quantificar o tradeoff acurácia × comunicação e dizer se o pré-treino auto-supervisionado o reduz, inclusive quando a federação reúne domínios diferentes"* — e depois as três RQs, uma a uma |
| 19 | Fecho | Obrigado! / Perguntas? |

As três RQs, como aparecem no deck:

- **RQ1** — qual o tradeoff, em acurácia e comunicação, de treinar um modelo de HAR
  de forma federada cross-device em vez de centralizada?
- **RQ2** — o pré-treino auto-supervisionado reduz a perda de acurácia? Quanto
  aumenta o custo de comunicação?
- **RQ3** — qual o impacto de domínios diferentes em uma federação? O SSL reduz
  esse impacto? A que custo de comunicação?

São as RQs de [`../perguntas_de_pesquisa.md`](../perguntas_de_pesquisa.md) reduzidas
a uma linha cada. Uma diferença a registrar: a RQ1 do documento inclui
**privacidade** no tradeoff, como discussão qualitativa; a RQ1 do deck fala só de
acurácia e comunicação.

## Sobre as figuras

As duas figuras autorais do deck — o diagrama de Fed-SSL (slides 3 e 13) e o
diagrama de dados de HAR (slide 7) — foram geradas por scripts matplotlib que
**não estão mais no repositório**; foram removidos junto com os PNG/PDF soltos
depois que o deck ficou pronto. O PDF é o registro. As demais figuras do
background (slides 8–11) são de terceiros.
