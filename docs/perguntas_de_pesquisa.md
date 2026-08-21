# Perguntas de pesquisa

> Formulação fechada com o orientador (**2026-08-19**), que substitui a versão de
> 14/08 (duas RQs). Três RQs planas, sem substrutura. O material de apoio — as 6 RQs
> do benchmark, o portão FINER, as restrições de medição, a bibliografia dos
> frameworks — está em [`perguntas_de_pesquisa_apoio.md`](perguntas_de_pesquisa_apoio.md).
>
> **Esta formulação é independente da grade rodada até aqui e do desenho
> experimental.** Ela diz o que o trabalho pergunta; como cada RQ será medida vem
> depois, em outro documento, e não deve puxar a pergunta para o que já existe.
>
> Frameworks: **template de objetivo de Wohlin** (cap. 8) para o guarda-chuva e
> **PICOC** (Kitchenham & Charters 2007) para dar comparador explícito a cada RQ.

---

## 1. O objetivo

> **Analisar** a federação do treinamento de modelos de HAR (a transição do
> centralizado para o federado cross-device)
> **com o propósito de** quantificar seu tradeoff e testar se o pré-treino
> auto-supervisionado o reduz — inclusive quando a federação reúne domínios
> diferentes
> **com respeito a** acurácia, custo de comunicação e privacidade
> **do ponto de vista de** quem precisa treinar sem centralizar os dados dos
> usuários, por privacidade — e para quem o treino centralizado é o **teto de
> referência**
> **no contexto de** HAR por IMU de smartphone, em federação cross-device com
> clientes = usuários reais.

### Em uma frase

> **"Quanto custa descentralizar os dados dos usuários? O pré-treino
> auto-supervisionado diminui esse custo — mesmo quando os usuários vêm de
> domínios diferentes?"**

### Privacidade: o que se pode e o que não se pode afirmar

Privacidade entra no tradeoff da RQ1 como **discussão qualitativa**, não como
métrica medida — não há, hoje, cache de nenhuma métrica de privacidade (ex.: curva
acurácia vs. ε de DP-FedAvg).

- **Pode afirmar** — propriedade *arquitetural*, verificável: nenhuma janela bruta
  sai do dispositivo; o que trafega são deltas de parâmetros. O termo correto para
  isso é **minimização de dados**.
- **Não pode afirmar** — que os dados do usuário estão protegidos. Compartilhar
  atualizações de parâmetros não resiste a inversão de gradiente (*Deep Leakage from
  Gradients*, Zhu et al. 2019; *Inverting Gradients*, Geiping et al. 2020). Sem
  privacidade diferencial ou agregação segura, "preserva a privacidade" é derrubável
  com uma citação.
- Curva de acurácia vs. ε (DP-FedAvg) fica nomeada como **trabalho futuro**.

---

## 2. As três RQs (PICOC)

A intervenção **muda a cada uma**: na RQ1 o que se aplica é *federar*; na RQ2 é
*pré-treinar*; na RQ3 é *misturar domínios na federação*. Por isso são três RQs e
não uma com subitens.

### RQ1 — o tradeoff

> **Qual o tradeoff, em acurácia, comunicação e privacidade, de treinar um modelo
> de HAR de forma federada cross-device em vez de centralizada?**

| | |
|---|---|
| **P** | modelos de HAR sobre janelas de IMU de smartphone, treinados de forma **supervisionada, sem pré-treino** |
| **I** | treino **federado cross-device** (clientes = usuários reais) |
| **C** | treino **centralizado** sobre exatamente as mesmas janelas — teto de referência |
| **O** | acurácia de teste (e F1-macro) por regime de rótulos; MB transmitidos; discussão qualitativa de privacidade (minimização de dados vs. o que não se pode afirmar) |
| **C** | HAR por IMU de smartphone, em federação cross-device com usuários reais |

**H:** acurácia federada < centralizada; o custo em comunicação é estritamente
positivo (o centralizado não transmite nada).

### RQ2 — o efeito do pré-treino auto-supervisionado

> **O pré-treino auto-supervisionado reduz a perda de desempenho em acurácia do
> modelo? E o quanto ele aumenta o custo de comunicação?**

| | |
|---|---|
| **P** | os mesmos modelos e a mesma federação da RQ1 |
| **I** | **pré-treino auto-supervisionado** antes do treino federado |
| **C** | a **mesma federação sem pré-treino** — exatamente o braço medido na RQ1 |
| **O** | (a) a perda de acurácia da RQ1, recalculada sob pré-treino; (b) os MB **adicionais** transmitidos pela fase de pré-treino |
| **C** | idem RQ1 |

**H:** a perda de acurácia é menor com pré-treino; e o pré-treino **adiciona**
comunicação, por ser uma fase federada a mais. A pergunta é a **razão entre as
duas** — quantos pontos de acurácia cada MB extra compra.

### RQ3 — o impacto de domínios diferentes na federação

> **Qual o impacto de termos diferentes domínios em uma federação? O pré-treino
> auto-supervisionado reduz esse impacto? Qual o custo em comunicação?**

| | |
|---|---|
| **P** | os mesmos modelos e o mesmo treino federado das RQ1/RQ2 |
| **I** | federação cross-device com clientes de **domínios diferentes** (datasets/dispositivos distintos reunidos no mesmo pool), com e sem pré-treino auto-supervisionado |
| **C** | a federação **de domínio único** já medida na RQ1 (sem pré-treino) e na RQ2 (com pré-treino) |
| **O** | (a) diferença de acurácia entre federação de domínio único e federação multi-domínio; (b) o quanto o pré-treino auto-supervisionado reduz essa diferença; (c) MB adicionais de comunicação envolvidos |
| **C** | HAR por IMU de smartphone, em federação cross-device com usuários reais de domínios diferentes |

**H:** a federação multi-domínio piora a acurácia em relação à de domínio único
(RQ1); o pré-treino auto-supervisionado reduz parte dessa piora, a um custo de
comunicação adicional.

**A RQ2 e a RQ3 são as que podem dar errado, e é por isso que são a contribuição.**
A RQ1 é régua: "federado ≤ centralizado" é esperado por todos, e reportá-la é
calibração. Já RQ2 e RQ3 têm partes que podem apontar em direções opostas — o SSL
pode recuperar acurácia e ainda assim não se pagar em comunicação; a
heterogeneidade de domínio pode doer mais (ou menos) do que a intuição sugere. Uma
RQ cuja resposta esperada não é "sim" é uma RQ de verdade.
