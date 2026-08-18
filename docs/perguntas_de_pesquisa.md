# Perguntas de pesquisa

> Formulação do caderno (**2026-08-14**, `foto_caderno.png`), que substitui a versão
> de 13/08. Duas RQs planas, sem substrutura. O material de apoio — as 6 RQs do
> benchmark, o portão FINER, as restrições de medição, a bibliografia dos frameworks
> — está em [`perguntas_de_pesquisa_apoio.md`](perguntas_de_pesquisa_apoio.md).
>
> **Esta formulação é independente da grade rodada até aqui.** Ela diz o que o
> trabalho pergunta; o que já foi medido é assunto do desenho experimental, que vem
> depois e não deve puxar a pergunta para o que já existe.
>
> Frameworks: **template de objetivo de Wohlin** (cap. 8) para o guarda-chuva e
> **PICOC** (Kitchenham & Charters 2007) para dar comparador explícito a cada RQ.

---

## 1. O objetivo

> **Analisar** a federação do treinamento de modelos de HAR (a transição do
> centralizado para o federado cross-device)
> **com o propósito de** quantificar seu custo e testar se o pré-treino
> auto-supervisionado o reduz
> **com respeito a** acurácia e custo de comunicação
> **do ponto de vista de** quem precisa treinar sem centralizar os dados dos
> usuários, por privacidade — e para quem o treino centralizado é o **teto de
> referência**
> **no contexto de** HAR por IMU de smartphone, em federação cross-device com
> clientes = usuários reais.

### Em uma frase

> **"Quanto custa descentralizar os dados dos usuários? O pré-treino
> auto-supervisionado diminui esse custo?"**

### Os três movimentos

1. **Medir quanto custa** federar o treinamento de um modelo de HAR, contra um teto
   centralizado, em prol da privacidade dos dados.
2. **Testar se o pré-treino auto-supervisionado torna essa troca mais barata.**
3. ❓ **Medir quanto se ganha** em relação a não colaborar com ninguém.

**O item 3 está em aberto, deliberadamente.** Ele não tem RQ correspondente na §2.
Incluí-lo acrescenta um comparador (o usuário treinando sozinho) e um terceiro
degrau à escada; deixá-lo fora mantém o trabalho num eixo só. É a pergunta a levar
para o orientador — não uma promessa já assumida:

```
  centralizado   ← teto de referência
       ↑           RQ1: quanto falta? · RQ2: o SSL encurta isso?
  federado
       ┆           ❓ em aberto: vale medir este degrau?
  solo (usuário sozinho)
```

### A inversão que importa

**O objeto é a mudança de paradigma; o SSL é o tratamento.** É o que tira o trabalho
do formato "benchmark com um eixo a mais": o SSL deixa de ser o que se descreve e
passa a ser o que se testa contra o custo da federação.

### Privacidade: o que se pode e o que não se pode afirmar

Privacidade é **premissa declarada com precisão**, não eixo medido.

- **Pode afirmar** — propriedade *arquitetural*, verificável: nenhuma janela bruta
  sai do dispositivo; o que trafega são deltas de parâmetros. O termo correto para
  isso é **minimização de dados**.
- **Não pode afirmar** — que os dados do usuário estão protegidos. Compartilhar
  atualizações de parâmetros não resiste a inversão de gradiente (*Deep Leakage from
  Gradients*, Zhu et al. 2019; *Inverting Gradients*, Geiping et al. 2020). Sem
  privacidade diferencial ou agregação segura, "preserva a privacidade" é derrubável
  com uma citação.
- Curva de acurácia vs ε (DP-FedAvg) fica nomeada como **trabalho futuro**.

---

## 2. As duas RQs (PICOC)

A intervenção **muda entre as duas**: na RQ1 o que se aplica é *federar*; na RQ2 é
*pré-treinar*. Por isso são duas RQs e não uma com subitens — o comparador de cada
uma é outro.

### RQ1 — o custo

> **Qual o custo, em acurácia e em comunicação, de treinar um modelo de HAR de forma
> federada cross-device em vez de centralizada?**

| | |
|---|---|
| **P** | modelos de HAR sobre janelas de IMU de smartphone, treinados de forma **supervisionada, sem pré-treino** |
| **I** | treino **federado cross-device** (clientes = usuários reais) |
| **C** | treino **centralizado** sobre exatamente as mesmas janelas — teto de referência |
| **O** | acurácia de teste (e F1-macro) por regime de rótulos; MB transmitidos |
| **C** | HAR por IMU de smartphone, em federação cross-device com usuários reais |

**H:** acurácia federada < centralizada; o custo em comunicação é estritamente
positivo (o centralizado não transmite nada).

### RQ2 — o efeito do pré-treino auto-supervisionado

> **O pré-treino auto-supervisionado reduz o custo em acurácia? E o quanto ele
> aumenta em comunicação?**

| | |
|---|---|
| **P** | os mesmos modelos e a mesma federação da RQ1 |
| **I** | **pré-treino auto-supervisionado** antes do treino federado |
| **C** | a **mesma federação sem pré-treino** — exatamente o braço medido na RQ1 |
| **O** | (a) o custo da RQ1, recalculado sob pré-treino; (b) os MB **adicionais** transmitidos pela fase de pré-treino |
| **C** | idem RQ1 |

**H:** o custo em acurácia é menor com pré-treino; e o pré-treino **adiciona**
comunicação, por ser uma fase federada a mais. A pergunta é a **razão entre as
duas** — quantos pontos de acurácia cada MB extra compra.

**A RQ2 é a que pode dar errado, e é por isso que ela é a contribuição.** A RQ1 é
régua: "federado ≤ centralizado" é esperado por todos, e reportá-la é calibração.
Já a RQ2 tem duas metades que podem apontar em direções opostas — o SSL pode
recuperar acurácia e ainda assim não se pagar em comunicação. Uma RQ cuja resposta
esperada não é "sim" é uma RQ de verdade.
