# Pitch — reuniões semanais do grupo

*Roteiro de apresentação curta (~5 min) para a rodada de pitches das reuniões
semanais com o orientador. Escrito em 2026-08-19; a data da apresentação ainda
não está fixada. Quando fixar, este roteiro migra para
`docs/apresentacao_<DD_MM>/`.*

## Contexto e objetivo da apresentação

- Grupo de ~**16 alunos**; cada um faz um pitch breve da própria pesquisa.
- O grupo está montando uma **planilha de nomes × tópicos** para depois formar
  **sub-grupos de 2–3 alunos** com temas relacionados.
- Há outros orientandos em **foundation models para séries temporais / dados
  sequenciais**, e o público tem noção prévia de **SSL** e **HAR** — o pitch
  **não precisa** traduzir o tema nem se defender de estar deslocado.
- **O objetivo do pitch é apresentar a própria pesquisa.** Não é recrutar
  colaborador nem propor interseções — a formação dos sub-grupos acontece depois,
  a partir da planilha.

### Meus tópicos na planilha

`Federated SSL` · `Domain Shift` · `Data Efficiency on the Edge` · `On Device Pretrain`

O **slide 5** existe para dar corpo a esses quatro tópicos — é o slide que
conversa diretamente com o que está na planilha.

## Escopo: o que fica de fora, e por quê

**Sem resultados.** Os experimentos rodados até aqui vão ser **redesenhados** para
responder especificamente às RQs atuais (`perguntas_de_pesquisa.md`, formulação
de 2026-08-19). Apresentar número da grade antiga criaria compromisso com um
desenho que está mudando. Este é um pitch de **direção**, não de entrega.

Opcional, uma frase e nenhum número, se quiser sinalizar que o projeto não é
papel: *"o eixo centralizado já está medido; o federado cross-device está em
redesenho para responder exatamente a estas RQs."*

**Sem detalhe de dataset.** Nada de listar as 6 bases do DAGHAR nem as 6 classes
de atividade. Mencionar "bases com posições de sensor diferentes" é útil e cabe
no slide 5; a enumeração não.

---

## A estrutura

Funil: **tema geral → motivação → termos → desafios do meu recorte → RQs**.

| # | Slide | Papel |
|---|---|---|
| 1 | O tema, objetivamente | nomear FL, SSL, HAR, domínio, Fed-SSL |
| 2 | Por que é importante | motivação |
| 3 | O que é SSL | background rápido |
| 4 | Fed-SSL: onde a área está | estado, problemas conhecidos, lacuna |
| 5 | Desafios do meu recorte | os 4 tópicos da planilha |
| 6 | RQs e o que espero atingir | fecho |

### 1 — O tema, objetivamente

**No slide:** uma frase de objetivo, com os termos nomeados:

> Pré-treino auto-supervisionado federado (**Fed-SSL**) para **reconhecimento de
> atividade humana (HAR)** com sensores de smartphone, quando os clientes vêm de
> **domínios diferentes**.

Abaixo, o diagrama: clientes (usuários) → pré-treino SSL federado → fine-tuning →
modelo. Marcar que **o dado bruto nunca sai do cliente**.

Asset pronto: `docs/pitch_assets/fig_pitch_slide1_fedssl.{png,pdf}` (o PDF é
vetorial, melhor para projetar). Regerar com
`poetry run python scripts/analysis/build_pitch_diagram.py --outdir docs/pitch_assets`
— quando a data da apresentação fixar, o asset migra junto com este roteiro para
`docs/apresentacao_<DD_MM>/`.

**Falar:** nomear FL, SSL, HAR e domínio já aqui — quem só prestar atenção a um
slide, presta a este.

### 2 — Por que é importante

**No slide**, dois argumentos:

- **Privacidade** — nenhuma janela bruta sai do dispositivo; o que trafega são
  deltas de parâmetros. O termo é **minimização de dados**.
- **Rótulo é caro na borda, dado não-rotulado é abundante** — quase todo dado
  coletado é descartado por não ter rótulo. O SSL usa esse dado. É o argumento
  que justifica *SSL* federado, e não só FL.

> ⚠️ Dizer **minimização de dados**, nunca "preserva a privacidade".
> Compartilhar atualizações de parâmetros não resiste a inversão de gradiente
> (Zhu et al. 2019; Geiping et al. 2020) — é derrubável com uma citação, e o
> público tem repertório para fazer isso. Ver `perguntas_de_pesquisa.md §1`.

### 3 — O que é SSL

**No slide:** tarefa de **pretexto** (sem rótulo) → treina o **backbone** →
congela/afina → tarefa **alvo** (poucos rótulos). Uma linha: *"técnicas
investigadas: LFR, TF-C, SimCLR"*.

**Falar:** 30–40 s. Com este público é recapitulação, não aula.

> ⚠️ "Investigadas" cobre o SimCLR, que ainda não está implementado (`scripts/ssl/`
> tem `pretrain_lfr.py` e `pretrain_tfc.py`). Só não deixar parecer que os três já
> rodaram.

### 4 — Fed-SSL: onde a área está

**No slide:** técnicas existentes + os problemas conhecidos:

- inconsistência e **desalinhamento dos espaços de representação** entre clientes;
- agregação de encoders treinados sob distribuições locais distintas;
- custo de comunicação de uma fase de treino a mais.

Fechar com **a lacuna**, em uma linha — é ela que justifica o trabalho existir.

### 5 — Desafios do meu recorte

Quatro itens curtos, mapeando 1:1 nos tópicos da planilha:

| Tópico da planilha | No slide |
|---|---|
| **Domain Shift** | clientes com posição de sensor, dispositivo e taxa de coleta (Hz) diferentes |
| **Data Efficiency on the Edge** | todo o dado não-rotulado no pré-treino, poucos rótulos no fine-tuning |
| **On Device Pretrain** | restrições de cliente: batch mínimo viável, Hz de coleta, compute heterogêneo |
| **Federated SSL** | o desalinhamento do slide 4, agravado por clientes que nem compartilham a posição do sensor |

### 6 — RQs e o que espero atingir

**No slide**, as três RQs em uma linha cada (fonte: `perguntas_de_pesquisa.md`):

- **RQ1** — qual o tradeoff, em acurácia, comunicação e privacidade, de treinar
  HAR de forma federada cross-device em vez de centralizada?
- **RQ2** — o pré-treino auto-supervisionado reduz a perda de acurácia? E quanto
  aumenta o custo de comunicação?
- **RQ3** — qual o impacto de domínios diferentes na federação? O SSL reduz esse
  impacto? A que custo de comunicação?

**O que espero atingir:** quantificar o tradeoff **acurácia × comunicação** e
dizer se o pré-treino auto-supervisionado o reduz — inclusive quando a federação
reúne domínios diferentes.

---

## Notas de execução

**O slide 4 é o que estoura o tempo.** SSL do zero + pretexto/alvo/backbone +
três técnicas + panorama de Fed-SSL + problemas de alinhamento + limitações da
literatura é conteúdo de três slides. Por isso o background está **partido em
dois** (3 e 4), com as técnicas reduzidas a uma linha. Se o tempo for 10 min,
cabe folgado; em 5 min, este é o ponto de risco.

**Não repetir o slide 4 no slide 5.** A separação é:

- **slide 4** = o que a literatura de Fed-SSL já tropeça — genérico, qualquer
  modalidade;
- **slide 5** = o que só aparece no meu recorte — HAR, IMU, multi-domínio, borda.

O **desalinhamento de espaços de representação** é a ponte: aparece no 4 como
problema conhecido da área e volta no 5 como *"e no meu caso piora, porque os
clientes não compartilham nem a posição do sensor"*.

**Cortes, se o tempo apertar:** fundir 3 e 4 em um único slide de background
(perdendo as limitações da literatura, que viram uma frase falada). Não cortar o
5 nem o 6 — são o que a planilha e o orientador esperam ver.
