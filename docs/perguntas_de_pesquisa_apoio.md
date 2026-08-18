# Perguntas de pesquisa — material de apoio

> **Documento vivo, não arquivo.** Nasceu em **2026-08-14** do enxugamento de
> [`perguntas_de_pesquisa.md`](perguntas_de_pesquisa.md), que ficou complexo demais e
> foi reduzido ao objetivo + as duas RQs. Tudo o que saiu de lá está aqui, sem
> alteração de conteúdo — e continua **válido**, não superado. Por isso não foi para
> [`_arquivo/`](_arquivo/).
>
> A §3 (restrições de medição) tem uma **armadilha ativa** — o `n_shots` aplicado por
> cliente — que inverteria a conclusão de quem rodar a coluna centralizada
> ingenuamente. Leia-a antes de desenhar qualquer experimento novo.

---

## 1. Por que as RQs mudaram

A grade atual foi desenhada em torno de **Δ de heterogeneidade** (`Δ_custo-do-shift`,
`Δ_valor-estrangeiro`, `Δ_feature-skew` — `plano_fedssl.md §2.2`). Isso é
vocabulário de *replicação*: "o achado do benchmark sobrevive à federação?" — e a
resposta já está medida (`wilcoxon_pares.md §7`: sobrevive, 7/8 de concordância de
sinal). É exatamente o que o orientador chamou de reprodução.

O eixo novo é de **decisão**: quem constrói um sistema HAR sob restrição de
privacidade deve federar, e o que o SSL muda nessa conta.

Enquadramento que reposiciona o trabalho sem refazer nada: a grade do benchmark é
`encoder × técnica × refinamento × fração de rótulo × dataset`. **A nossa é essa
grade com um eixo a mais — a topologia.** Não é réplica; é extensão deliberada do
fatorial.

### As 6 RQs do benchmark, para referência (B1 §I)

Todas são perguntas fatoriais sobre uma grade — cada uma pega um eixo e pergunta
"qual nível é melhor" ou "quanto esse eixo importa". Nenhuma é de mecanismo.

- **RQ1**: What is the best-performing encoder overall in supervised and SSL settings for HAR tasks?
- **RQ2**: How is the performance of encoders affected by the freeze refinement strategy?
- **RQ3**: What has a greater impact on final performance: changing the pre-training technique or the encoder architecture?
- **RQ4**: How does encoder performance vary across different target datasets?
- **RQ5**: What is the impact of the labeled data fraction on the performance of different encoders?
- **RQ6**: What is the minimum amount of labeled data required to achieve near-maximum performance, and what is the gain provided by using SSL in scarse data scenarios?

Duas delas são o modelo formal que seguimos: a **RQ3** é atribuição de variância
(não "qual é melhor", mas "quanto cada fator importa") e a **RQ6** é composta —
magnitude + ganho do SSL na mesma pergunta, que é a forma adotada aqui.

> ⚠️ Estas 6 RQs não estavam em lugar nenhum do repo e o IEEE bloqueia scraping —
> o `B1_daLuz_benchmark.pdf` precisou voltar ao disco para transcrevê-las. Esta é a
> única cópia versionada.

---

## 2. O portão FINER

| | RQ1 | RQ2 |
|---|---|---|
| **F**easible | ✅ comparador roda com código existente e validado; backbones prontos | ⚠️ o mais caro; precisa do seletor por usuário |
| **I**nteresting | ⚠️ `.1` morna, `.2` genuína | ✅ resposta não óbvia dos dois lados |
| **N**ovel | ⚠️ provisório (ver abaixo) | ❌ `.1` é resultado canônico |
| **E**thical | ✅ dados públicos e de-identificados | ✅ idem |
| **R**elevant | ✅ decide o desenho sob restrição de privacidade | ✅ é a decisão que sobra |

> O parágrafo "as metades `.1` são régua, não contribuição" migrou para o documento
> principal (§2), porque é enunciado sobre as RQs e não sobre o portão.

O que sustenta o ineditismo das `.2`: a verificação do repo não encontrou **nenhum**
trabalho federando LFR ou TF-C, nem qualquer *"comparação controlada pré-treino
centralizado vs federado do MESMO método SSL de séries temporais com budget
pareado"* (`estado_da_arte.md §2.1`).

⚠️ **O portão N está provisoriamente aprovado, não aprovado.** O levantamento foi
feito por snippets e abstracts, não leitura integral; o próprio `estado_da_arte.md`
registra que *"nenhuma afirmação de 'primeiro' deve ir ao artigo sem essa
verificação final"* (§6.1 lista como fechar).

---

## 3. Consequências para a medição (fatos, não desenho)

Levantados durante a sabatina de 13/08. **Não constituem desenho experimental
aprovado** — são restrições que qualquer desenho terá de respeitar.

1. **A escada já existe para o pré-treino.** `single:X:10` é 1 cliente com
   *exatamente as mesmas* 1.920 janelas do `device:X:10` — controle centralizado
   pareado, não "isolado". 320 células com os 3 degraus completos.
2. **Achado que motiva a RQ1.2:** no RealWorld_thigh, 5,4× mais dado de pré-treino
   (1.920 → 10.338 janelas) compra só **+1,97 pp (LFR) / +1,28 pp (TF-C)**, enquanto
   federar custa ~1,5 pp. **A perda da federação não é sobre acesso a dado — é sobre
   agregação.** No MotionSense, onde `@full` é +11% de dado, o Δ é ~0 (consistência
   interna).
3. **O agregado mente.** O custo de federar o pré-treino vai de **+5,44 pp**
   (LFR+ResNet-SE no RW: federar *ajuda*) a **−10,96 pp** (LFR+RNN no MS). Sinal
   troca por encoder e por alvo. Nunca citar a média por método.
4. ⚠️ **Armadilha do pareamento de rótulos.** `n_shots` é aplicado **por cliente**
   (`run_cross_device.py:17`). `device:X:10 --shots 5` = 300 janelas rotuladas;
   `single:X:10 --shots 5` = **30**. Rodar a coluna centralizada ingenuamente
   compararia 300 contra 30 rótulos e **inverteria a conclusão**. Correção acordada:
   aplicar o corte few-shot **por usuário** e só então colapsar num cliente — janelas
   idênticas por construção. Não afeta o degrau `full` nem as 320 células já medidas
   (lá o `single` é spec de *pré-treino*, que não usa rótulo).
5. **O braço federado está convergido em R=150.** Argmax da validação na mediana da
   rodada 80 (LFR) / 106 (sup.) / 44 (TF-C), apenas +1,04 pp acima da média das
   rodadas 121–150. Mata de antemão a objeção "seu federado foi sub-treinado".
6. **Pareamento de orçamento:** mesmas janelas, mesmo R e E, seleção por validação
   nos dois braços — compara platô com platô. O centralizado avança mais por rodada
   (acesso sequencial ao dado todo), e isso **é** o que centralizar significa, não um
   confundidor. "Rodadas até o platô" vira resultado secundário.

---

## 4. Em aberto

1. **Fechar a verificação de ineditismo** (`estado_da_arte.md §6.1`) — bloqueia
   qualquer afirmação de "primeiro".
2. **Métrica** (nível M do GQM): acurácia e/ou F1-macro. O foco de qualidade está
   decidido; a métrica não.
3. **Teste estatístico** para as hipóteses `.2` (efeito de interação). A máquina do
   `wilcoxon_pares.py` é reusável, mas o estimando muda.
4. **`uplink_mb` não cobre o pré-treino** (`mapa_experimentos.md §9`) — a segunda
   dimensão do foco de qualidade está incompleta.
5. **Contexto de alcance médio vs 2 domínios.** O objetivo afirma sobre "HAR por IMU
   de smartphone cross-device"; a grade tem 2 domínios. Dívida a pagar em ameaças à
   validade.
6. **Desenho experimental das RQs** — deliberadamente não decidido. Só depois de as
   RQs estarem fechadas com o orientador.

---

## 5. Material para estudar os frameworks

Ordem de leitura sugerida: **Kitchenham & Charters** (é o que fala a língua da
computação) → **Wohlin cap. 8** (o template de objetivo) → **Basili** (a raiz do
GQM) → **Farrugia** (FINER, curto).

### GQM e o template de objetivo (Wohlin)

| recurso | tipo | nota |
|---|---|---|
| [Basili, Caldiera & Rombach, *The Goal Question Metric Approach* (1994)](https://www.cs.umd.edu/users/mvz/handouts/gqm.pdf) | PDF, ~10 pp | A fonte. Curto e direto; é o que define a hierarquia Objetivo → Pergunta → Métrica |
| [Basili, *The Role of Empirical Study in Software Engineering*](https://www.microsoft.com/en-us/research/video/the-role-of-empirical-study-in-software-engineering/) | **vídeo** (Microsoft Research) | Palestra do próprio Basili; contextualiza por que o GQM existe |
| [Basili, slides da mesma palestra (UMD, 2006)](https://www.cs.umd.edu/~basili/presentations/2006/Role%20of%20E%20in%20SE%20Irvine.pdf) | PDF de slides | Versão para folhear rápido |
| Wohlin, Runeson, Höst, Ohlsson, Regnell & Wesslén, *Experimentation in Software Engineering*, Springer, 2ª ed. 2012 | livro | **O template de objetivo está no cap. 8 (scoping).** É o capítulo que importa aqui; o resto do livro é desenho de experimento, útil na fase seguinte |
| van Solingen & Berghout, *The Goal/Question/Metric Method*, McGraw-Hill, 1999 | livro | Guia prático, se o artigo de 1994 ficar abstrato demais |

### PICO / PICOC

| recurso | tipo | nota |
|---|---|---|
| Kitchenham & Charters, *Guidelines for Performing Systematic Literature Reviews in Software Engineering*, EBSE-2007-01 | relatório técnico, livre | **A referência canônica de PICOC em computação.** É daqui que sai a tradução dos elementos do PICOC para termos de CS. Buscar por "EBSE-2007-01" |
| Petticrew & Roberts, *Systematic Reviews in the Social Sciences*, Blackwell, 2006 | livro | Origem do "C" de Context sobre o PICO clínico |
| Richardson, Wilson, Nishikawa & Hayward, *The well-built clinical question*, ACP Journal Club, 1995 | artigo, 3 pp | O PICO original. Vale ler para entender que ele nasceu para *busca*, não para estudo primário |
| [Kitchenham et al., *Systematic literature reviews in SE: a systematic literature review*](https://www.sciencedirect.com/science/article/pii/S2215016122002746) e [Ralph et al., *On the Pragmatic Design of Literature Studies in SE*](https://arxiv.org/pdf/1612.03583) | artigos | Como a comunidade de fato usa (e critica) esses frameworks |
| [Guia de frameworks de pergunta — Western Sydney University](https://subjectguides.library.westernsydney.edu.au/c.php?g=944483&p=6864427) | guia de biblioteca | Tabela comparando PICO, PICOC, PEO, SPIDER, CoCoPop etc. Útil para justificar por que **descartamos** os outros |

### FINER

| recurso | tipo | nota |
|---|---|---|
| Hulley, Cummings, Browner, Grady & Newman, *Designing Clinical Research*, 4ª ed. | livro, **cap. 2** | A fonte do FINER. Só o cap. 2 interessa |
| Farrugia, Petrisor, Farrokhyar & Bhandari, *Research questions, hypotheses and objectives*, Can J Surg 53(4):278-81, 2010 | artigo, 4 pp | **Comece por aqui.** Cobre FINER **e** PICO em quatro páginas; é o resumo mais eficiente que existe |

### Complementares (não são frameworks de pergunta, mas ajudam)

- **Easterbrook, Singer, Storey & Damian**, *Selecting Empirical Methods for Software
  Engineering Research* (in *Guide to Advanced Empirical SE*, Springer 2008) — como
  casar **tipo de pergunta** com **método empírico**. Diretamente útil para a fase
  seguinte (desenho).
- **Shaw**, *Writing Good Software Engineering Research Papers*, ICSE 2003 — taxonomia
  de tipos de pergunta e de validação em computação. Curto e citável.

> ⚠️ **Vídeos são escassos para PICOC e FINER.** O único vídeo verificado nesta lista
> é o do Basili. Para os outros, o material bom é texto — e as buscas por vídeo caem
> em canais de enfermagem baseada em evidências, que ensinam PICO clínico e não a
> adaptação para computação. Se quiser vídeo mesmo assim, busque por "systematic
> literature review software engineering Kitchenham" em vez de "PICOC".

### Descartados, com justificativa

Frameworks de pergunta avaliados e não adotados: **CoCoPop** (desenhado para
prevalência), **SPIDER** (qualitativo), **PEO** (a topologia aqui é *atribuída*, não
observada), **SPICE/ECLIPSE** (política pública).
