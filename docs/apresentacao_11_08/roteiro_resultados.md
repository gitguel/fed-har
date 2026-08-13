# Roteiro da seção de resultados — apresentação 11/08

> Assets gerados por
> [`scripts/analysis/build_fedssl_slides.py`](../../scripts/analysis/build_fedssl_slides.py)
> (figuras 1–6) e
> [`build_fedssl_anexos.py`](../../scripts/analysis/build_fedssl_anexos.py)
> (anexos A–F), ambos com `--outdir docs/apresentacao_11_08`. Cada figura sai em
> **PDF vetorial** (para o slide) e **PNG** (para conferência rápida). Todos os
> números vêm de `results/derived/` — nenhum treino novo.
>
> A seção entra **entre o slide 6** (grades de treino) **e o slide "O que não
> vemos…"**, que passa a ser o último.

---

## A âncora: as hipóteses, como você as formulou

Estas são as quatro afirmações que a seção precisa provar, **no texto original**
(pedido de 11/08). A seção de resultados existe para responder a elas, uma por uma,
e o veredicto de cada uma está em [As hipóteses depois dos resultados](#as-hipóteses-depois-dos-resultados).

> **H1.** Cross domain prejudica a federação SL.
> **H2.** Alguns pares de encoder + SSL ajudam nos cenários cross domain em relação ao SL.
> **H3.** Mostramos quais pares ajudam, quais atrapalham e quais ainda são inconclusivos.
> **H4.** Mostramos isso com rigor estatístico de "mesmo nível" do benchmark.

**Mapa hipótese → figura:**

| | hipótese | figura | slide |
|---|---|---|---|
| **H1** | cross domain prejudica o SL federado | `fig1_custo_domain_shift` | 7 |
| **H2** | alguns pares SSL ajudam no cross | `fig2_ssl_no_cross` | 8 |
| **H3** | quais ajudam / atrapalham / inconclusivos | `fig3_placar_por_par` | 9 |
| **H4** | rigor de mesmo nível do benchmark | `fig4_replica_benchmark` + `fig5_piso_de_poder` | 10–11 |
| **H5** | *(emergente)* qual o custo de federar o **pré-treino** | `fig6_custo_de_federar_pretreino` | 12 |

**Duas observações honestas antes de começar**, porque um orientador atento vai
fazê-las:

- **H3 e H4 não são hipóteses** — são *entregáveis*. H3 é a resolução de H2 em três
  camadas (quem ajuda, quem empata, quem não dá pra dizer) e H4 é a régua com que
  H1–H3 são medidas. Vale enunciá-las assim no slide: fica claro que a seção tem
  **dois** claims empíricos e **duas** promessas de método.
- **H5 não estava na sua lista.** Ela nasceu do Exp. 2 e é a pergunta *anterior* a
  H2: antes de perguntar se o SSL ajuda, é preciso saber se **federar o pré-treino**
  estraga a representação. Ela entrou porque é ela que abre a discussão final.

**Sugestão de slide-âncora (6½):** as quatro linhas do bloco acima, sem números.
Você volta a ele no slide de veredicto (ver a seção final deste roteiro), e a plateia
sabe o tempo inteiro qual pergunta está sendo respondida.

---

## ⚠️ Antes de tudo: a hipótese 1 precisa ser reescrita

Você formulou como *"cross domain prejudica a federação SL"*. **Os dados sustentam
metade disso**, e a metade que não se sustenta é justamente a que um orientador
atento vai atacar primeiro. Existem **duas comparações diferentes** escondidas na
palavra "cross":

| comparação | o que muda | resultado |
|---|---|---|
| **Substituir** — `cross5+5` vs in-domain | 10 clientes nos dois lados; metade vira estrangeira | **−3,72 pp** mediano · n=40 · **p = 1,4e−09** |
| **Acrescentar** — `cross10+10` vs in-domain | 20 clientes contra 10; o dado próprio fica intacto | **−0,55 pp** · n=40 · **p = 0,29 → n.s.** |

E, no `cross10+10`, o regime de **1 rótulo por classe é positivo: +1,1 pp** — dado
estrangeiro *ajuda* quando o rótulo é escasso.

**A formulação que os dados sustentam:**

> *Misturar domínios **a orçamento fixo de clientes** custa caro ao SL federado, e
> **o custo cresce com o orçamento de rótulos** (MotionSense: −4,4 pp no 1-shot →
> −7,8 pp no full). Já **acrescentar** um domínio estrangeiro é praticamente de
> graça — e no regime de 1 rótulo por classe, ajuda.*

Isso é **mais forte** do que a versão original, não mais fraco: vira um resultado
com estrutura (o custo tem uma direção e uma causa) em vez de uma constatação
genérica. E monta o palco para a hipótese 2 — o SSL ajuda mais exatamente onde o
shift dói mais.

Já está registrado em `docs/mapa_experimentos.md` §10.4 que o Δ do `cross10+10` é
*"acrescentar um segundo domínio"*, não *"dobrar o dado"*. A figura 1 mostra os
dois painéis lado a lado exatamente para essa distinção não passar batida.

---

## Slide 7 — `fig1_custo_domain_shift.pdf`

**Título sugerido:** *"O que o domain shift custa — e por que a resposta depende da pergunta"*

**A frase:** "Quando eu troco metade dos meus clientes por clientes de outro
domínio, eu perco quase 4 pontos — e perco **mais** quanto mais rótulo eu tenho.
Quando eu só **acrescento** o outro domínio, não perco nada, e no few-shot eu ganho."

**O que apontar, em ordem:**
1. Painel da esquerda, linha vermelha: −4,4 → −7,8. **O custo cresce com o rótulo.**
2. O `p = 1,4e−09` embaixo do título: não é ruído.
3. Painel da direita: quase tudo em cima do zero, `n.s.`
4. Os dois pontos positivos do 1-shot (+1,8 RW, +0,3 MS).

**Guardrail:** os pontos translúcidos são as médias por encoder. Estão ali de
propósito — mostram que o sinal não vem de um encoder só. Se perguntarem, é a
resposta pronta.

**A pergunta que vem:** *"por que o custo cresce com o rótulo? Não era pra ser o
contrário?"* — É o achado contraintuitivo. A leitura: com muito rótulo o modelo
tem capacidade de se especializar no domínio próprio, e o cliente estrangeiro
passa a atrapalhar essa especialização. Com pouco rótulo, qualquer sinal ajuda.
Isso **contraria** a intuição de que "SSL/dado extra cura shift no regime pobre" —
e é justamente por isso que vale mostrar.

---

## Slide 8 — `fig2_ssl_no_cross.pdf`

**Título sugerido:** *"Dentro das federações mistas, o SSL recupera parte do custo"*

**A frase:** "Nas duas federações cross-domain, TF-C+RNN e TF-C+CNN-PFF batem o
FedAvg supervisionado com placar perfeito de 10/10. O LFR+TSTCC entra na
`cross10+10`."

**O que apontar:** os marcadores **preenchidos** — são os que sobrevivem ao
Bonferroni. E o `+22,0` / `+24,1` do TF-C+RNN, que é o efeito grande da seção.

**Guardrail obrigatório:** com `n = 10` o piso de p depois do Bonferroni é
**0,0156** — *só* um placar de 10/10 passa. Então "não passou" aqui inclui casos
como LFR+TSTCC no `cross5+5` (+2,8 pp, 8/10), que é efeito plausível sem poder
para ser declarado. Diga isso antes de perguntarem; é o que separa cautela de
descuido.

---

## Slide 9 — `fig3_placar_por_par.pdf`

**Título sugerido:** *"Quem ajuda, quem atrapalha, quem ainda não dá pra dizer"*

**A frase:** "Dos 8 pares, 3 vencem no centralizado e 5 no federado. Nenhum par
piora o baseline de forma significante."

**Cuidado com a palavra "atrapalha".** O único Δ negativo é `LFR + cnnpff`
(−0,4 pp no centralizado) e ele é **n.s.** A afirmação honesta é: *"nenhum par
atrapalha de forma detectável; o LFR+CNN-PFF é o único que fica do lado negativo,
e é exatamente o par que **também** aparece negativo no benchmark (62,2% contra
63,4%, Tab. 7 deles)"*. Isso é um ponto forte — a réplica pega até o perdedor.

**A leitura de três camadas** (é isso que responde a hipótese 3):
- **Ajudam:** TF-C+RNN, TF-C+CNN-PFF, LFR+RNN (centralizado); +LFR+TSTCC e
  LFR+ResNet-SE (federado).
- **Empatam:** TF-C+ResNet-SE, LFR+ResNet-SE no centralizado — `+0,2` e `+0,3` pp
  são empate honesto, não "efeito pequeno".
- **Inconclusivos:** TF-C+TSTCC (+1,8 pp, 15/24) — efeito plausível que o nosso `n`
  não resolve. **Isso não é a mesma coisa que empate**, e a distinção vale ser dita.

**Detalhe que impressiona se você mencionar:** no federado passam *mais* pares que
no centralizado, e isso **não é** um efeito da federação — é que a grade federada
tem 30 configurações contra 24. Mais `n`, mais poder.

---

## Slide 10 — `fig4_replica_benchmark.pdf`

**Título sugerido:** *"Por que temos o direito de invocar o benchmark"*

**A frase:** "Não replicamos só o nível do benchmark — replicamos a **estrutura**
do efeito. A ordem dos 8 pares é praticamente idêntica: ρ = 0,98, sinal 8/8."

**O ponto de método:** bater em nível (MAE 1,56 pp contra a Tabela 10) não autoriza
o claim, porque o claim usa o **Δ**. Esta figura compara o Δ deles com o nosso.

**Diga também que somos mais conservadores:** quase todo Δ nosso é *menor* que o
deles. Antecipa a acusação de resultado inflado.

---

## Slide 11 — `fig5_piso_de_poder.pdf`

**Título sugerido:** *"O que 4 seeds compram — e o que não compram"*

**A frase:** "O `n` do teste nunca foi o número de seeds. São 4 seeds em ambos os
casos, mas o poder vem das 24 ou 30 configurações pareadas."

**Este é o slide que preempta a pergunta mais provável do seu orientador**
(*"com 4 seeds dá pra afirmar isso?"*). A resposta tem três partes:

1. As 4 seeds servem para **estabilizar cada célula** antes de parear; o `n` do
   teste é o número de configurações. O benchmark do da Luz tem **3** seeds e
   afirma coisas pela mesma razão.
2. Para claim **agregado**, 4 bastam. Para claim **célula-a-célula**, não — e nem 8
   resolveriam a um custo razoável (`dp(Δ) ≈ 2,5 pp`, IC95 de ±4 pp contra efeito
   mediano de 3 pp).
3. Onde `n` é pequeno, o problema é **estrutural**: com as 5 configurações de uma
   federação única, o piso de p é 0,0625 — acima de α **antes** de qualquer
   correção. Nada ali pode ser significante, nem em princípio. Não é ausência de
   efeito, é ausência de teste.

**Sobre "rigor de mesmo nível do benchmark":** você está **subvendendo**. Conferido
no PDF deles: todo teste que reportam é *encoder contra encoder dentro de um
método*. **Não existe teste de SSL contra supervisionado em lugar nenhum do
paper** — o *"SSL significantly outperform supervised baselines"* é legenda de
tabela, sem teste por trás, e a comparação SL vs SSL deles (Tab. 11) é
melhor-de-24 contra melhor-de-6, com contagem de células verdes. Contagem não é
teste. **O nosso teste é mais forte do que o que eles reportam.** Diga isso — com
essas palavras, que são verificáveis.

---

## Slide 12 — `fig6_custo_de_federar_pretreino.pdf`  (Exp. 2)

**Título sugerido:** *"E quanto custa federar o pré-treino?"*

**A frase:** "Todo o resto da apresentação compara *com* pré-treino contra *sem*.
Esta é a pergunta anterior: **espalhar o pré-treino por 10 clientes piora a
representação?** A resposta é: custa cerca de **1 ponto**. É barato."

**Por que o controle é honesto:** o `single:X` tem **as mesmas 1.920 janelas, dos
mesmos 10 usuários, num cliente só**. Muda uma coisa e só uma — se o dado está
espalhado ou concentrado. O fine-tuning depois é idêntico nos dois braços. É o que
separa *"o ganho federado é menor porque a loss precisa de lote grande"* de
*"é menor porque há menos dado"* — duas explicações com consequências opostas.

**O número:** −0,97 pp mediano, n = 80 configurações, **p = 4,7e−03**. Detectável,
e pequeno.

**O achado de verdade não é o total — é a inversão.** Os dois métodos pagam em
regimes **opostos**, e é isso que a figura mostra:

| | 1 rótulo | 2 | 5 | 10 | full |
|---|---|---|---|---|---|
| **TF-C** | **−5,7** | −0,7 | −1,3 | −0,9 | **−0,0** |
| **LFR** | **+0,4** | −0,1 | −1,4 | −1,5 | **−1,5** |

O TF-C paga quase tudo no **1-shot** e zera no `full`; o LFR não paga nada no
few-shot e paga ~1,5 pp do 5-shot em diante. **Isso é consistente com a teoria:** o
TF-C é contrastivo e depende de negativos — fragmentar em 10 clientes machuca a
representação, e é no regime pobre de rótulo que a representação carrega o
resultado sozinha. Com rótulo farto, o fine-tuning conserta.

**Verifiquei que o padrão por regime é estável** nas quatro combinações de
métrica × regra de rodada (acurácia/F1 × média-das-20-últimas/`argmax(val)`). Já o
**agregado por método não é** — ele media uma inversão e troca de ordem conforme a
convenção. Por isso a figura tem regime no eixo x, e por isso **não diga**
*"o LFR federa melhor que o TF-C"* (nem o contrário): a frase correta é
**"cada um paga num regime diferente"**.

**Se perguntarem por par:** só TF-C + TSTCC sobrevive ao Bonferroni (−1,4 pp, 9/10).
O LFR + RNN tem o maior Δ (−4,5 pp) e **não** passa (8/10, p_bonf = 0,078) — exemplo
perfeito de "efeito plausível sem poder", que amarra direto no slide 11.

**Por que isso importa para a discussão final:** no agregado, federar o pré-treino é
barato (~1 pp) — então **o gargalo geral não é a fragmentação**. Mas a inversão dá
uma recomendação bem mais específica que "é barato":

- **Se o cenário-alvo é few-shot**, o custo do TF-C (−5,7 pp) é real e é exatamente
  o que BYOL/SimSiam/MAE consertariam — os objetivos sem dependência de lote que já
  estão no seu último slide. Aqui a ablação tem alvo claro.
- **Se o cenário-alvo é rótulo farto**, quem paga é o LFR, e o caminho é outro.

Ou seja: a fig. 6 não fecha a discussão, ela **transforma "o que fazer a seguir" numa
pergunta sobre qual regime interessa** — que é a pergunta que você quer fazer ao seu
orientador no slide seguinte.

---

## As hipóteses depois dos resultados

Fecha o círculo: mesma ordem, mesmas quatro afirmações, agora com veredicto. **Duas
foram reescritas, duas se confirmaram — e as duas reescritas ficaram mais fortes,
não mais fracas.**

| | como formulada | veredicto | evidência |
|---|---|---|---|
| **H1** | cross domain prejudica a federação SL | ⚠️ **reescrita** — vale para *substituir*, não para *acrescentar* | −3,72 pp (n=40, p=1,4e−09) vs −0,55 pp (n.s.) |
| **H2** | alguns pares SSL ajudam no cross | ✅ **confirmada** | TF-C+RNN e TF-C+CNN-PFF, 10/10 nas duas federações mistas |
| **H3** | quais ajudam / atrapalham / inconclusivos | ⚠️ **reescrita** — a categoria "atrapalha" saiu **vazia** | 3/8 centralizado · 5/8 federado · 0 pioras significantes |
| **H4** | rigor de mesmo nível do benchmark | ✅ **superada** — o nosso teste é mais forte que o deles | eles não testam SSL vs SL em lugar nenhum |
| **H5** | *(emergente)* custo de federar o pré-treino | 🆕 **respondida, com uma inversão** | −0,97 pp (n=80, p=4,7e−03), mas TF-C paga no 1-shot e LFR no full |

### As formulações que os dados sustentam

**H1 — trocar domínio custa; somar domínio é de graça.**

> *Misturar domínios **a orçamento fixo de clientes** custa −3,72 pp ao SL federado,
> e **o custo cresce com o orçamento de rótulos** (MotionSense: −4,4 pp no 1-shot →
> −7,8 pp no full). Já **acrescentar** um domínio estrangeiro é estatisticamente
> indistinguível de zero — e no regime de 1 rótulo por classe, **ajuda** (+1,1 pp).*

O que mudou: a palavra "cross" escondia duas comparações. A reescrita troca uma
constatação genérica por um resultado **com direção e com causa** — e monta o palco
para H2, porque o shift dói mais exatamente onde o SSL vai recuperar. Detalhe na
seção *"Antes de tudo: a hipótese 1 precisa ser reescrita"*, no início do roteiro.

**H2 — confirmada, e mais específica do que "alguns pares".**

> *Nas duas federações cross-domain, **TF-C + RNN** e **TF-C + CNN-PFF** batem o
> FedAvg supervisionado com **placar perfeito de 10/10** e efeitos grandes
> (+22,0 e +24,1 pp). **LFR + TSTCC** entra na `cross10+10`.*

O que mudou: nada no sentido, só na precisão — "alguns pares" virou nome, placar e
tamanho de efeito. O guardrail continua obrigatório: com n=10 o piso de p pós-Bonferroni
é 0,0156, então **só** 10/10 passa, e "não passou" inclui efeitos plausíveis sem poder.

**H3 — três camadas, e "atrapalha" não é uma delas.**

> *Dos 8 pares, **3 vencem no centralizado e 5 no federado**. **Nenhum par piora o
> baseline de forma detectável**: o único Δ negativo é LFR + CNN-PFF (−0,4 pp,
> n.s.) — e é exatamente o par que **também** aparece negativo no benchmark
> (62,2% contra 63,4%, Tab. 7 deles). Há ainda uma terceira camada, distinta de
> empate: **inconclusivos** (TF-C + TSTCC, +1,8 pp, 15/24).*

O que mudou: você esperava um pódio com perdedores. Não há perdedores — e a réplica
pega até o perdedor *deles*, o que é um ponto a favor, não uma decepção. A camada
"inconclusivo" (efeito plausível, `n` insuficiente) é a que dá honestidade ao placar,
e ela **não** é a mesma coisa que empate.

**H4 — você estava subvendendo.**

> *Não é "mesmo nível": conferido no PDF deles, **todo teste que o benchmark reporta é
> encoder contra encoder dentro de um método**. Não existe teste de SSL contra
> supervisionado em lugar nenhum do paper — o "SSL significantly outperform supervised
> baselines" é legenda de tabela, e a comparação SL vs SSL (Tab. 11 deles) é
> melhor-de-24 contra melhor-de-6 com contagem de células verdes. **Contagem não é
> teste.** Além disso, replicamos a **estrutura** do efeito, não só o nível: ρ = 0,98
> na ordem dos 8 pares, sinal 8/8, e quase todo Δ nosso é **menor** que o deles.*

O que mudou: de "estamos à altura" para "estamos acima" — com uma afirmação
verificável, não retórica. Isso vem com o dever de declarar as duas ressalvas da
seção seguinte; é o que separa um claim forte de um claim inflado.

**H5 — a que você não formulou, e a que abre a discussão.**

> *Espalhar o pré-treino por 10 clientes custa **−0,97 pp** (n=80, p=4,7e−03) — o
> gargalo geral **não é** a fragmentação. Mas o agregado esconde uma **inversão**:
> **TF-C paga quase tudo no 1-shot** (−5,7 pp) e zera no `full`; **o LFR não paga nada
> no few-shot** e paga ~1,5 pp do 5-shot em diante. **Cada método paga num regime
> diferente** — nunca diga que um "federa melhor" que o outro.*

É esta hipótese que transforma o último slide de "o que fazer a seguir" em **"qual
regime nos interessa?"** — que é a pergunta que você quer fazer ao seu orientador.

**Sugestão de slide de veredicto (12½):** a tabela de cinco linhas acima, com os dois
⚠️ em destaque. É o retorno ao slide-âncora 6½ e a rampa natural para a discussão:
duas hipóteses mudaram de forma ao encostar nos dados, e é a mudança — não a
confirmação — que dá o que conversar.

---

## As duas ressalvas que você deve declarar (não esperar que perguntem)

1. **As configurações pareadas não são independentes.** O mesmo dataset aparece em
   4 regimes, o mesmo encoder em 6 datasets. Wilcoxon assume independência,
   Bonferroni não conserta dependência. Os `p` **ordenam evidência** com segurança;
   tratá-los como probabilidades exatas seria exagero. O benchmark tem o mesmo
   problema — não estamos abaixo do padrão da área.
2. **F7 continua em aberto.** A seleção few-shot usa o split de validação **cheio**,
   o que infla os regimes pobres de todas estas células. Como o viés atinge os dois
   braços da diferença, o Δ sofre menos que o nível — mas "menos" não é "nada".
   Está em `docs/metodo_e_auditoria.md` §F7, e já aparece como último item do slide
   "O que não vemos…" (*protocolo de seleção com menos dados de validação*).

---

## Cola de números

| | |
|---|---|
| substituição de domínio (SL) | −3,72 pp · n=40 · p=1,4e−09 |
| adição de domínio (SL) | −0,55 pp · n=40 · p=0,29 (n.s.) · **+1,1 pp no 1-shot** |
| pares que vencem | 3/8 centralizado · 5/8 federado |
| maior efeito | TF-C + RNN: +15,1 pp centralizado · +23,1 pp federado |
| agregado por método | LFR +1,27 / TF-C +6,88 (centr.) · +1,66 / +4,46 (fed.) — todos vencem |
| réplica do benchmark | MAE 1,56 pp em nível · sinal 8/8 · ρ = +0,98 no Δ |
| piso de p | n=5 → 0,0625 · n=10 → 0,0020 (0,0156 pós-Bonferroni) |
| custo de federar o pré-treino | −0,97 pp · n=80 · p=4,7e−03 |
| ↳ a inversão | TF-C: −5,7 pp no 1-shot → −0,0 no full · LFR: +0,4 → −1,5 |

---

## Uma sugestão de ordem

A sequência 1 → 2 → 3 conta uma história ("o shift custa" → "o SSL recupera" →
"eis quais pares"); 4 → 5 é a defesa metodológica; e a 6 abre a discussão final,
porque o resultado dela **reordena a fila de próximos passos** do último slide.

Se o tempo apertar, **a 3, a 5 e a 6** são as que eu manteria: o placar, a régua
que o defende, e a que dá o que discutir. As figuras 1 e 2 são as que mais
dependem de você narrar bem — sem narração viram tabela.

**Os dois slides de texto (6½ âncora e 12½ veredicto) não são o que se corta primeiro.**
Eles custam ~40 s cada e são o que dá forma ao resto: sem eles a seção é uma sequência
de seis figuras; com eles é um argumento que abre com quatro perguntas e fecha com
quatro respostas — duas delas diferentes do que você esperava. Se precisar cortar um,
corte o 6½ e enuncie as hipóteses de viva-voz sobre o slide 6; o 12½ é o que emenda
na discussão final.

---

## Anexos A–F — o material de apoio no fim do deck

Seis figuras que **abrem os agregados** das figuras 1–6 nas dimensões que elas
escondem: encoder, domínio/alvo e regime de rótulos. Não entram na narrativa — ficam
depois do último slide, para quando a pergunta vier.

### Se perguntarem X, vá para o anexo Y

| a pergunta provável | anexo | responde com |
|---|---|---|
| *"qual encoder sofre mais com o shift?"* | **A** | 4 painéis (alvo × comparação), uma linha por encoder |
| *"esse ganho vale nos dois alvos, ou é de um só?"* | **B** | 8 pares × 6 colunas (federação × alvo) |
| *"em que regime de rótulo o par ajuda?"* | **C** | 8 pares × regime, centralizado e federado |
| *"qual domínio responde ao SSL?"* | **D** | 8 pares × 6 datasets (centralizado) |
| *"qual encoder paga o custo de federar?"* | **E** | 2 painéis (TF-C, LFR), uma linha por encoder |
| *"a réplica bate em todo regime ou só na mediana?"* | **F** | 4 painéis, nosso Δ contra o do benchmark |

**⚠️ Diga isto antes de mostrar qualquer anexo:** uma célula destes anexos tem n = 4,
5 ou 6 configurações pareadas, e o piso do Wilcoxon (2/2ⁿ) fica em 0,125 / 0,0625 /
0,031 — **acima de α/8 em todos os casos**. Aqui não há teste possível, só descrição.
É o argumento da fig. 5 aplicado ao próprio material de apoio, e dizê-lo primeiro é o
que impede que um número solto do anexo vire um claim.

### O que cada anexo mostra — e o que nele surpreende

**A — `anexoA_shift_por_encoder.pdf`.** O custo do shift **não é o mesmo para os
quatro**, e nem sequer tem o mesmo ranking nos dois alvos:

| | pior encoder | melhor encoder |
|---|---|---|
| RealWorld · substituir | cnnpff −4,2 | rnn −1,2 |
| RealWorld · acrescentar | resnetse5 −0,3 | rnn +2,6 |
| MotionSense · substituir | **resnetse5 −10,1** | tstcc −2,0 |
| MotionSense · acrescentar | resnetse5 −4,2 | tstcc +1,1 |

O melhor no RealWorld (rnn) é o que desaba no `full` do MotionSense (−11,9); o pior no
MotionSense (resnetse5) é dos mais estáveis no RealWorld. **Não há um "encoder robusto
a shift"** — e essa é a resposta honesta se pedirem um.

**B — `anexoB_placar_por_federacao_e_alvo.pdf`.** O TF-C + RNN vence nas **seis**
colunas (5/5 regimes em todas) — é o resultado mais sólido da apresentação. E o
contraexemplo útil: **TF-C + TSTCC é negativo no `cross5+5` alvo RealWorld (−4,2, 2/5)
e no `cross10+10` alvo RealWorld (−2,0, 1/5)**, enquanto é +5,0 (5/5) no in-domain
MotionSense. Um par pode ser bom em média e ruim numa federação específica; a fig. 2
não tem como mostrar isso.

**C — `anexoC_placar_por_regime.pdf`.** No federado, o ganho mediano sobre os 8 pares é
**+4,8 pp no 1-shot e +1,6 pp no `full`**, com um platô em ~2,5 pp entre o 2 e o 10 (não
é um declínio monótono — não afirme que é). O SSL vale mais onde o rótulo é escasso, que
é o argumento inteiro da linha. O olho vai direto para o
**−16,6 pp (0/6) do TF-C + ResNet-SE no 1-shot centralizado**: é a pior célula de todo o
material, e é o mesmo par que fica em +4,6 no regime de 100. Se perguntarem, a leitura é
que o pré-treino TF-C atrapalha a inicialização do ResNet-SE quando quase não há rótulo
para corrigi-la — e casa com a Tabela 10 do benchmark, onde esse par também cai no 1-shot
(39,6 contra 50,3 do supervisionado).

**D — `anexoD_placar_por_dataset.pdf`.** O domínio que mais responde é o **KuHar
(+5,9 pp mediano sobre os 8 pares)**; o que menos é o **WISDM (+0,8)**. E o "perdedor"
da fig. 3 tem endereço: **LFR + CNN-PFF é −11,9 no KuHar (1/4)** e positivo no
RealWorld_waist e no UCI. O Δ negativo daquele par não está espalhado — está num
domínio.

**E — `anexoE_custo_federar_por_encoder.pdf`.** Destrincha a inversão da fig. 6 e mostra
que **ela também não é uniforme dentro do método**. No TF-C, o tombo do 1-shot vem de
**três dos quatro** encoders (tstcc −10,5, rnn −8,0, cnnpff −6,8) — o **resnetse5 é a
exceção, e ganha (+2,2)**. No LFR o custo é quase todo do **rnn** (−4,2 mediano), com
cnnpff em ~0. Ou seja: a recomendação "se o alvo é few-shot, o TF-C paga" tem exceção
nomeada, e vale dizê-la antes que alguém ache.

**F — `anexoF_replica_por_regime.pdf`.** A réplica não depende do agregado: a
concordância de **ordem** se sustenta nos quatro regimes (ρ = +1,00 no 1-shot, +0,95 no
10, +0,93 no 100, **+0,83 no `full`** — o pior). A concordância de **sinal** cai no
regime de 10 rótulos (5/8), e a figura mostra por quê: lá quase todo Δ está colado no
zero, onde trocar de sinal não custa nada. É a resposta pronta para *"e se a réplica só
funcionar no regime que te convém?"*.
