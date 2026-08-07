# Wilcoxon pareado por par (técnica SSL × encoder)

> Escrito em **2026-08-06**. Gerado por
> [`scripts/analysis/wilcoxon_pares.py`](../scripts/analysis/wilcoxon_pares.py);
> saídas em `results/derived/wilcoxon/`. Continuação direta da auditoria de
> variância (§11–12 de `notebooks/fedssl_cross_device_avaliation.ipynb`), que
> concluiu que a variância não está alta — o `n` de seeds é que está baixo — e
> deixou "Wilcoxon pareado sobre células" na fila como a rota de custo zero.

---

## 1. O que é o Wilcoxon pareado

É um teste **não-paramétrico** para a pergunta "essa diferença é sistemática ou é sorteio?".

O mecanismo, concretamente. Você tem pares — a mesma configuração medida nos dois braços. Para cada par calcula a diferença Δ. Aí:

1. Joga fora o sinal e **ordena os |Δ| do menor para o maior**, atribuindo postos 1, 2, 3…
2. Soma os postos dos Δ positivos (W⁺) e dos negativos (W⁻).
3. Se os dois braços fossem equivalentes, cada Δ teria 50% de chance de cair para cada lado, e W⁺ ≈ W⁻. O `p` é a probabilidade de ver um desequilíbrio ao menos tão grande por acaso.

Dois pontos que mudam tudo para a nossa situação:

**O `n` do teste é o número de pares, não o número de seeds.** Esta é a chave. O teste não conta repetições — conta *configurações*. Se eu tenho 24 configurações (6 datasets × 4 regimes) e o SSL ganha em 24 delas, isso é `p = 2/2²⁴ ≈ 10⁻⁷`, e não importa se cada configuração foi medida 3 ou 4 vezes.

**Postos, não valores.** Ele não assume normalidade e é robusto a outliers — o que importa é *quantas vezes* e *em que ordem de magnitude* um braço ganha. Isso serve bem para acurácia agregada sobre datasets com patamares muito diferentes.

Usei **Bonferroni**: como faço 8 testes ao mesmo tempo (2 métodos × 4 encoders), multiplico cada `p` por 8. Sem isso, com 8 testes a 5%, esperaria ~0,4 falsos positivos só de sorte.

## 2. Não — o ±16 pp **não** é isso

São coisas ortogonais. O ±16 pp das tabelas é **desvio padrão** — dispersão entre as ~200 configurações que a linha agrega (1-shot ~40% vs `full` ~85%, KuHar vs MotionSense). É descritivo. O Wilcoxon é **inferencial** e opera sobre a diferença *pareada dentro* da configuração, onde essa dispersão toda cancela. Um estudo pode ter ±20 pp de dispersão e `p = 10⁻⁷`, porque a variação que assusta na tabela não é a variação que entra no teste.

## 3. O benchmark faz Wilcoxon — mas não o que se imagina

Conferido no PDF inteiro. Todo teste reportado (Fig. 3, Tabelas 7, 8, 9, 13) é **encoder contra encoder, dentro de um método**. É daí que saem os "(3-0, +3)" das tabelas e o grafo de precedência que coroa o ResNet-SE-5.

**Não há teste de SSL contra supervisionado em lugar nenhum do paper.** O *"SSL techniques significantly outperform supervised baselines"* é legenda da Tabela 10, sem teste por trás. E a comparação SL vs SSL deles é a **Tabela 11**, que é melhor-de-24 (6 encoders × 4 SSL) contra melhor-de-6 (6 encoders supervisionados) — um max contra outro max, com contagem de células verdes (`≥+1: 45`) e vermelhas (`≤-1: 1`). Contagem não é teste, e max-vs-max é estimador enviesado a favor do lado com mais candidatos.

Ou seja: **o teste rodado aqui é mais forte do que o que eles reportam.**

## 4. Onde estão os pares que não dão certo

Não estão escondidos. A seleção é só a Tabela 11; as matrizes completas estão no corpo do artigo:

| Tabela | Página | O que traz |
|---|---|---|
| **7** | 13 | 6 encoders × 5 métodos, média geral — todos os 30 pares |
| **9** | 17 | 6 encoders × 6 datasets × 5 métodos, full FT — **180 células** |
| **10** | 18 | 6 encoders × 8 regimes de rótulo × 5 métodos — 240 células |
| **13** | 21 | idem Tab. 9, mas freeze |

E os perdedores estão lá, explicitamente. O caso mais direto para nós: **LFR+CNN-PFF rende 62,2% contra 63,4% do supervisionado** (Tab. 7) — o par piora o baseline, e o placar dele é (0-2, −2). No RW-Thigh e no KuHar o LFR+CNN-PFF chega a 59,8% e 41,1%, contra 58,9% e 45,2% do supervisionado. Nada disso aparece na Tabela 11, porque a Tabela 11 só mostra o vencedor de cada linha.

## 5. A nossa réplica bate — e bate no que importa

Nível, contra a Tabela 10 (48 células, 4 encoders × 3 métodos × 4 regimes):

```
MAE 1,56 pp · viés +0,54 pp · maior desvio 4,7 pp
supervisionado MAE 1,62 · LFR 1,32 · TF-C 1,73
```

Mas bater em nível não é o que autoriza o claim — o claim usa o **Δ**. Então comparei o Δ deles (SSL − SL, da Tab. 10) com o nosso:

| par | Δ benchmark | Δ nosso |
|---|---|---|
| TF-C + rnn | +14,45 | **+15,14** |
| TF-C + cnnpff | +10,80 | **+8,63** |
| LFR + rnn | +7,30 | **+5,95** |
| TF-C + tstcc | +4,50 | +1,77 |
| LFR + tstcc | +3,80 | +1,15 |
| TF-C + resnetse5 | +2,30 | +0,20 |
| LFR + resnetse5 | +0,95 | +0,27 |
| LFR + cnnpff | −1,50 | **−0,39** |

**Concordância de sinal 8/8, Spearman ρ = +0,98.** A ordem dos pares é praticamente idêntica, inclusive o único par negativo. Isso é o que dá direito de invocar o benchmark: não replicamos só o número, replicamos a *estrutura* do efeito.

## 6. O Wilcoxon nos nossos dados

**Centralizado** (24 configurações = 6 datasets × 4 regimes, seeds mediadas):

| par | Δ mediano | Δ>0 | p (Bonf.) | |
|---|---|---|---|---|
| TF-C + rnn | +15,14 | 24/24 | 1,5e-04 | **vence** |
| TF-C + cnnpff | +8,63 | 22/24 | 2,4e-04 | **vence** |
| LFR + rnn | +5,95 | 24/24 | 9,5e-07 | **vence** |
| TF-C + tstcc | +1,77 | 15/24 | 1,000 | n.s. |
| LFR + tstcc | +1,15 | 15/24 | 1,000 | n.s. |
| TF-C + resnetse5 | +0,20 | 12/24 | 1,000 | n.s. |
| LFR + resnetse5 | +0,27 | 12/24 | 1,000 | n.s. |
| LFR + cnnpff | −0,39 | 10/24 | 1,000 | n.s. |

**Federado** (30 configurações = 4 specs/6 alvos × 5 regimes):

| par | Δ mediano | Δ>0 | p (Bonf.) | |
|---|---|---|---|---|
| TF-C + rnn | +23,07 | 30/30 | 1,5e-08 | **vence** |
| TF-C + cnnpff | +5,55 | 30/30 | 1,5e-08 | **vence** |
| LFR + tstcc | +3,43 | 27/30 | 1,2e-04 | **vence** |
| LFR + rnn | +2,92 | 21/30 | 0,0021 | **vence** |
| TF-C + resnetse5 | +1,87 | 19/30 | 0,177 | n.s. |
| LFR + resnetse5 | +1,72 | 22/30 | 0,030 | **vence** |
| TF-C + tstcc | +0,43 | 18/30 | 1,000 | n.s. |
| LFR + cnnpff | +0,34 | 19/30 | 1,000 | n.s. |

E o **agregado por método** (empilhando os 4 encoders — a rota do da Luz): LFR e TF-C vencem o supervisionado nos dois setups, com folga (centralizado: +1,3 pp `p=4e-3` e +6,9 pp `p=1e-7`; federado: +1,7 pp `p=1e-10` e +4,5 pp `p=8e-14`).

**Concordância centralizado ↔ federado: 7/8 de sinal, ρ = +0,79.** O único desalinhamento é LFR+cnnpff (−0,39 centralizado, +0,34 federado) — e os dois são não-significantes, então não é contradição, é ruído em torno de zero. A conclusão do benchmark **sobrevive à federação**, com um detalhe interessante: no federado *mais* pares passam, porque a nossa grade federada tem 30 configurações contra 24 e os efeitos são um pouco maiores.

## 7. Sim, dá para concluir sobre 5+5 e 10+10

| par | 5+5 | 10+10 |
|---|---|---|
| TF-C + cnnpff | +3,97 · 10/10 · **vence** | +5,94 · 10/10 · **vence** |
| TF-C + rnn | +22,01 · 10/10 · **vence** | +24,11 · 10/10 · **vence** |
| LFR + tstcc | +2,77 · 8/10 · n.s. | +3,50 · 10/10 · **vence** |
| resto | n.s. | n.s. |
| **agregado LFR** | +1,47 · p=0,005 · **vence** | +2,10 · p=4e-5 · **vence** |
| **agregado TF-C** | +3,97 · p=2e-5 · **vence** | +4,54 · p=4e-6 · **vence** |

Aqui aparece o limite duro que é preciso saber: com **n = 10** configurações, o menor `p` bilateral possível é 2/2¹⁰ = 0,00195; depois do Bonferroni ×8 fica 0,0156. Ou seja, **só um placar perfeito de 10/10 sobrevive** — o que é exatamente o que os três vencedores fizeram. Não há meio-termo nessa escala.

E as federações únicas (`RealWorld_thigh:10` e `MotionSense:10`) têm apenas **5** configurações. Piso de `p` = 0,0625, já acima de 0,05 antes de qualquer correção. **Nenhum resultado ali pode ser declarado significante, nem em princípio** — nem o TF-C+rnn com Δ de +26,2 pp e 5/5. Isso não é ausência de efeito, é ausência de teste. Para essas duas federações a única rota é o agregado por método (n=20), que passa.

## 8. As 4 seeds bastam?

**Para claim agregado, sim — e as 3 deles também bastavam, pelo mesmo motivo.** O `n` do teste nunca foi o número de seeds. As seeds servem para *estabilizar cada célula* antes de parear; o poder vem das 24 ou 30 configurações. Rodei também a versão com cada seed como observação (n = 96 e 120) e ela não muda nenhum veredito — só encolhe os `p`, o que é ilusório: 4 seeds da mesma configuração não são 4 observações independentes. Por isso a versão com seeds mediadas é a primária.

**Para claim célula-a-célula, não — e nem 8 seeds resolveriam a um custo razoável.** Continua valendo o §12: dp(Δ) ≈ 2,5 pp com n=4 dá IC95 de ±4 pp, e o efeito mediano é 3 pp. Mas o Wilcoxon torna essa limitação muito menos importante do que parecia, porque quase todo claim que queremos fazer é agregado.

### Duas ressalvas honestas

As configurações pareadas **não são independentes** — o mesmo dataset aparece em 4 regimes, o mesmo encoder em 6 datasets. O Wilcoxon assume independência e o Bonferroni não conserta dependência. Os `p` ordenam evidência com segurança; tratá-los como probabilidades exatas seria exagero. O benchmark tem exatamente o mesmo problema, então não estamos abaixo do padrão da área — mas convém dizer na seção de método em vez de deixar o revisor achar.

Segunda: "n.s." significa **não distinguível**, nunca "não ajuda". LFR+resnetse5 com +0,27 pp centralizado é honestamente um empate; TF-C+tstcc com +1,77 pp e 15/24 é um efeito plausível que o nosso `n` não resolve.
