# Resultados centralizados — transfer, `comb2target` e skyline

*Consolidado em 2026-07-27 (origem: `analise_domain_shift.md`, escrito em
2026-07-21, §4.1 acrescentada em 2026-07-24; renomeado para cá com `git mv`). Reúne o que já foi **medido** no
eixo centralizado, mais o baseline federado cross-silo rebaixado a preliminar.
Todo número aqui é regenerável dos caches (§5).*

**Onde vive o quê:** os fatos dos datasets (posição do sensor, classes, usuários,
skews) estão em [`dados_daghar.md`](dados_daghar.md); o desenho do eixo federado
cross-device, em [`plano_fedssl.md`](plano_fedssl.md); hiperparâmetros e desvios
em relação ao benchmark, em [`metodo_e_auditoria.md`](metodo_e_auditoria.md).

---

## 1. O transfer centralizado segue a posição do sensor

As 6 bases caem em dois grupos cinemáticos — **cintura** `{KuHar, UCI,
RealWorld_waist}` e **perna/bolso** `{MotionSense, WISDM, RealWorld_thigh}`
(justificativa em `dados_daghar.md §2`). A hipótese é que a transferência siga a
similaridade de posição: sensores em posições parecidas veem distribuições de
aceleração/rotação parecidas.

Todas as medidas abaixo: acurácia, **full finetuning**, média sobre **4 encoders
× 4 seeds**, pares cross-domain (`source ≠ target`, excluindo `combined`).

**(a) Dentro do grupo vs entre grupos** — o teste direto da hipótese:

| Método | mesmo grupo de posição | entre grupos | gap |
|---|---|---|---|
| SL | 0.526 | 0.420 | **+0.105** |
| LFR | 0.546 | 0.410 | +0.136 |
| TF-C | 0.610 | 0.415 | **+0.195** |

Leitura: o transfer já é sistematicamente melhor dentro da mesma posição no
supervisionado (+10,5 pp). O **SSL amplifica o transfer intra-posição** (TF-C
sobe para 0.610) mas **quase não atravessa posições diferentes** (0.415 ≈ SL
0.420). Conclusão honesta e não-óbvia: **o SSL fortalece a transferência entre
sensores parecidos; ele não "resolve" o gap entre cintura e perna.**

**(b) Melhores pares de transfer natural (SL)** — todos dentro do grupo:

| Par | acc SL | grupo |
|---|---|---|
| RealWorld_waist → UCI | **0.686** | cintura |
| RealWorld_thigh → MotionSense | 0.586 | perna |
| RealWorld_waist → KuHar | 0.582 | cintura |
| MotionSense → RealWorld_thigh | 0.572 | perna |
| WISDM → MotionSense | 0.563 | perna |
| RealWorld_thigh → WISDM | 0.562 | perna |

**(c) Onde o SSL mais recompra performance (Δ acc vs SL)** — o par de maior ganho
é intra-grupo perna:

| Par | SL | LFR | TF-C | Δ(TF-C−SL) |
|---|---|---|---|---|
| RealWorld_thigh → MotionSense | 0.586 | 0.634 | **0.782** | **+0.196** |
| MotionSense → RealWorld_thigh | 0.572 | 0.613 | 0.710 | +0.137 |

## 2. Casos, exceções e o efeito do corpus `combined`

- **RW_waist → UCI (0.686)** é o melhor transfer *natural* (ambos cintura): o
  caso onde o domain shift já é pequeno sem SSL.
- **RW_thigh ↔ MotionSense** é o par de maior **Δ(SSL−SL)** (perna) — onde o SSL
  mais ajuda. Por isso é o par escolhido como **prova-de-conceito** do federado.
  ⚠️ *Escolher o par pelo maior Δ é seleção-no-desfecho: vale para validar que o
  pipeline converge, não como evidência da tese geral de mitigação.*
- **KuHar é um outlier fraco do grupo cintura**: pares envolvendo KuHar
  transferem 0.448 (vs 0.470 sem KuHar); dentro do grupo, KuHar → UCI é só 0.341
  (contra RW_waist → UCI 0.686). Causa provável: 100 Hz original, *waist bag*
  frouxo (sensor não rígido ao corpo) e usuários minúsculos (mediana ~10
  janelas). Registrar como limitação ao usar KuHar como "caso difícil".
- **O corpus `combined` degrada como fonte única, menos sob SSL.** Treinar/
  finetunar no `combined` e avaliar por alvo, contra o especialista in-domain
  (média sobre alvos, full-ft): SL −0.053, LFR −0.046, **TF-C −0.026** (RW_thigh
  chega a reverter: +0.040 no TF-C). Um único modelo generalista custa acurácia,
  mas o SSL atenua o custo.
- **Mas `combined` como pré-treino + finetune no alvo (`comb2target`) ganha.**
  É o argumento a favor de juntar dado não-rotulado de vários domínios no
  pré-treino — a proposta de valor do SSL federado. O quanto ganha depende de
  contra quem se compara, que é o assunto da §3.

## 3. O skyline SL-`combined` (ablação do `comb2target`)

*Medido em 2026-07-23/24 (3 encoders) e completado com o `tstcc` no cluster em
2026-07-27, por `scripts/ssl/sl_comb2target_eval.py`; cache
`results/sl_comb2target_eval_transfer.csv`. **Grade completa: 4608/4608 células**
(4 encoders × 6 fontes × 4 seeds × 2 protocolos × 4 regimes × 6 alvos), sem NaN.*

Para atribuir o ganho do `comb2target` ao **SSL** e não à etapa
`pré-treino → especialização no alvo`, o comparador é o mesmo pipeline com o
backbone vindo do modelo **supervisionado** treinado no `combined`
(`checkpoints/supervised/<enc>/combined/seed<N>/best.ckpt`), com cabeça,
otimizador, protocolos e regimes idênticos aos do `downstream_eval.py`.

**Este comparador NÃO é um baseline pareado — é um *skyline*.** Orçamento de
supervisão de cada braço:

| Braço | Rótulos no pré-treino | Seleção de checkpoint |
|---|---|---|
| LFR / TF-C | **0** (train+val do `combined` sem rótulos) | época fixa (LFR 600, TF-C 100), sem ES |
| SL-`combined` | **36.788** (train) | `best.ckpt` por `val_loss` **rotulada** (+5.844 janelas) |

Todo o viés aponta a favor do SL-`combined`: mais rótulos **e** seleção de modelo
supervisionada. Além disso, ele é **inviável sob a premissa do eixo federado**
(dado não-rotulado abundante, rótulo escasso) — logo não compete com o SSL, ele
o limita superiormente.

> **Base de encoders pareada.** A versão desta tabela anterior a 2026-07-27
> comparava o skyline (então só 3 encoders) com LFR/TF-C agregados sobre 4 — Δ
> não pareado. Com o `tstcc` medido, os três braços têm os **mesmos 4 encoders** e
> a comparação é legítima. O gerador de assets calcula essa base do próprio dado e
> a declara na legenda de cada tabela/figura (§5), então o erro não pode voltar.

Acurácia média, in-domain (`source == target`), full finetuning, **4 encoders**
× 6 alvos × 4 seeds. O Δ é **pareado por seed** (Δ de cada seed, depois média e
dp entre as 4) — é ele que diz o que é efeito e o que é ruído:

| shots | skyline | LFR | TF-C | Δ(LFR) | Δ(TF-C) |
|---|---|---|---|---|---|
| 1 | **0.528** ± 0.042 | 0.410 ± 0.044 | 0.439 ± 0.043 | −0.118 ± 0.044 | −0.089 ± 0.051 |
| 10 | 0.689 ± 0.012 | 0.591 ± 0.016 | **0.699** ± 0.017 | −0.098 ± 0.016 | +0.011 ± 0.019 |
| 100 | 0.751 ± 0.007 | 0.721 ± 0.006 | **0.807** ± 0.007 | −0.030 ± 0.011 | **+0.056** ± 0.010 |
| full | 0.778 ± 0.008 | 0.772 ± 0.008 | **0.833** ± 0.002 | −0.006 ± 0.014 | **+0.055** ± 0.007 |

(± = desvio-padrão **entre seeds**, convenção da apresentação.)

**Leitura por alvo no regime `full`:**

| Alvo | skyline | LFR | Δ(LFR) | TF-C | Δ(TF-C) |
|---|---|---|---|---|---|
| KuHar | 0.685 | 0.585 | **−0.099** | 0.774 | +0.089 |
| MotionSense | 0.850 | 0.860 | +0.011 | 0.909 | +0.059 |
| RealWorld_thigh | 0.687 | 0.682 | −0.005 | 0.740 | +0.053 |
| RealWorld_waist | 0.718 | 0.722 | +0.004 | 0.757 | +0.040 |
| UCI | 0.895 | 0.919 | +0.024 | 0.929 | +0.034 |
| WISDM | 0.833 | 0.861 | +0.028 | 0.888 | +0.056 |

- **TF-C vence o skyline em 6/6 alvos** (+0.034 a +0.089) gastando **zero rótulo**
  no pré-treino. Na média, o ganho é sólido **a partir de 100 shots**
  (+0.056 ± 0.010 e +0.055 ± 0.007, ~5× o dp). **Aos 10 shots é empate**
  (+0.011 ± 0.019 — o dp é maior que o efeito), e em 1-shot o skyline ganha.
- **No regime `full` o LFR empata com o skyline** (−0.006 ± 0.014, efeito dentro
  do ruído) e vence em **4 dos 6** alvos (MotionSense, RW_waist, UCI, WISDM). O
  déficit residual é dominado pelo **KuHar** (−0.099), já registrado como outlier
  fraco (§2). Em baixo-dado o LFR fica claramente abaixo.

Em cross-domain (`source ≠ target`) o skyline fica acima dos dois métodos SSL em
todos os regimes (Δ TF-C −0.045 a −0.127; Δ LFR −0.083 a −0.119) — esperado, já
que o pré-treino supervisionado multi-domínio produz uma representação já
alinhada a classes, que é exatamente a vantagem que 36,8k rótulos compram.

**Como reportar estes números (regra de redação):**

1. O achado positivo é do **TF-C**: ele **supera um comparador com vantagem de
   supervisão** in-domain **a partir de 100 shots** (+0.055, 6/6 alvos), gastando
   zero rótulo no pré-treino. Vencer um comparador enviesado contra você é
   evidência **mais forte** que vencer o SL in-domain, não mais fraca.
   ⚠️ **Não dizer "a partir de 10 shots"** — aos 10 shots o Δ é +0.011 ± 0.019,
   ou seja, empate. A versão anterior deste doc afirmava isso porque o Δ estava
   calculado entre bases de encoders diferentes.
2. Para o LFR, reportar a **medição**, nunca o veredito ("LFR perde" / "LFR
   degrada"). A comparação não é pareada em rótulos, então não sustenta juízo de
   mérito sobre o método. Enunciado defensável: *"com dados rotulados suficientes
   no alvo, o LFR alcança o pré-treino supervisionado gastando zero rótulo no
   pré-treino (empate dentro do ruído, vencendo em 4 dos 6 alvos); em baixo-dado,
   não."*
3. **A ablação é um pacote**: entra inteira (TF-C favorável + LFR desfavorável) ou
   fica inteira fora por escopo. Reportar só a metade favorável seria
   cherry-picking, já que as duas vêm da mesma medição.
4. Rótulo canônico em tabelas/figuras: **"skyline (pré-treino supervisionado,
   36,8k rótulos)"** — não "baseline". O rótulo faz o trabalho argumentativo.
5. **Sempre declarar a base de encoders** agregada na legenda. Foi a ausência
   disso que deixou passar a comparação 3-vs-4 encoders por 3 dias.

Em aberto: um SL-`combined` **pareado em rótulos** (pré-treino supervisionado
usando só N rótulos do `combined`) seria o baseline de fato justo; custo não
trivial, só se o `comb2target` for promovido a seção do artigo.

## 4. Preliminar: o baseline federado cross-silo

> Este resultado foi **rebaixado a motivação** na virada para o eixo cross-device
> (decisão com o orientador, 2026-07-21). Não é o controle do Fed-SSL e não é a
> contribuição; entra como "mesmo na topologia mais simples, domain shift custa".
> O controle honesto passou a ser Δ(cross-domain − in-domain) no cross-device —
> ver `plano_fedssl.md`.

Grade completa em `results/federated_eval.csv`: **128 runs** (8 cenários ×
4 encoders × 4 seeds, R=50) = 39.168 linhas. Acurácia média na rodada 50:

| Cenário | Partição | acc (4 enc) |
|---|---|---|
| 2 | união dos 6 train sets em 6 fatias IID | **0.743** |
| 1 | 1 dataset por cliente (non-IID por domínio) | **0.680** |
| 3–8 | 1 dataset dividido IID em 6 clientes | 0.453 – 0.542 |

**O custo do domain shift nesta topologia é 6,3 pp** (cenário 2 − cenário 1).

⚠️ **Correção de 2026-07-27:** o valor "≈8 pp" repetido em vários documentos vinha
da grade antiga de **3 encoders** (0.664 vs 0.740 = 7,6 pp). Com o `tstcc`
incluído a grade fechou em 128 runs e o gap é **6,3 pp**. Ao citar, dizer a base.

## 5. Proveniência e reprodução

- Caches: `results/supervised_eval_transfer.csv`, `results/ssl_lfr_eval_transfer.csv`,
  `results/ssl_tfc_eval_transfer.csv`, `results/ssl_{lfr,tfc}_comb2target_eval_transfer.csv`,
  `results/sl_comb2target_eval_transfer.csv` (§3, skyline), `results/federated_eval.csv` (§4).
- Agregação padrão: **acc, `n_shots="full"`, `protocol="finetune"`, média sobre
  encoders × seeds**; pares cross-domain excluem `source==target` e
  `source=="combined"`. Grupos de posição conforme `dados_daghar.md §2`.
- **Base de encoders:** todas as seções agregam os **4 encoders** (desde
  2026-07-27, quando o `tstcc` do skyline foi medido).
  `scripts/analysis/build_presentation_assets.py` calcula a interseção por método
  do próprio dado (`enc_base`) e a imprime na legenda de cada asset — por isso as
  tabelas da apresentação não podem mais divergir desta seção.
- **Δ pareado por seed** (§3): o Δ é calculado seed a seed e só então promediado;
  o dp reportado é o dessa distribuição. Diferença de médias marginais esconderia
  a variância que decide se +0.011 é efeito ou ruído.
- Assets da apresentação: `poetry run python scripts/analysis/build_presentation_assets.py
  --outdir docs/apresentacoes/<nome>`. Apresentação já entregue é registro
  imutável e **não** é regerada.
- Referências bibliográficas dos datasets: `dados_daghar.md §5`.
