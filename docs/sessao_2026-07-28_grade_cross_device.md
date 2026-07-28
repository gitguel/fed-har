# Sessão de 2026-07-28 — grade cross-device fechada

*Registro da sessão. Não é doc vivo: os números aqui valem para os commits
citados e envelhecem. Para o estado atual rode
`poetry run python scripts/analysis/cache_status.py` e leia o notebook
`notebooks/cross_device_avaliation.ipynb`.*

Commits da sessão: `99ecb5a` (primeira grade), `5b3ee9d` (k=5 R=50 + best.ckpt),
`5496f68` (notebook, fim do padrão builder, F7), `20449ab` (grade final R=150).

---

## Grade k=5, R=150, quatro encoders — fechada

**96/96 jobs, zero falhas.** Makespan **123 min** (estimativa era 2h20; foram
2h03). Média 10,0 min/job: resnetse5 20,2 · tstcc 8,5 · cnnpff 6,0 · rnn 5,4. 96
`best.ckpt` gravados (337 MB, fora do git). Notebook re-executado: **24/24
células completas**, 0 erros.

### R=150 valeu a pena — mas não pelo motivo que previmos

Ganho de teste **pareado** por `(encoder, arm, seed, alvo)`, n=128, contra o R=50
do `5b3ee9d`:

| encoder | rodada escolhida (50 → 150) | Δval | **Δtest** | pares que melhoraram |
|---|---|---|---|---|
| **rnn** | 43,7 → 111,4 | +4,19 pp | **+6,29 pp** | 81% |
| tstcc | 40,6 → 124,8 | +3,88 | +1,77 | 75% |
| resnetse5 | 26,5 → 73,6 | +1,86 | +1,34 | 59% |
| cnnpff | 38,7 → 112,8 | +1,36 | **+0,11** | 44% |

O `rnn` era o caso que motivou a opção B e foi o que mais ganhou — a decisão se
pagou. Mas o `cnnpff` ganhou **1,36 pp de validação e 0,11 pp de teste**, com
menos da metade dos pares melhorando.

### O achado que importa: a seleção está escolhendo ruído

Comparação entre o `argmax` da validação e simplesmente **usar a média do platô**
(R76–150):

| encoder | argmax ganha (val) | dp do platô | argmax ganha (test) |
|---|---|---|---|
| **resnetse5** | +11,44 pp | 3,44 pp | **+6,04 pp** |
| tstcc | +3,52 | 1,66 | +1,23 |
| cnnpff | +2,37 | 1,09 | **+0,10** |
| rnn | +2,30 | 1,23 | **−0,03** |

Em `cnnpff` e `rnn`, o ganho de validação do `argmax` é ~2× o desvio do platô —
ou seja, **é exatamente o tamanho do ruído**, e não sobrevive ao teste. Só no
`resnetse5` a seleção faz trabalho real, e por um motivo identificável: a curva
dele genuinamente pica e degrada, enquanto as dos outros três ficam planas
(inclinação final entre −0,28 e +0,19 pp/10 rodadas).

**Consequência prática:** "rodada escolhida" só é interpretável como "melhor
rodada" no `resnetse5`. Nos outros três, oferecer 150 candidatos em vez de 50
produz um máximo de validação mais alto que não existe no teste. É literalmente o
efeito que Oliver et al. descreve na §4.6, documentado no **F7** de
`metodo_e_auditoria.md` no mesmo dia — apareceu no nosso próprio dado, e com o
split de validação **inteiro**, não com o pequeno. Isso fortalece o F7: o
problema não é só do regime few-shot.

O protocolo não foi alterado — a decisão de manter a seleção pela validação está
tomada e os números do commit a seguem. Mas isso deve entrar na ablação do F7
como caso já medido.

### Resultado principal (R=150, rodada escolhida pela validação)

| arm / alvo | resnetse5 | cnnpff | rnn | tstcc |
|---|---|---|---|---|
| in10-RW | 0,680 ±0,058 | 0,701 ±0,018 | 0,589 ±0,051 | **0,711** ±0,027 |
| iid-RW | 0,710 ±0,035 | **0,753** ±0,018 | 0,626 ±0,018 | 0,714 ±0,021 |
| in10-MS | **0,891** ±0,044 | 0,837 ±0,019 | 0,721 ±0,038 | 0,853 ±0,016 |
| iid-MS | **0,905** ±0,024 | 0,846 ±0,007 | 0,723 ±0,040 | 0,845 ±0,020 |
| cross5+5 / RW | 0,647 ±0,037 | 0,651 ±0,029 | 0,592 ±0,019 | **0,661** ±0,038 |
| cross5+5 / MS | 0,801 ±0,043 | 0,776 ±0,028 | 0,602 ±0,033 | **0,812** ±0,009 |
| cross10+10 / RW | 0,647 ±0,041 | **0,704** ±0,031 | 0,615 ±0,014 | 0,693 ±0,040 |
| cross10+10 / MS | **0,877** ±0,040 | 0,828 ±0,012 | 0,667 ±0,039 | 0,810 ±0,012 |

O `rnn` subiu bastante com R=150 mas segue último em todo cenário. O `resnetse5`
passou a liderar no MotionSense (0,891/0,905), invertendo o quadro do R=50.

### Os dois contrastes

**Feature skew (`device − iid`)**: `RW` −0,052 a −0,003 · `MS` −0,014 a +0,009.
Continua **sem sustentar afirmação** — troca de sinal entre alvos, e o único
valor consistente (`cnnpff`/RW, −0,052 ±0,008) é isolado. Conclusão inalterada em
três grades sucessivas.

**Domain shift (`cross5+5 − in-domain`)**: **negativo em 7 das 8 células**,
−0,119 a +0,003, com o efeito maior no MotionSense (−0,041 a −0,119). Segue sendo
o eixo real, agora com magnitude maior que na grade anterior.

---

## Decisões da sabatina (fechadas nesta sessão)

| # | ramo | decisão |
|---|---|---|
| 1 | fonte do notebook | lê `results/fed_cross_device_parts/`, **nunca** o CSV consolidado (que fica velho durante uma grade) |
| 2 | dados parciais | só células com as 4 seeds; incompleta não aparece |
| 3 | fine-tuning | **federado (principal) + centralizado (secundário)**, notebooks separados |
| 4 | onde moram os rótulos | **distribuído** (`L` por classe em todo cliente); concentrado vira ablação |
| 5 | validação no few-shot | **completa**, seguindo Minerva/da Luz — registrado como **F7**, revisão prioritária |
| 6 | protocolo | **full fine-tune**; sem linear readout no federado |
| 7 | ladder | **{1, 2, 5, 10, full}** por classe por cliente |
| 8 | `R` | **150**, o mesmo do baseline ("mais justo") |
| 9 | seeds | **4**; se o ruído dominar, rodar outras 4 |

Convenção nova do repo: **notebooks são editados à mão**, sem builder script
(`CLAUDE.md`, seção "Notebooks"). Os três `_build_*_nb.py` foram removidos.

## Aberto

- **Ablação do F7** — o protocolo de seleção com validação honesta. Agora tem
  dado próprio a favor (a tabela do platô acima), não só literatura.
- **Cenário de rótulos concentrados** (poucos clientes rotulados) — o eixo mais
  realista, adiado para depois do distribuído.
- **Cronograma alternado** (`plano_fedssl.md:252`) — segue vivo, já que o full
  fine-tune não o inviabiliza. Candidato a contribuição nomeada.
- **Baseline de transfer na comparação do Fed-SSL** — exigência do B10 (Oliver
  et al.); os números já existem em `supervised_eval_transfer.csv`.
- **Aposentar ou não o cross-silo** — `federated_avaliation.ipynb` e
  `results/federated_eval.csv` seguem citados em `resultados.md §4`; o código é
  que está DEPRECATED.
