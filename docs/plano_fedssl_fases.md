# Plano de execução FedSSL em fases — começar 2026-07-16

*Escrito em 2026-07-15. Operacionaliza `plano_fedssl_simulado.md` (que define
o design: FedAvg manual, Modo A one-shot / Modo B multi-round, spikes S0–S3 e
gates G-EQ/G-IID). Decisão desta sessão (Miguel): depois do smoke test,
implantar em **2 fases**, começando com um subconjunto de 3 bases — o número
de clientes cresce rápido no cross-device, então validamos o pipeline inteiro
(treino federado → downstream → notebook) no recorte barato antes de escalar.*

## 0. Contexto herdado (fechado em 2026-07-15)

- Avaliação SSL centralizada **concluída e revisada**: `ssl_lfr_avaliation`,
  `ssl_tfc_avaliation` (com comb2target §9.2) e `ssl_methods_comparison`,
  todos com conclusões. Achados que orientam este plano:
  - **TF-C + finetune domina** (move CNN-PFF e RNN em +17 a +32 pp in-domain
    few-shot; supera SL até @100%); LFR só vence no linear readout.
  - comb→target: pré-treino multi-domínio custa ~0 (Δ +0.8 pp vs
    especialista) — backbone global + finetune local é a config promissora.
- Spike S0 (init idêntico por seed entre fontes): **PASS** — Modo A pode usar
  os checkpoints do Exp. 1 sem retreinar.

## 1. Escolha das 3 bases da Fase 1: **MotionSense, UCI, WISDM**

Critérios (dados de 2026-07-15):

| Base | Δ TF-C in-domain (1/10-shot, pp F1) | usuários (min janelas) | custo | veredito |
|---|---|---|---|---|
| KuHar | **+12.8 / +19.3** (maior sinal) | 57 (min **1**) — exige agrupamento D-K | baixo | **Fase 2** — sinal ótimo, mas o cross-device dela depende de decisão ainda não validada com o orientador (§10.7 do plano-mãe) |
| MotionSense | +8.6 / +20.7 | 17 (min 165) — **ideal** | baixo (3.6k janelas) | **Fase 1** |
| UCI | +2.1 / +6.8 | 21 (min 99) — ok c/ assert F2 | baixo (2.4k) | **Fase 1** |
| WISDM | −0.6 / +13.2 | 36 (min 235) — saudável | médio (8.7k) | **Fase 1** — e é o maior nº de clientes "limpo": estressa o simulador |
| RealWorld ×2 | +1.9–5.8 / +15.3–17.0 | 10 (min ~957) | **alto** (10.3k cada) | Fase 2 |

Racional: as 3 escolhidas cobrem sinal SSL forte (MotionSense), fraco/médio
(UCI, WISDM @1-shot) e uma faixa de 17→36 clientes por base, sem depender da
decisão D-K nem das bases caras. Combo cross-silo da Fase 1:
`MotionSense+UCI+WISDM` (3 clientes-silo; união ~14.7k janelas ≈ custo de um
pré-treino médio).

## 2. Sequência

### Etapa 0 — Smoke test (pré-requisito, ~meio dia)

Do §6 do plano-mãe, na ordem: **S1** (sopa tfc×cnnpff×seed0 + 1 downstream),
**S2** (loop 3 rodadas TF-C silo, combo da Fase 1), **S3** (LFR 2 rodadas ×
bloco 6), **G-EQ1** (1 cliente R=1×E=100 ≡ centralizado — valida o simulador).
G-EQ2 e G-IID podem rodar em paralelo com a Fase 1 (não bloqueiam).

### Fase 1 — 3 bases × {TF-C, LFR} × 4 encoders + notebook

**Infra a implementar** (§3–4 do plano-mãe):

1. `scripts/ssl/pretrain_fed.py` — `fedavg()`, `local_pretrain()`,
   `run_fedssl()`; checkpoints em
   `checkpoints/ssl_fed/<method>/<encoder>/<partition>/<combo>/seed<N>/`.
2. `scripts/federated/partitions.py` — `make_ssl_client_datasets()` com
   `silo` e `device-<dataset>` (só as 3 bases precisam funcionar; assert F2
   para shard < batch). `iid` só para o gate G-IID.
3. `scripts/ssl/downstream_eval.py` — flag `--ckpt-dir`.
4. Runner de grade no padrão `gpu_pool.py` + concat para
   `results/ssl_fed_eval_transfer.csv` (schema do §7 do plano-mãe).

**Grade da Fase 1** (medir o 1º job de cada bloco antes de extrapolar):

| Bloco | O quê | Runs |
|---|---|---|
| F1-A | Modo A soup: combos de `{MS,UCI,WISDM}` (|S|≥2: 4 combos) × 2 métodos × 4 encoders × 4 seeds — **só média + downstream** (ckpts do Exp. 1 já existem, S0 PASS) | 128 downstreams, 0 pré-treinos |
| F1-B | Modo B silo, combo `MotionSense+UCI+WISDM`: 2 métodos × 4 encoders × 4 seeds (TF-C R=100×1; LFR R=100×bloco 6) | 32 runs |
| F1-C | Modo B cross-device in-domain, `device-<d>` para as 3 bases: 2 métodos × 4 encoders × **seed 0 primeiro** (scan), 4 seeds só nos pares com sinal | 24 runs (scan) |

Braços de comparação já medidos: especialistas centralizados (Exp. 1) e
`central(MS+UCI+WISDM)` **não existe** — rodar 2×4×4 = 32 pré-treinos
centralizados do combo OU aceitar `combined` como referência superior; decidir
pelo custo após medir F1-B (registrar no notebook qual referencial se usou).

**Notebook**: `notebooks/_build_fedssl_nb.py` → `fedssl_avaliation.ipynb`
(padrão dos builders atuais), lendo só `results/ssl_fed_eval_transfer.csv` +
caches centralizados. Seções: setup · cobertura · soup vs central vs fedavg_R
por combo · cross-device vs especialista · curva por rodada (milestones) ·
custo de comunicação · conclusões. **Parametrizar as bases via constante
`DATASETS_WAVE`** para a Fase 2 só estender a lista.

**Gate de saída da Fase 1**: notebook executado com as 3 bases, G-EQ1 PASS
registrado, e leitura clara de soup vs multi-round vs centralizado em ≥ 1 par
(método, encoder). Commit ao final de cada bloco.

### Fase 2 — bases restantes no mesmo pipeline

1. **KuHar**: implementar agrupamento D-K (6 super-clientes) em
   `make_ssl_client_datasets` — antes, validar D-K com o orientador (§10.7).
2. **RealWorld_thigh/waist**: entram direto (partições limpas, só caras).
3. Estender grades: F2-A (soup: combos que incluem as novas bases — os
   destaques, não o power set inteiro; escolher à luz do F1-A), F2-B (silo
   `all6` = onda B1 do plano-mãe, 32 runs), F2-C (cross-device das 3 novas),
   e LODO (`lodo-<d>`×6) como análogo federado do comb2target.
4. Notebook: adicionar as bases a `DATASETS_WAVE` e re-executar; seção nova
   só se o LODO entrar.

Fora de escopo das 2 fases (decidir depois): ablação R×E (B3), fedbn,
finetuning federado no Flower (Fase 6 do plano antigo), A1 scan dos 57 combos.

## 3. Riscos específicos deste recorte

- Sopa catastrófica no F1-A → já previsto (§9 do plano-mãe); o F1-B mede o
  quanto o multi-round recupera.
- LFR federado caro (600 ep de Trainer/run) → se o 1º job do F1-B estourar,
  cortar LFR do F1-B para {rnn} (único encoder onde LFR ganha muito) e manter
  LFR completo só no F1-A (que é grátis).
- WISDM cross-device: 36 clientes × 100 rodadas sequenciais → medir; se >2×
  o pré-treino centralizado, reduzir para R=50×E=2 e registrar.
