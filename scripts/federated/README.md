# scripts/federated/ — Federação cross-silo (Flower)

> **⚠️ DEPRECATED como desenho/controle (2026-07-21).** A federação **cross-silo**
> desta pilha (cenários 1–8) foi **abandonada** (decisão com o orientador). O
> `results/federated_eval.csv` (96 runs, ~8 pp de domain shift) **não é deletado**:
> vira **resultado PRELIMINAR/motivação**, não a contribuição. O eixo ativo agora é
> **cross-device** (clientes = usuários), via `partition_users.py` +
> `scripts/ssl/pretrain_fed.py` (a fazer) — design em
> `docs/plano_fedssl_simulado.md` e análise em `docs/analise_domain_shift.md`. O
> **finetuning federado** (Exp. 2/3) permanece no Flower.

Integração com **Flower** (`flwr` 1.31 + `ray` 2.55) na configuração cross-silo
do projeto: **FedAvg** por simulação local, avaliação **centralizada por domínio**
(acurácia + F1-macro) e **custo de comunicação**. Reusa todo o pipeline dos
baselines supervisionados (`scripts/common.py`, os `build_model` de
`scripts/supervised/`, e a função `evaluate` de `scripts/eval_transfer.py`).

## Arquivos

| Arquivo | Papel |
|---|---|
| `partitions.py` | `make_client_datasets(scenario, seed)` — monta os shards por cliente |
| `client.py` | `FlowerClient(NumPyClient)`: treina o encoder no shard local (loop torch leve, Adam + CrossEntropy, sem `L.Trainer`) e devolve os pesos |
| `server.py` | Estratégia `FedAvg` + `evaluate_fn` centralizada (avalia nos 6 test sets de domínio) |
| `run_federated.py` | Entrypoint da simulação; grava `results/federated_eval.csv` |

## Cenários de particionamento

| Cenário | Partição | Tipo FL |
|---|---|---|
| 1 | 1 dataset DAGHAR por cliente (6 clientes = 6 datasets) | non-IID por domínio (**oficial**) |
| 2 | união dos 6 train sets, dividida em 6 fatias IID disjuntas | IID global (controle) |
| 3..8 | 1 dataset dividido IID nos 6 clientes (um por cenário) | IID intra-domínio (ablação) |

> **Escopo do Flower (decisão de 2026-07-13)**: esta pilha (Flower + ray) é
> usada para o treino **supervisionado** federado — o baseline já medido e o
> finetuning federado dos Exp. 2/3. O **pré-treino SSL federado NÃO usa
> Flower**: é uma simulação exata de FedAvg em loop Python
> (`scripts/ssl/pretrain_fed.py`, a fazer), incluindo a partição
> **cross-device por usuário** — design em `docs/plano_fedssl_simulado.md`.
> A `partitions.py` daqui ganha `make_ssl_client_datasets(partition, combo,
> seed)` para servir os dois mundos.

A comparação **1 vs 2** (mesmo volume total de dados) isola o efeito do *domain
shift*: espera-se que o Cenário 2 (IID) fique próximo do baseline `combined`
centralizado e o Cenário 1 (non-IID) fique abaixo.

## Uso

```bash
# Spike (validação rápida)
poetry run python scripts/federated/run_federated.py \
    --encoder resnetse5 --scenario 1 --seed 0 --rounds 3 --local-epochs 1

# Run completo (default: 50 rodadas, 1 época local)
poetry run python scripts/federated/run_federated.py \
    --encoder resnetse5 --scenario 1 --seed 0 --rounds 50
```

Argumentos: `--encoder {resnetse5,cnnpff,rnn}`, `--scenario {1..8}`,
`--seed {0..3}`, `--rounds`, `--local-epochs`, `--num-clients` (default 6).

Rodar via **tmux** para experimentos longos (ver `CLAUDE.md`); a GPU MX570A roda
1 cliente por vez (`num_gpus=1.0`), em sequência.

## Saída — `results/federated_eval.csv`

Colunas (incremental, mesmo padrão de cache de `eval_transfer.py`):

```
encoder, scenario, seed, round, target, test_acc, test_f1_macro,
uplink_bytes, downlink_bytes
```

Uma linha por (rodada × domínio-alvo). A rodada 0 registra a avaliação dos pesos
iniciais (sem custo de comunicação). `uplink_bytes`/`downlink_bytes` = bytes do
`state_dict` × nº de clientes, por rodada.

## Notas de implementação

- Os workers Ray são processos separados: `run_federated.py` exporta `PYTHONPATH`
  (`scripts/` + raiz) via `ray_init_args` para que `common`/`federated`/`minerva`
  importem dentro dos atores.
- Calibração medida no spike (resnetse5, MX570A): ~5–6 s/rodada (6 clientes +
  avaliação nos 6 domínios).

## A fazer (próximos passos do plano)

> ⚠️ A grade cross-silo (cenários 1–8) está **concluída** e **deprecada como
> controle**. Os próximos passos migraram para o eixo **cross-device**, fora desta
> pilha Flower cross-silo:

- **Baseline supervisionado federado cross-device** (novo controle do Fed-SSL):
  3 experimentos — (1) in-domain RW_thigh, (2) in-domain MotionSense, (3)
  cross-domain RW_thigh+MotionSense — usando `partition_users.py`. Controle honesto
  do custo de domain shift = **Δ(cross-domain − in-domain)**. Design em
  `docs/plano_fedssl_simulado.md` / `docs/plano_fedssl_fases.md`.
- Finetuning federado a partir de checkpoint SSL (Exp. 2/3) — permanece no Flower.
