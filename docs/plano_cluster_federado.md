# Plano — Rodar a grade federada no cluster (8× TITAN Xp 12 GiB)

> **Para quem é este documento:** é um *handoff* para a sessão do Claude Code que
> será iniciada **dentro do cluster**, depois de clonar este repositório lá. Ele
> descreve o que já existe, o que precisa ser adaptado para 8 GPUs, como validar
> e como rodar a grade completa. A investigação do ambiente, as modificações e o
> teste (spike) devem ser feitos **na sessão do cluster**, não antes.

---

## 0. TL;DR para a sessão do cluster

Execute nesta ordem (detalhes nas seções correspondentes):

1. **Investigar o ambiente** (§3): `nvidia-smi`, detectar se é nó único vs SLURM,
   versão de CUDA/driver, Python, se `poetry` existe.
2. **Setup** (§4): `poetry install`, baixar o DAGHAR, smoke-test de import.
3. **Adaptar para 8 GPUs** (§5): paralelizar a grade no nível de *run* (um run
   por GPU via `CUDA_VISIBLE_DEVICES`). Principal trabalho de código.
4. **Validar com o spike** (§6): rodar 1 combinação com `--rounds 3` e conferir
   que a acurácia sobe (não fica no acaso) e que a paralelização não vaza GPU.
5. **Rodar a grade completa** (§7): 96 runs, em tmux, com log e resume.
6. **Verificar e reportar** (§8): cache completo (96 × 7 rodadas-chave) e sanidade.

---

## 1. Objetivo

Rodar a **grade federada completa** do projeto no cluster:

- **96 runs** = **8 cenários × 3 encoders × 4 seeds**, **R = 50** rodadas de FedAvg.
- Encoders: `resnetse5`, `cnnpff`, `rnn`.
- Seeds: `0, 1, 2, 3`.
- Cenários (ver `scripts/federated/README.md` e `partitions.py`):
  - **1** — 1 dataset DAGHAR por cliente (6 clientes) → **non-IID por domínio** (oficial).
  - **2** — união dos 6 train sets em 6 fatias IID disjuntas → **IID global** (controle).
  - **3..8** — 1 dataset dividido IID em 6 clientes → **IID intra-domínio** (ablação).

Saída: `results/federated_eval.csv` com colunas
`encoder, scenario, seed, round, target, test_acc, test_f1_macro, uplink_bytes, downlink_bytes`
(uma linha por rodada × domínio-alvo).

A comparação **cenário 1 vs 2** (mesmo volume total de dados) isola o efeito do
*domain shift*. A ablação 3..8 isola o custo da federação sem heterogeneidade.

---

## 2. Estado atual do código (já implementado, **não** refazer)

`scripts/federated/` já tem o pipeline FedAvg funcionando por simulação Flower
(`flwr >=1.31` + `ray >=2.55`, ambos no `pyproject.toml`). Reusa todo o pipeline
supervisionado (`scripts/common.py`, `build_model` de `scripts/supervised/`, e a
função `evaluate` de `scripts/eval_transfer.py`).

| Arquivo | Papel |
|---|---|
| `partitions.py` | `make_client_datasets(scenario, seed, num_clients)` — shards por cliente |
| `client.py` | `FlowerClient(NumPyClient)`: loop torch leve (Adam + CrossEntropy), reusa os encoders; `get/set_parameters` via `state_dict` |
| `server.py` | `FedAvg` + `evaluate_fn` centralizada (avalia nos 6 test sets de domínio) |
| `run_federated.py` | Roda **1** combinação (encoder, cenário, seed); grava o cache |
| `run_all.py` | Driver da grade — itera combos, **subprocesso isolado** por run, **resume** via cache |

**Estado do cache (`results/federated_eval.csv`):** só contém o **spike** anterior
(`resnetse5`, cenário 1, seed 0, rodadas 0–3) feito na MX570A. **A grade NÃO foi
rodada.** No cluster é começar a grade essencialmente do zero (o resume vai pular
só as poucas linhas do spike, se o `--rounds` bater).

**Contratos importantes para não quebrar:**

- `run_all.py` considera uma combinação "pronta" se a **rodada final (`--rounds`)**
  daquela `(encoder, scenario, seed)` já está no cache. Mantenha `--rounds 50`
  consistente entre `run_all.py` e qualquer paralelização nova, senão o resume
  não reconhece o que já rodou.
- `run_federated.py` escreve o cache com `drop_duplicates(subset=KEY, keep="last")`
  onde `KEY = [encoder, scenario, seed, round, target]`. **Cada run escreve o
  cache inteiro de volta** → se vários runs escreverem o **mesmo arquivo ao mesmo
  tempo**, há corrida de escrita (ver §5, ponto crítico).
- Ray workers são processos separados: `run_federated.py` exporta `PYTHONPATH`
  (`scripts/` + raiz) via `ray_init_args["runtime_env"]["env_vars"]`. **Sem isso,
  o `fit` falha silenciosamente** (`ModuleNotFoundError: No module named 'federated'`)
  e a acurácia fica no acaso. Qualquer nova variável de ambiente que os workers
  precisem (ex.: `CUDA_VISIBLE_DEVICES`) tem que ser considerada aqui.

---

## 3. Investigação inicial do cluster (primeira coisa a fazer na sessão)

Antes de qualquer código, descobrir o tipo de ambiente — isso decide a estratégia
de orquestração:

```bash
nvidia-smi                     # confirma 8 GPUs (índices 0..7), memória, driver, CUDA
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nproc                          # núcleos de CPU (afeta num_workers / paralelismo)
python --version               # precisa de >=3.11,<4.0
which poetry || pipx --version # como as deps serão instaladas
sinfo 2>/dev/null && echo "==> SLURM presente" || echo "==> sem SLURM"
squeue 2>/dev/null; scontrol show partition 2>/dev/null | head
```

Decisão a tomar com base no resultado:

- **Nó único com as 8 GPUs visíveis via SSH** (caminho mais simples e provável):
  paraleliza a grade com `CUDA_VISIBLE_DEVICES`, um run por GPU. Ver §5A.
- **Cluster SLURM** (cada GPU/nó atrás de fila): cada run vira um job; usar um
  array job ou submeter N jobs limitados pela conta/partição. Ver §5B.
- **Outro scheduler (PBS/LSF/k8s):** adaptar a ideia de §5B ao submissor local.

> Mesmo em SLURM, normalmente se aloca **1 nó com várias GPUs interativo** e aí o
> caminho §5A funciona dentro do job. Confirmar política do cluster (tempo máximo
> de job interativo, GPUs por job).

---

## 4. Setup do ambiente (internet liberada no cluster)

```bash
# 1. Dependências
poetry install

# 2. Dataset DAGHAR (standardized_view)
wget "https://zenodo.org/records/13987073/files/standardized_view.zip?download=1" \
    -O daghar_standardized_view.zip
mkdir -p datasets/DAGHAR
unzip -o daghar_standardized_view.zip -d datasets/DAGHAR/
rm daghar_standardized_view.zip

# 3. Smoke-test de import (pega quebras de ambiente antes de gastar GPU)
poetry run python -c "
import sys; from pathlib import Path
sys.path.insert(0, 'scripts')
from common import DATASETS, SEEDS, make_datamodule
from federated.partitions import make_client_datasets
from federated.client import BUILD_MODEL
import flwr, ray, torch
print('datasets:', DATASETS)
print('cuda:', torch.cuda.is_available(), 'n_gpus:', torch.cuda.device_count())
print('flwr', flwr.__version__, 'ray', ray.__version__)
"
```

Esperado: `cuda: True`, `n_gpus: 8`, os 6 datasets DAGHAR listados, sem erro de import.

---

## 5. Adaptação para 8 GPUs (principal trabalho de código)

**Estado hoje:** `run_federated.py` usa `client_resources={"num_gpus": 1.0}` → 1
cliente por vez, e `run_all.py` roda os 96 subprocessos **em série**. Numa única
MX570A isso era o certo; em 8 TITAN Xp desperdiça 7/8 da máquina.

**Estratégia recomendada — paralelismo no nível de *run* (não dentro do run):**
cada um dos 96 runs é independente e single-GPU. Em vez de mexer no paralelismo
interno do Flower/Ray (mais frágil), dispare **até 8 runs simultâneos, um por
GPU**, fixando `CUDA_VISIBLE_DEVICES=<id>` em cada subprocesso. Ganho ~linear
(≈8×) e robusto: se um run falha, os outros seguem.

### 5A. Caminho nó único — pool de 8 workers no `run_all.py`

Modificar `run_all.py` para manter um **pool de 8 slots** (um por GPU). Cada slot
roda `run_federated.py` num subprocesso com:

```python
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # cada run enxerga só "sua" GPU (vira cuda:0)
subprocess.Popen([...], env=env)
```

Pontos a cuidar (validar no spike, §6):

- **Propagar `CUDA_VISIBLE_DEVICES` aos workers Ray.** `run_federated.py` define
  `ray_init_args["runtime_env"]["env_vars"]` só com `PYTHONPATH`. Como o Ray é
  iniciado **dentro** do subprocesso (que já tem `CUDA_VISIBLE_DEVICES` no env),
  ele deve detectar apenas 1 GPU e atribuí-la ao cliente — mas **confirmar no
  spike** que cada run usa exatamente 1 GPU distinta (`nvidia-smi` durante o teste).
  Se não propagar, adicionar `CUDA_VISIBLE_DEVICES` explicitamente ao `env_vars`
  do `runtime_env`.
- **CRÍTICO — corrida de escrita no cache.** Hoje cada `run_federated.py` lê e
  reescreve `results/federated_eval.csv` inteiro no fim. Com até 8 runs
  terminando em paralelo, isso corrompe/perde linhas. Escolher **uma** solução:
  - (Recomendado) Cada run grava um **CSV parcial próprio** em
    `results/federated_parts/<encoder>_<scenario>_<seed>.csv`; um passo final
    (no `run_all.py`, após o pool drenar) concatena tudo em `federated_eval.csv`.
    Implica passar um `--out` opcional a `run_federated.py`.
  - (Alternativa) Serializar a escrita com um `filelock` no cache.
  A opção de CSV parcial é mais simples e elimina a corrida de raiz; mantém o
  mesmo esquema de colunas.
- **Resume.** Manter a lógica de "pular se a rodada final já está no cache" — no
  esquema de CSV parcial, checar se o parcial daquela combinação já tem `round==50`.
- **Memória:** os modelos são minúsculos (~127K params) e o batch é pequeno; 1
  run por GPU usa uma fração dos 12 GiB. Não tentar empacotar vários runs por GPU
  antes de medir — o ganho marginal não compensa o risco de OOM/contenda.

### 5B. Caminho SLURM (se for o caso)

- Transformar a grade num **job array** (`#SBATCH --array=0-95`), mapeando o índice
  do array → `(encoder, scenario, seed)` (mesma ordem de `run_all.py`).
- Cada tarefa do array pede `--gres=gpu:1` e roda **um** `run_federated.py`.
- Resolver a escrita concorrente com o **mesmo esquema de CSV parcial** da §5A
  (cada tarefa grava seu parcial; um job de agregação no fim concatena).
- Respeitar limites da conta/partição (nº máx. de jobs simultâneos, walltime).

> Decidir 5A vs 5B só **depois** da investigação da §3. Não pré-commitar a SLURM
> se um nó interativo com 8 GPUs estiver disponível (5A é bem mais simples).

---

## 6. Validação no cluster (spike) — antes da grade

Rodar **1 combinação curta** e conferir que tudo funciona no ambiente novo
**antes** de lançar 96 runs:

```bash
# Spike de 3 rodadas (rápido). Fixa a GPU 0 explicitamente.
CUDA_VISIBLE_DEVICES=0 poetry run python scripts/federated/run_federated.py \
    --encoder resnetse5 --scenario 1 --seed 0 --rounds 3 --local-epochs 1
```

Critérios de aceite do spike:

1. **Acurácia sobe** entre `round=0` (≈ acaso, ~0.0–0.17) e `round=3` (deve
   passar de ~0.5 em vários domínios, como no spike da MX570A). Se ficar no acaso
   → o `fit` não treinou (provável `PYTHONPATH`/import nos workers Ray, ver §2).
2. **`nvidia-smi` durante o run** mostra uso em **exatamente uma** GPU.
3. O cache recebe as linhas esperadas (3 rodadas × 6 domínios + rodada 0).

**Teste de paralelização** (depois de implementar §5A): lançar 2 runs em GPUs
diferentes ao mesmo tempo e confirmar via `nvidia-smi` que cada um ocupa a sua
GPU e que **ambos** os resultados chegam ao cache sem se sobrescrever:

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python scripts/federated/run_federated.py \
    --encoder resnetse5 --scenario 1 --seed 0 --rounds 3 --out results/federated_parts/a.csv &
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/federated/run_federated.py \
    --encoder cnnpff   --scenario 2 --seed 0 --rounds 3 --out results/federated_parts/b.csv &
wait
```

Recalibrar o tempo/rodada na TITAN Xp a partir do spike (na MX570A era ~5–6 s/rodada)
para estimar o walltime da grade (§7).

---

## 7. Rodar a grade completa (96 runs)

Sempre em **tmux** (ver `CLAUDE.md`), com log via `tee`:

```bash
tmux new-session -d -s fed-grid
tmux send-keys -t fed-grid \
  'cd /caminho/para/fed-har && poetry run python scripts/federated/run_all.py 2>&1 | tee logs/fed-grid.log' \
  Enter
```

(Se a §5A adicionou um flag de paralelismo, ex.: `--max-parallel 8`, incluí-lo aqui.)

Acompanhar:

```bash
tmux attach -t fed-grid          # entrar (sair com Ctrl+b d, sem matar)
tail -f logs/fed-grid.log        # acompanhar fora do tmux
nvidia-smi -l 5                  # ver as 8 GPUs ocupadas
tmux ls                          # listar sessões
tmux kill-session -t fed-grid    # encerrar
```

**Estimativa grosseira de custo** (recalibrar com o spike, §6): R=50 a ~5 s/rodada
≈ ~5 min/run. 96 runs em série ≈ **~8 h**; com paralelismo 8× ≈ **~1 h** de
*walltime*. A TITAN Xp deve ser ≥ a MX570A, então tende a ser menos.

**Resume:** se a grade cair no meio, basta relançar `run_all.py` — ele pula as
combinações cuja rodada final (50) já está no cache/parciais. Para forçar
re-execução, `--force`. Para rodar subconjuntos: `--scenario 1 2`, `--encoder resnetse5`,
`--seed 0`.

**Ordem de prioridade** (já é o default do `run_all.py`): cenário externo →
encoder → seed, começando pelos cenários 1 e 2 (os mais informativos), depois a
ablação 3..8. Assim os resultados que importam saem primeiro mesmo que a grade
seja interrompida.

---

## 8. Verificação final e entrega

Ao terminar, conferir que o cache está completo e coerente:

```bash
poetry run python -c "
import pandas as pd
df = pd.read_csv('results/federated_eval.csv')
done = df[df['round']==50].groupby(['encoder','scenario','seed']).ngroups
print('combinações completas (round=50):', done, '/ 96')
print('linhas totais:', len(df))
print(df[df['round']==50].groupby('scenario')['test_acc'].mean())
"
```

Esperado: **96** combinações com `round==50`; cada combinação com 51 rodadas
(0..50) × 6 domínios. Sanidade científica: **cenário 2 (IID global)** deve ficar
próximo do baseline `combined` centralizado e **acima** do **cenário 1 (non-IID
por domínio)** — se não, investigar antes de declarar pronto.

Entregáveis:

- `results/federated_eval.csv` completo (96 runs).
- Resumo no final da sessão: tempo total, GPUs usadas, qualquer ajuste de código
  feito na §5, e a comparação 1 vs 2 (efeito do domain shift).
- Atualizar a memória do projeto (`project_supervised.md`) com o estado "grade
  federada concluída no cluster".

---

## 9. Prompt sugerido para abrir a sessão no cluster

> "Estou no cluster com 8 TITAN Xp. Siga `docs/plano_cluster_federado.md`:
> investigue o ambiente (§3), faça o setup (§4), implemente a paralelização em 8
> GPUs (§5), valide com o spike (§6) e então rode a grade completa de 96 runs
> (§7). Me mostre o resultado do spike antes de lançar a grade."

---

## Apêndice — referências rápidas

- Visão geral do projeto e padrão de scripts: `CLAUDE.md`.
- Detalhes do pipeline federado: `scripts/federated/README.md`.
- Notas internas do plano SSL/federado: `docs/notas_internas_projeto_ssl_federado_har.md`.
- Constantes compartilhadas (`DATASETS`, `SEEDS`, `BEST_LR`, `BATCH_SIZE`,
  `make_datamodule`): `scripts/common.py`.
</content>
</invoke>
