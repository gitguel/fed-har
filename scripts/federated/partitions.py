"""Particionamento dos dados em clientes para a simulação federada (Flower).

Define os cenários de particionamento do plano federado, sempre reusando
`common.make_datamodule` para ler os CSVs do DAGHAR — exatamente o mesmo
pipeline dos baselines supervisionados.

| Cenário | Partição                                                | Tipo FL           |
|---------|---------------------------------------------------------|-------------------|
| 1       | 1 dataset DAGHAR por cliente (6 clientes = 6 datasets)  | non-IID (domínio) |
| 2       | união dos 6 train sets, dividida em 6 fatias IID        | IID global        |
| 3..8    | 1 dataset dividido IID nos 6 clientes (um por cenário)  | IID intra-domínio |

- Cenário 1 é o experimento comprometido no plano oficial (non-IID por domínio).
- Cenário 2 é o controle homogêneo: dividir a *união* em fatias **disjuntas**
  (não duplicar dados) mantém o volume total igual ao do Cenário 1, isolando o
  efeito do domain shift na comparação 1 vs 2.
- Cenários 3..8 isolam o custo da federação sem heterogeneidade, por domínio.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from common import DAGHAR_ROOT, DATASETS, make_datamodule

NUM_CLIENTS = 6

# Cenários 3..8 -> um sub-dataset DAGHAR individual cada (na ordem de DATASETS).
SCENARIO_TO_DATASET = {3 + i: name for i, name in enumerate(DATASETS)}


def _train_dataset(dataset_name: str, seed: int) -> Dataset:
    """Carrega o split de treino de um sub-dataset DAGHAR como `Dataset` torch.

    Reusa `make_datamodule` (mesma leitura/feature_prefixes/cast dos baselines).
    Com os parâmetros padrão (`data_percentage=1.0`, `samples_per_class=None`),
    o `MultiModalHARSeriesDataModule` devolve o train set completo, sem subamostrar.
    """
    dm = make_datamodule(dataset_name, seed=seed, num_workers=0)
    dm.setup("fit")
    dataset, _domain_labels = dm.datasets["train"]
    return dataset


def _iid_shards(dataset: Dataset, num_clients: int, seed: int) -> list[Subset]:
    """Divide `dataset` em `num_clients` fatias IID disjuntas.

    Os índices são embaralhados deterministicamente por `seed` e repartidos em
    blocos de tamanho ~igual (o resto é distribuído aos primeiros clientes).
    """
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    base, rem = divmod(n, num_clients)
    shards: list[Subset] = []
    start = 0
    for client in range(num_clients):
        size = base + (1 if client < rem else 0)
        shards.append(Subset(dataset, perm[start : start + size]))
        start += size
    return shards


def make_client_datasets(
    scenario: int, seed: int, num_clients: int = NUM_CLIENTS
) -> list[Dataset]:
    """Constrói a lista de datasets de treino, um por cliente, para um cenário.

    Retorna uma lista com `num_clients` elementos (datasets torch map-style),
    onde o elemento `i` é o shard do cliente `i`.
    """
    if scenario == 1:
        if num_clients != len(DATASETS):
            raise ValueError(
                f"Cenário 1 exige num_clients == {len(DATASETS)} (1 cliente por dataset DAGHAR)."
            )
        return [_train_dataset(name, seed) for name in DATASETS]

    if scenario == 2:
        full = ConcatDataset([_train_dataset(name, seed) for name in DATASETS])
        return _iid_shards(full, num_clients, seed)

    if scenario in SCENARIO_TO_DATASET:
        dataset = _train_dataset(SCENARIO_TO_DATASET[scenario], seed)
        return _iid_shards(dataset, num_clients, seed)

    raise ValueError(f"Cenário inválido: {scenario} (use 1, 2, ou 3..8).")


# ---------------------------------------------------------------------------
# Partições do FedSSL simulado (plano_fedssl_simulado.md §4)
# ---------------------------------------------------------------------------
def parse_combo(combo: str) -> list[str]:
    """Expande o nome do combo em lista ordenada de datasets DAGHAR.

    `"A+B"` = os datasets listados; `"all6"` = todos; `"lodo-<d>"` = todos
    menos `<d>`. A ordem canônica é a de `DATASETS` (o nome do combo deve
    seguir essa ordem para o layout de checkpoints ser estável).
    """
    if combo == "all6":
        return list(DATASETS)
    if combo.startswith("lodo-"):
        held_out = combo[len("lodo-"):]
        if held_out not in DATASETS:
            raise ValueError(f"Dataset desconhecido no lodo: {held_out!r}")
        return [d for d in DATASETS if d != held_out]
    names = combo.split("+")
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise ValueError(f"Datasets desconhecidos no combo {combo!r}: {unknown}")
    return sorted(names, key=DATASETS.index)


def _user_shards(dataset_name: str, seed: int) -> list[tuple[str, Subset]]:
    """1 cliente por usuário do train.csv (cross-device).

    A ordem das linhas do CSV é a ordem dos índices do dataset torch (o
    CSVReader lê o arquivo sequencialmente), então basta agrupar os índices
    pela coluna `user`.
    """
    users = pd.read_csv(DAGHAR_ROOT / dataset_name / "train.csv", usecols=["user"])["user"]
    dataset = _train_dataset(dataset_name, seed)
    if len(users) != len(dataset):
        raise RuntimeError(
            f"{dataset_name}: train.csv tem {len(users)} linhas mas o dataset "
            f"torch tem {len(dataset)} amostras — ordem/leitura divergente."
        )
    shards = []
    for uid, idx in users.groupby(users, sort=True).groups.items():
        shards.append((f"user_{uid}", Subset(dataset, list(idx))))
    return shards


def make_ssl_client_datasets(
    partition: str, combo: str, seed: int
) -> list[tuple[str, Dataset]]:
    """Clientes do pré-treino federado simulado: lista de `(id_client, shard)`.

    - `silo`            — 1 cliente por dataset do combo.
    - `iid`             — união do combo em 6 fatias IID (gate G-IID).
    - `device-<dataset>`— 1 cliente por usuário do train do dataset (que deve
                          pertencer ao combo). Pesos de agregação = len(shard).
    """
    names = parse_combo(combo)
    if partition == "silo":
        return [(name, _train_dataset(name, seed)) for name in names]
    if partition == "iid":
        full = ConcatDataset([_train_dataset(name, seed) for name in names])
        return [(f"iid{i}", s) for i, s in enumerate(_iid_shards(full, NUM_CLIENTS, seed))]
    if partition.startswith("device-"):
        dataset_name = partition[len("device-"):]
        if dataset_name not in names:
            raise ValueError(f"{partition!r}: {dataset_name!r} não está no combo {combo!r}.")
        return _user_shards(dataset_name, seed)
    raise ValueError(f"Partição inválida: {partition!r} (use silo, iid ou device-<dataset>).")
