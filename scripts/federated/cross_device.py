"""Partição de clientes do eixo CROSS-DEVICE (clientes = usuários).

Função única compartilhada pelo baseline supervisionado FedAvg e pelo pré-treino
Fed-SSL: os dois braços têm de ver **exatamente os mesmos clientes**, senão o
Δ(Fed-SSL − baseline) carrega diferença de partição junto com o efeito do método.

Desenho decidido na sabatina de 2026-07-27 (`docs/plano_fedssl.md §2.1`):

- **Orçamento fixo** de `budget` janelas por cliente (default 192 = 3 batches de
  64), amostradas de forma **estratificada por classe, proporcional** à
  distribuição do usuário e determinística por seed. Com isso todos os braços têm
  o mesmo volume por cliente e os pesos do FedAvg ficam **uniformes**.
- **Seleção de clientes fixa e aninhada**: um sorteio único (`SELECTION_SEED`)
  define uma ordem de usuários por dataset, gravada em `client_selection.csv`.
  Os braços tomam prefixos dessa ordem, então os 5 clientes de um braço são
  sempre um subconjunto dos 10 do outro — o Δ significa "troquei 5 pessoas por
  5 estrangeiras", não "troquei o elenco inteiro".

Gramática do `spec`: ``<modo>:<datasets>:<contagens>``

| spec | Clientes | Para quê |
|---|---|---|
| ``device:RealWorld_thigh:10`` | 10 (1 por usuário) | braço in-domain |
| ``device:RealWorld_thigh+MotionSense:5+5`` | 10 | braço cross-domain |
| ``iid:RealWorld_thigh:10`` | 10 fatias IID | controle de **feature skew**: mesmas janelas do `device`, só a estrutura da partição muda |
| ``single:RealWorld_thigh:10`` | 1 | **gate**: mesmas janelas num cliente só, R=1 deve reproduzir o centralizado |

Uso:
    # (re)gerar a seleção fixa — só precisa rodar uma vez, o CSV é versionado
    python scripts/federated/cross_device.py --regenerate-selection

    # inspecionar um braço sem treinar nada
    python scripts/federated/cross_device.py --describe device:RealWorld_thigh+MotionSense:5+5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from common import DAGHAR_ROOT, DATASETS, LABEL_COLUMN
from federated.partitions import _train_dataset

# Sorteio único da ordem dos clientes (data da sabatina). Trocar este valor
# reembaralha quem entra em cada braço — não faça isso sem regerar o CSV e
# reportar, porque muda a população do experimento.
SELECTION_SEED = 20260727
DEFAULT_BUDGET = 192  # 3 batches de 64: mínimo que dá ao TF-C >1 passo por época
SELECTION_CSV = Path(__file__).resolve().parent / "client_selection.csv"
MODES = ("device", "iid", "single")


# --------------------------------------------------------------- metadados
def _user_labels(dataset_name: str) -> pd.DataFrame:
    """`user` e rótulo de cada janela do train, na ordem das linhas do CSV.

    A ordem das linhas é a ordem dos índices do dataset torch (o CSVReader lê o
    arquivo sequencialmente) — a mesma premissa de `partitions._user_shards`.
    """
    return pd.read_csv(DAGHAR_ROOT / dataset_name / "train.csv",
                       usecols=["user", LABEL_COLUMN])


# --------------------------------------------------- seleção fixa e aninhada
def build_selection(datasets=DATASETS) -> pd.DataFrame:
    """Sorteia UMA ordem de usuários por dataset (permutação semeada)."""
    rng = np.random.default_rng(SELECTION_SEED)
    rows = []
    for name in datasets:
        # str em toda parte: o RealWorld usa ids como "proband13" e o
        # MotionSense usa inteiros.
        counts = _user_labels(name).user.astype(str).value_counts()
        users = np.array(sorted(counts.index))  # ordem estável antes do sorteio
        rng.shuffle(users)
        for rank, user in enumerate(users):
            rows.append({"dataset": name, "rank": rank, "user": user,
                         "n_windows": int(counts[user])})
    return pd.DataFrame(rows)


def load_selection() -> pd.DataFrame:
    if not SELECTION_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTION_CSV} não existe. Rode:\n"
            f"  poetry run python scripts/federated/cross_device.py --regenerate-selection")
    return pd.read_csv(SELECTION_CSV, dtype={"user": str})


def pick_users(dataset_name: str, n: int, budget: int | None) -> list[str]:
    """Os `n` primeiros usuários elegíveis, na ordem fixa do manifesto.

    Elegível = tem pelo menos `budget` janelas. Como a ordem é fixa, o prefixo de
    5 é subconjunto do prefixo de 10 (seleção **aninhada**).
    """
    sel = load_selection()
    sel = sel[sel.dataset == dataset_name].sort_values("rank")
    if budget is not None:
        sel = sel[sel.n_windows >= budget]
    if len(sel) < n:
        raise ValueError(
            f"{dataset_name}: pedidos {n} clientes com >= {budget} janelas, "
            f"mas só {len(sel)} usuários são elegíveis.")
    return sel.user.head(n).tolist()


# ------------------------------------------------------ orçamento por cliente
def _allocate(counts: np.ndarray, budget: int) -> np.ndarray:
    """Quantas janelas tirar de cada classe: proporcional, somando exatamente `budget`.

    Maior resto, respeitando a disponibilidade de cada classe.
    """
    if counts.sum() < budget:
        raise ValueError(f"orçamento {budget} > janelas disponíveis {counts.sum()}")
    props = counts / counts.sum()
    exact = props * budget
    quota = np.minimum(np.floor(exact).astype(int), counts)
    order = np.argsort(-(exact - np.floor(exact)))  # maior parte fracionária primeiro
    i = 0
    while quota.sum() < budget:
        c = order[i % len(order)]
        if quota[c] < counts[c]:
            quota[c] += 1
        i += 1
    return quota


def budget_indices(labels: np.ndarray, pool: np.ndarray, budget: int,
                   rng: np.random.Generator) -> list[int]:
    """Amostra `budget` índices de `pool`, estratificada por classe."""
    y = labels[pool]
    classes, counts = np.unique(y, return_counts=True)
    quota = _allocate(counts, budget)
    picked: list[int] = []
    for cls, k in zip(classes, quota):
        idx = pool[y == cls]
        picked.extend(rng.choice(idx, size=int(k), replace=False).tolist())
    # O pareamento entre braços depende de todo cliente fechar o orçamento; sem
    # este assert um cliente curto passaria despercebido e quebraria o desenho.
    assert len(picked) == budget, f"orçamento não fechou: {len(picked)} != {budget}"
    return sorted(int(i) for i in picked)


# ------------------------------------------------------------------ a API
def parse_spec(spec: str) -> tuple[str, list[str], list[int]]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"spec inválido: {spec!r} (use <modo>:<datasets>:<contagens>)")
    mode, names, counts = parts
    if mode not in MODES:
        raise ValueError(f"modo inválido: {mode!r} (use {'/'.join(MODES)})")
    names = names.split("+")
    counts = [int(c) for c in counts.split("+")]
    if len(names) != len(counts):
        raise ValueError(f"spec {spec!r}: {len(names)} datasets para {len(counts)} contagens")
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise ValueError(f"datasets desconhecidos em {spec!r}: {sorted(unknown)}")
    return mode, names, counts


def make_clients(spec: str, seed: int,
                 budget: int | None = DEFAULT_BUDGET) -> list[tuple[str, Subset]]:
    """Clientes de um braço cross-device: lista de `(id_cliente, Subset)`.

    `budget=None` desliga o orçamento fixo (cenário "natural", ablação futura):
    cada cliente fica com todas as suas janelas.
    """
    mode, names, counts = parse_spec(spec)
    rng = np.random.default_rng(seed)

    per_user: list[tuple[str, Dataset, list[int]]] = []
    for name, n in zip(names, counts):
        meta = _user_labels(name)
        dataset = _train_dataset(name, seed)
        if len(meta) != len(dataset):
            raise RuntimeError(
                f"{name}: train.csv tem {len(meta)} linhas mas o dataset torch tem "
                f"{len(dataset)} amostras — ordem/leitura divergente.")
        labels = meta[LABEL_COLUMN].to_numpy()
        users = meta.user.astype(str).to_numpy()
        for user in pick_users(name, n, budget):
            pool = np.where(users == user)[0]
            idx = (budget_indices(labels, pool, budget, rng) if budget is not None
                   else sorted(int(i) for i in pool))
            per_user.append((f"{name}/{user}", dataset, idx))

    if mode == "device":
        return [(cid, Subset(ds, idx)) for cid, ds, idx in per_user]

    # `iid` e `single` reusam EXATAMENTE as mesmas janelas do braço `device`
    # correspondente — é isso que faz do `iid` um controle pareado de feature
    # skew (mesmo dado, mesmos rótulos, mesmo domínio; muda só a partição).
    # ConcatDataset desloca os índices de cada dataset; `pool` já vem no mesmo
    # espaço de índices dele.
    datasets, pool, offset = [], [], 0
    for _cid, ds, idx in per_user:
        if not datasets or datasets[-1] is not ds:
            if datasets:
                offset += len(datasets[-1])
            datasets.append(ds)
        pool.extend(offset + i for i in idx)
    union = ConcatDataset(datasets)

    if mode == "single":
        return [("single", Subset(union, pool))]
    perm = np.random.default_rng(seed).permutation(len(pool))
    return [(f"iid{k}", Subset(union, [pool[i] for i in chunk]))
            for k, chunk in enumerate(np.array_split(perm, sum(counts)))]


# ---------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--regenerate-selection", action="store_true",
                    help="Sorteia de novo a ordem fixa dos clientes e grava o CSV.")
    ap.add_argument("--describe", metavar="SPEC",
                    help="Monta o braço e imprime a composição (não treina nada).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = ap.parse_args()

    if args.regenerate_selection:
        sel = build_selection()
        sel.to_csv(SELECTION_CSV, index=False)
        print(f"gravado {SELECTION_CSV} ({len(sel)} usuários)")
        for name in DATASETS:
            d = sel[sel.dataset == name]
            print(f"  {name:18s} {len(d):>3} usuários | elegíveis a B={DEFAULT_BUDGET}: "
                  f"{(d.n_windows >= DEFAULT_BUDGET).sum()}")

    if args.describe:
        clients = make_clients(args.describe, seed=args.seed, budget=args.budget)
        total = sum(len(s) for _, s in clients)
        print(f"\nspec={args.describe} seed={args.seed} budget={args.budget}")
        print(f"  {len(clients)} clientes, {total} janelas no total")
        for cid, shard in clients:
            print(f"    {cid:<34} {len(shard):>5} janelas")


if __name__ == "__main__":
    main()
