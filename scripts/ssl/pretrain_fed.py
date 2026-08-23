"""Pré-treino SSL federado SIMULADO (FedAvg manual, sem Flower).

Implementa o §3 de `docs/plano_fedssl_simulado.md`: um loop sequencial de
FedAvg sobre clientes definidos por `federated.partitions.make_ssl_client_datasets`
(`silo`, `iid` ou `device-<dataset>`), com os builders SSL centralizados já
auditados (`build_lfr` / `build_tfc`). Cada run ocupa 1 GPU e itera clientes
sequencialmente; a grade paraleliza runs (gpu_pool), não clientes.

Decisões herdadas do plano:
- `seed_everything(seed)` no init global; antes de cada fit local,
  `seed_everything(seed*10**6 + round*10**3 + client)` (shuffles distintos e
  reprodutíveis). Otimizador recriado a cada rodada (Adam reseta — FedAvg padrão).
- DPP do LFR: seleção executada UMA vez, pelo "servidor", sobre 128 amostras da
  união do combo; clientes herdam a estrutura já reduzida (num_targets=None
  nas cópias => o setup() do Lightning não re-seleciona).
- FedAvg default agrega TUDO (inclusive buffers de BatchNorm); `skip_prefixes`
  fica disponível para variantes fedbn futuras.
- Clientes usam SÓ o train.csv (validation/test do DAGHAR são disjuntos por
  usuário — verificado em 2026-07-17); não há loop de validação no pré-treino,
  então o CSV de rodadas loga tempo e bytes analíticos, não val_loss.
- TF-C: `drop_last=True` (máscara NT-Xent fixa em 2×batch) + assert de loader
  não-vazio (réplica do F2 da auditoria) — shard de usuário menor que o batch
  aborta o run com mensagem clara.
- LFR: `--local-epochs` conta épocas do Trainer; use múltiplos de
  (predictor_training_epochs+1)=6 ("bloco 6") para fechar épocas efetivas de
  backbone inteiras por rodada.

Saída:
    checkpoints/ssl_fed/<method>/<encoder>/<partition>/<combo>/seed<N>/
        backbone.ckpt            # final, formato idêntico ao centralizado
        round<R>/backbone.ckpt   # milestones R ∈ {1,5,10,25,50,100}
        state_last.pt            # estado global completo + rodada (resume)
        rounds.csv               # round, n_clients, secs, uplink_mb, downlink_mb
    logs/ssl_fed/<method>/<encoder>/<partition>/<combo>/seed<N>.log (via runner)

Uso:
    # Smoke S2: 3 rodadas TF-C silo no combo da Fase 1
    python scripts/ssl/pretrain_fed.py --method tfc --encoder cnnpff \
        --partition silo --combo MotionSense+UCI+WISDM --rounds 3 --local-epochs 1 --seed 0

    # Gate G-EQ1: 1 cliente, R=1×E=100 ≡ centralizado
    python scripts/ssl/pretrain_fed.py --method tfc --encoder cnnpff \
        --partition silo --combo MotionSense --rounds 1 --local-epochs 100 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

from common import BATCH_SIZE, PROJECT_ROOT, SEEDS  # noqa: E402
from encoders import ENCODERS  # noqa: E402
from federated.partitions import make_ssl_client_datasets, parse_combo  # noqa: E402
from federated.cross_device import DEFAULT_BUDGET, make_clients  # noqa: E402
from pretrain_lfr import build_lfr  # noqa: E402
from pretrain_tfc import build_tfc  # noqa: E402

# Sobrescrevível por env: a grade das RQs escreve em checkpoints/rqs/ssl_fed/
# para não colidir com os backbones da grade B=192 já rodada no cluster.
FED_CKPT_ROOT = Path(os.environ.get("FEDHAR_FEDSSL_CKPT_ROOT", PROJECT_ROOT / "checkpoints" / "ssl_fed"))
METHODS = ["lfr", "tfc"]
MILESTONES = {1, 5, 10, 25, 50, 100}
DPP_SELECTION_SIZE = 128  # amostras da união usadas na seleção (LFR, servidor)


def fed_ckpt_dir(method: str, encoder: str, partition: str, combo: str, seed: int) -> Path:
    return FED_CKPT_ROOT / method / encoder / partition / combo / f"seed{seed}"


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------
def fedavg(states: list[dict], weights: list[float],
           skip_prefixes: tuple[str, ...] = ()) -> dict:
    """Média ponderada de state_dicts (pesos n_k/n).

    Tensores float: média ponderada; inteiros (ex.: `num_batches_tracked` do
    BatchNorm): média arredondada, mantendo o dtype. Chaves com prefixo em
    `skip_prefixes` são copiadas do primeiro estado (o global da rodada
    anterior deve ser passado como `states[0]` nesse caso) — o default agrega
    TUDO (FedAvg padrão).
    """
    total = float(sum(weights))
    norm = [w / total for w in weights]
    out: dict = {}
    for key in states[0]:
        if any(key.startswith(p) for p in skip_prefixes):
            out[key] = states[0][key].clone()
            continue
        ref = states[0][key]
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for state, w in zip(states, norm):
            acc += state[key].to(torch.float64) * w
        if ref.dtype.is_floating_point:
            out[key] = acc.to(ref.dtype)
        else:
            out[key] = acc.round().to(ref.dtype)
    return out


def state_bytes(state: dict, skip_prefixes: tuple[str, ...] = ()) -> int:
    """Bytes analíticos de comunicação de um state_dict (uplink = downlink)."""
    return sum(
        t.numel() * t.element_size()
        for k, t in state.items()
        if not any(k.startswith(p) for p in skip_prefixes)
    )


# ---------------------------------------------------------------------------
# DataModule de shard (Subset + protocolo do PretrainDataModule)
# ---------------------------------------------------------------------------
class ShardPretrainDataModule(L.LightningDataModule):
    """Serve um shard de cliente como train_dataloader único (sem validação).

    Mesmo contrato do `PretrainDataModule` centralizado, mas sobre um Dataset
    já construído (Subset/ConcatDataset). `drop_last=True` para TF-C; o assert
    replica o F2 da auditoria (loader vazio treinaria silenciosamente 0 steps).
    """

    def __init__(self, shard: Dataset, batch_size: int, num_workers: int,
                 drop_last: bool) -> None:
        super().__init__()
        self.shard = shard
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.shard,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )
        assert len(loader) > 0, (
            f"Shard com {len(self.shard)} amostras gera 0 batches "
            f"(batch={self.batch_size}, drop_last={self.drop_last}) — F2."
        )
        return loader


# ---------------------------------------------------------------------------
# Modelo global e treino local
# ---------------------------------------------------------------------------
def build_global_model(method: str, encoder: str, union: Dataset) -> L.LightningModule:
    """Constrói o modelo global inicial no "servidor".

    Para o LFR, executa aqui (uma única vez) a seleção DPP dos 6 projetores
    sobre `DPP_SELECTION_SIZE` amostras da união do combo — hipótese de
    "amostra pública" do plano — e desliga `num_targets` para que os fits dos
    clientes (que rodam `setup('fit')` de novo) não re-selecionem.
    """
    if method == "tfc":
        return build_tfc(encoder)
    model = build_lfr(encoder)
    idx = np.random.choice(len(union), size=DPP_SELECTION_SIZE,
                           replace=DPP_SELECTION_SIZE > len(union))
    sample = torch.from_numpy(np.array([
        s[0] if isinstance(s, (tuple, list)) else s for s in (union[i] for i in idx)
    ]))
    model._select_targets(sample)
    model.num_targets = None  # cópias/clientes herdam a seleção, sem repeti-la
    return model


def local_pretrain(method: str, template: L.LightningModule, global_state: dict,
                   shard: Dataset, local_epochs: int, seed: int,
                   num_workers: int) -> dict:
    """1 cliente × 1 rodada: parte do estado global, treina, devolve o estado."""
    seed_everything(seed, workers=True)
    model = copy.deepcopy(template)
    model.load_state_dict(global_state)
    dm = ShardPretrainDataModule(shard, batch_size=BATCH_SIZE,
                                 num_workers=num_workers,
                                 drop_last=(method == "tfc"))
    trainer = L.Trainer(
        max_epochs=local_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        deterministic="warn",
    )
    trainer.fit(model, datamodule=dm)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def extract_backbone_state(model: L.LightningModule, global_state: dict) -> dict:
    """Recorta do estado global o state_dict do backbone (formato centralizado)."""
    model.load_state_dict(global_state)
    return {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def run_fedssl(method: str, encoder: str, partition: str, combo: str,
               rounds: int, local_epochs: int, seed: int,
               num_workers: int = 4, force: bool = False,
               skip_prefixes: tuple[str, ...] | None = None,
               budget: int | None = DEFAULT_BUDGET) -> Path:
    # Projetores do LFR: aleatórios, CONGELADOS e idênticos em todos os clientes
    # (DPP feita uma vez pelo servidor). São 96,2% do state_dict — agregá-los é um
    # no-op que custaria 38,6 MB por cliente por rodada em cada sentido.
    # Ver docs/plano_fedssl.md §2.3.
    if skip_prefixes is None:
        skip_prefixes = ("projectors",) if method == "lfr" else ()

    out_dir = fed_ckpt_dir(method, encoder, partition.replace(":", "-"), combo, seed)
    final_ckpt = out_dir / "backbone.ckpt"
    if final_ckpt.exists() and not force:
        print(f"[skip] backbone já existe: {final_ckpt}", flush=True)
        return final_ckpt
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, workers=True)
    # `partition` no formato <modo>:<datasets>:<contagens> é o eixo cross-device
    # (função única compartilhada com o baseline supervisionado); qualquer outro
    # valor cai na API herdada do cross-silo.
    if ":" in partition:
        clients = make_clients(partition, seed, budget=budget)
    else:
        clients = make_ssl_client_datasets(partition, combo, seed)
    weights = [len(shard) for _, shard in clients]
    union = ConcatDataset([shard for _, shard in clients])
    template = build_global_model(method, encoder, union)
    global_state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}

    start_round = 0
    resume_path = out_dir / "state_last.pt"
    if resume_path.exists() and not force:
        saved = torch.load(resume_path, map_location="cpu")
        global_state, start_round = saved["state"], saved["round"]
        print(f"[resume] retomando da rodada {start_round}: {resume_path}", flush=True)

    mb = state_bytes(global_state, skip_prefixes) / 2**20
    print(f"=== [FedSSL {method}/{encoder}] {partition}/{combo} seed{seed}: "
          f"{len(clients)} clientes ({sum(weights)} janelas), R={rounds}×E={local_epochs}, "
          f"~{mb:.1f} MB/rodada/sentido ===", flush=True)

    rounds_csv = out_dir / "rounds.csv"
    log_mode = "a" if start_round > 0 and rounds_csv.exists() else "w"
    with open(rounds_csv, log_mode, newline="") as fh:
        log = csv.writer(fh)
        if log_mode == "w":
            log.writerow(["round", "n_clients", "secs", "uplink_mb", "downlink_mb"])
        for r in range(start_round + 1, rounds + 1):
            t0 = time.time()
            states = []
            for ci, (cid, shard) in enumerate(clients):
                states.append(local_pretrain(
                    method, template, global_state, shard, local_epochs,
                    seed=seed * 10**6 + r * 10**3 + ci, num_workers=num_workers,
                ))
                print(f"  [r{r}] cliente {ci + 1}/{len(clients)} ({cid}, "
                      f"n={len(shard)}) ok", flush=True)
            global_state = fedavg(states, weights, skip_prefixes)
            secs = time.time() - t0
            log.writerow([r, len(clients), f"{secs:.1f}",
                          f"{mb * len(clients):.2f}", f"{mb * len(clients):.2f}"])
            fh.flush()
            torch.save({"state": global_state, "round": r}, resume_path)
            if r in MILESTONES:
                ms_dir = out_dir / f"round{r}"
                ms_dir.mkdir(exist_ok=True)
                torch.save(extract_backbone_state(template, global_state),
                           ms_dir / "backbone.ckpt")
            print(f"[r{r}/{rounds}] agregado em {secs:.1f}s", flush=True)

    torch.save(extract_backbone_state(template, global_state), final_ckpt)
    print(f"[OK] backbone federado salvo -> {final_ckpt}", flush=True)
    return final_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--partition", required=True,
                        help="cross-device: <modo>:<datasets>:<contagens> "
                             "(ex.: device:RealWorld_thigh:10). Herdado: silo | iid | device-<dataset>")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="Janelas por cliente no eixo cross-device (0 = natural, sem parear).")
    parser.add_argument("--combo", default="",
                        help="Datasets unidos por '+', 'all6' ou 'lodo-<dataset>'.")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, required=True,
                        help="Épocas do Trainer por cliente/rodada (LFR: múltiplos de 6).")
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Re-treina mesmo com backbone/resume existentes.")
    args = parser.parse_args()

    if ":" not in args.partition:
        parse_combo(args.combo)  # valida cedo (só no eixo herdado)
    for seed in args.seed:
        run_fedssl(args.method, args.encoder, args.partition, args.combo,
                   args.rounds, args.local_epochs, seed,
                   num_workers=args.num_workers, force=args.force,
                   budget=args.budget or None)


if __name__ == "__main__":
    main()
