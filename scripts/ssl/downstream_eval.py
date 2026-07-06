"""Avaliação downstream dos backbones SSL (LFR) — espelha `eval_transfer.py`.

Estágios B+C do pipeline SSL. Para cada (encoder, fonte, semente, protocolo,
regime de dados):
  B) treina uma cabeça de classificação sobre o backbone pré-treinado, no train
     rotulado da *fonte*, em 2 protocolos:
       - `linear`   : backbone congelado (`requires_grad=False`) + cabeça treinável.
       - `finetune` : backbone descongelado + cabeça, ponta-a-ponta.
     e em 4 regimes de dados rotulados (`n_shots`): 1 / 10 / 100 amostras-por-classe
     e `full` (100%).
  C) avalia (backbone + cabeça) nos `test.csv` dos 6 datasets-alvo (acc + F1-macro).

Protocolo de refinamento = o MESMO do baseline supervisionado (Seção V-F do
paper e YAMLs finetune/lfr_<enc>.yaml do repo oficial):
  - Cabeça MLP `Linear(enc_dim -> 128) -> ReLU -> Linear(128 -> 6)`
    (`common.build_prediction_head`, idêntica ao SL).
  - Adam lr=1e-4 nos DOIS protocolos, batch 64.
  - Até 100 épocas com parada antecipada (paciência 50 sobre a val_loss da
    fonte) e restauração do melhor estado (equivale ao `best.ckpt` do SL).
  - Congelamento como no `SimpleSupervisedModel(freeze_backbone=True)`: apenas
    `requires_grad=False`; BatchNorm/Dropout do backbone seguem em modo train.

Fontes = 6 datasets DAGHAR + `combined`; alvos = só os 6 reais.

`--pretrain-source` (opcional) desacopla a fonte do *backbone* da fonte do
refinamento: com `--pretrain-source combined --source KuHar`, o backbone vem do
pré-treino no `combined` mas a cabeça/finetune usa só o train rotulado do
KuHar. Isola o efeito do corpus de pré-treino no cenário especialista
(experimento "comb→target"). Nesse modo o cache padrão muda para
`results/ssl_lfr_pre<PS>_eval_transfer.csv` para não contaminar o principal.

Saída (incremental) — `results/ssl_lfr_eval_transfer.csv`:
    encoder, source, seed, protocol, n_shots, target, test_acc, test_f1_macro

Uso:
    # Spike (um combo, todos os protocolos/regimes)
    python scripts/ssl/downstream_eval.py --encoder resnetse5 --source combined \
        --seed 0 --protocol both --shots all

    # Grade completa
    python scripts/ssl/downstream_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

from common import (  # noqa: E402
    BATCH_SIZE,
    DATASETS,
    NUM_CLASSES,
    PROJECT_ROOT,
    SEEDS,
    build_prediction_head,
    make_datamodule,
    normalize_shots,
    subsampled_train_loader,
)
from encoders import ENCODERS, build_backbone  # noqa: E402
from eval_transfer import DEVICE, evaluate, test_loader  # noqa: E402
from pretrain_lfr import SOURCES, ssl_ckpt_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
TARGETS = DATASETS
PROTOCOLS = ["linear", "finetune"]
CACHE = PROJECT_ROOT / "results" / "ssl_lfr_eval_transfer.csv"
COLS = ["encoder", "source", "seed", "protocol", "n_shots",
        "target", "test_acc", "test_f1_macro"]
KEY = ["encoder", "source", "seed", "protocol", "n_shots", "target"]

DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4                 # paper: 1e-4 p/ freeze E full fine-tuning
EARLY_STOP_PATIENCE = 50          # mesma paciência do baseline SL


# ---------------------------------------------------------------------------
# Modelo de sondagem: backbone -> cabeça MLP (a mesma do baseline SL)
# ---------------------------------------------------------------------------
class Probe(nn.Module):
    def __init__(self, backbone: nn.Module, enc_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = build_prediction_head(enc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = z.view(z.size(0), -1)
        return self.head(z)


def load_probe(encoder: str, pretrain_source: str, seed: int) -> Probe:
    """Constrói o backbone, injeta os pesos pré-treinados (LFR) e acopla a cabeça."""
    backbone, enc_dim = build_backbone(encoder)
    ckpt = ssl_ckpt_dir(encoder, pretrain_source, seed) / "backbone.ckpt"
    state = torch.load(ckpt, map_location="cpu")
    backbone.load_state_dict(state, strict=True)
    return Probe(backbone, enc_dim)


@torch.no_grad()
def _val_loss(probe: Probe, loader, crit: nn.Module) -> float:
    """Cross-entropy média (por amostra) no split de validação da fonte."""
    probe.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        total += crit(probe(x), y).item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def train_probe(
    probe: Probe,
    train_loader,
    val_loader,
    protocol: str,
    lr: float,
    epochs: int,
    patience: int = EARLY_STOP_PATIENCE,
) -> None:
    """Treina a cabeça (`linear`) ou tudo (`finetune`), com ES + melhor estado.

    Espelha o pipeline `train` do benchmark: até `epochs` épocas, parada
    antecipada com `patience` sobre a val_loss e avaliação final no melhor
    estado (análogo ao `best.ckpt` usado pelo baseline supervisionado).
    """
    probe.to(DEVICE)
    freeze = protocol == "linear"
    for p in probe.backbone.parameters():
        p.requires_grad_(not freeze)
    params = probe.head.parameters() if freeze else probe.parameters()

    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(epochs):
        # Congelamento só via requires_grad (semântica do SimpleSupervisedModel
        # do benchmark): BN/Dropout do backbone permanecem em modo train.
        probe.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            loss = crit(probe(x), y)
            loss.backward()
            opt.step()

        vloss = _val_loss(probe, val_loader, crit)
        if vloss < best_loss:
            best_loss, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)


# ---------------------------------------------------------------------------
# Cache (mesmo padrão de eval_transfer.py)
# ---------------------------------------------------------------------------
def load_cache() -> pd.DataFrame:
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        df["n_shots"] = df["n_shots"].astype(str)
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", choices=ENCODERS, nargs="+", default=ENCODERS)
    parser.add_argument("--source", choices=SOURCES, nargs="+", default=SOURCES)
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--protocol", choices=["linear", "finetune", "both"], default="both")
    parser.add_argument("--shots", nargs="+", default=["all"],
                        help="Regimes: inteiros, 'full', ou 'all' (1 10 100 full). Padrão: all.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="LR do refinamento (paper: 1e-4 p/ ambos os protocolos).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recalcula tudo, ignorando o cache.")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV de saída (parcial por-run p/ paralelização). Padrão: cache global.")
    parser.add_argument("--pretrain-source", choices=SOURCES, default=None,
                        help="Fonte do backbone, se diferente da fonte do refinamento "
                             "(ex.: 'combined' p/ o experimento comb→target).")
    args = parser.parse_args()

    global CACHE
    if args.out:
        CACHE = Path(args.out)
    elif args.pretrain_source:
        CACHE = PROJECT_ROOT / "results" / f"ssl_lfr_pre{args.pretrain_source}_eval_transfer.csv"

    protocols = PROTOCOLS if args.protocol == "both" else [args.protocol]
    shot_regimes: List[Union[int, str]] = normalize_shots(args.shots)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    done = set()
    if not args.force and not cache.empty:
        ok = cache.dropna(subset=["test_f1_macro"])
        done = set(zip(ok.encoder, ok.source, ok.seed.astype(int), ok.protocol,
                       ok.n_shots.astype(str), ok.target))

    print(f"Device: {DEVICE} | cache: {CACHE} ({len(cache)} linhas)", flush=True)

    new_rows = []
    for encoder in args.encoder:
        for source in args.source:
            for seed in args.seed:
                backbone_src = args.pretrain_source or source
                ckpt = ssl_ckpt_dir(encoder, backbone_src, seed) / "backbone.ckpt"
                if not ckpt.exists():
                    print(f"[miss] backbone ausente: {ckpt}", flush=True)
                    continue
                for protocol in protocols:
                    for n_shots in shot_regimes:
                        shots_str = str(n_shots)
                        todo = [t for t in TARGETS if args.force or
                                (encoder, source, seed, protocol, shots_str, t) not in done]
                        if not todo:
                            continue

                        seed_everything(seed, workers=True)
                        probe = load_probe(encoder, backbone_src, seed)
                        dm = make_datamodule(source, seed, num_workers=args.num_workers)
                        loader = subsampled_train_loader(
                            dm, n_shots, seed, batch_size=BATCH_SIZE,
                            num_workers=args.num_workers,
                        )
                        train_probe(probe, loader, dm.val_dataloader(), protocol,
                                    lr=args.lr, epochs=args.epochs, patience=args.patience)

                        probe.eval()
                        for target in todo:
                            acc, f1 = evaluate(probe, test_loader(target))
                            new_rows.append(dict(
                                encoder=encoder, source=source, seed=seed,
                                protocol=protocol, n_shots=shots_str, target=target,
                                test_acc=acc, test_f1_macro=f1,
                            ))
                            print(f"[OK] {encoder} {source}->{target} seed{seed} "
                                  f"{protocol} shots={shots_str}: acc={acc:.4f} f1={f1:.4f}",
                                  flush=True)
                        del probe
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

    if not new_rows:
        print("Nada a calcular — cache já completo. Use --force para recalcular.")
        return

    cache = (
        pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    cache.to_csv(CACHE, index=False)
    print(f"\n[CACHE] {len(new_rows)} linhas novas/atualizadas -> {CACHE} (total {len(cache)})")


if __name__ == "__main__":
    main()
