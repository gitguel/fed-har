"""Ablação SL para o experimento comb→target — isola o efeito do SSL.

O `comb2target` (SSL) faz `pré-treino no combined → finetune no alvo`. O ganho
observado vs os baselines SL pode vir do SSL OU só da etapa de especialização no
alvo. Este script é o baseline que fecha essa lacuna: usa o **backbone do modelo
SUPERVISIONADO treinado no `combined`** (`checkpoints/supervised/<enc>/combined/
seed<N>/best.ckpt`) como ponto de partida e refina no alvo com **exatamente o
mesmo downstream** do `downstream_eval.py` (mesmo `Probe`, cabeça nova, protocolos
`linear`/`finetune`, regimes 1/10/100/full, Adam 1e-4, ES paciência 50).

Assim, na diagonal `source==target`:
    Δ(SSL comb2target − SL comb2target)  isola o ganho atribuível ao **SSL**,
não à etapa de finetune no alvo. A grade completa (source×target) espelha
`results/ssl_{lfr,tfc}_comb2target_eval_transfer.csv` célula a célula.

Diferença vs `downstream_eval.py`: lá o backbone vem de um checkpoint SSL
(`build_backbone`/`build_tfc_backbone` + `backbone.ckpt`); aqui vem do modelo
supervisionado `combined` (`build_model().backbone`). Cabeça, treino e avaliação
são idênticos.

Saída (incremental) — `results/sl_comb2target_eval_transfer.csv`:
    encoder, source, seed, protocol, n_shots, target, test_acc, test_f1_macro
(mesmo esquema dos caches SSL comb2target; o método é SL, implícito no nome.)

Uso:
    # Spike (1 combo, 1 protocolo, 1 regime) — mede custo antes de extrapolar
    python scripts/ssl/sl_comb2target_eval.py --encoder resnetse5 \
        --source MotionSense --seed 0 --protocol finetune --shots 10

    # Grade completa
    python scripts/ssl/sl_comb2target_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from lightning.pytorch import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

from common import (  # noqa: E402
    BATCH_SIZE,
    COMBINED_DATASET_NAME,
    DATASETS,
    FULL_SHOTS,
    PROJECT_ROOT,
    SEEDS,
    make_datamodule,
    normalize_shots,
    subsampled_train_loader,
    supervised_ckpt_dir,
)
from encoders import ENC_DIM, ENCODERS  # noqa: E402
from downstream_eval import Probe, train_probe, DEFAULT_EPOCHS, DEFAULT_LR, EARLY_STOP_PATIENCE  # noqa: E402
from eval_transfer import BUILD_MODEL, DEVICE, evaluate, test_loader  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração — fontes de refinamento são os 6 datasets reais (nunca combined,
# que aqui é o corpus de PRÉ-TREINO, não de finetune); alvos são os 6 reais.
# ---------------------------------------------------------------------------
SOURCES = DATASETS
TARGETS = DATASETS
PROTOCOLS = ["linear", "finetune"]
CACHE = PROJECT_ROOT / "results" / "sl_comb2target_eval_transfer.csv"
COLS = ["encoder", "source", "seed", "protocol", "n_shots",
        "target", "test_acc", "test_f1_macro"]
KEY = ["encoder", "source", "seed", "protocol", "n_shots", "target"]


def load_sl_combined_probe(encoder: str, seed: int) -> Probe:
    """Backbone do modelo SUPERVISIONADO `combined` (full) + cabeça nova.

    Reconstrói o modelo supervisionado, injeta os pesos do `best.ckpt` treinado no
    `combined`, extrai `.backbone` e acopla a mesma cabeça MLP do downstream SSL.
    Cada chamada devolve um probe fresco (o finetune não vaza entre configs).
    """
    ckpt = supervised_ckpt_dir(encoder, COMBINED_DATASET_NAME, seed, FULL_SHOTS) / "best.ckpt"
    full = BUILD_MODEL[encoder]()
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    full.load_state_dict(state, strict=True)
    backbone = full.backbone
    if encoder == "cnnpff":
        # mesmo ajuste do build_backbone: CNN_PF_Backbone -> (B, 768) achatado.
        backbone.flatten = True
    return Probe(backbone, ENC_DIM[encoder])


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
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recalcula tudo, ignorando o cache.")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV de saída (parcial por-run). Padrão: cache global.")
    args = parser.parse_args()

    global CACHE
    if args.out:
        CACHE = Path(args.out)

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
    total_new = 0
    for encoder in args.encoder:
        for seed in args.seed:
            ckpt = supervised_ckpt_dir(encoder, COMBINED_DATASET_NAME, seed, FULL_SHOTS) / "best.ckpt"
            if not ckpt.exists():
                print(f"[miss] SL-combined ausente: {ckpt}", flush=True)
                continue
            for source in args.source:
                for protocol in protocols:
                    for n_shots in shot_regimes:
                        shots_str = str(n_shots)
                        todo = [t for t in TARGETS if args.force or
                                (encoder, source, seed, protocol, shots_str, t) not in done]
                        if not todo:
                            continue

                        seed_everything(seed, workers=True)
                        probe = load_sl_combined_probe(encoder, seed)
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
                            print(f"[OK] {encoder} comb->{source}->{target} seed{seed} "
                                  f"{protocol} shots={shots_str}: acc={acc:.4f} f1={f1:.4f}",
                                  flush=True)
                        del probe
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

                        # Flush incremental: grava o cache a cada config concluído,
                        # tornando o run crash-safe e resume-friendly (a próxima
                        # invocação pula o que já está no cache via `done`).
                        if new_rows:
                            cache = (
                                pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                                .drop_duplicates(subset=KEY, keep="last")
                                .sort_values(KEY)
                                .reset_index(drop=True)
                            )
                            cache.to_csv(CACHE, index=False)
                            total_new += len(new_rows)
                            new_rows = []

    if total_new == 0:
        print("Nada a calcular — cache já completo. Use --force para recalcular.")
        return

    print(f"\n[CACHE] {total_new} linhas novas/atualizadas -> {CACHE} (total {len(cache)})")


if __name__ == "__main__":
    main()
