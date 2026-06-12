"""Avaliação de transfer learning cross-dataset (zero-shot) dos baselines supervisionados.

Para cada (encoder, fonte, semente), carrega o `best.ckpt` treinado na *fonte* e
avalia por inferência direta (modelo + cabeça congelados) no `test.csv` de cada um
dos 6 datasets-alvo. Calcula **acurácia** e **F1-macro**.

As "fontes" são os 6 datasets individuais + o modelo `combined` (treinado na união
dos 6). Os "alvos" são sempre apenas os 6 datasets reais — `combined` nunca é alvo.

Saída (incremental) — `results/supervised_eval_transfer.csv`:
    colunas: encoder, source, seed, target, test_acc, test_f1_macro

Uso:
    python scripts/eval_transfer.py            # calcula só o que falta no cache
    python scripts/eval_transfer.py --force    # recalcula tudo do zero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Coloca a pasta `scripts/` no sys.path para importar `common` e os scripts de treino.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from common import (
    CHECKPOINTS_ROOT,
    COMBINED_DATASET_NAME,
    DATASETS,
    PROJECT_ROOT,
    SEEDS,
    make_datamodule,
)
from supervised.train_resnetse5 import build_model as build_resnetse5
from supervised.train_cnnpff import build_model as build_cnnpff
from supervised.train_rnn import build_model as build_rnn

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ENCODERS = ["resnetse5", "cnnpff", "rnn"]
BUILD_MODEL = {
    "resnetse5": build_resnetse5,
    "cnnpff": build_cnnpff,
    "rnn": build_rnn,
}
SOURCES = [COMBINED_DATASET_NAME] + DATASETS  # combined primeiro; alvos = só os 6 reais
TARGETS = DATASETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = PROJECT_ROOT / "results" / "supervised_eval_transfer.csv"
COLS = ["encoder", "source", "seed", "target", "test_acc", "test_f1_macro"]
KEY = ["encoder", "source", "seed", "target"]


# ---------------------------------------------------------------------------
# Carregamento de modelo e dados
# ---------------------------------------------------------------------------
def load_checkpoint(encoder: str, ckpt_path: Path) -> torch.nn.Module:
    """Reconstrói o modelo via o `build_model` do treino e injeta o state_dict."""
    model = BUILD_MODEL[encoder]()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval().to(DEVICE)


_loaders: dict = {}

def test_loader(dataset: str):
    """Cacheia o test_dataloader de cada alvo para não reler os CSVs."""
    if dataset not in _loaders:
        dm = make_datamodule(dataset, seed=0, num_workers=0)
        dm.setup("test")
        _loaders[dataset] = dm.test_dataloader()
    return _loaders[dataset]


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader) -> tuple[float, float]:
    """Roda o modelo no loader e retorna (acurácia, F1-macro)."""
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
        y_true.append(np.asarray(y))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def load_cache() -> pd.DataFrame:
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Recalcula todas as combinações, ignorando o cache.")
    args = parser.parse_args()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    # Combinações já avaliadas COM F1-macro preenchido (a não ser que --force).
    done = set()
    if not args.force and not cache.empty:
        ok = cache.dropna(subset=["test_f1_macro"])
        done = set(zip(ok.encoder, ok.source, ok.seed.astype(int), ok.target))

    print(f"Device: {DEVICE} | cache: {CACHE} ({len(cache)} linhas)", flush=True)

    new_rows = []
    for encoder in ENCODERS:
        for source in SOURCES:
            for seed in SEEDS:
                ckpt = CHECKPOINTS_ROOT / encoder / source / f"seed{seed}" / "best.ckpt"
                if not ckpt.exists():
                    continue
                todo = [t for t in TARGETS
                        if args.force or (encoder, source, seed, t) not in done]
                if not todo:
                    continue
                model = load_checkpoint(encoder, ckpt)
                for target in todo:
                    acc, f1 = evaluate(model, test_loader(target))
                    new_rows.append(dict(encoder=encoder, source=source, seed=seed,
                                         target=target, test_acc=acc, test_f1_macro=f1))
                    print(f"[OK] {encoder} {source}->{target} seed{seed}: "
                          f"acc={acc:.4f} f1={f1:.4f}", flush=True)
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    if not new_rows:
        print("Nada a calcular — cache já completo. Use --force para recalcular.")
        return

    # Combina com o cache antigo: linhas recalculadas substituem as antigas (keep=last).
    cache = (
        pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    cache.to_csv(CACHE, index=False)
    print(f"\n[CACHE] {len(new_rows)} linhas novas/atualizadas -> {CACHE} "
          f"(total {len(cache)})")


if __name__ == "__main__":
    main()
