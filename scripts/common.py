"""Utilitários compartilhados pelos scripts de baseline supervisionado.

Usado por `train_supervised_resnetse5.py`, `train_supervised_cnnpff.py` e
`train_supervised_rnn.py`. O protocolo espelha o pipeline supervisionado do
benchmark DAGHAR (da Luz et al., IEEE Access 2026): mesmos datasets, mesma
cabeça MLP, mesma parada antecipada e mesma taxa de aprendizado por encoder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

# Coloca a raiz do projeto no sys.path para que `import minerva` funcione
# mesmo quando o script é executado de qualquer diretório.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minerva.data.data_modules.har import MultiModalHARSeriesDataModule  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes do protocolo (pipeline supervisionado do benchmark DAGHAR)
# ---------------------------------------------------------------------------
DAGHAR_ROOT = PROJECT_ROOT / "datasets" / "DAGHAR" / "standardized_view"
_SUP_CKPT_DEFAULT = PROJECT_ROOT / "checkpoints" / "supervised"


def checkpoints_root() -> Path:
    """Raiz dos checkpoints supervisionados, resolvida NA HORA DA CHAMADA.

    Sobrescrevível por `FEDHAR_SUP_CKPT_ROOT` para que a grade das RQs
    (docs/desenho_experimental.md) escreva em `checkpoints/rqs/supervised/` sem
    tocar nos artefatos antigos.

    Por que funcao e nao constante: `rqs/config.py` seta a env com `setdefault`,
    entao quem importasse `common` ANTES de `rqs.config` congelava a raiz antiga
    e passava a procurar checkpoint no lugar errado -- em silencio, sem erro. Um
    driver nessa ordem concluiria que a grade nao existe e retreinaria tudo.
    Resolvendo a cada chamada, a ordem de import deixa de importar.
    """
    return Path(os.environ.get("FEDHAR_SUP_CKPT_ROOT", _SUP_CKPT_DEFAULT))
LOGS_ROOT = PROJECT_ROOT / "logs" / "supervised"

DATASETS: List[str] = [
    "UCI",
    "MotionSense",
    "KuHar",
    "WISDM",
    "RealWorld_thigh",
    "RealWorld_waist",
]
# Pseudo-dataset que une os 6 sub-datasets DAGHAR num único treino. O DataModule
# concatena (`ConcatDataset`) os splits de todos eles. Usado pelo treino conjunto
# (`train_supervised_combined.py`) e tratado como uma "fonte" extra nas análises
# de transfer learning do notebook.
COMBINED_DATASET_NAME = "combined"
SEEDS: List[int] = [0, 1, 2, 3]

# Regimes de dados rotulados usados no treino downstream (few-shot + total). O
# eixo `n_shots` atravessa tanto o baseline supervisionado quanto a avaliação SSL.
# `"full"` = 100% do train set; os demais = amostras-por-classe (estratificado).
FULL_SHOTS = "full"
SHOT_REGIMES: List[Union[int, str]] = [1, 10, 100, FULL_SHOTS]

# Convenções padronizadas do DAGHAR para formato e rótulos das amostras.
INPUT_CHANNELS = 6
INPUT_TIMESTEPS = 60
INPUT_SHAPE = (INPUT_CHANNELS, INPUT_TIMESTEPS)
NUM_CLASSES = 6  # padronizado: {0:Sentar,1:Em pé,2:Andar,3:Subir,4:Descer,5:Correr}
FEATURE_PREFIXES = ("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z")
LABEL_COLUMN = "standard activity code"

# Hiperparâmetros de treino retirados do paper (Seção V-D + ablação Tabela 12).
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 50          # épocas sem melhora em val_loss
MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"

# Melhor taxa de aprendizado por encoder (Apêndice A, Tabela 12).
BEST_LR = {
    "resnetse5": 1e-4,
    "cnnpff":    1e-4,
    "rnn":       1e-4,
    "tstcc":     1e-4,
}

# Dimensão da camada oculta da cabeça de predição (paper, Seção V-C; Tabela 2).
HIDDEN_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Callback customizado: salva um checkpoint logo após a primeira época.
# ---------------------------------------------------------------------------
class FirstEpochCheckpoint(Callback):
    """Salva um checkpoint chamado ``first.ckpt`` ao fim da época 0."""

    def __init__(self, dirpath: Path, filename: str = "first") -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self._saved = False

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._saved:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"{self.filename}.ckpt"
        trainer.save_checkpoint(ckpt_path)
        self._saved = True


# ---------------------------------------------------------------------------
# Cabeça de predição padrão: Linear(in -> 128) -> ReLU -> Linear(128 -> 6)
# ---------------------------------------------------------------------------
def build_prediction_head(in_features: int, num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """Cabeça MLP do paper (Seção V-C): uma camada oculta de 128 unidades,
    ativação ReLU e uma camada de saída com ``num_classes`` unidades."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, HIDDEN_HEAD_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_HEAD_DIM, num_classes),
    )


# ---------------------------------------------------------------------------
# Fábrica de DataModule
# ---------------------------------------------------------------------------
def make_datamodule(
    dataset_name: str,
    seed: int,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> MultiModalHARSeriesDataModule:
    """Cria o DataModule padrão do DAGHAR.

    Para um sub-dataset individual, ``dataset_name`` é o nome da pasta. Para o
    treino conjunto, passe ``COMBINED_DATASET_NAME`` ("combined"): o DataModule
    recebe a lista dos 6 caminhos e concatena os splits de todos eles.
    """
    if dataset_name == COMBINED_DATASET_NAME:
        dataset_path = [DAGHAR_ROOT / name for name in DATASETS]
        missing = [str(p) for p in dataset_path if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Sub-datasets DAGHAR ausentes para o treino conjunto: "
                f"{missing}. Você baixou o `standardized_view`?"
            )
    else:
        dataset_path = DAGHAR_ROOT / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Sub-dataset DAGHAR não encontrado em {dataset_path}. "
                "Você baixou o `standardized_view`?"
            )
    return MultiModalHARSeriesDataModule(
        data_path=dataset_path,
        feature_prefixes=FEATURE_PREFIXES,
        label=LABEL_COLUMN,
        features_as_channels=True,
        cast_to="float32",
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Regimes de dados (few-shot): subamostragem estratificada do train
# ---------------------------------------------------------------------------
def _train_dataset(dm: MultiModalHARSeriesDataModule):
    """Devolve o `Dataset` de treino (após `setup('fit')`).

    O DataModule guarda cada split como uma tupla ``(dataset, domain_labels)``.
    """
    dataset = dm.datasets["train"]
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    return dataset


def _dataset_labels(dataset) -> np.ndarray:
    """Extrai o vetor de rótulos de um dataset que devolve ``(x, y)`` por índice."""
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def few_shot_indices(
    dataset,
    n_per_class: Optional[Union[int, str]],
    seed: int,
) -> List[int]:
    """Índices de um subconjunto com ``n_per_class`` amostras por classe.

    Seleção determinística (RNG semeado) e estratificada por classe. Se uma
    classe tiver menos que ``n_per_class`` amostras, usa todas as disponíveis.
    ``n_per_class`` igual a ``None`` ou ``"full"`` devolve todos os índices.
    """
    if n_per_class is None or n_per_class == FULL_SHOTS:
        return list(range(len(dataset)))

    labels = _dataset_labels(dataset)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[: int(n_per_class)].tolist())
    selected.sort()
    return selected


def subsampled_train_loader(
    dm: MultiModalHARSeriesDataModule,
    n_per_class: Optional[Union[int, str]],
    seed: int,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """`DataLoader` de treino restrito a ``n_per_class`` amostras por classe.

    Faz `dm.setup('fit')` se necessário, embrulha o dataset de treino num
    `Subset` com os índices de :func:`few_shot_indices` e devolve o loader.
    Para ``n_per_class in (None, 'full')`` usa o train completo.
    """
    if "train" not in dm.datasets:
        dm.setup("fit")
    train_ds = _train_dataset(dm)
    idx = few_shot_indices(train_ds, n_per_class, seed)
    subset = Subset(train_ds, idx)
    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def supervised_ckpt_dir(encoder: str, dataset: str, seed: int, n_shots: Union[int, str] = FULL_SHOTS) -> Path:
    """Diretório de checkpoints de um treino supervisionado.

    O regime ``"full"`` mantém o layout histórico (sem subpasta ``shots``) para
    preservar os 84 checkpoints já existentes; os regimes few-shot ficam num
    subdiretório ``shots<K>``.
    """
    base = checkpoints_root() / encoder / dataset / f"seed{seed}"
    return base if n_shots == FULL_SHOTS else base / f"shots{n_shots}"


# ---------------------------------------------------------------------------
# Fábrica do Trainer
# ---------------------------------------------------------------------------
def build_callbacks(run_dir: Path) -> List[Callback]:
    """Três checkpoints (primeiro / melhor / último) e parada antecipada."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    first_ckpt = FirstEpochCheckpoint(dirpath=run_dir, filename="first")
    # ModelCheckpoint com save_last=True escreve tanto `best.ckpt` quanto `last.ckpt`.
    #
    # `enable_version_counter=False` (2026-08-24): por padrao o Lightning NAO
    # sobrescreve -- se `best.ckpt` ja existe, ele grava `best-v1.ckpt`. Como a
    # avaliacao le `best.ckpt`, um re-treino com `--force` queimava GPU e o
    # resultado era descartado em silencio: o numero reportado continuava o da
    # corrida velha. Foi assim que 3 celulas com corridas patologicas
    # sobreviveram a uma tentativa de refazer.
    best_and_last = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    early_stop = EarlyStopping(
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        patience=EARLY_STOP_PATIENCE,
    )
    return [first_ckpt, best_and_last, early_stop]


def build_trainer(run_dir: Path, log_dir: Path, max_epochs: int = MAX_EPOCHS) -> L.Trainer:
    log_dir = Path(log_dir)
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(log_dir.parent), name=log_dir.name)
    return L.Trainer(
        max_epochs=max_epochs,
        callbacks=build_callbacks(run_dir),
        logger=logger,
        accelerator="auto",
        devices=1,
        deterministic="warn",
        log_every_n_steps=10,
        enable_progress_bar=True,
    )


# ---------------------------------------------------------------------------
# Execução de uma única combinação e laço pela grade completa
# ---------------------------------------------------------------------------
def run_single(
    model_factory: Callable[[], L.LightningModule],
    encoder_name: str,
    dataset: str,
    seed: int,
    num_workers: int = 4,
    max_epochs: int = MAX_EPOCHS,
    n_shots: Union[int, str] = FULL_SHOTS,
) -> None:
    """Treina, de ponta a ponta, uma combinação (encoder, dataset, semente).

    ``n_shots`` seleciona o regime de dados rotulados: ``"full"`` treina com o
    train completo (comportamento histórico); um inteiro treina com esse número
    de amostras por classe (few-shot). Validação e teste usam sempre 100%.
    """
    seed_everything(seed, workers=True)
    model = model_factory()
    dm = make_datamodule(dataset, seed, num_workers=num_workers)

    run_dir = supervised_ckpt_dir(encoder_name, dataset, seed, n_shots)
    shots_tag = "full" if n_shots == FULL_SHOTS else f"shots{n_shots}"
    log_dir = LOGS_ROOT / encoder_name / dataset / f"seed{seed}" / shots_tag
    trainer = build_trainer(run_dir=run_dir, log_dir=log_dir, max_epochs=max_epochs)

    print(
        f"\n=== [{encoder_name}] dataset={dataset} semente={seed} "
        f"n_shots={n_shots} max_epochs={max_epochs} -> checkpoints={run_dir} ===",
        flush=True,
    )
    if n_shots == FULL_SHOTS:
        trainer.fit(model, datamodule=dm)
    else:
        train_loader = subsampled_train_loader(
            dm, n_shots, seed, batch_size=BATCH_SIZE, num_workers=num_workers
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dm.val_dataloader())
    trainer.test(model, datamodule=dm, ckpt_path="best")


def run_grid(
    model_factory: Callable[[], L.LightningModule],
    encoder_name: str,
    datasets: Iterable[str] = DATASETS,
    seeds: Iterable[int] = SEEDS,
    num_workers: int = 4,
    max_epochs: int = MAX_EPOCHS,
    shot_regimes: Iterable[Union[int, str]] = (FULL_SHOTS,),
) -> None:
    """Treina toda a grade (dataset, semente, regime) para um único encoder."""
    for dataset in datasets:
        for seed in seeds:
            for n_shots in shot_regimes:
                run_single(
                    model_factory=model_factory,
                    encoder_name=encoder_name,
                    dataset=dataset,
                    seed=seed,
                    num_workers=num_workers,
                    max_epochs=max_epochs,
                    n_shots=n_shots,
                )


# ---------------------------------------------------------------------------
# CLI comum: permite que cada script aceite --dataset / --seed / --num-workers.
# ---------------------------------------------------------------------------
def make_argparser(encoder_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Treinamento supervisionado de baseline para o encoder {encoder_name} no DAGHAR."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS + [COMBINED_DATASET_NAME],
        nargs="+",
        default=DATASETS,
        help=(
            "Subconjunto de datasets DAGHAR para treinar (padrão: todos os seis). "
            f"Use '{COMBINED_DATASET_NAME}' para treinar com todos juntos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Sementes a serem rodadas (padrão: 0 1 2 3).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Workers do DataLoader (padrão: 4).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Máximo de épocas por treino (padrão: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=[FULL_SHOTS],
        help=(
            "Regimes de dados a treinar: inteiros (amostras-por-classe), 'full' "
            "(100%%) ou 'all' (1 10 100 full). Padrão: full."
        ),
    )
    return parser


def normalize_shots(values: Iterable[Union[int, str]]) -> List[Union[int, str]]:
    """Converte tokens de CLI (`1 10 100 full` / `all`) em regimes canônicos."""
    values = list(values)
    if any(str(v).lower() == "all" for v in values):
        return list(SHOT_REGIMES)
    out: List[Union[int, str]] = []
    for v in values:
        v = str(v).lower()
        out.append(FULL_SHOTS if v == FULL_SHOTS else int(v))
    return out


def __getattr__(name):
    """Mantem `CHECKPOINTS_ROOT` acessivel (PEP 562), resolvido na hora do acesso.

    Legado: codigo novo deve chamar `checkpoints_root()`. Este atalho existe para
    nao quebrar os notebooks antigos, que leem a grade supervisionada original.
    """
    if name == "CHECKPOINTS_ROOT":
        return checkpoints_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
