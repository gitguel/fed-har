"""Constantes da grade das RQ1/RQ2 — ver `docs/desenho_experimental.md`.

Fonte única dos parâmetros. Se um número aqui divergir do documento, **o
documento manda** e a divergência é bug. `verifica_clientes()` confere a tabela
de clientes contra os CSVs; todo driver chama antes de enfileirar job.

Importe este módulo ANTES de `common` — ele fixa as raízes de checkpoint das RQs
por variável de ambiente, para não tocar nos artefatos das grades antigas.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))

# Raízes isoladas: nada da grade das RQs escreve por cima do que já existe.
os.environ.setdefault("FEDHAR_SUP_CKPT_ROOT", str(_ROOT / "checkpoints" / "rqs" / "supervised"))
os.environ.setdefault("FEDHAR_FEDSSL_CKPT_ROOT", str(_ROOT / "checkpoints" / "rqs" / "ssl_fed"))

from common import FULL_SHOTS, PROJECT_ROOT  # noqa: E402

# --- Desenho (docs/desenho_experimental.md §2 e §3) -------------------------
# D2: KuHar fora. Clientes = usuários do split `train`.
FEDERACOES: dict[str, int] = {
    "UCI": 21,
    "MotionSense": 17,
    "WISDM": 36,
    "RealWorld_thigh": 10,
    "RealWorld_waist": 10,
}
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
K_LEVELS = [1, 2, 4, FULL_SHOTS]      # rótulos por classe POR CLIENTE
SEEDS_RQ = [0, 1, 2]                  # 3 seeds na 1ª onda; a 4ª é incremento
METHODS = ["tfc", "lfr"]              # D1: SimCLR fora da 1ª onda

R_FT = 150                            # rodadas de fine-tuning federado
R_PRE = 100                           # rodadas de pré-treino federado
E_LOCAL = 5                           # épocas EFETIVAS de backbone por rodada (D6)
LOCAL_EPOCHS_PRE = {"tfc": E_LOCAL, "lfr": 6 * E_LOCAL}   # LFR alterna 1:5
BUDGET = 0                            # D3: partição natural
MAX_EPOCHS_CENTR = 100                # D4: protocolo do benchmark, com early stopping
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]    # D6: busca S1 (só LR, E=5 fixo)

# --- Saídas ----------------------------------------------------------------
RESULTS = PROJECT_ROOT / "results" / "rqs"
LOGS = PROJECT_ROOT / "logs" / "rqs"
CKPT = PROJECT_ROOT / "checkpoints" / "rqs"


def spec(dataset: str) -> str:
    """Spec cross-device na configuração MÁXIMA de clientes daquele dataset."""
    return f"device:{dataset}:{FEDERACOES[dataset]}"


def shots_centralizado(dataset: str, k):
    """Regime centralizado pareado com `k` por cliente: `k × n_clientes`."""
    return FULL_SHOTS if k == FULL_SHOTS else k * FEDERACOES[dataset]


def verifica_clientes() -> None:
    """Confere FEDERACOES contra os CSVs. Aborta se divergir — o pareamento
    centralizado (`k × n_clientes`) depende deste número estar certo."""
    import pandas as pd
    view = PROJECT_ROOT / "datasets" / "DAGHAR" / "standardized_view"
    for ds, n in FEDERACOES.items():
        real = pd.read_csv(view / ds / "train.csv", usecols=["user"]).user.nunique()
        if real != n:
            raise SystemExit(
                f"[CONFIG] {ds}: config diz {n} clientes, o train.csv tem {real}. "
                f"Corrija FEDERACOES e o §2.1 de docs/desenho_experimental.md."
            )
    print(f"[CONFIG] {len(FEDERACOES)} federações conferidas contra os CSVs.", flush=True)
