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
# D6: busca S1 (só LR, E=5 fixo). Estendida em 2026-08-24: com a grade original
# de 4 pontos, 11 das 20 células escolhiam o TETO (3e-3) no regime k=full e a
# curva de val_acc ainda subia no último ponto (até +3,5 pp de 1e-3 para 3e-3 em
# cnnpff/RealWorld_thigh e tstcc/MotionSense). Grade truncada à direita vira LR
# sub-ótima travada em toda a RQ1, então o teto subiu para 3e-2.
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

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


def parcial_pronto(path, linhas: int, lr: float | None = None) -> bool:
    """Um parcial conta como pronto? Completo E treinado com a LR vigente.

    O teste de LR entrou em 2026-08-24 (D9). Antes, o skip dos drivers olhava so
    a contagem de linhas, entao um parcial de uma LR antiga sobrevivia a uma
    mudanca da busca S1 e a grade ficava com LRs misturadas, em silencio -- na
    virada para LR por regime, 9 dos 22 parciais federados completos estavam
    divergentes, um deles 10x fora. Parcial sem a coluna `lr` (anterior a essa
    data) e tratado como desconhecido, logo refeito.

    Vive aqui, e nao em cada driver, porque duas copias de uma checagem sutil
    divergem: e exatamente o bug que ela existe para evitar.
    """
    import pandas as pd
    if not path.exists():
        return False
    df = pd.read_csv(path).dropna(subset=["test_f1_macro"])
    if len(df) < linhas:
        return False
    if lr is None:
        return True
    if "lr" not in df.columns or df["lr"].isna().all():
        return False
    return bool((df["lr"].astype(float) - float(lr)).abs().max() < 1e-12)


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
