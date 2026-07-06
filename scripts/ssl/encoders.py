"""Fábrica de *backbones* (encoders) para os scripts SSL.

Reusa os `build_model()` dos baselines supervisionados (`scripts/supervised/`)
e devolve apenas o `backbone` limpo + a dimensão de features (`enc_dim`), sem a
cabeça de classificação. Um único ponto de verdade para `pretrain_lfr.py` e
`downstream_eval.py`.

`enc_dim` por encoder (ver docs dos scripts supervisionados):
    resnetse5 -> 64   (ResNet-SE-5, `fc_input_features`)
    cnnpff    -> 768  (CNN-PFF, `fc_input_channels`)
    rnn       -> 320  (BiGRU, `RnnEncoder.encoding_size`)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from supervised.train_resnetse5 import build_model as build_resnetse5  # noqa: E402
from supervised.train_cnnpff import build_model as build_cnnpff  # noqa: E402
from supervised.train_rnn import build_model as build_rnn  # noqa: E402

ENCODERS = ["resnetse5", "cnnpff", "rnn"]

_BUILDERS = {
    "resnetse5": build_resnetse5,
    "cnnpff": build_cnnpff,
    "rnn": build_rnn,
}

# Dimensão de saída do backbone (após achatamento) por encoder.
ENC_DIM = {
    "resnetse5": 64,
    "cnnpff": 768,
    "rnn": 320,
}


def build_backbone(encoder: str) -> Tuple[torch.nn.Module, int]:
    """Constrói o backbone do encoder e devolve `(backbone, enc_dim)`.

    Descarta a cabeça de classificação do modelo supervisionado — só o extrator
    de features é usado no pré-treino SSL e reaproveitado no downstream.
    """
    if encoder not in _BUILDERS:
        raise ValueError(f"Encoder desconhecido: {encoder}. Opções: {ENCODERS}")
    model = _BUILDERS[encoder]()
    backbone = model.backbone
    if encoder == "cnnpff":
        # O CNN_PF_Backbone devolve (B, 64, 12); a config oficial do LFR usa
        # `flatten=True` interno para sair (B, 768), como os outros dois
        # backbones (que já devolvem 2D). Não afeta o state_dict.
        backbone.flatten = True
    return backbone, ENC_DIM[encoder]
