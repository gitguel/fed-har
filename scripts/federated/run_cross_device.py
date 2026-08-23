"""FedAvg supervisionado e fine-tuning Fed-SSL no eixo CROSS-DEVICE — sem Flower.

O loop local do `client.py` sempre foi torch puro (Adam + CrossEntropy); o que o
Flower acrescentava era a casca `NumPyClient` e a orquestração Ray. No
cross-device isso não paga: os clientes rodam sequencialmente na mesma GPU, e
tirar o Ray elimina de quebra o gotcha de `PYTHONPATH` que já mordeu o projeto.

Este runner e o `scripts/ssl/pretrain_fed.py` consomem a **mesma**
`cross_device.make_clients`, para que o Δ(Fed-SSL − baseline) não carregue
diferença de partição junto com o efeito do método.

**Os dois braços vivem aqui** (sabatina de 2026-07-29). Três flags opcionais
transformam o baseline no downstream federado do Fed-SSL:

    --method {none,lfr,tfc}   arquitetura/origem do modelo global
    --init-ckpt PATH          backbone.ckpt do pré-treino federado
    --shots {1,2,5,10,full}   rótulos por classe POR CLIENTE (ladder)

Com os defaults (`none` / sem ckpt / `full`) o caminho é **bit-idêntico** ao
baseline supervisionado — é o gate **G-DEG**, e é por isso que os dois braços
compartilham um script em vez de dois. `--method none` usa o modelo supervisionado
(`BUILD_MODEL`); `lfr` e `tfc` usam o `Probe` do downstream centralizado
(`ssl/downstream_eval.py`), que é a mesma cabeça MLP sobre o backbone do método.

⚠️ O `Probe` do TF-C tem 2,2–4,0× os parâmetros do modelo supervisionado (backbone
composto: duas cópias do encoder + dois projetores). Isso é **constitutivo do
método** — o benchmark do da Luz declara que *"performance differences observed
under TF-C should be interpreted in light of this composite backbone
configuration"* e não compensa a diferença. Nós seguimos a mesma convenção e
reportamos `uplink_mb` ao lado da acurácia. Ver `metodo_e_auditoria.md`.

A ladder recorta **dentro do shard de orçamento** (`budget=192`), não dos dados
completos do usuário: é o que faz o degrau `full` coincidir exatamente com o
baseline já medido, e mantém os pesos do FedAvg uniformes (todo cliente com `6L`).

Saída: `results/fed_cross_device.csv` (ou `--out`)
    encoder, spec, budget, seed, local_epochs, round, target,
    test_acc, test_f1_macro, val_acc, val_f1_macro, uplink_mb, downlink_mb,
    method, n_shots, pretrain_rounds

Uso:
    # gate: 1 cliente com as mesmas janelas, R=1, deve reproduzir o centralizado
    python scripts/federated/run_cross_device.py --gate \
        --spec single:RealWorld_thigh:10 --encoder resnetse5 --local-epochs 5 --seed 0

    # baseline (comportamento histórico)
    python scripts/federated/run_cross_device.py \
        --spec device:RealWorld_thigh:10 --encoder resnetse5 --rounds 150 --local-epochs 5

    # um degrau da ladder do baseline
    python scripts/federated/run_cross_device.py \
        --spec device:RealWorld_thigh:10 --encoder rnn --rounds 150 --local-epochs 5 --shots 5

    # fine-tuning federado de um backbone pré-treinado
    python scripts/federated/run_cross_device.py --method tfc \
        --init-ckpt checkpoints/ssl_fed/tfc/rnn/device-RealWorld_thigh-10/seed0/backbone.ckpt \
        --spec device:RealWorld_thigh:10 --encoder rnn --rounds 150 --local-epochs 5 --shots 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Precisa estar definido ANTES de o CUDA inicializar, senão
# `use_deterministic_algorithms` falha nas GEMMs do cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl"))

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import ConcatDataset, DataLoader, Subset

from common import (BATCH_SIZE, BEST_LR, FULL_SHOTS, NUM_CLASSES, PROJECT_ROOT,
                    SEEDS, few_shot_indices, make_datamodule)
from downstream_eval import BUILD_SSL_BACKBONE, Probe
from eval_transfer import BUILD_MODEL, DEVICE, evaluate, test_loader
from federated.cross_device import DEFAULT_BUDGET, make_clients, parse_spec
from pretrain_fed import fedavg, state_bytes

CACHE = PROJECT_ROOT / "results" / "fed_cross_device.csv"
COLS = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target",
        "test_acc", "test_f1_macro", "val_acc", "val_f1_macro",
        "uplink_mb", "downlink_mb", "method", "n_shots", "pretrain_rounds",
        "pretrain_spec"]
METHODS = ["none", "lfr", "tfc"]
SHOT_LEVELS = [1, 2, 5, 10, FULL_SHOTS]

_val_loaders: dict = {}


def val_loader(dataset: str):
    """`validation.csv` do alvo — split de seleção, nunca de reporte.

    Existe para que o `best.ckpt` seja escolhido **sem olhar o teste**: com k=5 a
    curva do FedAvg tem pico cedo e degrada, então "melhor rodada" é uma escolha
    real de modelo, e fazê-la no teste inflaria o número reportado.
    """
    if dataset not in _val_loaders:
        dm = make_datamodule(dataset, seed=0, num_workers=0)
        dm.setup("fit")
        _val_loaders[dataset] = dm.val_dataloader()
    return _val_loaders[dataset]


def set_deterministic() -> None:
    """Torna o treino bit-reprodutível na GPU.

    Sem isto, dois treinos idênticos divergem ~1e-2 nos pesos depois de algumas
    épocas (algoritmos não-determinísticos do cuDNN no backward das convoluções)
    — medido em 2026-07-28, e foi o que reprovou a primeira execução do gate.
    Fora do gate não é obrigatório: a grade tem 4 seeds e a variação entra na
    barra de erro.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def local_train(model: torch.nn.Module, dataset, epochs: int, lr: float,
                seed: int, num_workers: int = 0) -> dict:
    """Treino local de um cliente: o mesmo loop de `client.py`, sem a casca Flower.

    Devolve o `state_dict` em CPU. `seed` semeia o shuffle do loader — os dois
    caminhos do gate (federado com 1 cliente e centralizado) chamam esta mesma
    função com a mesma seed, que é o que torna a comparação bit-a-bit legítima.
    """
    seed_everything(seed, workers=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    assert len(loader) > 0, f"cliente com {len(dataset)} amostras gera 0 batches"
    model = model.to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_global(encoder: str, method: str) -> torch.nn.Module:
    """Modelo global do braço: supervisionado (`none`) ou `Probe` do método SSL.

    O `Probe` (`ssl/downstream_eval.py`) é backbone do método + a MESMA cabeça MLP
    do baseline (`common.build_prediction_head`), então no LFR ele é
    parâmetro-a-parâmetro idêntico ao modelo supervisionado. No TF-C não é — ver
    o aviso do docstring do módulo.
    """
    if method == "none":
        return BUILD_MODEL[encoder]()
    backbone, enc_dim = BUILD_SSL_BACKBONE[method](encoder)
    return Probe(backbone, enc_dim)


def _fresh(encoder: str, state: dict, method: str = "none") -> torch.nn.Module:
    model = build_global(encoder, method)
    model.load_state_dict(state)
    return model


def initial_state(encoder: str, method: str, init_ckpt: Path | None) -> dict:
    """Estado global da rodada 0: aleatório, ou com o backbone pré-treinado dentro.

    Implementa o gate **G-LOAD**: o `strict=True` pega incompatibilidade de chaves
    (o `TFC_Backbone` tem layout próprio) e a comparação com a inicialização
    aleatória pega o no-op silencioso da família F2 — um pré-treino em que algum
    cliente gerou 0 batches devolve os pesos globais intactos e "termina" sem erro.
    """
    model = build_global(encoder, method)
    if init_ckpt is None:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    before = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    model.backbone.load_state_dict(
        torch.load(init_ckpt, map_location="cpu"), strict=True)
    after = model.backbone.state_dict()
    if all(torch.equal(before[k], after[k].cpu()) for k in before):
        raise RuntimeError(
            f"G-LOAD: {init_ckpt} é idêntico à inicialização aleatória — o "
            f"pré-treino não treinou nada (0 passos de gradiente?).")
    print(f"[G-LOAD] backbone carregado de {init_ckpt} (strict=True, pesos != init)",
          flush=True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def apply_shots(clients: list, n_shots, seed: int) -> list:
    """Recorta `n_shots` rótulos por classe DENTRO do shard de cada cliente.

    `n_shots` é **por classe e por cliente** — o pareamento com o centralizado é
    `n_shots × n_clientes` (docs/desenho_experimental.md §3). Com `--budget 192` a
    ladder mora dentro do orçamento; com `--budget 0` (partição natural, o padrão
    das RQ1/RQ2) ela mora nos dados completos do usuário.

    Duas invariantes são checadas, porque quebrá-las desencaixa os degraus da
    ladder em silêncio:

    1. o recorte preserva **todas as classes que aquele cliente tem** — se sumir
       uma, `few_shot_indices` passa a percorrer um `np.unique` diferente e o
       consumo do RNG desloca entre degraus;
    2. **todos os clientes veem o mesmo conjunto de classes** — senão os shards
       deixam de ser comparáveis entre si e o peso do FedAvg mistura cobertura
       com volume.

    Ambas são dataset-agnósticas de propósito: UCI tem 5 classes e WISDM tem 4
    (`dados_daghar.md §1`), então comparar com `NUM_CLASSES` seria errado.
    """
    if n_shots == FULL_SHOTS:
        return clients
    out, cobertura = [], []
    for cid, shard in clients:
        todas = {int(shard[i][1]) for i in range(len(shard))}
        idx = few_shot_indices(shard, n_shots, seed)
        present = {int(shard[i][1]) for i in idx}
        if present != todas:
            raise ValueError(
                f"cliente {cid}: o recorte de {n_shots} shots perdeu as classes "
                f"{sorted(todas - present)} — a ladder deixa de ser aninhada.")
        cobertura.append((cid, todas))
        out.append((cid, Subset(shard, idx)))
    ref_cid, ref = cobertura[0]
    for cid, classes in cobertura[1:]:
        if classes != ref:
            raise ValueError(
                f"clientes com conjuntos de classes diferentes: {ref_cid} tem "
                f"{sorted(ref)} e {cid} tem {sorted(classes)} — o pareamento "
                f"`n_shots × n_clientes` deixa de valer (ver desenho_experimental.md §3).")
    return out


def _eval_all(model: torch.nn.Module, targets: list[str]) -> list[tuple[str, float, float]]:
    model = model.to(DEVICE).eval()
    return [(t, *evaluate(model, test_loader(t))) for t in targets]


def _eval_val(model: torch.nn.Module, targets: list[str]) -> list[tuple[str, float, float]]:
    model = model.to(DEVICE).eval()
    return [(t, *evaluate(model, val_loader(t))) for t in targets]


def run(spec: str, encoder: str, rounds: int, local_epochs: int, seed: int,
        budget: int | None, targets: list[str], num_workers: int = 0,
        ckpt_dir: Path | None = None, method: str = "none",
        init_ckpt: Path | None = None, n_shots=FULL_SHOTS,
        pretrain_rounds: int = 0, pretrain_spec: str = "",
        lr: float | None = None) -> pd.DataFrame:
    # `pretrain_spec` == "" significa "o backbone veio do MESMO spec do finetuning"
    # (ou não há backbone). Só é preenchido quando as duas populações diferem —
    # é o que distingue as células que separam pré-treino de fine-tuning.
    seed_everything(seed, workers=True)
    clients = apply_shots(make_clients(spec, seed, budget=budget), n_shots, seed)
    weights = [len(shard) for _, shard in clients]
    lr = BEST_LR[encoder] if lr is None else float(lr)

    global_state = initial_state(encoder, method, init_ckpt)
    mb = state_bytes(global_state) / 2**20
    tag = "FedAvg-SL" if method == "none" else f"FedSSL-{method}"
    print(f"=== [{tag} {encoder}] {spec} seed{seed} shots={n_shots}: "
          f"{len(clients)} clientes ({sum(weights)} janelas), "
          f"R={rounds}×E={local_epochs}, "
          f"{mb:.2f} MB/cliente/rodada/sentido ===", flush=True)

    rows = []
    best_val, best_round = -1.0, -1
    for r in range(1, rounds + 1):
        t0 = time.time()
        states = [
            local_train(_fresh(encoder, global_state, method), shard, local_epochs,
                        lr, seed=seed * 10**6 + r * 10**3 + ci,
                        num_workers=num_workers)
            for ci, (_cid, shard) in enumerate(clients)
        ]
        global_state = fedavg(states, weights)
        model = _fresh(encoder, global_state, method)
        scores = _eval_all(model, targets)
        vscores = {t: (a, f) for t, a, f in _eval_val(model, targets)}
        for tgt, acc, f1 in scores:
            vacc, vf1 = vscores[tgt]
            rows.append({"encoder": encoder, "spec": spec,
                         "budget": budget if budget is not None else "natural",
                         "seed": seed, "local_epochs": local_epochs, "round": r,
                         "target": tgt, "test_acc": acc, "test_f1_macro": f1,
                         "val_acc": vacc, "val_f1_macro": vf1,
                         "uplink_mb": mb * len(clients),
                         "downlink_mb": mb * len(clients),
                         "method": method, "n_shots": str(n_shots),
                         "pretrain_rounds": pretrain_rounds,
                         "pretrain_spec": pretrain_spec})

        # Seleção do `best` na validação (média sobre os alvos do spec) — o teste
        # nunca entra na decisão, só no reporte.
        mean_val = sum(a for a, _ in vscores.values()) / len(vscores)
        if ckpt_dir is not None and mean_val > best_val:
            best_val, best_round = mean_val, r
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": global_state, "round": r,
                        "val_acc_mean": mean_val, "encoder": encoder,
                        "spec": spec, "seed": seed, "local_epochs": local_epochs,
                        "method": method, "n_shots": str(n_shots),
                        "pretrain_rounds": pretrain_rounds,
                        "pretrain_spec": pretrain_spec},
                       ckpt_dir / "best.ckpt")
        print(f"[r{r}/{rounds}] {time.time() - t0:5.1f}s | "
              + ", ".join(f"{t}={a:.4f}" for t, a, _ in scores)
              + f" | val={mean_val:.4f}" + (" *" if best_round == r else ""),
              flush=True)
    if ckpt_dir is not None:
        print(f"  best.ckpt = rodada {best_round} (val {best_val:.4f}) -> {ckpt_dir}",
              flush=True)
    return pd.DataFrame(rows, columns=COLS)


def gate(spec: str, encoder: str, local_epochs: int, seed: int,
         budget: int | None, targets: list[str]) -> bool:
    """G-EQ1: federado com 1 cliente e R=1 tem de reproduzir o centralizado.

    Valida de uma vez o caminho de dados (`make_clients`), o loop local e o
    `fedavg` — com 1 cliente a média é a identidade, então qualquer divergência
    aponta bug de estado/seed, não de agregação.
    """
    set_deterministic()
    clients = make_clients(spec, seed, budget=budget)
    if len(clients) != 1:
        raise ValueError(f"o gate exige 1 cliente; {spec!r} deu {len(clients)}")
    shard = clients[0][1]
    print(f"gate: {spec} -> 1 cliente, {len(shard)} janelas, E={local_epochs}, seed={seed}")

    seed_everything(seed, workers=True)
    init = {k: v.detach().cpu().clone()
            for k, v in BUILD_MODEL[encoder]().state_dict().items()}
    local_seed = seed * 10**6 + 1 * 10**3 + 0

    fed_state = fedavg([local_train(_fresh(encoder, init), shard, local_epochs,
                                    BEST_LR[encoder], seed=local_seed)], [len(shard)])
    cen_state = local_train(_fresh(encoder, init), shard, local_epochs,
                            BEST_LR[encoder], seed=local_seed)

    max_dev = max((fed_state[k].to(torch.float64) - cen_state[k].to(torch.float64))
                  .abs().max().item() for k in fed_state)
    fed = _eval_all(_fresh(encoder, fed_state), targets)
    cen = _eval_all(_fresh(encoder, cen_state), targets)
    print(f"\n  maior divergência de peso federado-vs-centralizado: {max_dev:.3e}")
    print(f"  {'alvo':<18}{'federado':>22}{'centralizado':>22}")
    ok = True
    for (t, fa, ff), (_t, ca, cf) in zip(fed, cen):
        print(f"  {t:<18}{f'{fa:.6f}/{ff:.6f}':>22}{f'{ca:.6f}/{cf:.6f}':>22}")
        ok &= abs(fa - ca) < 1e-9 and abs(ff - cf) < 1e-9
    print(f"\n  GATE {'PASS' if ok and max_dev < 1e-6 else 'FAIL'} "
          f"(acc/F1 idênticos e pesos equivalentes)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="<modo>:<datasets>:<contagens>")
    ap.add_argument("--encoder", required=True, choices=sorted(BUILD_MODEL))
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None,
                    help="LR do cliente. Padrão: BEST_LR[encoder]. Usado pela busca S1 do eixo federado (docs/desenho_experimental.md).")
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="Janelas por cliente (0 = natural, sem parear).")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Alvos de avaliação. Default: os datasets do spec.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="CSV de saída (parcial por-run). Default: cache global.")
    ap.add_argument("--ckpt-dir", default=None,
                    help="Se dado, salva o `best.ckpt` (melhor rodada pela "
                         "validação) em <ckpt-dir>/seed<N>/. Sem isto, nada é salvo.")
    ap.add_argument("--gate", action="store_true",
                    help="Roda o G-EQ1 (1 cliente, R=1 ≡ centralizado) e sai.")
    ap.add_argument("--method", choices=METHODS, default="none",
                    help="Arquitetura do modelo global: 'none' = supervisionado "
                         "(baseline); 'lfr'/'tfc' = Probe do método SSL.")
    ap.add_argument("--init-ckpt", default=None,
                    help="backbone.ckpt do pré-treino federado, injetado no "
                         "backbone do modelo global (gate G-LOAD).")
    ap.add_argument("--shots", default=FULL_SHOTS,
                    help=f"Rótulos por classe POR CLIENTE: {SHOT_LEVELS}. "
                         f"Padrão '{FULL_SHOTS}' (= o shard de orçamento inteiro).")
    ap.add_argument("--pretrain-rounds", type=int, default=0,
                    help="Só proveniência: quantas rodadas de pré-treino geraram "
                         "o --init-ckpt (vai para o CSV).")
    ap.add_argument("--pretrain-spec", default="",
                    help="Só proveniência: o spec da população que gerou o "
                         "--init-ckpt, quando DIFERE do --spec do fine-tuning "
                         "(ex.: pré-treino em RW, fine-tuning em MS). Vazio = "
                         "mesma população. Vai para o CSV — é o que distingue "
                         "as células que separam pré-treino de fine-tuning.")
    args = ap.parse_args()

    _mode, names, _counts = parse_spec(args.spec)
    targets = args.targets or names
    budget = args.budget or None
    n_shots = FULL_SHOTS if str(args.shots) == FULL_SHOTS else int(args.shots)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else None
    if init_ckpt is not None and args.method == "none":
        ap.error("--init-ckpt exige --method lfr|tfc (o baseline parte do zero).")
    if args.pretrain_spec and init_ckpt is None:
        ap.error("--pretrain-spec sem --init-ckpt: não há backbone cuja "
                 "proveniência registrar.")
    if args.pretrain_spec == args.spec:
        ap.error("--pretrain-spec igual ao --spec: deixe vazio, senão as células "
                 "de mesma população ficam com duas codificações no CSV.")

    if args.gate:
        ok = gate(args.spec, args.encoder, args.local_epochs, args.seed[0],
                  budget, targets)
        sys.exit(0 if ok else 1)

    frames = [run(args.spec, args.encoder, args.rounds, args.local_epochs, s,
                  budget, targets, args.num_workers,
                  ckpt_dir=Path(args.ckpt_dir) / f"seed{s}" if args.ckpt_dir else None,
                  method=args.method, init_ckpt=init_ckpt, n_shots=n_shots,
                  pretrain_rounds=args.pretrain_rounds,
                  pretrain_spec=args.pretrain_spec, lr=args.lr)
              for s in args.seed]
    new = pd.concat(frames, ignore_index=True)
    cache_path = Path(args.out) if args.out else CACHE
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=COLS)
    # `method` e `n_shots` FAZEM parte da chave: sem eles o dedup colapsaria o
    # degrau `full` e o `L=1` da mesma célula numa linha só, em silêncio.
    out = pd.concat([cache, new], ignore_index=True).drop_duplicates(
        subset=["encoder", "spec", "budget", "seed", "local_epochs", "round",
                "target", "method", "n_shots"],
        keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    print(f"\n{len(new)} linhas novas -> {cache_path} ({len(out)} no total)")


if __name__ == "__main__":
    main()
