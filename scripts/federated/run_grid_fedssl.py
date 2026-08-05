"""Driver da grade Fed-SSL cross-device — pré-treino federado + fine-tuning federado.

Desenho fechado na sabatina de 2026-07-29. **Não** reusa
`run_grid_cross_device.py`: aquele driver e o `results/fed_cross_device.csv` que
ele produziu ficam **congelados** com a grade commitada em `20449ab`. O motivo é
concreto — o `consolidate()` de lá faz `PART_DIR.glob("*.csv")` e deduplica por uma
`KEY` que **não** conhece `method`/`n_shots`; misturar os parciais novos no mesmo
diretório colapsaria, em silêncio, o degrau `full` e o `L=1` da mesma célula.

Duas fases, com dependência real entre elas:

  Fase 1 — pré-treino federado (`ssl/pretrain_fed.py`), R=100
      2 métodos × 4 encoders × 4 specs × 4 seeds = 128 runs
      LFR usa `--local-epochs 30` (= 6k, épocas EFETIVAS de backbone casadas com o
      TF-C, que usa 5). Medido em 2026-07-29: a época bruta do LFR custa 0,87× a do
      TF-C neste hardware, então o LFR sai ~4,6× mais caro por rodada. É o preço de
      casar o eixo certo — ver `plano_fedssl.md §2`.

  Fase 2 — fine-tuning federado (`federated/run_cross_device.py`), R=150
      baseline (`method=none`) : 4 enc × 4 specs × 4 seeds × 4 degraus = 256 runs
      LFR e TF-C               : 4 enc × 4 specs × 4 seeds × 5 degraus = 320 cada
      total 896 runs
      O degrau `full` do baseline **não** é rodado aqui: já existe em
      `results/fed_cross_device.csv` (grade `20449ab`) e o notebook une os dois.
      A ladder do baseline não depende da Fase 1 — pode rodar em paralelo com ela.

Os `iid:*` ficaram de fora (ablação de segunda onda): o contraste `device − iid`
não sustentou afirmação em três grades sucessivas, e re-respondê-lo no braço SSL
custaria ~170 GPU-h.

Saída: `results/fedssl_cross_device.csv`, consolidado de
`results/fedssl_cross_device_parts/`. **Durante uma grade o CSV consolidado está
velho** — o driver só consolida no fim. Análises leem o diretório de parciais.

Uso:
    python scripts/federated/run_grid_fedssl.py --phase 2 --method none --gpus 2,3,4,5,6,7
    python scripts/federated/run_grid_fedssl.py --phase 1 --gpus 2,3,4,5,6,7
    python scripts/federated/run_grid_fedssl.py --consolidate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

import pandas as pd

from common import FULL_SHOTS, PROJECT_ROOT, SEEDS
from eval_transfer import BUILD_MODEL
from federated.cross_device import parse_spec
from gpu_pool import Job, detect_gpus, run_pool

FT_RUNNER = Path(__file__).resolve().parent / "run_cross_device.py"
PRE_RUNNER = Path(__file__).resolve().parents[1] / "ssl" / "pretrain_fed.py"

LOG_DIR = PROJECT_ROOT / "logs" / "fedssl_cross_device"
CACHE = PROJECT_ROOT / "results" / "fedssl_cross_device.csv"
PART_DIR = PROJECT_ROOT / "results" / "fedssl_cross_device_parts"
# Braço de população CRUZADA (--pretrain-spec): pré-treino e fine-tuning em
# populações diferentes. Cache próprio — ver `partial_path`.
XCACHE = PROJECT_ROOT / "results" / "fedssl_crossspec.csv"
XPART_DIR = PROJECT_ROOT / "results" / "fedssl_crossspec_parts"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "fedssl_cross_device"
PRE_CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "ssl_fed"

KEY = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target",
       "method", "n_shots"]

SPECS = [
    "device:RealWorld_thigh:10",                 # in10-RW
    "device:MotionSense:10",                     # in10-MS
    "device:RealWorld_thigh+MotionSense:10+10",  # cross10+10
    "device:RealWorld_thigh+MotionSense:5+5",    # cross5+5
]
METHODS = ["none", "lfr", "tfc"]
K = 5                    # épocas locais efetivas de backbone (as do baseline)
PRETRAIN_ROUNDS = 100    # sem seleção no pré-treino; valor da literatura
FINETUNE_ROUNDS = 150    # o mesmo do baseline, para casar o eixo de comparação
# Épocas do Trainer por cliente/rodada no PRÉ-TREINO. O LFR alterna em blocos de
# 6 (predictor_training_epochs+1), então 6k épocas brutas = k efetivas.
PRE_LOCAL_EPOCHS = {"lfr": 6 * K, "tfc": K}
# O baseline já tem o degrau `full` medido em results/fed_cross_device.csv.
SHOTS = {"none": [1, 2, 5, 10],
         "lfr": [1, 2, 5, 10, FULL_SHOTS],
         "tfc": [1, 2, 5, 10, FULL_SHOTS]}


def spec_tag(spec: str) -> str:
    return spec.replace(":", "-").replace("@", "-")


def split_budget(spec: str) -> tuple[str, str]:
    """Separa o sufixo de orçamento de pré-treino: `<spec>@full` -> (spec, "full").

    O sufixo existe porque `single:X:10` a budget 192 e o MESMO `single:X:10` sem
    teto são dois experimentos diferentes que, sem ele, dividiriam o mesmo
    `spec_tag` — logo o mesmo diretório de checkpoint e os mesmos parciais. O
    braço de budget 192 (Exp. 2, fechado em 2026-08-05) seria sobrescrito em
    silêncio. `@full` vira um segmento `combo` no caminho do `pretrain_fed.py` e
    um sufixo no tag, então os dois convivem visíveis no `ls`.

    Só o pré-treino aceita o sufixo. O budget do fine-tuning é outro eixo e fica
    fixo em 192 em toda a grade."""
    base, _, suffix = spec.partition("@")
    if suffix and suffix != "full":
        raise ValueError(f"sufixo de orçamento desconhecido em {spec!r}: "
                         f"só '@full' é aceito")
    return base, suffix


def pretrain_dir(method: str, encoder: str, spec: str, seed: int) -> Path:
    """Onde `pretrain_fed.py` grava — combo vazio some no `Path` (verificado).

    Espelha `fed_ckpt_dir(method, encoder, partition, combo, seed)`: o sufixo de
    orçamento (`@full`) é passado ao runner como `--combo`, então entra aqui como
    segmento próprio."""
    base, combo = split_budget(spec)
    return PRE_CKPT_ROOT / method / encoder / spec_tag(base) / combo / f"seed{seed}"


def partial_path(method: str, encoder: str, spec: str, shots, seed: int,
                 pre_spec: str = "") -> Path:
    """Parciais de população casada e de população cruzada vivem em diretórios
    SEPARADOS. Não é preciosismo: a `KEY` do `consolidate()` não conhece
    `pretrain_spec`, então misturar os dois colapsaria em silêncio a célula
    `pré-treino em RW / finetuning em RW` contra `pré-treino em MS / finetuning em
    RW` — exatamente o modo de falha que o cabeçalho deste arquivo descreve.

    Dentro do braço cruzado, cada **população de pré-treino** ganha o seu
    subdiretório (`pre-device-.../`, `pre-single-.../`). Motivo concreto: o mesmo
    diretório passou a hospedar dois experimentos com perguntas diferentes — o
    transfer de domínio (`pre-device-*`) e o Exp. 2 (`pre-single-*`) — e enquanto
    a distinção existia só numa COLUNA, qualquer `glob` ingênuo somava os dois na
    mesma média sem erro nenhum. A pasta torna a diferença visível no `ls`."""
    if pre_spec:
        return XPART_DIR / f"pre-{spec_tag(pre_spec)}" / (
            f"{method}_{encoder}_pre-{spec_tag(pre_spec)}"
            f"__ft-{spec_tag(spec)}_k{K}_shots{shots}_seed{seed}.csv")
    return PART_DIR / (f"{method}_{encoder}_{spec_tag(spec)}_k{K}"
                       f"_shots{shots}_seed{seed}.csv")


def done(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    return len(pd.read_csv(path).dropna(subset=["test_f1_macro"])) >= expected_rows


def consolidate() -> None:
    """Consolida os dois braços, cada um no seu cache. O braço cruzado deduplica
    por `KEY + pretrain_spec`, senão duas populações de pré-treino diferentes
    colidiriam na mesma linha."""
    # `rglob` no braço cruzado: os parciais moram um nível abaixo, num
    # subdiretório por população de pré-treino (ver `partial_path`).
    for part_dir, cache, key in ((PART_DIR, CACHE, KEY),
                                 (XPART_DIR, XCACHE, KEY + ["pretrain_spec"])):
        frames = [pd.read_csv(p) for p in sorted(part_dir.rglob("*.csv"))]
        if not frames:
            print(f"[CONCAT] nenhum parcial em {part_dir.name}.", flush=True)
            continue
        full = pd.concat(frames, ignore_index=True)
        full = full.drop_duplicates(subset=[c for c in key if c in full.columns],
                                    keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(cache, index=False)
        print(f"[CONCAT] {len(full)} linhas de {len(frames)} parciais -> {cache}",
              flush=True)


# ---------------------------------------------------------------------------
# Fase 1 — pré-treino federado
# ---------------------------------------------------------------------------
def pretrain_jobs(methods, encoders, specs, seeds, rounds, force):
    jobs, skipped = [], 0
    for method in methods:
        if method == "none":
            continue
        for encoder in encoders:
            for spec in specs:
                for seed in seeds:
                    out = pretrain_dir(method, encoder, spec, seed)
                    if not force and (out / "backbone.ckpt").exists():
                        skipped += 1
                        continue
                    base, combo = split_budget(spec)
                    label = f"pre_{method}_{encoder}_{spec_tag(spec)}_seed{seed}"
                    # `--budget 0` cai em `args.budget or None` no runner = sem
                    # teto por cliente (todas as janelas dos usuários do spec).
                    argv = [
                        sys.executable, str(PRE_RUNNER),
                        "--method", method, "--encoder", encoder,
                        "--partition", base, "--rounds", str(rounds),
                        "--local-epochs", str(PRE_LOCAL_EPOCHS[method]),
                        "--seed", str(seed),
                    ] + (["--combo", combo, "--budget", "0"] if combo else [])
                    jobs.append(Job(label, argv + (["--force"] if force else []),
                                    LOG_DIR / f"{label}.log"))
    return jobs, skipped


# ---------------------------------------------------------------------------
# Fase 2 — fine-tuning federado
# ---------------------------------------------------------------------------
def finetune_jobs(methods, encoders, specs, seeds, shots_filter, rounds, budget, force,
                  pre_spec=""):
    """`pre_spec` != "" desacopla a população do pré-treino da do fine-tuning: o
    backbone vem de `pre_spec` e os clientes do fine-tuning vêm de `spec`. É o que
    abre o Exp. 2 (`single:X` -> `device:X`, separa custo da federação do custo do
    orçamento de dado) e o transfer de domínio (`device:RW` -> `device:MS`)."""
    jobs, skipped, missing = [], 0, []
    for method in methods:
        if pre_spec and method == "none":
            continue                      # baseline não tem backbone a herdar
        for encoder in encoders:
            for spec in specs:
                if pre_spec == spec:
                    continue              # não é população cruzada; já está na grade base
                n_targets = len(parse_spec(spec)[1])
                for shots in SHOTS[method]:
                    if shots_filter and str(shots) not in shots_filter:
                        continue
                    for seed in seeds:
                        part = partial_path(method, encoder, spec, shots, seed, pre_spec)
                        if not force and done(part, rounds * n_targets):
                            skipped += 1
                            continue
                        ck_tag = (f"pre-{spec_tag(pre_spec)}__ft-{spec_tag(spec)}"
                                  if pre_spec else spec_tag(spec))
                        argv = [
                            sys.executable, str(FT_RUNNER),
                            "--spec", spec, "--encoder", encoder,
                            "--rounds", str(rounds), "--local-epochs", str(K),
                            "--seed", str(seed), "--budget", str(budget),
                            "--shots", str(shots), "--method", method,
                            "--out", str(part),
                            "--ckpt-dir", str(CKPT_DIR / method / encoder
                                              / ck_tag / f"shots{shots}"),
                        ] + (["--pretrain-spec", pre_spec] if pre_spec else [])
                        if method != "none":
                            ckpt = (pretrain_dir(method, encoder, pre_spec or spec, seed)
                                    / "backbone.ckpt")
                            if not ckpt.exists():
                                missing.append(str(ckpt.relative_to(PROJECT_ROOT)))
                                continue
                            argv += ["--init-ckpt", str(ckpt),
                                     "--pretrain-rounds", str(PRETRAIN_ROUNDS)]
                        label = (f"{method}_{encoder}_{ck_tag}"
                                 f"_shots{shots}_seed{seed}")
                        jobs.append(Job(label, argv, LOG_DIR / f"{label}.log"))
    if missing:
        print(f"[GRID] {len(missing)} células sem backbone pré-treinado (Fase 1 "
              f"incompleta) — puladas. Ex.: {missing[0]}", flush=True)
    return jobs, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all",
                    help="1 = pré-treino, 2 = fine-tuning, all = as duas em ordem.")
    ap.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--encoder", nargs="+", choices=sorted(BUILD_MODEL),
                    default=sorted(BUILD_MODEL))
    ap.add_argument("--spec", nargs="+", default=SPECS)
    ap.add_argument("--pretrain-spec", default="",
                    help="Desacopla as populações: o backbone vem DESTE spec e o "
                         "fine-tuning roda nos clientes do --spec. Vazio (padrão) = "
                         "grade base, populações casadas. Ex.: Exp.2 = "
                         "'--pretrain-spec single:RealWorld_thigh:10 --spec "
                         "device:RealWorld_thigh:10'; transfer de domínio = "
                         "'--pretrain-spec device:RealWorld_thigh:10 --spec "
                         "device:MotionSense:10'. Escreve em "
                         "results/fedssl_crossspec_parts/, nunca na grade base. "
                         "Sufixo '@full' tira o teto de janelas do PRÉ-treino "
                         "(ex.: 'single:RealWorld_thigh:10@full' = as 10.338 "
                         "janelas dos mesmos 10 usuários, contra 1.920 a budget "
                         "192) — é o eixo de volume do Exp. 2.")
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--shots", nargs="+", default=None,
                    help="Filtra degraus da ladder (ex.: 1 2). Padrão: todos.")
    ap.add_argument("--pretrain-rounds", type=int, default=PRETRAIN_ROUNDS)
    ap.add_argument("--rounds", type=int, default=FINETUNE_ROUNDS)
    ap.add_argument("--budget", type=int, default=192)
    ap.add_argument("--gpus", type=str, default=None,
                    help="OBRIGATÓRIO na prática: a máquina é compartilhada. Ex.: '2,3,4,5,6,7'.")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Lista os jobs e sai.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.consolidate:
        consolidate()
        return

    # `@full` é eixo do PRÉ-treino. No fine-tuning o budget é fixo (192) e vem de
    # `--budget`; aceitar o sufixo lá daria um spec que `parse_spec` não entende.
    for s in args.spec:
        if "@" in s:
            ap.error(f"--spec {s!r}: o sufixo '@full' só vale em --pretrain-spec.")
    if args.pretrain_spec:
        split_budget(args.pretrain_spec)   # valida o sufixo cedo

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()]
            if args.gpus else detect_gpus()) or [0]
    if not args.gpus:
        print("[AVISO] --gpus não passado: usando TODAS as GPUs detectadas. "
              "A máquina é compartilhada — confira `nvidia-smi` antes.", flush=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    if args.pretrain_spec:
        XPART_DIR.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], 0
    if args.phase in ("1", "all"):
        # Com populações cruzadas a Fase 1 é a do PRÉ-TREINO, não a do fine-tuning.
        pre_specs = [args.pretrain_spec] if args.pretrain_spec else args.spec
        j, s = pretrain_jobs(args.method, args.encoder, pre_specs, args.seed,
                             args.pretrain_rounds, args.force)
        jobs += j
        skipped += s
    if args.phase in ("2", "all"):
        j, s = finetune_jobs(args.method, args.encoder, args.spec, args.seed,
                             args.shots, args.rounds, args.budget, args.force,
                             pre_spec=args.pretrain_spec)
        jobs += j
        skipped += s

    print(f"[GRID] fase={args.phase} | {len(jobs)} jobs a rodar "
          f"({skipped} já completos) em {len(gpus)} GPUs {gpus}.", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job.label}")
        return

    failed = run_pool(jobs, gpus, args.max_parallel or len(gpus))
    if failed:
        print(f"[GRID] {len(failed)} jobs falharam: {failed}", flush=True)
    if args.phase in ("2", "all"):
        consolidate()


if __name__ == "__main__":
    main()
