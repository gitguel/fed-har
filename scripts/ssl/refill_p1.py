#!/usr/bin/env python3
"""Ronda de monitoramento/refill da Fase 1 do Fed-SSL (pré-treino federado, R=100).

Grade: 2 métodos x 4 encoders x 4 specs x 4 seeds = 128 runs.
DONE      = <seed>/backbone.ckpt (final) existe
RODANDO   = processo pretrain_fed.py vivo com esses args
PENDENTE  = resto; se tem state_last.pt, retoma da rodada salva (o script resume sozinho)

Refill: mantém no máximo MAX_PER_GPU runs nossos em cada GPU de GPUS, e só lança se
sobrar memória para o encoder pedido (máquina compartilhada — outros usuários enchem
0,3,4,5; foi assim que 32 runs de tstcc morreram de OOM em rajada, 29/07 18h36-18h55).

Uso:  python3 scripts/ssl/refill_p1.py            # relatório + lança o que couber
      python3 scripts/ssl/refill_p1.py --dry-run  # só relatório
Roda de 15 em 15 min pelo crontab; log em logs/refill_p1_cron.log.
"""
import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/miguel.barreto/fed-har")
CKPT = ROOT / "checkpoints" / "ssl_fed"
LANCA = ROOT / "scripts" / "ssl" / "lanca_p1.sh"

TMUX_SESSION = "fedssl-p1-manual"

GPUS = [1, 2, 6, 7]          # 4 GPUs, sempre — o resto da máquina é de outras pessoas
MAX_PER_GPU = 5
# memória reservada por run, medida em 30/07: tstcc (backbone de 7,3M params no TF-C)
# pesa 1656 MiB; os demais encoders ficam em ~474 MiB. A folga cobre o pico da rodada.
MEM_PER_RUN = {"tstcc": 2000}
MEM_DEFAULT = 900
MIN_HEADROOM_MIB = 1500      # nunca encostar no teto: deixa isto livre depois de lançar

ROUNDS = 100
PRE_LOCAL_EPOCHS = {"tfc": 5, "lfr": 30}   # K=5 efetivas; LFR alterna em blocos de 6
SPECS = [
    "device:RealWorld_thigh:10",
    "device:MotionSense:10",
    "device:RealWorld_thigh+MotionSense:10+10",
    "device:RealWorld_thigh+MotionSense:5+5",
]
SEEDS = [0, 1, 2, 3]
# ordem da fila: por encoder, alternando os dois métodos (LFR é ~4,6x mais caro/rodada,
# então não fica todo para o fim); tstcc por último (o mais pesado)
QUEUE_ORDER = [(m, e) for e in ("resnetse5", "cnnpff", "rnn", "tstcc") for m in ("tfc", "lfr")]


def _sh(args, timeout=45):
    """stdout do comando; string vazia se travar (nvidia-smi engasga com GPU cheia)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"[aviso] falhou/timeout em: {' '.join(args[:2])}", flush=True)
        return ""


def seed_dir(method, enc, spec, seed):
    return CKPT / method / enc / spec.replace(":", "-") / f"seed{seed}"


def mem_needed(enc):
    return MEM_PER_RUN.get(enc, MEM_DEFAULT)


def running_jobs():
    """{(method, enc, spec, seed): (pid, etime)} dos processos pretrain_fed.py vivos."""
    out = _sh(["ps", "-eo", "pid,etime,args"])
    jobs = {}
    for line in out.splitlines():
        if "pretrain_fed.py" not in line or " grep " in line:
            continue
        pid, etime = line.split()[0], line.split()[1]
        m = re.search(r"--method (\S+).*--encoder (\S+).*--partition (\S+).*--seed (\d+)", line)
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
        # pais têm etime maior que os workers forkados; guarda o mais velho
        if key not in jobs or _secs(etime) > _secs(jobs[key][1]):
            jobs[key] = (int(pid), etime)
    return jobs


def _secs(etime):
    d, _, rest = etime.partition("-")
    if rest:
        days, parts = int(d), rest.split(":")
    else:
        days, parts = 0, etime.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def gpu_state(live_pids):
    """(nossos_jobs_por_gpu, mem_livre_por_gpu) — só conta pids da nossa grade."""
    uuid2idx = {}
    for line in _sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        uuid2idx[uuid] = int(idx)
    free = {}
    for line in _sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]).splitlines():
        idx, used, total = [int(x.strip()) for x in line.split(",")]
        free[idx] = total - used
    mine = {g: 0 for g in free}
    for line in _sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                     "--format=csv,noheader"]).splitlines():
        uuid, pid = [x.strip() for x in line.split(",")]
        if int(pid) in live_pids and uuid in uuid2idx:
            mine[uuid2idx[uuid]] = mine.get(uuid2idx[uuid], 0) + 1
    return mine, free


def resume_round(d):
    """Última rodada concluída, lida do rounds.csv (barato — sem torch.load)."""
    p = d / "rounds.csv"
    if not p.exists():
        return 0
    try:
        last = [l for l in p.read_text().splitlines() if l and not l.startswith("round,")][-1]
        return int(last.split(",")[0])
    except Exception:
        return -1


def ensure_tmux():
    """Garante a sessão tmux dona dos processos; recria se alguém a matou."""
    alive = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not alive:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", str(ROOT)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        print(f"[aviso] sessão tmux {TMUX_SESSION} não existia — recriada", flush=True)
    return True


def launch(gpu, method, enc, spec, seed):
    """Dispara o pré-treino DENTRO do tmux TMUX_SESSION, via lanca_p1.sh.

    Não use Popen daqui: um filho da chamada de ferramenta (ou do cron) morre quando a
    chamada termina — 30/07, 13 runs mortos em menos de 2 min sem escrever uma linha de
    log. O dono do processo tem de ser o servidor tmux, como manda o CLAUDE.md.
    """
    lbl = f"pre_{method}_{enc}_{spec.replace(':', '-')}_seed{seed}"
    subprocess.run(["tmux", "send-keys", "-t", TMUX_SESSION,
                    f"{LANCA} {gpu} {method} {enc} {spec} {seed}", "Enter"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    return f"LAUNCH gpu={gpu} {lbl} (via tmux {TMUX_SESSION})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"\n===== ronda {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"{'(dry-run)' if args.dry_run else ''} =====", flush=True)

    run = running_jobs()
    done, active, pending = [], [], []
    for method, enc in QUEUE_ORDER:
        for spec in SPECS:
            for seed in SEEDS:
                key = (method, enc, spec, seed)
                d = seed_dir(*key)
                if (d / "backbone.ckpt").exists():
                    done.append(key)
                elif key in run:
                    active.append((key, run[key]))
                else:
                    pending.append(key)

    # runs interrompidos (têm state_last.pt e ninguém rodando) vêm primeiro: retomam
    # de onde pararam e fecham células que a Fase 2 já espera
    pending.sort(key=lambda k: 0 if (seed_dir(*k) / "state_last.pt").exists() else 1)

    live_pids = {pid for pid, _ in run.values()}
    mine, free = gpu_state(live_pids)

    print(f"FASE 1 (pré-treino): {len(done)}/128 DONE | {len(active)} RODANDO | "
          f"{len(pending)} PENDENTES", flush=True)
    print("GPUs: " + "  ".join(
        f"g{g}={mine.get(g, 0)}job/{free[g]}MiB livre" for g in sorted(free)))

    if not args.dry_run and pending:
        ensure_tmux()

    # relança/lança enquanto houver slot
    launched = []
    for key in pending:
        need = mem_needed(key[1]) + MIN_HEADROOM_MIB
        cands = [g for g in GPUS
                 if mine.get(g, 0) < MAX_PER_GPU and free.get(g, 0) >= need]
        if not cands:
            break
        g = min(cands, key=lambda x: (mine.get(x, 0), -free[x]))
        r = resume_round(seed_dir(*key))
        tag = f" [retoma round{r}]" if r > 0 else ""
        if args.dry_run:
            launched.append(f"(dry) gpu={g} {key}{tag}")
        else:
            launched.append(launch(g, *key) + tag)
        mine[g] = mine.get(g, 0) + 1
        free[g] -= mem_needed(key[1])

    print(f"\nLANÇADOS nesta ronda: {len(launched)}")
    for l in launched:
        print("  " + l)

    print("\nRODANDO (mais velho primeiro):")
    for key, (pid, et) in sorted(active, key=lambda x: -_secs(x[1][1])):
        r = resume_round(seed_dir(*key))
        print(f"  {key[0]:3} {key[1]:9} {key[2]:42} seed{key[3]}  pid={pid:>7} "
              f"up={et:>9} round~{r}")

    print("\nPRÓXIMOS DA FILA:")
    for key in pending[len(launched):len(launched) + 8]:
        print(f"  {key[0]:3} {key[1]:9} {key[2]:42} seed{key[3]}")


if __name__ == "__main__":
    main()
