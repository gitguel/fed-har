# Prompt para a próxima sessão

Copie o bloco abaixo inteiro como primeira mensagem.

---

```
Repo fed-har, branch main. O baseline supervisionado FedAvg cross-device está
FECHADO (commit 20449ab): k=5, R=150, 4 encoders × 6 specs × 4 seeds = 96 runs,
zero falhas, gate federado-vs-centralizado PASS com divergência 0.000e+00 nos
quatro encoders. Leia docs/sessao_2026-07-28_grade_cross_device.md antes de
começar — ele traz os números, as decisões e o que ficou aberto.

Tarefa desta sessão: implementar e rodar o Fed-SSL, o braço que vai ser comparado
contra esse baseline.

DESENHO JÁ DECIDIDO (não reabrir, foi sabatinado em 28/07):
- Pré-treino federado SSL (TF-C e LFR) sobre os MESMOS clientes do baseline —
  usar cross_device.make_clients, que já é compartilhada, para que o
  Δ(Fed-SSL − baseline) não carregue diferença de partição.
- Fine-tuning FEDERADO e full (nada de linear readout no eixo federado).
- Ladder de rótulos {1, 2, 5, 10, full} amostras por classe POR CLIENTE,
  distribuída (todo cliente tem L por classe), aninhada por seed.
- Os DOIS braços varridos na mesma ladder: comparar Fed-SSL@1-shot contra
  FedAvg@1-shot, nunca contra FedAvg@full. O ponto `full` do baseline já existe.
- R=150 nos dois braços; 4 seeds; seleção da rodada pela validação (mesmo
  protocolo do baseline, val_acc no CSV + best.ckpt via --ckpt-dir).
- Validação: split completo, seguindo Minerva/da Luz. Está registrado como F7 em
  docs/metodo_e_auditoria.md e é ponto de revisão prioritário — NÃO mude o
  protocolo agora, mas não escreva nada que finja que o problema não existe.
- Uma segunda leitura barata, em notebook SEPARADO: fine-tuning centralizado do
  encoder pré-treinado federado (via downstream_eval.py), para responder "o
  pré-treino federado produz representação tão boa quanto a centralizada?".

ANTES DE IMPLEMENTAR: rode a skill grill-me sobre o plano de implementação. Não
sobre o desenho (já está fechado) — sobre COMO implementar: reuso de
scripts/ssl/pretrain_fed.py, onde entra a ladder, o gate que valida o novo
caminho, e o custo total da grade antes de disparar.

REGRAS DO REPO: nunca criar branch. Commit/push só quando eu pedir. Script longo
sempre em tmux com tee para logs/. Notebooks editados à mão (o padrão de builder
script foi abolido em 28/07). Uma experiência pesada por vez — a máquina tem 8
TITAN Xp e o gpu_pool paraleliza 1 job por GPU.

ARMADILHAS JÁ PAGAS, não repita:
- Nunca ler results/fed_cross_device.csv durante uma grade: o driver só consolida
  no fim, então durante a execução ele contém os números da grade ANTERIOR. Ler
  os parciais em results/fed_cross_device_parts/.
- O resume do driver compara "linhas >= R × alvos". Se mudar R para menos,
  apagar os parciais antigos antes, senão os jobs são pulados e sobram rodadas
  velhas misturadas.
- O piso de batch morde o PRÉ-TREINO, não o fine-tuning: budget=192 por cliente
  é 3 batches de 64 e foi escolhido por isso. A ladder de rótulos só afeta o
  CrossEntropy do downstream, que é por amostra.
- Em 3 dos 4 encoders o argmax da validação escolhe ruído num platô (só o
  resnetse5 tem pico real). Ao reportar "melhor rodada", verificar se o ganho
  sobre a média do platô sobrevive no teste.

Comece confirmando o estado com:
  poetry run python scripts/analysis/cache_status.py
```

---

## Contexto que o prompt assume (para você, não para colar)

- **Onde está o quê**: baseline em `results/fed_cross_device.csv` (22.400 linhas:
  k=5 R=150 nos 4 encoders + k=1 R=100 só resnetse5, guardado para ablação);
  notebook em `notebooks/cross_device_avaliation.ipynb`; checkpoints em
  `checkpoints/fed_cross_device/` (337 MB, gitignored).
- **Infra que já existe e deve ser reusada**: `scripts/federated/cross_device.py`
  (partição, `client_selection.csv` versionado), `scripts/ssl/pretrain_fed.py`
  (FedAvg manual do pré-treino), `scripts/gpu_pool.py`, `scripts/common.py`
  (`few_shot_indices` já é aninhada por seed).
- **Custo de referência para estimar a grade nova**: no baseline, R=150 e k=5
  custaram 20,2 min/job no resnetse5, 8,5 no tstcc, 6,0 no cnnpff, 5,4 no rnn,
  com 8 jobs em paralelo. A grade de 96 runs levou 2h03.
