# Texto sugerido — slides SSL centralizado (LFR, TF-C, SSL vs SL)

Assets nesta pasta: `resultados_ssl.xlsx` (5 abas) e 6 figuras PNG (300 dpi).
Todos os números: acurácia média sobre 4 encoders × 4 seeds; SSL avaliado em
protocolo *finetune* salvo indicação. Fonte: caches `results/ssl_{lfr,tfc}_eval_transfer.csv`,
`results/supervised_eval_transfer.csv`, `results/ssl_{lfr,tfc}_comb2target_eval_transfer.csv`.

---

## Slide A — LFR (1 slide)
**Figura:** `fig_lfr_data_efficiency.png` | **Tabela de apoio:** aba `encoder_x_tecnica`

- Setup: pré-treino LFR + linear readout / fine-tuning, mesmos 4 encoders e 4 regimes de rótulos.
- (Se usar o par com o Slide C) `fig_lfr_comb2target.png`: no LFR o backbone
  multi-domínio é apenas ~neutro no próprio domínio (5/6 datasets; KuHar −3,5 pp) —
  sem o ganho observado no TF-C.
- Na média, LFR ≈ SL (+1,3 pp in-domain @100%): o ganho é **concentrado no BiGRU**
  (+17 pp @10-shot, +3,4 pp @100%); ResNet-SE-5 neutro; CNN-PFF chega a piorar (−3,9 pp @100%).
- Replica o comportamento reportado no benchmark (da Luz et al., IEEE Access 2026):
  LFR ajuda encoders recorrentes, é neutro/negativo nos convolucionais.

## Slide B — TF-C (slide 1)
**Figura:** `fig_tfc_data_efficiency.png` | **Tabela de apoio:** abas `resumo_metodos`, `regimes`

- TF-C é a melhor técnica do benchmark — confirmado na nossa reprodução:
  **+6,2 pp in-domain @100% (82,8 vs 76,6) e +12,2 pp @10-shot (70,4 vs 58,2)** sobre SL.
- O ganho **não desaparece com 100% dos rótulos** (vence SL em 3/4 encoders no teto):
  "SSL só ajuda em few-shot" não vale para TF-C.
- Maiores beneficiários: BiGRU (+13,6 pp) e CNN-PFF (+6,0 pp) @100%.
- Ressalva: sob **linear readout** (fig `fig_ssl_vs_sl_fewshot_linear.png`) o TF-C
  só perde para o LFR no caso extremo de 1-shot (29,9 vs 38,7); lidera de 10 a
  100-shot e empata no teto (71,8 vs 73,7 @100%). Ainda assim, os ganhos grandes
  do TF-C exigem *finetune*.

## Slide C — TF-C (slide 2, opcional — ponte para o federado)
**Figura:** `fig_tfc_comb2target.png` (opcional: `fig_lfr_comb2target.png` ao lado)
**Tabela de apoio:** aba `comb2target`

- Decompomos o cenário "combined". As 3 barras compartilham o protocolo e diferem
  em pré-treino/finetune:
  - **Especialista** (azul): pré-treino e finetune só no próprio dataset;
  - **Pré-treino multi + finetune local** (verde): backbone pré-treinado sem rótulos
    no corpus dos 6 datasets, classificador ajustado só com os rótulos do alvo —
    proxy centralizado de "backbone global federado + finetune local";
  - **Finetune misturado / combined** (cinza): mesmo backbone multi-domínio, mas
    finetune com os rótulos dos 6 datasets juntos.
- Verde ≈ azul: **pré-treinar em multi-domínio custa ~nada no próprio domínio**
  (Δ médio +0,5 pp com TF-C; ≥ especialista em 4/6 datasets).
- Cinza < verde: a queda do cenário combined vem dos **rótulos misturados no
  finetune** (até −7 pp no WISDM), não do corpus de pré-treino → motiva o FedSSL
  cross-device: pré-treino global sem rótulos + adaptação local.
- Contraste com LFR (`fig_lfr_comb2target.png`): backbone multi-domínio apenas
  ~neutro (KuHar −3,5 pp) — mais um motivo para TF-C ser o método default do
  federado.

## Slide D — Comparação final SSL vs SL
**Figuras:** `fig_ssl_vs_sl_cenarios.png` (principal) e `fig_ssl_vs_sl_fewshot.png`
**Tabela:** aba `resumo_metodos` (colar como tabela do slide)

Conclusões (calibradas):
1. **SSL ajuda a tarefa dentro do domínio** — melhor técnica overall: **TF-C**
   (82,8 vs 76,6 do SL @100%; +12 pp @10-shot). Melhor encoder overall: **CNN-PFF
   com TF-C (85,6%)**. Melhor encoder por técnica: TF-C → CNN-PFF (85,6);
   LFR → TS-TCC (81,9); SL puro → ResNet-SE-5 (80,4).
2. **Pré-treino SSL mitiga (parcialmente) a queda em outros domínios** —
   cross-domain: TF-C 49,3 vs SL 46,3 (+3,0 pp @100%; +5,5 pp @10-shot).
   O domain shift continua dominante (~49% ≪ ~83% in-domain): SSL desloca o nível,
   não fecha o gap.
3. (Extra, bem fundamentada) **O caminho promissor para multi-domínio é pré-treino
   SSL no corpus agregado**: TF-C combined 80,2 vs SL 71,3 (+8,9 pp), e o
   comb2target mostra que esse backbone global não sacrifica o especialista →
   justificativa direta do experimento FedSSL cross-device.

Nota de rodapé sugerida: sob linear readout (backbone congelado — sem comparação
direta com SL, que treina o backbone) os ganhos são menores e LFR só supera TF-C
no 1-shot; os ganhos relevantes do SSL exigem finetune.
