# `ssl_benchmark_exemple/` — Notas para o projeto `fed-har`

> Documento de referência sobre o paper de benchmark contido nesta pasta e como ele guia as escolhas
> da implementação federada. Para o framework de implementação, ver `../minerva/FED_HAR_NOTES.md`.

---

## 1. O que é esta pasta

Contém um único PDF: **"Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human
Activity Recognition"** — paper de benchmark publicado em **IEEE Access, vol. 14, 2026, pp. 37451–37475**
(DOI `10.1109/ACCESS.2026.3669412`). 25 páginas.

**Autores**: Gustavo P. C. P. da Luz, Darlinne H. P. Soto, Otávio O. Napoli, Anderson Rocha, Levy
Boccato, Edson Borin (H.IAAC — UNICAMP) — **mesmo grupo que mantém o `minerva-ml`**.

Código oficial dos experimentos: <https://github.com/H-IAAC/benchmarking-encoders-ssl-har>.

---

## 2. Resumo executivo

Benchmark unificado e reprodutível que treinou **11 232 modelos** combinando 6 encoders × 4 técnicas
SSL × 6 sub-datasets DAGHAR × 8 regimes de dados × estratégias de refinamento (congelamento vs
*full fine-tuning*). Responde a 6 RQs (questões de pesquisa) sobre quais combinações de encoder ×
SSL × regime de dados rendem mais.

---

## 3. Encoders avaliados

Todos disponíveis em `../minerva/models/nets/`:

| Encoder         | Parâm. do backbone | Origem                                                           |
| --------------- | ------------------ | ---------------------------------------------------------------- |
| **ResNet-SE-5** | 0,13 M             | Mekruksavanich & Jitpattanakul (2022). Melhor encoder geral.     |
| **CNN-PFF**     | 0,05 M             | Ha & Choi (2016). Melhor encoder quando combinado com TF-C.      |
| **IMU Transformer** | 0,22 M         | Shavit & Klein (2021).                                           |
| **TS2Vec Encoder**  | 0,64 M         | Yue et al. (2022) — CNN 1D dilatada.                             |
| **TS-TCC Encoder**  | 3,04 M         | Eldele et al. (2021) — FCN de 3 blocos.                          |
| **RNN**             | 0,13 M         | Tonekaboni et al. (TNC) — BiGRU.                                 |

---

## 4. Técnicas SSL avaliadas

Todas disponíveis em `../minerva/models/ssl/`:

| Técnica   | Tipo                                | Hiperparâmetros chave (pré-treino)                       |
| --------- | ----------------------------------- | -------------------------------------------------------- |
| **TNC**   | Contrastivo por vizinhança temporal  | AdamW, lr `1e-5`, sem `drop_last`.                        |
| **TF-C**  | Consistência tempo × frequência      | Adam, lr `3e-4`, weight_decay `3e-4`, `drop_last=True`.   |
| **LFR**   | Projeção aleatória (random projection) | Adam, lr `3e-4`, weight_decay `3e-4`, betas `(0,9; 0,99)`. |
| **DIET**  | Índice da amostra como alvo (Datum IndEx as Target) | Adam, lr `3e-4`, weight_decay `3e-4`.       |

Todos pré-treinados por **100 épocas**, batch `64`, sem parada antecipada (mantém-se o último
checkpoint). Após o pré-treino, ajuste fino até **100 épocas**, parada antecipada com paciência de
50 épocas, mesmo batch, lr `1e-4` (`1e-3` para TS2Vec).

---

## 5. Datasets (DAGHAR `standardized_view` — o mesmo que já usamos)

| Dataset       | Classes | Total amostras |
| ------------- | ------- | -------------- |
| KuHar         | 6       | 1 392          |
| MotionSense   | 6       | 3 558          |
| RealWorld-Thigh | 6     | 10 338         |
| RealWorld-Waist | 6     | 10 332         |
| UCI HAR       | 5       | 2 420          |
| WISDM         | 4       | 8 748          |

- Taxa de amostragem **20 Hz**, janelas de **3 s não-sobrepostas** (60 timesteps × 6 canais).
- Divisão 70/20/10 com **garantia de que usuários distintos não compartilham divisão** (avalia
  generalização entre usuários — *cross-user*).
- *Few-shot* estratificado em 1/5/10/25/50/100/200 amostras por classe + máximo.
- Sementes 42, 43, 44 (3 repetições por configuração).

---

## 6. Resultados principais (para guiar o setup federado)

1. **RQ1 — Melhor encoder geral**: `ResNet-SE-5` domina (sem arestas de derrota no grafo Wilcoxon).
   Combinação específica vencedora: `CNN-PFF + TF-C` (acurácia média **76,9 ± 14,6 %**).
2. **RQ2 — Refinamento**: *full fine-tuning* > congelamento em quase todos os cenários.
   `ResNet-SE-5 + Full FT` chega a `71,2 % ± 13,8 %` em média.
3. **RQ3 — Encoder vs técnica SSL**: impacto comparável (5,57 pp vs 5,67 pp). A **interação** é o
   determinante real → não basta combinar os "melhores" isolados.
4. **RQ4 — Por dataset**: `ResNet-SE-5` consistente em todos; `TS2Vec` em UCI; `CNN-PFF + TF-C`
   domina RW-Thigh/RW-Waist/WISDM.
5. **RQ5 — Fração de dados rotulados**: `ResNet-SE-5` excelente em regime escasso; `TS2Vec` toma a
   liderança quando há muitos rótulos. `CNN-PFF` consistente com TF-C, exceto em regimes
   extremamente escassos.
6. **RQ6 — Eficiência de rótulos**: SSL com **TF-C atinge ≥ 95 %** do pico de acurácia com **apenas
   25–50 amostras por classe** na maioria dos datasets. Demais técnicas SSL alcançam 90 % com 100
   amostras/classe. Supervisionado puro precisa de muito mais dados.

---

## 7. Tempo de inferência (Tabela 15 — referência prática)

- RNN / CNN-PFF / ResNet-SE-5 / IMU Transformer / TS-TCC: **~0.6–1.5 ms/amostra** (4–10 MACs M).
- TS2Vec: **~50 ms/amostra**, 76.5 MACs M — **~50× mais lento**.
- TF-C dobra esses tempos (duas cópias do encoder + FFT).

**Relevante para FL**: encoders pesados (TS2Vec, TS-TCC) podem inviabilizar dispositivos de borda
(*edge*) → ResNet-SE-5 e CNN-PFF são candidatos mais práticos para clientes federados reais.

---

## 8. Por que esse paper é diretamente útil para a implementação federada

- Fornece **baselines centralizados sólidos** para cada combinação (encoder × SSL × dataset ×
  *few-shot*) — qualquer configuração federada deve ser comparada contra eles para mostrar ganho/queda
  de FL vs. centralizado.
- Indica **quais combinações priorizar primeiro**: começar por `ResNet-SE-5 + LFR` (já temos pipeline
  LFR pronto) e `CNN-PFF + TF-C` (melhor combinação do paper).
- Define o **protocolo experimental a replicar**: mesmas sementes (42 / 43 / 44), mesmas divisões
  (70/20/10), mesmos regimes *few-shot* (1 / 5 / 10 / 25 / 50 / 100 / 200 / máx), mesma cabeça MLP
  (`entrada → 128 → 6`) para comparabilidade.
- Sugere **estratégias de refinamento federado**: *full fine-tuning* é dominante; o congelamento só
  vale em contextos de eficiência computacional → na FL, isso significa que o cliente provavelmente
  deve ajustar todo o modelo agregado, não apenas a cabeça.

---

## 9. Referências rápidas

| Recurso                                  | Caminho/URL                                                       |
| ---------------------------------------- | ----------------------------------------------------------------- |
| PDF do paper                             | `Benchmarking_Encoders_and_Self-Supervised_Learning_for_Smartphone-Based_Human_Activity_Recognition.pdf` |
| Código oficial dos experimentos          | <https://github.com/H-IAAC/benchmarking-encoders-ssl-har>          |
| DOI                                      | `10.1109/ACCESS.2026.3669412`                                     |
| Notas sobre o framework usado            | `../minerva/FED_HAR_NOTES.md`                                     |
