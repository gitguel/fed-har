# Tabelas da apresentação

Compiladas com **tectonic** (motor TeX autocontido, binário único, sem root) — o
mesmo que gerou as tabelas da apresentação de 21/07. Fontes **newtx**, filetes
**booktabs**.

Cada tabela produz três `.tex` e um `.pdf`:

| Arquivo | O que é |
|---|---|
| `<nome>_body.tex` | só o `tabular` — **fonte única**, os outros dois o reusam |
| `<nome>.tex` | ambiente `table` com `\caption` e `\label`, para `\input{}` num paper |
| `<nome>_standalone.tex` | documento `standalone` — é o que compila no `.pdf` |
| `<nome>.pdf` | **a tabela recortada, pronta para o slide** |

## Reproduzir

```bash
poetry run python <gerador>          # lê results/, escreve .tex, chama o tectonic
```

Se o `tectonic` não estiver instalado:

```bash
curl -sSL -o /tmp/t.tar.gz https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz
tar xzf /tmp/t.tar.gz -C /tmp && mv /tmp/tectonic ~/.local/bin/ && chmod +x ~/.local/bin/tectonic
```

Ele resolve os pacotes sozinho (há cache em `~/.cache/Tectonic`).

## Detalhe de implementação que vale saber

O `standalone` mede o `tabular` num `\savebox` e usa `\wd\tabbox` como largura da
`minipage`. Sem isso, a nota de rodapé — mais larga que a tabela — entrava como
`\multicolumn` e **esticava a última coluna**, abrindo um vão no meio da tabela.

## As tabelas

| Arquivo | Conteúdo |
|---|---|
| `tab21_parametros` | configuração da federação cross-device (§2.1, 2 colunas) |
| `tab22_federacoes` | as quatro federações (§2.2, 3 colunas) |
| `t1_centralizado_encoder_dataset` | centralizado: encoder × dataset × técnica, rótulo cheio |
| `t2_centralizado_regime` | centralizado: técnica × regime × encoder |
| `t3_crossdevice_encoder_federacao` | cross-device: encoder × federação × técnica, rótulo cheio |
| `t4_crossdevice_regime` | cross-device: técnica × regime × encoder |

Todos os números saem dos caches em `results/`. Nada foi treinado para a apresentação.
