# retrieval-eval-pt

Independent demo, not affiliated with BIP (Business Integration Partners).

Uma ferramenta pequena para medir a qualidade da busca de um RAG em português. Ela roda offline, só com a biblioteca padrão do Python, e tem poucas centenas de linhas.

Ela compara dois jeitos de preparar o texto antes da busca BM25:

- `raw`: minúsculas e divisão por palavra.
- `pt`: remove acentos, tira palavras vazias (de, para, como) e corta sufixos comuns (plural, -ção).

Para cada jeito, calcula Recall@k, MRR, a média de tokens de contexto que iriam para o modelo e o custo estimado desse contexto. Também lista as perguntas em que a busca não achou o documento certo.

## Como rodar

```bash
pip install pytest
PYTHONPATH=src python -m retrieval_eval.cli          # usa os dados de exemplo
PYTHONPATH=src python -m retrieval_eval.cli --corpus meu.jsonl --golden meu_golden.jsonl -k 5 --price-per-million 3.0
PYTHONPATH=src pytest -q
```

Formato dos arquivos (uma linha JSON por item):

```
corpus:  {"id": "d01", "title": "...", "text": "..."}
golden:  {"query": "...", "relevant": ["d01"]}
```

## Resultado nos dados de exemplo

Os dados em `data/` são inventados: 16 trechos de uma política interna de TI fictícia e 18 perguntas escritas por mim, algumas sem acento ou com plural diferente do texto. Com k=3:

| analyzer | Recall@3 | MRR | tokens de contexto (média) |
|---|---|---|---|
| raw | 0,94 | 0,85 | 76 |
| pt | 1,00 | 0,96 | 48 |

A pergunta "como instalar impressora" não acha nada com `raw`: o texto diz "impressoras" e o `raw` só compara a palavra inteira. Com `pt` o documento certo fica em primeiro lugar. O contexto também fica menor, porque o `pt` traz menos documentos irrelevantes.

O preço por milhão de tokens é um valor de exemplo (1,0 USD). Troque pelo da sua conta.

## What was tested

- 14 testes automatizados (`pytest`), todos offline: normalização de acentos, stemmer, BM25, Recall@k, MRR, estimativa de custo, consistência dos dados de exemplo, a comparação raw contra pt e a saída da linha de comando.
- O mesmo conjunto roda no GitHub Actions (`.github/workflows/test.yml`), junto com a execução da ferramenta nos dados de exemplo.
- Os números da tabela acima saem desse comando e são conferidos por testes.

## What was not tested

- Dados reais. O corpus e as perguntas são pequenos e inventados, e as perguntas foram escritas por quem também escreveu o stemmer. A diferença entre `raw` e `pt` deve ser menor ou diferente em texto real.
- Busca vetorial, reranker e qualquer chamada a um modelo. Aqui só existe BM25.
- O stemmer é simples (lista de sufixos) e erra em muitas palavras. Não substitui um stemmer completo como o RSLP.
- A contagem de tokens usa 4 caracteres por token. Para cobrança real use o tokenizador do seu provedor.
- Só foi rodado em Python 3.12 (CI) e 3.15 (máquina local).
