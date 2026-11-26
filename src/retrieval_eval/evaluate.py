import json
from pathlib import Path

from .bm25 import BM25
from .cost import context_cost
from .metrics import recall_at_k, reciprocal_rank
from .text import ANALYZERS


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(corpus: list[dict], golden: list[dict], analyzer: str, k: int = 3,
             price_per_million: float = 0.0) -> dict:
    tok = ANALYZERS[analyzer]
    index = BM25([tok(d["title"] + " " + d["text"]) for d in corpus])
    ids = [d["id"] for d in corpus]
    rows, rec, rr, tokens, usd = [], 0.0, 0.0, 0, 0.0
    for g in golden:
        top = index.rank(tok(g["query"]), k)
        ranked = [ids[i] for i in top]
        rel = set(g["relevant"])
        r, m = recall_at_k(ranked, rel, k), reciprocal_rank(ranked, rel)
        t, u = context_cost([corpus[i]["text"] for i in top], price_per_million)
        rec, rr, tokens, usd = rec + r, rr + m, tokens + t, usd + u
        rows.append({"query": g["query"], "relevant": sorted(rel), "ranked": ranked, "recall": r, "rr": m})
    n = len(golden) or 1
    return {"analyzer": analyzer, "k": k, "queries": len(golden), "recall_at_k": rec / n,
            "mrr": rr / n, "avg_context_tokens": tokens / n, "avg_context_usd": usd / n,
            "misses": [r for r in rows if r["recall"] == 0.0], "rows": rows}


def compare(corpus, golden, k=3, price_per_million=0.0) -> list[dict]:
    return [evaluate(corpus, golden, a, k, price_per_million) for a in ANALYZERS]
