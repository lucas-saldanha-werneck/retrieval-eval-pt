import argparse
import sys

from .evaluate import compare, load_jsonl


def render(results: list[dict]) -> str:
    k = results[0]["k"]
    lines = [f"| analyzer | Recall@{k} | MRR | avg context tokens | avg context USD |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['analyzer']} | {r['recall_at_k']:.2f} | {r['mrr']:.2f} | "
                     f"{r['avg_context_tokens']:.0f} | {r['avg_context_usd']:.6f} |")
    for r in results:
        lines.append(f"\nMisses with '{r['analyzer']}' ({len(r['misses'])}):")
        lines += [f"- {m['query']}  (expected {', '.join(m['relevant'])})" for m in r["misses"]] or ["- none"]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="retrieval-eval", description=__doc__)
    p.add_argument("--corpus", default="data/example_corpus.jsonl")
    p.add_argument("--golden", default="data/example_golden.jsonl")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--price-per-million", type=float, default=1.0,
                   help="USD per 1M input tokens (placeholder, set your own)")
    a = p.parse_args(argv)
    print(render(compare(load_jsonl(a.corpus), load_jsonl(a.golden), a.k, a.price_per_million)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
