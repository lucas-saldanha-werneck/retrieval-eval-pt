def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for pos, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / pos
    return 0.0
