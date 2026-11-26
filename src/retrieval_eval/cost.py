def estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token). Use your provider's tokenizer for billing."""
    return max(1, round(len(text) / 4)) if text else 0


def context_cost(texts: list[str], usd_per_million_input_tokens: float) -> tuple[int, float]:
    tokens = sum(estimate_tokens(t) for t in texts)
    return tokens, tokens * usd_per_million_input_tokens / 1_000_000
