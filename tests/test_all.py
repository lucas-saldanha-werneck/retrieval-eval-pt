import json
from pathlib import Path

from retrieval_eval.bm25 import BM25
from retrieval_eval.cli import main, render
from retrieval_eval.cost import context_cost, estimate_tokens
from retrieval_eval.evaluate import compare, evaluate, load_jsonl
from retrieval_eval.metrics import recall_at_k, reciprocal_rank
from retrieval_eval.text import fold, stem, tokenize_pt, tokenize_raw

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_jsonl(ROOT / "data/example_corpus.jsonl")
GOLDEN = load_jsonl(ROOT / "data/example_golden.jsonl")


def test_fold_strips_accents_and_case():
    assert fold("Ação") == "acao"


def test_stem_keeps_short_words():
    assert stem("gas") == "gas"
    assert stem("certificacoes") == "certific"


def test_pt_tokenizer_drops_stopwords_and_accents():
    assert tokenize_pt("As férias do gestor") == ["feria", "gestor"]


def test_raw_tokenizer_keeps_accents():
    assert tokenize_raw("Férias") == ["férias"]


def test_bm25_ranks_matching_doc_first_and_skips_zero_scores():
    idx = BM25([["gato", "preto"], ["cachorro"], ["gato"]])
    assert idx.rank(["gato"], 5)[0] in (0, 2)
    assert 1 not in idx.rank(["gato"], 5)
    assert idx.rank(["peixe"], 5) == []


def test_bm25_empty_corpus():
    assert BM25([]).rank(["x"], 3) == []


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_cost():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefgh") == 2
    tokens, usd = context_cost(["abcdefgh"] * 2, 10.0)
    assert tokens == 4 and abs(usd - 0.00004) < 1e-12


def test_example_data_is_consistent():
    ids = {d["id"] for d in CORPUS}
    assert len(ids) == len(CORPUS) == 16
    assert all(set(g["relevant"]) <= ids for g in GOLDEN)


def test_accent_folding_finds_unaccented_query():
    docs = [{"id": "a", "title": "Férias", "text": "pedir férias ao gestor"},
            {"id": "b", "title": "Outro", "text": "assunto diferente"}]
    gold = [{"query": "ferias", "relevant": ["a"]}]
    assert evaluate(docs, gold, "raw", k=1)["recall_at_k"] == 0.0
    assert evaluate(docs, gold, "pt", k=1)["recall_at_k"] == 1.0


def test_pt_analyzer_beats_raw_on_example_set():
    raw, pt = compare(CORPUS, GOLDEN, k=3)
    assert pt["mrr"] > raw["mrr"]
    assert pt["recall_at_k"] >= raw["recall_at_k"]
    assert pt["avg_context_tokens"] < raw["avg_context_tokens"]


def test_known_miss_for_raw_analyzer():
    raw = evaluate(CORPUS, GOLDEN, "raw", k=3)
    assert [m["query"] for m in raw["misses"]] == ["como instalar impressora"]


def test_cli_runs_and_prints_table(capsys):
    assert main(["--corpus", str(ROOT / "data/example_corpus.jsonl"),
                 "--golden", str(ROOT / "data/example_golden.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "| pt |" in out and "| raw |" in out


def test_render_no_misses():
    r = {"analyzer": "x", "k": 3, "recall_at_k": 1, "mrr": 1, "avg_context_tokens": 1,
         "avg_context_usd": 0.0, "misses": []}
    assert "- none" in render([r])
