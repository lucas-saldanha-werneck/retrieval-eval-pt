"""Two analyzers: 'raw' (plain lowercase split) and 'pt' (accent folding, stopwords, light stemming)."""
import re
import unicodedata

STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem e ou que se "
    "ao aos como mais ja foi sao ser tem ter pode posso preciso quero qual quais quando onde "
    "meu minha meus minhas seu sua".split()
)
SUFFIXES = ("amentos", "amento", "acoes", "acao", "adores", "ador", "idades", "idade",
            "mente", "oes", "ais", "eis", "res", "ns", "s")


def fold(text: str) -> str:
    """Lowercase and strip accents: 'Senha' == 'sênha'-style typos aside, 'ação' -> 'acao'."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def stem(word: str) -> str:
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def tokenize_raw(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def tokenize_pt(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold(text))
    return [stem(w) for w in words if w not in STOPWORDS]


ANALYZERS = {"raw": tokenize_raw, "pt": tokenize_pt}
