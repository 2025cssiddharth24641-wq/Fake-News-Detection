import re
import string
import unicodedata
from typing import Iterable, List

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOPWORDS = set(ENGLISH_STOP_WORDS)
_LEMMATIZER = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _ensure_nltk_resources() -> None:
    resources = [("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]
    for lookup, name in resources:
        try:
            nltk.data.find(lookup)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_resources()


def _normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = _normalize_repeated_characters(text)
    text = text.lower().translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _sanitize_text(text)
    if not cleaned:
        return ""
    tokens = []
    for token in cleaned.split(" "):
        if len(token) < 2:
            continue
        if token in _STOPWORDS:
            continue
        lemma = _LEMMATIZER.lemmatize(token)
        if len(lemma) < 2:
            continue
        tokens.append(lemma)
    return " ".join(tokens)


def clean_texts(texts: Iterable[str]) -> List[str]:
    return [clean_text(text) for text in texts]


def is_meaningful_input(text: str, min_word_count: int = 8, min_char_count: int = 40) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.strip()
    if len(normalized) < min_char_count:
        return False
    words = re.findall(r"[A-Za-z]+", normalized)
    if len(words) < min_word_count:
        return False
    cleaned = clean_text(normalized)
    cleaned_words = cleaned.split()
    return len(cleaned_words) >= max(6, min_word_count - 1)


def cleaned_word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(clean_text(text).split())
