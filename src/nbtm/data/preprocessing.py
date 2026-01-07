"""Text preprocessing utilities."""

from __future__ import annotations

import re
import string
from typing import Callable, List, Optional, Set

# Common English stopwords
ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "could", "ought", "of", "as",
}

# Korean stopwords (common particles and suffixes)
KOREAN_STOPWORDS: Set[str] = {
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "까지", "부터", "에게", "한테", "께",
    "이다", "있다", "하다", "되다", "않다", "없다", "같다",
    "그", "저", "이것", "저것", "그것", "여기", "저기", "거기",
    "및", "등", "또는", "그리고", "하지만", "그러나", "따라서",
}


def word_tokenize(text: str) -> List[str]:
    """
    Word tokenizer that handles punctuation.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Remove punctuation attached to words
    text = re.sub(r"[^\w\s]", " ", text)
    # Split on whitespace
    tokens = text.split()
    # Filter empty tokens
    return [t for t in tokens if t]


class TextPreprocessor:
    """
    Text preprocessing pipeline.

    Applies configurable preprocessing steps:
    - Lowercasing
    - Punctuation removal
    - Stopword removal
    - Minimum word length filtering
    - Custom filters
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stopwords: Optional[Set[str]] = None,
        min_word_length: int = 2,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove stopwords
            stopwords: Custom stopword set (default: English + Korean)
            min_word_length: Minimum word length to keep
            tokenizer: Custom tokenizer function
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.tokenizer = tokenizer or word_tokenize

        # Combine default stopwords
        if stopwords is None:
            self.stopwords = ENGLISH_STOPWORDS | KOREAN_STOPWORDS
        else:
            self.stopwords = stopwords

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess a single text.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = self.tokenizer(text)

        # Filter tokens
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_word_length:
                continue

            # Skip stopwords
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue

            filtered.append(token)

        return filtered

    def preprocess_documents(
        self, documents: List[str]
    ) -> List[List[str]]:
        """
        Preprocess multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of tokenized documents
        """
        return [self.preprocess(doc) for doc in documents]

    def __call__(self, text: str) -> List[str]:
        """Apply preprocessing."""
        return self.preprocess(text)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TextPreprocessor("
            f"lowercase={self.lowercase}, "
            f"remove_punctuation={self.remove_punctuation}, "
            f"remove_stopwords={self.remove_stopwords}, "
            f"min_word_length={self.min_word_length})"
        )
