"""Vocabulary management for topic models."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import json


class Vocabulary:
    """
    Vocabulary for mapping words to indices.

    Attributes:
        word2idx: Mapping from word to index
        idx2word: Mapping from index to word
        word_counts: Word frequency counts
    """

    UNK_TOKEN = "<UNK>"
    UNK_IDX = 0

    def __init__(self) -> None:
        """Initialize empty vocabulary."""
        self.word2idx: dict[str, int] = {self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: dict[int, str] = {self.UNK_IDX: self.UNK_TOKEN}
        self.word_counts: Counter[str] = Counter()
        self._frozen = False

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx

    def __getitem__(self, word: str) -> int:
        """Get index for word."""
        return self.word2idx.get(word, self.UNK_IDX)

    def add_word(self, word: str) -> int:
        """
        Add word to vocabulary.

        Args:
            word: Word to add

        Returns:
            Index of the word
        """
        if self._frozen:
            return self.word2idx.get(word, self.UNK_IDX)

        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_counts[word] += 1
        return self.word2idx[word]

    def add_words(self, words: Iterable[str]) -> List[int]:
        """Add multiple words to vocabulary."""
        return [self.add_word(word) for word in words]

    def get_word(self, idx: int) -> str:
        """Get word for index."""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def get_words(self, indices: Iterable[int]) -> List[str]:
        """Get words for multiple indices."""
        return [self.get_word(idx) for idx in indices]

    @classmethod
    def from_documents(
        cls,
        documents: List[List[str]],
        min_count: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> Vocabulary:
        """
        Build vocabulary from documents.

        Args:
            documents: List of tokenized documents
            min_count: Minimum word frequency to include
            max_vocab_size: Maximum vocabulary size (None for unlimited)

        Returns:
            Vocabulary instance
        """
        vocab = cls()

        # Count all words
        for doc in documents:
            for word in doc:
                vocab.word_counts[word] += 1

        # Filter by min_count
        filtered_words = [
            (word, count)
            for word, count in vocab.word_counts.items()
            if count >= min_count
        ]

        # Sort by frequency (descending)
        filtered_words.sort(key=lambda x: x[1], reverse=True)

        # Limit vocabulary size
        if max_vocab_size is not None:
            filtered_words = filtered_words[:max_vocab_size]

        # Reset and add filtered words
        vocab.word2idx = {cls.UNK_TOKEN: cls.UNK_IDX}
        vocab.idx2word = {cls.UNK_IDX: cls.UNK_TOKEN}

        for word, _ in filtered_words:
            idx = len(vocab.word2idx)
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word

        return vocab

    def freeze(self) -> None:
        """Freeze vocabulary (no new words can be added)."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze vocabulary."""
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Check if vocabulary is frozen."""
        return self._frozen

    def to_indices(self, words: List[str]) -> List[int]:
        """Convert words to indices."""
        return [self[word] for word in words]

    def to_words(self, indices: List[int]) -> List[str]:
        """Convert indices to words."""
        return [self.get_word(idx) for idx in indices]

    def save(self, path: Path | str) -> None:
        """Save vocabulary to file."""
        path = Path(path)
        data = {
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> Vocabulary:
        """Load vocabulary from file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_counts = Counter(data["word_counts"])
        return vocab

    def most_common(self, n: int = 10) -> List[tuple[str, int]]:
        """Get most common words."""
        return self.word_counts.most_common(n)

    def __repr__(self) -> str:
        """String representation."""
        return f"Vocabulary(size={len(self)}, frozen={self._frozen})"
