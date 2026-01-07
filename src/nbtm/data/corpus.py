"""Corpus loading and management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Union

from .preprocessing import TextPreprocessor
from .vocabulary import Vocabulary


class Corpus:
    """
    Document corpus for topic modeling.

    Manages a collection of documents with vocabulary.
    """

    def __init__(
        self,
        documents: List[List[str]],
        vocabulary: Optional[Vocabulary] = None,
        raw_documents: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize corpus.

        Args:
            documents: Tokenized documents (list of word lists)
            vocabulary: Pre-built vocabulary (built from documents if None)
            raw_documents: Original raw text (optional, for reference)
        """
        self.documents = documents
        self.raw_documents = raw_documents

        if vocabulary is None:
            self.vocabulary = Vocabulary.from_documents(documents)
        else:
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> List[str]:
        """Get document by index."""
        return self.documents[idx]

    def __iter__(self) -> Iterator[List[str]]:
        """Iterate over documents."""
        return iter(self.documents)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        preprocessor: Optional[TextPreprocessor] = None,
        min_word_count: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> Corpus:
        """
        Create corpus from raw texts.

        Args:
            texts: List of raw text documents
            preprocessor: Text preprocessor (default TextPreprocessor used if None)
            min_word_count: Minimum word frequency for vocabulary
            max_vocab_size: Maximum vocabulary size

        Returns:
            Corpus instance
        """
        if preprocessor is None:
            preprocessor = TextPreprocessor()

        # Preprocess documents
        documents = preprocessor.preprocess_documents(texts)

        # Build vocabulary
        vocabulary = Vocabulary.from_documents(
            documents,
            min_count=min_word_count,
            max_vocab_size=max_vocab_size,
        )

        # Filter documents to only include vocabulary words
        filtered_docs = [
            [word for word in doc if word in vocabulary]
            for doc in documents
        ]

        return cls(
            documents=filtered_docs,
            vocabulary=vocabulary,
            raw_documents=texts,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        preprocessor: Optional[TextPreprocessor] = None,
        min_word_count: int = 1,
        max_vocab_size: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> Corpus:
        """
        Load corpus from text file.

        Each line is treated as a document.

        Args:
            path: Path to text file
            preprocessor: Text preprocessor
            min_word_count: Minimum word frequency
            max_vocab_size: Maximum vocabulary size
            encoding: File encoding

        Returns:
            Corpus instance
        """
        path = Path(path)
        with open(path, encoding=encoding) as f:
            texts = [line.strip() for line in f if line.strip()]

        return cls.from_texts(
            texts,
            preprocessor=preprocessor,
            min_word_count=min_word_count,
            max_vocab_size=max_vocab_size,
        )

    def get_statistics(self) -> dict:
        """Get corpus statistics."""
        total_words = sum(len(doc) for doc in self.documents)
        doc_lengths = [len(doc) for doc in self.documents]

        return {
            "num_documents": len(self.documents),
            "vocab_size": len(self.vocabulary),
            "total_words": total_words,
            "avg_doc_length": total_words / len(self.documents) if self.documents else 0,
            "min_doc_length": min(doc_lengths) if doc_lengths else 0,
            "max_doc_length": max(doc_lengths) if doc_lengths else 0,
        }

    def filter_documents(
        self,
        min_length: int = 1,
        max_length: Optional[int] = None,
    ) -> Corpus:
        """
        Filter documents by length.

        Args:
            min_length: Minimum document length
            max_length: Maximum document length (None for no limit)

        Returns:
            New Corpus with filtered documents
        """
        filtered_docs = []
        filtered_raw = [] if self.raw_documents else None

        for i, doc in enumerate(self.documents):
            if len(doc) < min_length:
                continue
            if max_length is not None and len(doc) > max_length:
                continue
            filtered_docs.append(doc)
            if self.raw_documents:
                filtered_raw.append(self.raw_documents[i])

        return Corpus(
            documents=filtered_docs,
            vocabulary=self.vocabulary,
            raw_documents=filtered_raw,
        )

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"Corpus(documents={stats['num_documents']}, "
            f"vocab={stats['vocab_size']}, "
            f"avg_length={stats['avg_doc_length']:.1f})"
        )


def load_corpus(
    path: Union[str, Path],
    preprocessor: Optional[TextPreprocessor] = None,
    min_word_count: int = 5,
    max_vocab_size: Optional[int] = None,
) -> Corpus:
    """
    Convenience function to load corpus from file.

    Args:
        path: Path to corpus file
        preprocessor: Text preprocessor
        min_word_count: Minimum word frequency
        max_vocab_size: Maximum vocabulary size

    Returns:
        Corpus instance
    """
    return Corpus.from_file(
        path,
        preprocessor=preprocessor,
        min_word_count=min_word_count,
        max_vocab_size=max_vocab_size,
    )
