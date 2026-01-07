"""Dataset abstraction for topic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .corpus import Corpus
from .vocabulary import Vocabulary


@dataclass
class DataSplit:
    """Train/validation/test split of a corpus."""

    train: Corpus
    val: Optional[Corpus] = None
    test: Optional[Corpus] = None


def train_val_test_split(
    corpus: Corpus,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
) -> DataSplit:
    """
    Split corpus into train/validation/test sets.

    Args:
        corpus: Source corpus
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_state: Random seed
        shuffle: Whether to shuffle before splitting

    Returns:
        DataSplit with train/val/test corpora
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    n = len(corpus)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    # Calculate split points
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    def create_subset(idxs: np.ndarray) -> Corpus:
        docs = [corpus.documents[i] for i in idxs]
        raw = (
            [corpus.raw_documents[i] for i in idxs]
            if corpus.raw_documents
            else None
        )
        return Corpus(
            documents=docs,
            vocabulary=corpus.vocabulary,  # Share vocabulary
            raw_documents=raw,
        )

    train_corpus = create_subset(train_indices)
    val_corpus = create_subset(val_indices) if len(val_indices) > 0 else None
    test_corpus = create_subset(test_indices) if len(test_indices) > 0 else None

    return DataSplit(train=train_corpus, val=val_corpus, test=test_corpus)


class TopicModelDataset:
    """
    Dataset wrapper for topic models.

    Provides batch iteration and document access.
    """

    def __init__(
        self,
        corpus: Corpus,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initialize dataset.

        Args:
            corpus: Source corpus
            batch_size: Batch size for iteration
            shuffle: Shuffle documents each epoch
            random_state: Random seed
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state)

        self._indices = np.arange(len(corpus))

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.corpus)

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch."""
        return (len(self) + self.batch_size - 1) // self.batch_size

    @property
    def vocabulary(self) -> Vocabulary:
        """Get vocabulary."""
        return self.corpus.vocabulary

    @property
    def documents(self) -> List[List[str]]:
        """Get all documents."""
        return self.corpus.documents

    def get_batch(self, batch_idx: int) -> List[List[str]]:
        """Get a batch by index."""
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self))
        indices = self._indices[start:end]
        return [self.corpus.documents[i] for i in indices]

    def iterate_batches(self) -> List[List[List[str]]]:
        """
        Iterate over all batches.

        Yields:
            Batches of documents
        """
        if self.shuffle:
            self.rng.shuffle(self._indices)

        batches = []
        for i in range(self.num_batches):
            batches.append(self.get_batch(i))
        return batches

    def reset(self, shuffle: bool = True) -> None:
        """Reset dataset for new epoch."""
        if shuffle:
            self.rng.shuffle(self._indices)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TopicModelDataset("
            f"documents={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"num_batches={self.num_batches})"
        )
