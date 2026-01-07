"""Topic coherence metrics.

Implements various coherence measures for evaluating topic quality:
- UMass: Document co-occurrence based (Mimno et al., 2011)
- UCI: Pointwise mutual information based (Newman et al., 2010)
- NPMI: Normalized PMI (Bouma, 2009)
- C_V: Combined measure (Roder et al., 2015)
"""

from __future__ import annotations

from collections import Counter
from typing import List, Literal, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


class TopicCoherence:
    """
    Compute topic coherence scores.

    Coherence measures how semantically similar the top words
    of a topic are to each other.

    Example:
        >>> coherence = TopicCoherence(documents, measure="c_v")
        >>> score = coherence.compute(model.get_all_topic_words())
    """

    MEASURES = ["u_mass", "c_uci", "c_npmi", "c_v"]

    def __init__(
        self,
        documents: List[List[str]],
        measure: Literal["u_mass", "c_uci", "c_npmi", "c_v"] = "c_v",
        top_n: int = 10,
        window_size: int = 110,
    ) -> None:
        """
        Initialize coherence calculator.

        Args:
            documents: Corpus documents for computing co-occurrences
            measure: Coherence measure to use
            top_n: Number of top words per topic
            window_size: Sliding window size for C_V (not used for others)
        """
        if measure not in self.MEASURES:
            raise ValueError(f"Unknown measure: {measure}. Choose from {self.MEASURES}")

        self.documents = documents
        self.measure = measure
        self.top_n = top_n
        self.window_size = window_size

        # Precompute frequencies
        self._compute_frequencies()

    def _compute_frequencies(self) -> None:
        """Precompute word and co-occurrence frequencies."""
        self.word_doc_count: Counter[str] = Counter()
        self.pair_doc_count: Counter[tuple[str, str]] = Counter()
        self.num_docs = len(self.documents)

        for doc in self.documents:
            unique_words = set(doc)
            for word in unique_words:
                self.word_doc_count[word] += 1

            word_list = list(unique_words)
            for i, w1 in enumerate(word_list):
                for w2 in word_list[i + 1:]:
                    pair = tuple(sorted([w1, w2]))
                    self.pair_doc_count[pair] += 1

    def compute(
        self,
        topic_words: List[List[tuple[str, float]]],
    ) -> float:
        """
        Compute coherence for all topics.

        Args:
            topic_words: List of (word, prob) tuples for each topic

        Returns:
            Average coherence score across topics
        """
        # Extract just the words
        topics = [
            [word for word, _ in words[:self.top_n]]
            for words in topic_words
        ]

        scores = [self._topic_coherence(words) for words in topics]
        return np.mean(scores)

    def compute_per_topic(
        self,
        topic_words: List[List[tuple[str, float]]],
    ) -> List[float]:
        """
        Compute coherence for each topic separately.

        Args:
            topic_words: List of (word, prob) tuples for each topic

        Returns:
            List of coherence scores
        """
        topics = [
            [word for word, _ in words[:self.top_n]]
            for words in topic_words
        ]
        return [self._topic_coherence(words) for words in topics]

    def _topic_coherence(self, words: List[str]) -> float:
        """Compute coherence for a single topic."""
        if self.measure == "u_mass":
            return self._umass_coherence(words)
        elif self.measure == "c_uci":
            return self._uci_coherence(words)
        elif self.measure == "c_npmi":
            return self._npmi_coherence(words)
        else:  # c_v
            return self._cv_coherence(words)

    def _umass_coherence(self, words: List[str]) -> float:
        """
        UMass coherence (Mimno et al., 2011).

        Based on document co-occurrence.
        Range: (-inf, 0], higher is better.
        """
        score = 0.0
        n_pairs = 0

        for i, w2 in enumerate(words[1:], 1):
            for w1 in words[:i]:
                pair = tuple(sorted([w1, w2]))
                d_w1_w2 = self.pair_doc_count.get(pair, 0)
                d_w1 = self.word_doc_count.get(w1, 0)

                if d_w1 > 0:
                    score += np.log((d_w1_w2 + 1) / d_w1)
                    n_pairs += 1

        return score / n_pairs if n_pairs > 0 else 0.0

    def _uci_coherence(self, words: List[str]) -> float:
        """
        UCI coherence (Newman et al., 2010).

        Based on pointwise mutual information.
        Range: (-inf, +inf), higher is better.
        """
        score = 0.0
        n_pairs = 0
        eps = 1e-12

        for i, w1 in enumerate(words):
            for w2 in words[i + 1:]:
                pair = tuple(sorted([w1, w2]))
                p_w1 = self.word_doc_count.get(w1, 0) / self.num_docs
                p_w2 = self.word_doc_count.get(w2, 0) / self.num_docs
                p_w1_w2 = self.pair_doc_count.get(pair, 0) / self.num_docs

                if p_w1 > eps and p_w2 > eps:
                    pmi = np.log((p_w1_w2 + eps) / (p_w1 * p_w2))
                    score += pmi
                    n_pairs += 1

        return score / n_pairs if n_pairs > 0 else 0.0

    def _npmi_coherence(self, words: List[str]) -> float:
        """
        Normalized PMI coherence (Bouma, 2009).

        Range: [-1, 1], higher is better.
        """
        score = 0.0
        n_pairs = 0
        eps = 1e-12

        for i, w1 in enumerate(words):
            for w2 in words[i + 1:]:
                pair = tuple(sorted([w1, w2]))
                p_w1 = self.word_doc_count.get(w1, 0) / self.num_docs
                p_w2 = self.word_doc_count.get(w2, 0) / self.num_docs
                p_w1_w2 = self.pair_doc_count.get(pair, 0) / self.num_docs

                if p_w1_w2 > eps and p_w1 > eps and p_w2 > eps:
                    pmi = np.log((p_w1_w2 + eps) / (p_w1 * p_w2))
                    npmi = pmi / (-np.log(p_w1_w2 + eps))
                    score += npmi
                    n_pairs += 1

        return score / n_pairs if n_pairs > 0 else 0.0

    def _cv_coherence(self, words: List[str]) -> float:
        """
        C_V coherence (Roder et al., 2015).

        Combines NPMI with word similarity measures.
        Simplified implementation using NPMI.
        Range: [0, 1], higher is better.
        """
        # Simplified: use NPMI and scale to [0, 1]
        npmi = self._npmi_coherence(words)
        return (npmi + 1) / 2


def compute_coherence(
    model: BaseTopicModel,
    documents: List[List[str]],
    measure: str = "c_v",
    top_n: int = 10,
) -> float:
    """
    Convenience function to compute coherence.

    Args:
        model: Trained topic model
        documents: Reference corpus
        measure: Coherence measure
        top_n: Number of top words

    Returns:
        Coherence score
    """
    coherence = TopicCoherence(
        documents=documents,
        measure=measure,
        top_n=top_n,
    )
    return coherence.compute(model.get_all_topic_words(top_n))
