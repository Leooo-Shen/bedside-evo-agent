"""Retriever module for memory search.

Implements the Search operation: R_t = R(M_t, x_t)
where R retrieves relevant memory entries based on similarity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Memory, MemoryEntry


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    entry: MemoryEntry
    score: float
    rank: int


class Retriever(ABC):
    """
    Abstract base class for memory retrieval.

    The retriever implements: R_t = R(M_t, x_t)
    """

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant memory entries.

        Args:
            query: Query text (x_t)
            memory: Memory store (M_t)
            top_k: Number of entries to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of retrieval results sorted by relevance
        """
        pass

    def update_embeddings(self, memory: Memory) -> None:
        """Update embeddings for all memory entries."""
        for entry in memory:
            if entry.embedding is None:
                text = entry.to_text(include_trajectory=True, include_feedback=True)
                entry.embedding = self.encode(text)


class EmbeddingRetriever(Retriever):
    """
    Embedding-based retriever using sentence transformers.

    Uses BAAI/bge-base-en-v1.5 as the default encoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize embedding retriever.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cuda, cpu, or None for auto)
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_embeddings = cache_embeddings
        self._model = None
        self._embedding_cache: Dict[str, List[float]] = {}

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingRetriever. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()

        if self.cache_embeddings:
            self._embedding_cache[text] = embedding_list

        return embedding_list

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently."""
        to_encode = []
        to_encode_idx = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self._embedding_cache:
                results[i] = self._embedding_cache[text]
            else:
                to_encode.append(text)
                to_encode_idx.append(i)

        if to_encode:
            embeddings = self.model.encode(to_encode, convert_to_numpy=True)
            for idx, (text, emb) in zip(to_encode_idx, zip(to_encode, embeddings)):
                emb_list = emb.tolist()
                results[idx] = emb_list
                if self.cache_embeddings:
                    self._embedding_cache[text] = emb_list

        return results

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant memory entries.

        Implements: R_t = Top-k_{m_i ∈ M_t} φ(x_t, m_i)
        where φ is cosine similarity.
        """
        if len(memory) == 0:
            return []

        query_embedding = self.encode(query)

        results = []
        for entry in memory:
            if entry.embedding is None:
                text = entry.to_text(include_trajectory=True, include_feedback=True)
                entry.embedding = self.encode(text)

            score = cosine_similarity(query_embedding, entry.embedding)

            if score >= threshold:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(entry=entry, score=score, rank=i + 1)
            for i, (entry, score) in enumerate(results[:top_k])
        ]

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()


class TFIDFRetriever(Retriever):
    """
    TF-IDF based retriever that doesn't require external models.

    Uses Term Frequency-Inverse Document Frequency for text similarity.
    """

    def __init__(self):
        """Initialize TF-IDF retriever."""
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Filter out very short tokens and common stopwords
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
            'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that',
            'these', 'those', 'am', 'it', 'its', 'he', 'she', 'they', 'them',
            'his', 'her', 'their', 'what', 'which', 'who', 'whom'
        }
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for a document."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize by document length
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}

    def _build_vocabulary(self, documents: List[str]) -> None:
        """Build vocabulary and compute IDF from documents."""
        # Count document frequency for each term
        doc_freq: Dict[str, int] = {}
        all_tokens = set()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_tokens.update(tokens)
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Build vocabulary
        self._vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        # Compute IDF: log(N / df) + 1
        n_docs = len(documents) if documents else 1
        self._idf = {
            token: np.log(n_docs / (df + 1)) + 1
            for token, df in doc_freq.items()
        }
        self._doc_count = n_docs

    def _to_tfidf_vector(self, text: str) -> List[float]:
        """Convert text to TF-IDF vector."""
        if not self._vocabulary:
            return []

        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        vector = [0.0] * len(self._vocabulary)
        for token, freq in tf.items():
            if token in self._vocabulary:
                idx = self._vocabulary[token]
                idf = self._idf.get(token, 1.0)
                vector[idx] = freq * idf

        return vector

    def encode(self, text: str) -> List[float]:
        """Encode text to TF-IDF vector."""
        return self._to_tfidf_vector(text)

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant memory entries using TF-IDF similarity.
        """
        if len(memory) == 0:
            return []

        # Build vocabulary from all memory entries + query
        documents = []
        for entry in memory:
            text = entry.to_text(include_trajectory=True, include_feedback=True)
            documents.append(text)
        documents.append(query)

        self._build_vocabulary(documents)

        # Encode query
        query_vector = self._to_tfidf_vector(query)

        # Compute similarity with each memory entry
        results = []
        for entry in memory:
            text = entry.to_text(include_trajectory=True, include_feedback=True)
            entry_vector = self._to_tfidf_vector(text)

            score = cosine_similarity(query_vector, entry_vector)

            if score >= threshold:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(entry=entry, score=score, rank=i + 1)
            for i, (entry, score) in enumerate(results[:top_k])
        ]


class RecencyRetriever(Retriever):
    """Retrieve most recent entries (for baseline comparison)."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        """Generate placeholder embedding."""
        return [0.0] * self.embedding_dim

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve most recent entries."""
        recent = memory.get_recent(top_k)

        return [
            RetrievalResult(entry=entry, score=1.0 - (i * 0.1), rank=i + 1)
            for i, entry in enumerate(recent)
        ]
