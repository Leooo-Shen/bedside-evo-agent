"""Embedding-based action matcher for recommendation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _cosine_similarity_matrix(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("Embeddings must be rank-2 arrays.")
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError(f"Embedding dimensions do not match: {lhs.shape[1]} != {rhs.shape[1]}")
    lhs_norms = np.linalg.norm(lhs, axis=1, keepdims=True)
    rhs_norms = np.linalg.norm(rhs, axis=1, keepdims=True)
    lhs_safe = np.divide(lhs, lhs_norms, out=np.zeros_like(lhs), where=lhs_norms > 0)
    rhs_safe = np.divide(rhs, rhs_norms, out=np.zeros_like(rhs), where=rhs_norms > 0)
    return lhs_safe @ rhs_safe.T


@dataclass(frozen=True)
class EmbeddingMatchPair:
    prediction_index: int
    gt_index: int
    similarity: float


class EmbeddingActionMatcher:
    def __init__(
        self,
        *,
        model_name: str,
        similarity_threshold: float,
        device: Optional[str],
    ):
        self.model_name = str(model_name)
        self.similarity_threshold = float(similarity_threshold)
        self.device = str(device) if device is not None else None
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding-based matching. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("encode_texts requires at least one text.")
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        if matrix.ndim != 2:
            raise ValueError(f"Unexpected embedding matrix shape: {matrix.shape}")
        return matrix

    def match(
        self,
        *,
        prediction_texts: Sequence[str],
        gt_texts: Sequence[str],
    ) -> Dict[str, Any]:
        if not prediction_texts or not gt_texts:
            return {
                "matched_pairs": [],
                "matched_prediction_indices": [],
                "pair_similarities": [],
                "similarity_threshold": float(self.similarity_threshold),
            }

        prediction_embeddings = self.encode_texts(prediction_texts)
        gt_embeddings = self.encode_texts(gt_texts)
        similarity_matrix = _cosine_similarity_matrix(prediction_embeddings, gt_embeddings)

        candidates: List[EmbeddingMatchPair] = []
        for prediction_index in range(similarity_matrix.shape[0]):
            for gt_index in range(similarity_matrix.shape[1]):
                similarity = float(similarity_matrix[prediction_index, gt_index])
                if similarity < float(self.similarity_threshold):
                    continue
                candidates.append(
                    EmbeddingMatchPair(
                        prediction_index=int(prediction_index),
                        gt_index=int(gt_index),
                        similarity=float(similarity),
                    )
                )

        candidates = sorted(
            candidates,
            key=lambda pair: (-pair.similarity, pair.prediction_index, pair.gt_index),
        )

        matched_prediction_indices = set()
        matched_gt_indices = set()
        selected_pairs: List[EmbeddingMatchPair] = []
        for pair in candidates:
            if pair.prediction_index in matched_prediction_indices:
                continue
            if pair.gt_index in matched_gt_indices:
                continue
            selected_pairs.append(pair)
            matched_prediction_indices.add(pair.prediction_index)
            matched_gt_indices.add(pair.gt_index)

        selected_pairs = sorted(selected_pairs, key=lambda pair: pair.prediction_index)
        return {
            "matched_pairs": [
                {
                    "prediction_index": int(pair.prediction_index),
                    "gt_index": int(pair.gt_index),
                }
                for pair in selected_pairs
            ],
            "matched_prediction_indices": sorted(int(pair.prediction_index) for pair in selected_pairs),
            "pair_similarities": [
                {
                    "prediction_index": int(pair.prediction_index),
                    "gt_index": int(pair.gt_index),
                    "similarity": float(pair.similarity),
                }
                for pair in selected_pairs
            ],
            "similarity_threshold": float(self.similarity_threshold),
        }
