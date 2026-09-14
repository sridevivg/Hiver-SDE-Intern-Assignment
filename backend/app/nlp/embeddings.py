"""
SupportGraph AI — Semantic Embedding Generator

Generates dense semantic vector embeddings using local open-source sentence-transformers.
Runs 100% locally without external API dependencies or keys.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbeddingGenerator:
    """
    Wrapper around SentenceTransformer for local, batch-wise vector generation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info("Loading sentence-transformers model '%s'...", self.model_name)
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model '%s' loaded successfully.", self.model_name)
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of text strings into a 2D numpy array of embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for model inference (default: 64).
            show_progress_bar: Whether to display encoding progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
