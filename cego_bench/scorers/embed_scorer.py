"""Embedding-based semantic similarity scoring."""

import numpy as np
from typing import List, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingScorer:
    """Semantic similarity scorer using sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding scorer.

        Args:
            model_name: Name of the sentence transformer model

        Raises:
            ImportError: If sentence-transformers is not available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. Install with: "
                "pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = None
        self._cache = {}

    def _ensure_model_loaded(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logging.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching.

        Args:
            text: Input text

        Returns:
            Normalized embedding vector
        """
        # Simple cache based on text hash
        cache_key = hash(text.strip().lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._ensure_model_loaded()

        try:
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Failed to encode text: {e}")
            raise

    def score_similarity(self, original_texts: List[str], optimized_texts: List[str]) -> float:
        """Calculate semantic similarity between original and optimized texts.

        Args:
            original_texts: List of original text items
            optimized_texts: List of optimized text items

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not original_texts or not optimized_texts:
            return 0.0

        try:
            # Get embeddings for all texts
            original_combined = " ".join(original_texts)
            optimized_combined = " ".join(optimized_texts)

            orig_embedding = self._get_embedding(original_combined)
            opt_embedding = self._get_embedding(optimized_combined)

            # Calculate cosine similarity
            similarity = np.dot(orig_embedding, opt_embedding)

            # Ensure similarity is in [0, 1] range
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logging.error(f"Error calculating embedding similarity: {e}")
            return 0.0

    def score_item_similarity(self, original_text: str, optimized_text: str) -> float:
        """Calculate similarity between individual text items.

        Args:
            original_text: Original text
            optimized_text: Optimized text

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not original_text.strip() or not optimized_text.strip():
            return 0.0

        try:
            orig_embedding = self._get_embedding(original_text)
            opt_embedding = self._get_embedding(optimized_text)

            similarity = np.dot(orig_embedding, opt_embedding)
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logging.error(f"Error calculating item similarity: {e}")
            return 0.0

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "model_loaded": self.model is not None,
            "model_name": self.model_name
        }


def is_available() -> bool:
    """Check if embedding scorer is available."""
    return SENTENCE_TRANSFORMERS_AVAILABLE