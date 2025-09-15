"""TF-IDF based semantic similarity scoring."""

import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging


class TFIDFScorer:
    """TF-IDF based semantic similarity scorer."""

    def __init__(self,
                 max_features: int = 5000,
                 stop_words: str = 'english',
                 ngram_range: tuple = (1, 2)):
        """Initialize TF-IDF scorer.

        Args:
            max_features: Maximum number of features
            stop_words: Stop words to remove
            ngram_range: N-gram range for feature extraction
        """
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vectorizer = None

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)

        return text.strip()

    def score_similarity(self, original_texts: List[str], optimized_texts: List[str]) -> float:
        """Calculate TF-IDF based similarity between text collections.

        Args:
            original_texts: List of original text items
            optimized_texts: List of optimized text items

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not original_texts or not optimized_texts:
            return 0.0

        try:
            # Preprocess texts
            original_processed = [self._preprocess_text(text) for text in original_texts]
            optimized_processed = [self._preprocess_text(text) for text in optimized_texts]

            # Combine texts for comparison
            original_combined = " ".join(original_processed)
            optimized_combined = " ".join(optimized_processed)

            # Fit TF-IDF on combined corpus
            corpus = [original_combined, optimized_combined]
            corpus = [text for text in corpus if text.strip()]  # Remove empty

            if len(corpus) < 2:
                return 0.0

            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

            tfidf_matrix = vectorizer.fit_transform(corpus)

            if tfidf_matrix.shape[0] < 2:
                return 0.0

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Ensure similarity is in [0, 1] range
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logging.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0

    def score_item_similarity(self, original_text: str, optimized_text: str) -> float:
        """Calculate TF-IDF similarity between individual text items.

        Args:
            original_text: Original text
            optimized_text: Optimized text

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        return self.score_similarity([original_text], [optimized_text])

    def get_top_terms(self, texts: List[str], top_k: int = 10) -> List[tuple]:
        """Get top TF-IDF terms for analysis.

        Args:
            texts: List of texts to analyze
            top_k: Number of top terms to return

        Returns:
            List of (term, score) tuples
        """
        if not texts:
            return []

        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            combined_text = " ".join(processed_texts)

            if not combined_text.strip():
                return []

            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()

            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            term_scores = list(zip(feature_names, scores))

            # Sort by score and return top k
            term_scores.sort(key=lambda x: x[1], reverse=True)
            return term_scores[:top_k]

        except Exception as e:
            logging.error(f"Error getting top terms: {e}")
            return []

    def analyze_coverage(self, original_texts: List[str], optimized_texts: List[str]) -> dict:
        """Analyze term coverage between original and optimized texts.

        Args:
            original_texts: List of original texts
            optimized_texts: List of optimized texts

        Returns:
            Dictionary with coverage statistics
        """
        try:
            original_terms = self.get_top_terms(original_texts, top_k=100)
            optimized_terms = self.get_top_terms(optimized_texts, top_k=100)

            if not original_terms or not optimized_terms:
                return {"term_coverage": 0.0, "score_coverage": 0.0}

            original_term_set = set(term for term, _ in original_terms)
            optimized_term_set = set(term for term, _ in optimized_terms)

            # Term overlap
            overlap = len(original_term_set.intersection(optimized_term_set))
            term_coverage = overlap / len(original_term_set) if original_terms else 0.0

            # Score-weighted coverage
            original_dict = dict(original_terms)
            score_coverage = 0.0
            total_original_score = sum(score for _, score in original_terms)

            if total_original_score > 0:
                for term in optimized_term_set:
                    if term in original_dict:
                        score_coverage += original_dict[term] / total_original_score

            return {
                "term_coverage": term_coverage,
                "score_coverage": score_coverage,
                "original_unique_terms": len(original_term_set),
                "optimized_unique_terms": len(optimized_term_set),
                "shared_terms": overlap
            }

        except Exception as e:
            logging.error(f"Error analyzing coverage: {e}")
            return {"term_coverage": 0.0, "score_coverage": 0.0}