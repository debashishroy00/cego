"""ROUGE-based lexical overlap scoring for semantic similarity."""

import re
from typing import List, Dict, Set
from collections import Counter
import logging


class ROUGEScorer:
    """ROUGE-based lexical overlap scorer."""

    def __init__(self, use_stemming: bool = False):
        """Initialize ROUGE scorer.

        Args:
            use_stemming: Whether to apply basic stemming
        """
        self.use_stemming = use_stemming

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)

        if self.use_stemming:
            tokens = [self._simple_stem(token) for token in tokens]

        return tokens

    def _simple_stem(self, word: str) -> str:
        """Apply simple stemming (remove common suffixes).

        Args:
            word: Input word

        Returns:
            Stemmed word
        """
        if len(word) <= 3:
            return word

        # Simple suffix removal
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]

        return word

    def _get_ngrams(self, tokens: List[str], n: int) -> Set[tuple]:
        """Extract n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            Set of n-grams
        """
        if len(tokens) < n:
            return set()

        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def rouge_n(self, reference_texts: List[str], candidate_texts: List[str], n: int = 1) -> Dict[str, float]:
        """Calculate ROUGE-N scores.

        Args:
            reference_texts: Reference (original) texts
            candidate_texts: Candidate (optimized) texts
            n: N-gram size

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not reference_texts or not candidate_texts:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        try:
            # Tokenize and get n-grams
            ref_tokens = []
            for text in reference_texts:
                ref_tokens.extend(self._tokenize(text))

            cand_tokens = []
            for text in candidate_texts:
                cand_tokens.extend(self._tokenize(text))

            if not ref_tokens or not cand_tokens:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            ref_ngrams = self._get_ngrams(ref_tokens, n)
            cand_ngrams = self._get_ngrams(cand_tokens, n)

            if not ref_ngrams or not cand_ngrams:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Calculate overlap
            overlap = len(ref_ngrams.intersection(cand_ngrams))

            # Calculate precision, recall, F1
            precision = overlap / len(cand_ngrams)
            recall = overlap / len(ref_ngrams)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        except Exception as e:
            logging.error(f"Error calculating ROUGE-{n}: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def rouge_l(self, reference_texts: List[str], candidate_texts: List[str]) -> Dict[str, float]:
        """Calculate ROUGE-L (Longest Common Subsequence) scores.

        Args:
            reference_texts: Reference (original) texts
            candidate_texts: Candidate (optimized) texts

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not reference_texts or not candidate_texts:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        try:
            # Combine texts and tokenize
            ref_combined = " ".join(reference_texts)
            cand_combined = " ".join(candidate_texts)

            ref_tokens = self._tokenize(ref_combined)
            cand_tokens = self._tokenize(cand_combined)

            if not ref_tokens or not cand_tokens:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Calculate LCS length
            lcs_length = self._lcs_length(ref_tokens, cand_tokens)

            # Calculate precision, recall, F1
            precision = lcs_length / len(cand_tokens)
            recall = lcs_length / len(ref_tokens)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        except Exception as e:
            logging.error(f"Error calculating ROUGE-L: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of Longest Common Subsequence.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def score_similarity(self, original_texts: List[str], optimized_texts: List[str]) -> float:
        """Calculate overall ROUGE-based similarity score.

        Combines ROUGE-1, ROUGE-2, and ROUGE-L with equal weighting.

        Args:
            original_texts: List of original text items
            optimized_texts: List of optimized text items

        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        if not original_texts or not optimized_texts:
            return 0.0

        try:
            # Calculate different ROUGE metrics
            rouge1 = self.rouge_n(original_texts, optimized_texts, n=1)
            rouge2 = self.rouge_n(original_texts, optimized_texts, n=2)
            rougel = self.rouge_l(original_texts, optimized_texts)

            # Combine F1 scores with equal weighting
            f1_scores = [rouge1["f1"], rouge2["f1"], rougel["f1"]]
            combined_score = sum(f1_scores) / len(f1_scores)

            return float(max(0.0, min(1.0, combined_score)))

        except Exception as e:
            logging.error(f"Error calculating ROUGE similarity: {e}")
            return 0.0

    def detailed_scores(self, original_texts: List[str], optimized_texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Get detailed ROUGE scores for analysis.

        Args:
            original_texts: List of original text items
            optimized_texts: List of optimized text items

        Returns:
            Dictionary with detailed ROUGE scores
        """
        return {
            "rouge-1": self.rouge_n(original_texts, optimized_texts, n=1),
            "rouge-2": self.rouge_n(original_texts, optimized_texts, n=2),
            "rouge-l": self.rouge_l(original_texts, optimized_texts)
        }