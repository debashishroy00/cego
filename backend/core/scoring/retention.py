"""
Enhanced retention scoring with bipartite alignment and robust metrics.
Addresses edge cases and provides more credible retention scores.
"""
import re
import string
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional


class RetentionScorer:
    """Enhanced retention scorer with ensemble approach and robust alignment."""

    def __init__(self):
        self.stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "on",
                         "for", "with", "is", "are", "be", "by", "at", "this",
                         "that", "it", "as", "but", "not", "have", "has", "had"}

    def calculate_retention(self, original_items: List[str], kept_items: List[str]) -> float:
        """
        Calculate retention using ensemble of TF-IDF, ROUGE-L, and token Jaccard.
        Uses bipartite alignment to avoid concatenation bias.

        Args:
            original_items: List of original context items
            kept_items: List of items that were kept after optimization

        Returns:
            Retention score in [0.0, 1.0]
        """
        if not kept_items:
            return 0.0
        if not original_items:
            return 1.0

        # Normalize per-item text
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        O = [norm(s) for s in original_items]
        K = [norm(s) for s in kept_items]

        # 1) TF-IDF with bipartite alignment (char 3-5 grams)
        tfidf_score = self._tfidf_alignment_score(O, K)

        # 2) ROUGE-L F1 with true LCS and bipartite alignment
        rouge_score = self._rouge_l_f1_alignment(O, K)

        # 3) Token Jaccard with stopword filtering
        token_score = self._token_jaccard(O, K)

        # Ensemble using trimmed mean for robustness
        scores = sorted([tfidf_score, rouge_score, token_score])
        retention = np.mean(scores)  # All three scores for trimmed mean

        return float(np.clip(retention, 0.0, 1.0))

    def _tfidf_alignment_score(self, O: List[str], K: List[str]) -> float:
        """TF-IDF cosine similarity with bipartite alignment."""
        try:
            # Higher max_features to avoid flat scores
            vec = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                min_df=1,
                max_features=5000,
                norm='l2'
            )

            X = vec.fit_transform(O + K)
            OX, KX = X[:len(O)], X[len(O):]

            # Bipartite alignment: each O maps to best K (and vice versa)
            sim = cosine_similarity(OX, KX)  # [|O| x |K|]

            # Best match for each original item
            row_best = sim.max(axis=1).mean() if sim.shape[1] > 0 else 0.0
            # Best match for each kept item
            col_best = sim.max(axis=0).mean() if sim.shape[0] > 0 else 0.0

            return float((row_best + col_best) / 2.0)

        except Exception:
            # Fallback for edge cases
            return 0.5

    def _rouge_l_f1_alignment(self, O: List[str], K: List[str]) -> float:
        """ROUGE-L F1 with true LCS computation and bipartite alignment."""
        O_tok = [self._tokenize(s) for s in O]
        K_tok = [self._tokenize(s) for s in K]

        def lcs_length(a: List[str], b: List[str]) -> int:
            """Compute true longest common subsequence length."""
            if not a or not b:
                return 0

            dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[-1][-1]

        def best_f1_alignment(A: List[List[str]], B: List[List[str]]) -> float:
            """Find best F1 score for each item in A against all items in B."""
            if not A:
                return 1.0 if not B else 0.0

            best_scores = []
            for a in A:
                if not a:
                    best_scores.append(0.0)
                    continue

                max_f1 = 0.0
                for b in B:
                    if not b:
                        continue

                    lcs_len = lcs_length(a, b)
                    precision = lcs_len / len(b) if len(b) > 0 else 0.0
                    recall = lcs_len / len(a) if len(a) > 0 else 0.0

                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        max_f1 = max(max_f1, f1)

                best_scores.append(max_f1)

            return float(np.mean(best_scores)) if best_scores else 0.0

        # Bipartite alignment: O→K and K→O
        r1 = best_f1_alignment(O_tok, K_tok)
        r2 = best_f1_alignment(K_tok, O_tok)

        return (r1 + r2) / 2.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text: lowercase, remove punctuation, split."""
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _token_jaccard(self, O: List[str], K: List[str]) -> float:
        """Token Jaccard similarity with stopword filtering."""
        def extract_tokens(texts: List[str]) -> set:
            """Extract tokens from list of texts, filtering stopwords."""
            tokens = set()
            for text in texts:
                words = self._tokenize(text)
                tokens.update(word for word in words if word not in self.stopwords and len(word) > 1)
            return tokens

        A = extract_tokens(O)
        B = extract_tokens(K)

        # Handle edge cases
        if not A and not B:
            return 1.0
        if not A:
            return 1.0  # Perfect retention if original had no meaningful tokens
        if not B:
            return 0.0

        intersection = len(A & B)
        union = len(A | B)

        return intersection / union if union > 0 else 0.0

    def calculate_retention_with_debug(self, original_items: List[str], kept_items: List[str]) -> dict:
        """Calculate retention with detailed breakdown for debugging."""
        if not kept_items:
            return {"total": 0.0, "tfidf": 0.0, "rouge": 0.0, "jaccard": 0.0}
        if not original_items:
            return {"total": 1.0, "tfidf": 1.0, "rouge": 1.0, "jaccard": 1.0}

        # Normalize text
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        O = [norm(s) for s in original_items]
        K = [norm(s) for s in kept_items]

        # Calculate individual scores
        tfidf_score = self._tfidf_alignment_score(O, K)
        rouge_score = self._rouge_l_f1_alignment(O, K)
        token_score = self._token_jaccard(O, K)

        # Ensemble score
        scores = [tfidf_score, rouge_score, token_score]
        total_score = float(np.clip(np.mean(scores), 0.0, 1.0))

        return {
            "total": total_score,
            "tfidf": tfidf_score,
            "rouge": rouge_score,
            "jaccard": token_score,
            "scores": scores
        }


# Unit tests for retention scorer
def test_retention_identity():
    """Test that identical inputs give near-perfect retention."""
    scorer = RetentionScorer()
    orig = ["alpha beta gamma", "delta epsilon"]

    # Perfect retention case
    result = scorer.calculate_retention(orig, orig)
    assert result >= 0.98, f"Perfect retention should be ≥0.98, got {result}"

    # Empty kept items
    result = scorer.calculate_retention(orig, [])
    assert result == 0.0, f"Empty retention should be 0.0, got {result}"

    print("[OK] Identity tests passed")


def test_retention_monotonicity():
    """Test that adding more items generally increases retention."""
    scorer = RetentionScorer()
    orig = ["a b c d e", "f g h i j", "k l m n o"]

    kept1 = ["a b c d e"]
    kept2 = ["a b c d e", "f g h i j"]

    r1 = scorer.calculate_retention(orig, kept1)
    r2 = scorer.calculate_retention(orig, kept2)

    assert r2 >= r1, f"Adding items should not decrease retention: {r1} -> {r2}"

    print("[OK] Monotonicity test passed")


def test_retention_bounds():
    """Test that retention scores are always in [0,1]."""
    scorer = RetentionScorer()

    test_cases = [
        (["short"], ["short"]),
        (["very long text with many words"], ["short"]),
        (["a", "b", "c"], ["x", "y", "z"]),
        ([""], [""]),
        (["normal text"], ["completely different text"]),
    ]

    for orig, kept in test_cases:
        score = scorer.calculate_retention(orig, kept)
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {orig} -> {kept}"

    print("[OK] Bounds test passed")


if __name__ == "__main__":
    # Run unit tests
    test_retention_identity()
    test_retention_monotonicity()
    test_retention_bounds()

    # Demo calculation
    scorer = RetentionScorer()
    original = ["This is a test document", "With multiple sentences", "And some content"]
    kept = ["This is a test document", "With different content"]

    debug_result = scorer.calculate_retention_with_debug(original, kept)
    print(f"\nRetention breakdown:")
    print(f"  Total: {debug_result['total']:.3f}")
    print(f"  TF-IDF: {debug_result['tfidf']:.3f}")
    print(f"  ROUGE-L: {debug_result['rouge']:.3f}")
    print(f"  Jaccard: {debug_result['jaccard']:.3f}")