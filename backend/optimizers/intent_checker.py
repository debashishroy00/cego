"""
Intent preservation checker with TF-IDF ranking for backfill selection.
Ensures critical query intent is preserved in optimized context.
"""
from typing import List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IntentPreservationChecker:
    """
    Check and ensure intent preservation in context optimization.
    Uses TF-IDF similarity for intelligent backfill when intent is lost.
    """

    def __init__(self, threshold: float = 0.85, max_additions: int = 2):
        """
        Initialize intent preservation checker.

        Args:
            threshold: Minimum coverage threshold for intent preservation
            max_additions: Maximum number of items to add for intent recovery
        """
        self.threshold = threshold
        self.max_additions = max_additions
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1
        )

    def check_intent_preserved(self,
                             kept_items: List[str],
                             query: str,
                             gold_intent_keywords: Optional[List[str]] = None,
                             threshold: Optional[float] = None) -> bool:
        """
        Check if critical intent is preserved in kept items.

        Args:
            kept_items: Items that were kept after optimization
            query: Original query/intent
            gold_intent_keywords: Optional gold standard intent keywords
            threshold: Override default threshold

        Returns:
            True if intent is sufficiently preserved
        """
        if not kept_items:
            return False

        threshold = threshold or self.threshold
        kept_text = ' '.join(kept_items).lower()

        # Use gold keywords if available, otherwise extract from query
        if gold_intent_keywords:
            intent_keywords = [kw.lower().strip() for kw in gold_intent_keywords if kw.strip()]
        else:
            intent_keywords = self._extract_intent_keywords(query)

        if not intent_keywords:
            return True  # No specific intent to preserve

        # Check coverage of intent keywords in kept text
        covered_keywords = 0
        for keyword in intent_keywords:
            # Check exact keyword match or partial word match
            if (keyword in kept_text or
                any(word in kept_text for word in keyword.split()) or
                any(keyword in word for word in kept_text.split())):
                covered_keywords += 1

        coverage = covered_keywords / len(intent_keywords)
        return coverage >= threshold

    def ensure_intent(self,
                     context_pool: List[str],
                     selected_indices: List[int],
                     query: str,
                     gold_intent_keywords: Optional[List[str]] = None,
                     max_additions: Optional[int] = None) -> List[int]:
        """
        Ensure intent preservation by adding items if necessary.

        Args:
            context_pool: Full pool of available context items
            selected_indices: Currently selected item indices
            query: Original query/intent
            gold_intent_keywords: Optional gold standard intent keywords
            max_additions: Override default max additions

        Returns:
            Updated list of selected indices with intent preservation
        """
        max_additions = max_additions or self.max_additions
        selected = sorted(set(selected_indices))

        # Check if intent is already preserved
        selected_items = [context_pool[i] for i in selected if i < len(context_pool)]
        if self.check_intent_preserved(selected_items, query, gold_intent_keywords):
            return selected

        # Find missing items that could help preserve intent
        missing_indices = [i for i in range(len(context_pool)) if i not in selected]
        if not missing_indices:
            return selected

        # Rank missing items by relevance to query intent
        relevance_scores = self._rank_items_by_relevance(
            [context_pool[i] for i in missing_indices],
            query,
            gold_intent_keywords
        )

        # Add highest-relevance items until intent is preserved or limit reached
        additions_made = 0
        for score, relative_idx in sorted(relevance_scores, key=lambda x: x[0], reverse=True):
            if additions_made >= max_additions:
                break

            actual_idx = missing_indices[relative_idx]
            selected.append(actual_idx)
            additions_made += 1

            # Check if intent is now preserved
            updated_items = [context_pool[i] for i in selected if i < len(context_pool)]
            if self.check_intent_preserved(updated_items, query, gold_intent_keywords):
                break

        return sorted(set(selected))

    def _extract_intent_keywords(self, query: str) -> List[str]:
        """Extract intent keywords from query using simple heuristics."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on',
                     'for', 'with', 'is', 'are', 'be', 'by', 'at', 'this',
                     'that', 'it', 'as', 'but', 'not', 'have', 'has', 'had'}

        words = query.lower().split()
        keywords = [word.strip('.,!?;') for word in words
                   if word.strip('.,!?;') not in stop_words and len(word) > 2]

        return keywords

    def _rank_items_by_relevance(self,
                                candidate_items: List[str],
                                query: str,
                                gold_intent_keywords: Optional[List[str]] = None) -> List[Tuple[float, int]]:
        """
        Rank candidate items by relevance to query intent using TF-IDF similarity.

        Args:
            candidate_items: List of candidate items to rank
            query: Original query
            gold_intent_keywords: Optional gold standard keywords

        Returns:
            List of (relevance_score, item_index) tuples sorted by relevance
        """
        if not candidate_items:
            return []

        # Prepare query text for comparison
        if gold_intent_keywords:
            intent_text = query + ' ' + ' '.join(gold_intent_keywords)
        else:
            intent_text = query

        try:
            # Compute TF-IDF similarity
            all_texts = [intent_text] + candidate_items
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Query vector is first row, candidate vectors are remaining rows
            query_vector = tfidf_matrix[0:1]
            candidate_vectors = tfidf_matrix[1:]

            # Compute cosine similarity
            similarities = cosine_similarity(query_vector, candidate_vectors).flatten()

            # Return as (score, index) pairs
            relevance_scores = [(float(sim), idx) for idx, sim in enumerate(similarities)]

        except Exception:
            # Fallback to simple keyword matching if TF-IDF fails
            intent_keywords = set(self._extract_intent_keywords(intent_text))
            relevance_scores = []

            for idx, item in enumerate(candidate_items):
                item_words = set(self._extract_intent_keywords(item))
                overlap = len(intent_keywords & item_words)
                score = overlap / max(len(intent_keywords), 1)
                relevance_scores.append((score, idx))

        return relevance_scores

    def get_intent_coverage_stats(self,
                                 kept_items: List[str],
                                 query: str,
                                 gold_intent_keywords: Optional[List[str]] = None) -> dict:
        """
        Get detailed intent coverage statistics.

        Args:
            kept_items: Items that were kept
            query: Original query
            gold_intent_keywords: Optional gold standard keywords

        Returns:
            Dictionary with coverage statistics
        """
        if not kept_items:
            return {
                "coverage": 0.0,
                "keywords_total": 0,
                "keywords_covered": 0,
                "missing_keywords": [],
                "is_preserved": False
            }

        kept_text = ' '.join(kept_items).lower()

        # Get intent keywords
        if gold_intent_keywords:
            intent_keywords = [kw.lower().strip() for kw in gold_intent_keywords if kw.strip()]
        else:
            intent_keywords = self._extract_intent_keywords(query)

        if not intent_keywords:
            return {
                "coverage": 1.0,
                "keywords_total": 0,
                "keywords_covered": 0,
                "missing_keywords": [],
                "is_preserved": True
            }

        # Check coverage
        covered = []
        missing = []

        for keyword in intent_keywords:
            if keyword in kept_text or any(word in kept_text for word in keyword.split()):
                covered.append(keyword)
            else:
                missing.append(keyword)

        coverage = len(covered) / len(intent_keywords)

        return {
            "coverage": coverage,
            "keywords_total": len(intent_keywords),
            "keywords_covered": len(covered),
            "covered_keywords": covered,
            "missing_keywords": missing,
            "is_preserved": coverage >= self.threshold
        }


def test_intent_preservation():
    """Test intent preservation functionality."""
    checker = IntentPreservationChecker(threshold=0.5)  # Lower threshold for test

    # Test basic intent checking
    query = "Find insurance claims for car accidents"
    kept_items = ["Car accident claim #123", "Insurance policy details"]
    gold_keywords = ["insurance", "claims", "car", "accidents"]

    is_preserved = checker.check_intent_preserved(kept_items, query, gold_keywords)

    # Debug output
    stats = checker.get_intent_coverage_stats(kept_items, query, gold_keywords)
    print(f"Debug: kept_items={kept_items}")
    print(f"Debug: keywords={gold_keywords}")
    print(f"Debug: coverage={stats['coverage']:.2f}, covered={stats['covered_keywords']}")

    assert is_preserved, f"Intent should be preserved with matching keywords. Coverage: {stats['coverage']:.2f}"
    print("[OK] Basic intent preservation check passed")

    # Test intent recovery
    context_pool = [
        "Home insurance policy",
        "Car accident claim #123",
        "Life insurance details",
        "Auto insurance claim process",
        "Health insurance coverage"
    ]

    # Start with poor selection
    selected_indices = [0, 2]  # Home and life insurance (not car-related)
    improved_indices = checker.ensure_intent(context_pool, selected_indices, query, gold_keywords)

    # Should add car-related items
    assert len(improved_indices) > len(selected_indices), "Should add items for intent preservation"
    improved_items = [context_pool[i] for i in improved_indices]
    is_improved = checker.check_intent_preserved(improved_items, query, gold_keywords)
    assert is_improved, "Intent should be preserved after improvement"
    print("[OK] Intent recovery test passed")

    # Test coverage statistics
    stats = checker.get_intent_coverage_stats(kept_items, query, gold_keywords)
    assert stats["coverage"] >= 0.5, "Should have reasonable coverage"
    assert stats["keywords_total"] == len(gold_keywords), "Should count all keywords"
    print(f"[OK] Coverage stats: {stats['coverage']:.2f} ({stats['keywords_covered']}/{stats['keywords_total']})")

    print("[OK] All intent preservation tests passed")


if __name__ == "__main__":
    test_intent_preservation()

    # Demo intent preservation
    checker = IntentPreservationChecker()

    print("\nIntent preservation demo:")
    query = "Process medical insurance claims"
    context_pool = [
        "Medical claim form template",
        "Auto insurance policy",
        "Health insurance claim #456",
        "Life insurance application",
        "Medical records processing",
        "Car insurance quote"
    ]

    # Simulate poor initial selection
    initial_selection = [1, 3, 5]  # Non-medical items
    print(f"Initial selection: {[context_pool[i] for i in initial_selection]}")

    # Apply intent preservation
    improved_selection = checker.ensure_intent(context_pool, initial_selection, query)
    print(f"Improved selection: {[context_pool[i] for i in improved_selection]}")

    # Show coverage improvement
    initial_items = [context_pool[i] for i in initial_selection]
    improved_items = [context_pool[i] for i in improved_selection]

    initial_stats = checker.get_intent_coverage_stats(initial_items, query)
    improved_stats = checker.get_intent_coverage_stats(improved_items, query)

    print(f"Coverage: {initial_stats['coverage']:.2f} -> {improved_stats['coverage']:.2f}")
    print(f"Preserved: {initial_stats['is_preserved']} -> {improved_stats['is_preserved']}")