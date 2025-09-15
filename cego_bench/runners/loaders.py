"""JSONL dataset loading and test case management for CEGO benchmark."""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import re


@dataclass
class TestItem:
    """Single item in a test context."""
    id: str
    text: str
    domain_hint: Optional[str] = None
    is_junk_gt: Optional[bool] = None


@dataclass
class TestCase:
    """A single benchmark test case."""
    id: str
    query: str
    items: List[TestItem]
    gold: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate test case after initialization."""
        if not self.items:
            raise ValueError(f"Test case {self.id} has no items")
        if len(self.items) > 100:
            raise ValueError(f"Test case {self.id} has too many items ({len(self.items)})")


def count_tokens(text: str) -> int:
    """Estimate token count using simple word-based heuristic.

    Args:
        text: Input text

    Returns:
        Estimated token count (words * 1.3 to approximate subword tokenization)
    """
    if not text or not text.strip():
        return 0

    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text)
    return max(1, int(len(words) * 1.3))


def hash_test_case(test_case: TestCase) -> str:
    """Generate deterministic hash for test case caching.

    Args:
        test_case: Test case to hash

    Returns:
        SHA256 hex digest
    """
    content = {
        "id": test_case.id,
        "query": test_case.query,
        "items": [
            {
                "id": item.id,
                "text": item.text,
                "domain_hint": item.domain_hint,
                "is_junk_gt": item.is_junk_gt
            }
            for item in test_case.items
        ]
    }

    content_str = json.dumps(content, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()


def load_jsonl_dataset(file_path: Path) -> List[TestCase]:
    """Load test cases from JSONL file.

    Args:
        file_path: Path to JSONL dataset file

    Returns:
        List of parsed test cases

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If JSONL format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    test_cases = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {file_path}: {e}")

            # Parse test case
            try:
                items = []
                for item_data in data.get('items', []):
                    item = TestItem(
                        id=item_data['id'],
                        text=item_data['text'],
                        domain_hint=item_data.get('domain_hint'),
                        is_junk_gt=item_data.get('is_junk_gt')
                    )
                    items.append(item)

                test_case = TestCase(
                    id=data['id'],
                    query=data['query'],
                    items=items,
                    gold=data.get('gold'),
                    notes=data.get('notes')
                )

                test_cases.append(test_case)

            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid test case format at line {line_num} in {file_path}: {e}")

    if not test_cases:
        raise ValueError(f"No valid test cases found in {file_path}")

    return test_cases


def validate_dataset(test_cases: List[TestCase]) -> Dict[str, Any]:
    """Validate dataset and return statistics.

    Args:
        test_cases: List of test cases to validate

    Returns:
        Dictionary with validation statistics and warnings
    """
    stats = {
        "total_cases": len(test_cases),
        "total_items": sum(len(tc.items) for tc in test_cases),
        "avg_items_per_case": sum(len(tc.items) for tc in test_cases) / len(test_cases),
        "cases_with_gold": sum(1 for tc in test_cases if tc.gold),
        "cases_with_junk_labels": sum(1 for tc in test_cases
                                    if any(item.is_junk_gt is not None for item in tc.items)),
        "warnings": []
    }

    # Check for duplicate IDs
    ids = [tc.id for tc in test_cases]
    if len(ids) != len(set(ids)):
        stats["warnings"].append("Duplicate test case IDs found")

    # Check for very small/large cases
    item_counts = [len(tc.items) for tc in test_cases]
    if min(item_counts) < 3:
        stats["warnings"].append("Some test cases have < 3 items")
    if max(item_counts) > 50:
        stats["warnings"].append("Some test cases have > 50 items")

    # Check token estimates
    total_tokens = 0
    for tc in test_cases:
        case_tokens = count_tokens(tc.query)
        for item in tc.items:
            case_tokens += count_tokens(item.text)
        total_tokens += case_tokens

    stats["total_estimated_tokens"] = total_tokens
    stats["avg_tokens_per_case"] = total_tokens / len(test_cases)

    return stats


def filter_test_cases(test_cases: List[TestCase],
                     max_items: Optional[int] = None,
                     require_gold: bool = False,
                     require_junk_labels: bool = False) -> List[TestCase]:
    """Filter test cases based on criteria.

    Args:
        test_cases: Input test cases
        max_items: Maximum items per test case
        require_gold: Only include cases with gold labels
        require_junk_labels: Only include cases with junk ground truth

    Returns:
        Filtered test cases
    """
    filtered = test_cases

    if max_items is not None:
        filtered = [tc for tc in filtered if len(tc.items) <= max_items]

    if require_gold:
        filtered = [tc for tc in filtered if tc.gold]

    if require_junk_labels:
        filtered = [tc for tc in filtered
                   if any(item.is_junk_gt is not None for item in tc.items)]

    return filtered