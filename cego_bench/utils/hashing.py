"""Hashing utilities for caching and deduplication."""

import hashlib
import json
from typing import Any, Dict, List
from pathlib import Path


def hash_dict(data: Dict[str, Any]) -> str:
    """Create deterministic hash of dictionary.

    Args:
        data: Dictionary to hash

    Returns:
        SHA256 hex digest
    """
    # Sort keys for deterministic ordering
    content = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def hash_string(text: str) -> str:
    """Create hash of string.

    Args:
        text: String to hash

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_file(file_path: Path) -> str:
    """Create hash of file contents.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hex digest

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for memory efficiency
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_test_inputs(query: str, items: List[str], config_hash: str = "") -> str:
    """Create hash for test inputs to enable caching.

    Args:
        query: Test query
        items: List of test items
        config_hash: Optional configuration hash

    Returns:
        SHA256 hex digest combining all inputs
    """
    input_data = {
        "query": query,
        "items": items,
        "config_hash": config_hash
    }
    return hash_dict(input_data)


def short_hash(full_hash: str, length: int = 8) -> str:
    """Truncate hash to shorter version for display.

    Args:
        full_hash: Full hash string
        length: Desired length of shortened hash

    Returns:
        Truncated hash string
    """
    return full_hash[:length]


def create_cache_key(*components: Any) -> str:
    """Create cache key from multiple components.

    Args:
        components: Components to hash together

    Returns:
        SHA256 hex digest of all components
    """
    combined = json.dumps(components, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def verify_file_integrity(file_path: Path, expected_hash: str) -> bool:
    """Verify file integrity against expected hash.

    Args:
        file_path: Path to file
        expected_hash: Expected SHA256 hash

    Returns:
        True if file hash matches expected hash

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    actual_hash = hash_file(file_path)
    return actual_hash == expected_hash