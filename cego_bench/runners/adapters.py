"""HTTP adapters for calling CEGO optimization endpoints."""

import json
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from .loaders import TestCase


class OptimizerType(Enum):
    """Supported optimizer types."""
    PATTERN = "pattern"
    ENTROPY = "entropy"


@dataclass
class OptimizationRequest:
    """Request payload for optimization endpoints."""
    query: str
    items: List[str]


@dataclass
class OptimizationResult:
    """Result from optimization endpoint."""
    optimizer: OptimizerType
    success: bool
    latency_ms: float
    token_reduction_percentage: Optional[float] = None
    semantic_retention: Optional[float] = None
    confidence: Optional[float] = None
    optimized_context: Optional[List[str]] = None
    kept_indices: Optional[List[int]] = None  # Server-provided indices when available
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class CEGOAdapter:
    """HTTP adapter for CEGO optimization endpoints."""

    def __init__(self,
                 pattern_url: str = "http://127.0.0.1:8003/optimize/pattern",
                 entropy_url: str = "http://127.0.0.1:8003/optimize/entropy",
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize CEGO adapter.

        Args:
            pattern_url: Pattern optimizer endpoint URL
            entropy_url: Entropy optimizer endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.endpoints = {
            OptimizerType.PATTERN: pattern_url,
            OptimizerType.ENTROPY: entropy_url
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def optimize(self, test_case: TestCase, optimizer: OptimizerType) -> OptimizationResult:
        """Run optimization on test case.

        Args:
            test_case: Test case to optimize
            optimizer: Which optimizer to use

        Returns:
            Optimization result with metrics and timing
        """
        # Prepare request
        request_payload = {
            "query": test_case.query,
            "context_pool": [item.text for item in test_case.items]
        }

        url = self.endpoints[optimizer]
        start_time = time.perf_counter()

        # Execute request with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=request_payload,
                    timeout=self.timeout
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 200:
                    try:
                        result_data = response.json()
                        return self._parse_success_response(
                            optimizer, latency_ms, result_data, response
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        return OptimizationResult(
                            optimizer=optimizer,
                            success=False,
                            latency_ms=latency_ms,
                            error_message=f"Invalid response format: {e}"
                        )
                else:
                    # Handle HTTP errors
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_detail = response.json().get('detail', '')
                        if error_detail:
                            error_msg += f": {error_detail}"
                    except:
                        pass

                    return OptimizationResult(
                        optimizer=optimizer,
                        success=False,
                        latency_ms=latency_ms,
                        error_message=error_msg
                    )

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                latency_ms = (time.perf_counter() - start_time) * 1000
                return OptimizationResult(
                    optimizer=optimizer,
                    success=False,
                    latency_ms=latency_ms,
                    error_message="Request timeout"
                )

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                latency_ms = (time.perf_counter() - start_time) * 1000
                return OptimizationResult(
                    optimizer=optimizer,
                    success=False,
                    latency_ms=latency_ms,
                    error_message="Connection error"
                )

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return OptimizationResult(
                    optimizer=optimizer,
                    success=False,
                    latency_ms=latency_ms,
                    error_message=f"Unexpected error: {str(e)}"
                )

    def _parse_success_response(self,
                              optimizer: OptimizerType,
                              latency_ms: float,
                              data: Dict[str, Any],
                              response: requests.Response) -> OptimizationResult:
        """Parse successful API response.

        Args:
            optimizer: Optimizer type
            latency_ms: Request latency
            data: Response JSON data
            response: Raw HTTP response

        Returns:
            Parsed optimization result
        """
        try:
            # Extract required fields
            optimized_context = data.get('optimized_context')
            if optimized_context is None:
                raise KeyError("Missing optimized_context")

            # Convert to list of strings if needed
            if isinstance(optimized_context, str):
                optimized_context = [optimized_context]
            elif not isinstance(optimized_context, list):
                raise ValueError("optimized_context must be list or string")

            # Extract token reduction percentage - handle both API formats
            token_reduction = data.get('token_reduction_percentage')
            if token_reduction is None:
                # Check stats.reduction.token_reduction_pct (pattern optimizer format)
                stats = data.get('stats', {})
                if stats and 'reduction' in stats:
                    reduction_info = stats['reduction']
                    token_reduction = reduction_info.get('token_reduction_pct')

            # Extract semantic retention (API provides this reliably)
            semantic_retention = data.get('semantic_retention')

            # Extract confidence (entropy optimizer provides this)
            confidence = data.get('confidence')

            # Note: kept_indices not provided by API, will be calculated from context mapping
            return OptimizationResult(
                optimizer=optimizer,
                success=True,
                latency_ms=latency_ms,
                token_reduction_percentage=token_reduction,
                semantic_retention=semantic_retention,
                confidence=confidence,  # Entropy optimizer provides this
                optimized_context=optimized_context,
                kept_indices=None,  # Will calculate from context mapping
                raw_response=data
            )

        except (KeyError, ValueError, TypeError) as e:
            return OptimizationResult(
                optimizer=optimizer,
                success=False,
                latency_ms=latency_ms,
                error_message=f"Invalid response structure: {e}",
                raw_response=data
            )

    def health_check(self) -> Dict[OptimizerType, bool]:
        """Check if endpoints are healthy.

        Returns:
            Dictionary mapping optimizer types to health status
        """
        health = {}

        for optimizer_type, url in self.endpoints.items():
            try:
                # Simple health check with minimal payload
                response = self.session.post(
                    url,
                    json={"query": "test", "context_pool": ["test item"]},
                    timeout=5.0
                )
                health[optimizer_type] = response.status_code in [200, 400, 422]  # Allow validation errors
            except Exception:
                health[optimizer_type] = False

        return health

    def close(self):
        """Close the session."""
        self.session.close()


def create_adapter_from_config(config: Dict[str, Any]) -> CEGOAdapter:
    """Create adapter from configuration dictionary.

    Args:
        config: Configuration with endpoint URLs and settings

    Returns:
        Configured CEGO adapter
    """
    endpoints = config.get('endpoints', {})

    return CEGOAdapter(
        pattern_url=endpoints.get('pattern', 'http://127.0.0.1:8003/optimize/pattern'),
        entropy_url=endpoints.get('entropy', 'http://127.0.0.1:8003/optimize/entropy'),
        timeout=config.get('timeout', 30.0),
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay', 1.0)
    )