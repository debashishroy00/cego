"""High-resolution timing utilities for benchmarking."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager for high-resolution timing.

    Usage:
        with timer() as t:
            # do something
            pass
        print(f"Elapsed: {t['elapsed_ms']:.2f}ms")

    Yields:
        Dictionary that will contain elapsed time in milliseconds
    """
    timing_dict = {}
    start_time = time.perf_counter()

    try:
        yield timing_dict
    finally:
        end_time = time.perf_counter()
        timing_dict['elapsed_ms'] = (end_time - start_time) * 1000.0
        timing_dict['elapsed_seconds'] = end_time - start_time


class StopWatch:
    """Simple stopwatch for measuring elapsed time."""

    def __init__(self):
        """Initialize stopwatch."""
        self._start_time = None
        self._end_time = None

    def start(self) -> None:
        """Start the stopwatch."""
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> float:
        """Stop the stopwatch and return elapsed time in seconds.

        Returns:
            Elapsed time in seconds

        Raises:
            RuntimeError: If stopwatch was not started
        """
        if self._start_time is None:
            raise RuntimeError("Stopwatch not started")

        self._end_time = time.perf_counter()
        return self._end_time - self._start_time

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds

        Raises:
            RuntimeError: If stopwatch was not started or stopped
        """
        if self._start_time is None:
            raise RuntimeError("Stopwatch not started")

        if self._end_time is None:
            # Still running
            current_time = time.perf_counter()
            return (current_time - self._start_time) * 1000.0
        else:
            return (self._end_time - self._start_time) * 1000.0

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds

        Raises:
            RuntimeError: If stopwatch was not started
        """
        return self.elapsed_ms() / 1000.0

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._start_time = None
        self._end_time = None

    @property
    def is_running(self) -> bool:
        """Check if stopwatch is currently running."""
        return self._start_time is not None and self._end_time is None