"""Shared, lazy local-model runtime with bounded GPU admission."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, TypeVar

from deep_sea_explorer.domain.models import ModelHealth

from .errors import GpuOutOfMemory, InferenceQueueFull, InferenceTimeout, LocalModelError


T = TypeVar("T")


class ManagedAdapter(Protocol):
    name: str

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def health(self) -> ModelHealth: ...


class InferenceCoordinator:
    """Admits a bounded number of callers and serializes GPU work by default."""

    def __init__(self, max_concurrent: int = 1, max_queue: int = 4, queue_timeout_seconds: int = 180) -> None:
        if max_concurrent <= 0 or max_queue < 0 or queue_timeout_seconds <= 0:
            raise ValueError("local inference limits must be positive")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.queue_timeout_seconds = queue_timeout_seconds
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._waiting = 0
        self._active = 0

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    @contextmanager
    def slot(self):
        with self._lock:
            if self._active + self._waiting >= self.max_concurrent + self.max_queue:
                raise InferenceQueueFull("local inference queue is full")
            self._waiting += 1

        acquired = self._slots.acquire(timeout=self.queue_timeout_seconds)
        with self._lock:
            self._waiting -= 1
            if acquired:
                self._active += 1
        if not acquired:
            raise InferenceTimeout("timed out waiting for local GPU inference")

        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._slots.release()

    def execute(self, operation: Callable[[], T]) -> T:
        try:
            with self.slot():
                return operation()
        except LocalModelError:
            raise
        except MemoryError as error:
            raise GpuOutOfMemory("local model ran out of memory") from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise GpuOutOfMemory("local model ran out of GPU memory") from error
            raise


class LocalModelRuntime:
    """Keeps at most one model resident until capacity measurements justify more."""

    def __init__(self, coordinator: InferenceCoordinator) -> None:
        self.coordinator = coordinator
        self._resident: ManagedAdapter | None = None

    def invoke(self, adapter: ManagedAdapter, operation: Callable[[], T]) -> T:
        def run() -> T:
            self._activate(adapter)
            return operation()

        try:
            return self.coordinator.execute(run)
        except GpuOutOfMemory:
            if self._resident is adapter:
                adapter.unload()
                self._resident = None
            clear_cuda_cache()
            raise

    def stream(self, adapter: ManagedAdapter, operation: Callable[[], Iterator[T]]) -> Iterator[T]:
        def values() -> Iterator[T]:
            with self.coordinator.slot():
                self._activate(adapter)
                yield from operation()

        return values()

    def _activate(self, adapter: ManagedAdapter) -> None:
        if self._resident is adapter:
            return
        if self._resident is not None:
            self._resident.unload()
            clear_cuda_cache()
            self._resident = None
        adapter.load()
        self._resident = adapter

    @staticmethod
    def health(adapter: ManagedAdapter) -> ModelHealth:
        return adapter.health()


def clear_cuda_cache() -> None:
    """Best-effort cleanup; importing this module never imports GPU libraries."""
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
