"""Errors raised by the in-process local model runtime."""

from __future__ import annotations

from deep_sea_explorer.domain.exceptions import ModelUnavailableError, ValidationError


class LocalModelError(ModelUnavailableError):
    """Base class for failures that the model API can later map to stable errors."""


class ModelNotConfigured(LocalModelError):
    pass


class ModelLoadFailure(LocalModelError):
    pass


class InvalidModelInput(ValidationError):
    pass


class InferenceTimeout(LocalModelError):
    pass


class InferenceQueueFull(LocalModelError):
    pass


class GpuOutOfMemory(LocalModelError):
    pass


class ModelOutputInvalid(LocalModelError):
    pass
