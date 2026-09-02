"""Errors raised by the project's independent image-retrieval implementation."""


class ImageRetrievalError(RuntimeError):
    """Base class for retrieval failures that callers may surface or degrade from."""


class ImageIndexFormatError(ImageRetrievalError):
    """The on-disk index does not satisfy the expected portable layout."""


class ImageEmbeddingError(ImageRetrievalError):
    """An image cannot be encoded by the configured frozen encoder."""
