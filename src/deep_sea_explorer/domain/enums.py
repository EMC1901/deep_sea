from enum import StrEnum


class CaptureType(StrEnum):
    BIO = "bio"
    ENV = "env"
    SUBSTRATE = "substrate"
    GEOMORPHOLOGY = "geomorphology"


class StreamEventType(StrEnum):
    CHUNK = "chunk"
    FINAL = "final"
    IMAGE = "image"
    ERROR = "error"
