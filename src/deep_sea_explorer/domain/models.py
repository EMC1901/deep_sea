from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re

from .enums import CaptureType, StreamEventType


@dataclass(frozen=True, slots=True)
class CountItem:
    name: str
    count: int = 1


_COORDINATE_RE = re.compile(r"^[+-]\d{1,3}\.\d{1,8}$")


@dataclass(frozen=True, slots=True)
class MonitoringCoordinates:
    """Validated geographic coordinates read from a monitoring-image overlay."""

    longitude: Decimal
    latitude: Decimal

    @classmethod
    def from_text(cls, longitude: object, latitude: object) -> "MonitoringCoordinates":
        if not isinstance(longitude, str) or not isinstance(latitude, str):
            raise ValueError("monitoring coordinates must be strings")
        longitude, latitude = longitude.strip(), latitude.strip()
        if not _COORDINATE_RE.fullmatch(longitude) or not _COORDINATE_RE.fullmatch(latitude):
            raise ValueError("monitoring coordinates must be signed decimals with 1 to 8 decimal places")
        try:
            parsed_longitude = Decimal(longitude)
            parsed_latitude = Decimal(latitude)
        except InvalidOperation as error:
            raise ValueError("monitoring coordinates are invalid decimals") from error
        if not parsed_longitude.is_finite() or not parsed_latitude.is_finite():
            raise ValueError("monitoring coordinates must be finite")
        if not Decimal("-180") <= parsed_longitude <= Decimal("180"):
            raise ValueError("longitude is outside the geographic range")
        if not Decimal("-90") <= parsed_latitude <= Decimal("90"):
            raise ValueError("latitude is outside the geographic range")
        return cls(parsed_longitude, parsed_latitude)

    def as_payload(self) -> dict[str, str]:
        return {"LO": format(self.longitude, "f"), "LA": format(self.latitude, "f")}


@dataclass(frozen=True, slots=True)
class CaptureDecision:
    is_deepsea: bool
    is_typical: bool
    category: CaptureType
    description: str
    organisms: tuple[CountItem, ...] = ()
    env_features: tuple[CountItem, ...] = ()


@dataclass(frozen=True, slots=True)
class Capture:
    type: CaptureType
    image_data_uri: str
    description: str
    organisms: tuple[CountItem, ...] = ()
    env_features: tuple[CountItem, ...] = ()
    substrates: tuple[CountItem, ...] = ()
    geomorphologies: tuple[CountItem, ...] = ()


@dataclass(frozen=True, slots=True)
class Memo:
    timestamp: str
    content: str
    session_id: str
    capture: Capture | None = None
    captures: tuple[Capture, ...] = ()
    coordinates: MonitoringCoordinates | None = None
    statistics: dict[str, tuple[CountItem, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MonitoringAnalysis:
    """Single-image Qwen result emitted by the simplified monitoring pipeline."""

    description: str
    organisms: tuple[CountItem, ...] = ()
    env_features: tuple[CountItem, ...] = ()
    substrates: tuple[CountItem, ...] = ()
    geomorphologies: tuple[CountItem, ...] = ()
    coordinates: MonitoringCoordinates | None = None


@dataclass(frozen=True, slots=True)
class MonitoringTagMatch:
    """Structured output of the constrained first monitoring Qwen call."""

    organisms: tuple[CountItem, ...] = ()
    substrates: tuple[CountItem, ...] = ()
    geomorphologies: tuple[CountItem, ...] = ()
    unknown_categories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RagDocumentChunk:
    content: str
    doc_id: str
    chunk_id: int
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: StreamEventType
    text: str = ""
    content: str = ""
    prompt: str = ""


@dataclass(frozen=True, slots=True)
class ModelHealth:
    ready: bool
    detail: str = ""
    text_ready: bool | None = None
    vision_ready: bool | None = None


@dataclass(slots=True)
class SessionState:
    latest_video: str | None = None
    last_memo_embedding: tuple[float, ...] | None = None
    last_analyzed_video: str | None = None
    is_answering: bool = False
    cumulative_stats: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"bio": {}, "env": {}, "substrate": {}, "geomorphology": {}}
    )
    # Each completed queue-2 image compares against this per-session snapshot
    # only, then replaces it regardless of whether counts changed.
    previous_monitoring_coordinates: MonitoringCoordinates | None = None
    previous_monitoring_labels: dict[str, frozenset[str]] = field(
        default_factory=lambda: {"bio": frozenset(), "substrate": frozenset(), "geomorphology": frozenset()}
    )
    last_active_monotonic: float = 0.0
    # Event-driven monitoring state. Kept on the session so one process can resume
    # tracking immediately after a model task completes.
    active_tracks: dict[int, object] = field(default_factory=dict)
    last_scene_reference: str | None = None
    last_accepted_frame: str | None = None
    active_event_signature: str | None = None
    model_task_in_flight: bool = False
    pending_candidate: object | None = None
    last_model_call_time: float | None = None
