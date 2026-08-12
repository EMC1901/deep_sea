from __future__ import annotations

from dataclasses import dataclass, field

from .enums import CaptureType, StreamEventType


@dataclass(frozen=True, slots=True)
class CountItem:
    name: str
    count: int = 1


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


@dataclass(frozen=True, slots=True)
class Memo:
    timestamp: str
    content: str
    session_id: str
    capture: Capture | None = None
    captures: tuple[Capture, ...] = ()


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


@dataclass(slots=True)
class SessionState:
    latest_video: str | None = None
    last_memo_embedding: tuple[float, ...] | None = None
    last_analyzed_video: str | None = None
    is_answering: bool = False
    cumulative_stats: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"bio": {}, "env": {}}
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
