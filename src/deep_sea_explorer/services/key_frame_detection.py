"""Event-driven key-frame detection primitives.

The implementation deliberately keeps model-specific code behind small protocols so the
server can run with a real Ultralytics model, a remote detector, or a deterministic test
detector.  OpenCV/NumPy are used for the scene signal and no text embeddings are involved.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Detection:
    category: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrackState:
    track_id: int
    category: str
    confidence: float
    bbox: tuple[float, float, float, float]
    first_seen_time: float
    last_seen_time: float
    continuous_frames: int


@dataclass(frozen=True, slots=True)
class SceneMetrics:
    phash_distance: float
    hsv_distance: float
    edge_distance: float
    grid_distance: float
    changed: bool
    confirmed_frames: int = 0


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    candidate_id: str
    session_id: str
    captured_at: float
    current_image_path: Path
    reference_image_path: Path | None
    yolo_changes: tuple[dict[str, Any], ...]
    scene_change_metrics: dict[str, float | bool | int]
    trigger_type: str
    signature: str


@dataclass(frozen=True, slots=True)
class SurveyEventEvaluation:
    survey_value: bool
    event_type: str
    scene_changed: bool
    new_elements: tuple[dict[str, Any], ...]
    description: str
    confidence: float


class ObjectDetector(Protocol):
    def detect(self, image_path: Path) -> list[Detection]: ...


class NullObjectDetector:
    """Safe fallback used when Ultralytics or weights are unavailable."""

    def detect(self, image_path: Path) -> list[Detection]:
        return []


class YoloObjectDetector:
    def __init__(self, model_path: str, confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO  # type: ignore[import-not-found]

                    self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_path: Path) -> list[Detection]:
        if not self.model_path or not image_path.is_file():
            return []
        try:
            result = self._load().predict(str(image_path), conf=self.confidence, verbose=False)[0]
            names = result.names
            detections: list[Detection] = []
            for box, confidence, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            ):
                detections.append(Detection(str(names[int(class_id)]), float(confidence), tuple(map(float, box))))
            return detections
        except Exception:
            return []


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class ByteTrackState:
    """Small IoU tracker with ByteTrack-compatible lifecycle semantics."""

    def __init__(self, iou_threshold: float = 0.3, confirm_frames: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.confirm_frames = max(1, confirm_frames)
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1

    @property
    def tracks(self) -> dict[int, TrackState]:
        return dict(self._tracks)

    def update(self, detections: list[Detection], now: float | None = None) -> tuple[list[TrackState], list[TrackState]]:
        now = time.time() if now is None else now
        unmatched = set(self._tracks)
        current: list[TrackState] = []
        new_tracks: list[TrackState] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            best_id, best_iou = None, self.iou_threshold
            for track_id in unmatched:
                track = self._tracks[track_id]
                if track.category == detection.category:
                    score = _iou(track.bbox, detection.bbox)
                    if score > best_iou:
                        best_id, best_iou = track_id, score
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                state = TrackState(best_id, detection.category, detection.confidence, detection.bbox, now, now, 1)
                new_tracks.append(state)
            else:
                old = self._tracks[best_id]
                state = TrackState(best_id, detection.category, detection.confidence, detection.bbox, old.first_seen_time, now, old.continuous_frames + 1)
                unmatched.remove(best_id)
            self._tracks[best_id] = state
            current.append(state)
        for track_id in unmatched:
            self._tracks.pop(track_id, None)
        return current, [track for track in new_tracks if track.continuous_frames >= self.confirm_frames]


class SceneChangeDetector:
    def __init__(self, threshold: float = 0.22, confirm_frames: int = 3) -> None:
        self.threshold = threshold
        self.confirm_frames = max(1, confirm_frames)
        self._reference: Path | None = None
        self._streak = 0

    @property
    def reference(self) -> Path | None:
        return self._reference

    def compare(self, current: Path, reference: Path | None = None) -> SceneMetrics:
        import cv2
        import numpy as np

        reference = reference or self._reference
        if reference is None or not reference.is_file():
            self._reference = current
            self._streak = 0
            return SceneMetrics(0.0, 0.0, 0.0, 0.0, False, 0)
        old = cv2.imread(str(reference))
        new = cv2.imread(str(current))
        if old is None or new is None:
            return SceneMetrics(0.0, 0.0, 0.0, 0.0, False, self._streak)
        old = cv2.resize(old, (64, 64))
        new = cv2.resize(new, (64, 64))
        old_gray, new_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        old_hash = cv2.resize(old_gray, (8, 8)).astype(float)
        new_hash = cv2.resize(new_gray, (8, 8)).astype(float)
        phash = float(np.mean(np.abs((old_hash > old_hash.mean()) != (new_hash > new_hash.mean()))))
        old_hsv, new_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV), cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
        hsv = float(np.mean(np.abs(old_hsv.astype(float) - new_hsv.astype(float))) / 255.0)
        old_edge = cv2.Canny(old_gray, 80, 160)
        new_edge = cv2.Canny(new_gray, 80, 160)
        edge = float(np.mean(np.abs(old_edge.astype(float) - new_edge.astype(float))) / 255.0)
        grid = []
        for y in range(4):
            for x in range(4):
                a, b = old_gray[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16], new_gray[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
                grid.append(float(np.mean(np.abs(a.astype(float) - b.astype(float))) / 255.0))
        grid_distance = float(np.mean(grid))
        score = 0.35 * phash + 0.25 * hsv + 0.2 * edge + 0.2 * grid_distance
        if score >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        changed = self._streak >= self.confirm_frames
        return SceneMetrics(phash, hsv, edge, grid_distance, changed, self._streak)

    def accept(self, path: Path) -> None:
        self._reference = path
        self._streak = 0


def visual_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def make_candidate(session_id: str, image_path: Path, reference: Path | None, tracks: list[TrackState], metrics: SceneMetrics, trigger_type: str) -> CandidateEvent:
    yolo_changes = tuple({"track_id": t.track_id, "category": t.category, "confidence": t.confidence, "bounding_box": t.bbox, "continuous_frames": t.continuous_frames} for t in tracks)
    signature_input = f"{trigger_type}|{','.join(str(t.track_id) for t in tracks)}|{visual_fingerprint(image_path)}"
    signature = hashlib.sha256(signature_input.encode()).hexdigest()[:32]
    return CandidateEvent(str(uuid.uuid4()), session_id, time.time(), image_path, reference, yolo_changes, {"phash_distance": metrics.phash_distance, "hsv_distance": metrics.hsv_distance, "edge_distance": metrics.edge_distance, "grid_distance": metrics.grid_distance, "changed": metrics.changed, "confirmed_frames": metrics.confirmed_frames}, trigger_type, signature)
