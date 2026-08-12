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
from dataclasses import dataclass
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
    ssim_similarity: float = 1.0
    flow_magnitude: float = 0.0
    flow_coherence: float = 1.0
    phase_shift_x: float = 0.0
    phase_shift_y: float = 0.0
    phase_response: float = 0.0
    affine_angle_degrees: float = 0.0
    affine_response: float = 0.0
    affine_scale_x: float = 1.0
    affine_scale_y: float = 1.0
    affine_shear: float = 0.0
    motion_compensated: bool = False
    processing_backend: str = "cpu"


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    candidate_id: str
    session_id: str
    captured_at: float
    current_image_path: Path
    reference_image_path: Path | None
    yolo_changes: tuple[dict[str, Any], ...]
    scene_change_metrics: dict[str, float | bool | int | str]
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
    observed_elements: tuple[dict[str, Any], ...] = ()


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
                if len(box) != 4:
                    continue
                left, top, right, bottom = (float(value) for value in box)
                detections.append(
                    Detection(str(names[int(class_id)]), float(confidence), (left, top, right, bottom))
                )
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
    """Reject redundant frames using aligned low-resolution scene comparisons.

    CUDA acceleration is intentionally optional: the standard OpenCV wheels often lack
    CUDA support, so every operation has a deterministic CPU fallback.
    """

    def __init__(
        self,
        threshold: float = 0.22,
        confirm_frames: int = 3,
        *,
        analysis_size: tuple[int, int] = (160, 90),
        gpu_enabled: bool = True,
    ) -> None:
        self.threshold = threshold
        self.confirm_frames = max(1, confirm_frames)
        self.analysis_size = analysis_size
        self.gpu_enabled = gpu_enabled
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

        old, old_gray, backend = self._prepare_frame(old, cv2)
        new, new_gray, current_backend = self._prepare_frame(new, cv2)
        if current_backend == "cuda":
            backend = "cuda"
        aligned, aligned_gray, motion = self._align(current=new, current_gray=new_gray, reference_gray=old_gray, cv2=cv2, np=np)

        # Compare only the central image region so affine warping borders do not
        # look like a seabed change.
        old, aligned, old_gray, aligned_gray = self._crop_valid_region(
            old, aligned, old_gray, aligned_gray
        )
        old_hash = cv2.resize(old_gray, (8, 8)).astype(float)
        new_hash = cv2.resize(aligned_gray, (8, 8)).astype(float)
        phash = float(np.mean(np.abs((old_hash > old_hash.mean()) != (new_hash > new_hash.mean()))))
        old_hsv, new_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV), cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
        hsv = float(np.mean(np.abs(old_hsv.astype(float) - new_hsv.astype(float))) / 255.0)
        old_edge = cv2.Canny(old_gray, 80, 160)
        new_edge = cv2.Canny(aligned_gray, 80, 160)
        edge = float(np.mean(np.abs(old_edge.astype(float) - new_edge.astype(float))) / 255.0)
        grid_old = cv2.resize(old_gray, (64, 64))
        grid_new = cv2.resize(aligned_gray, (64, 64))
        grid = []
        for y in range(4):
            for x in range(4):
                a = grid_old[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
                b = grid_new[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
                grid.append(float(np.mean(np.abs(a.astype(float) - b.astype(float))) / 255.0))
        grid_distance = float(np.mean(grid))
        ssim = self._ssim(old_gray, aligned_gray, cv2, np)
        flow_magnitude, flow_coherence = self._flow_metrics(old_gray, aligned_gray, cv2, np)
        flow_score = min(flow_magnitude / 4.0, 1.0) * (1.0 - flow_coherence)
        score = (
            0.27 * phash
            + 0.18 * hsv
            + 0.15 * edge
            + 0.15 * grid_distance
            + 0.20 * (1.0 - ssim)
            + 0.05 * flow_score
        )
        if score >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        changed = self._streak >= self.confirm_frames
        return SceneMetrics(
            phash,
            hsv,
            edge,
            grid_distance,
            changed,
            self._streak,
            ssim,
            flow_magnitude,
            flow_coherence,
            motion["phase_shift_x"],
            motion["phase_shift_y"],
            motion["phase_response"],
            motion["affine_angle_degrees"],
            motion["affine_response"],
            motion["affine_scale_x"],
            motion["affine_scale_y"],
            motion["affine_shear"],
            bool(motion["motion_compensated"]),
            backend,
        )

    def _prepare_frame(self, image: Any, cv2: Any) -> tuple[Any, Any, str]:
        """Downsample before comparison, using CUDA when the installed OpenCV supports it."""
        if self.gpu_enabled:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    source = cv2.cuda_GpuMat()
                    source.upload(image)
                    reduced = cv2.cuda.resize(source, self.analysis_size, interpolation=cv2.INTER_AREA)
                    gray = cv2.cuda.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
                    return reduced.download(), gray.download(), "cuda"
            except (AttributeError, cv2.error):
                # A CUDA-capable driver does not guarantee that this OpenCV build
                # exposes every required operation.
                pass
        reduced = cv2.resize(image, self.analysis_size, interpolation=cv2.INTER_AREA)
        return reduced, cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY), "cpu"

    def _align(self, *, current: Any, current_gray: Any, reference_gray: Any, cv2: Any, np: Any) -> tuple[Any, Any, dict[str, float | bool]]:
        """Remove global translation then residual rotation/translation before scoring."""
        phase_shift_x = phase_shift_y = phase_response = 0.0
        affine_angle_degrees = affine_response = 0.0
        affine_scale_x = affine_scale_y = 1.0
        affine_shear = 0.0
        motion_compensated = False
        aligned_gray, aligned = current_gray, current
        try:
            (shift_x, shift_y), phase_response = cv2.phaseCorrelate(
                reference_gray.astype(np.float32), current_gray.astype(np.float32)
            )
            phase_shift_x, phase_shift_y = float(shift_x), float(shift_y)
            max_phase_x = self.analysis_size[0] * 0.15
            max_phase_y = self.analysis_size[1] * 0.15
            if phase_response >= 0.05 and abs(shift_x) <= max_phase_x and abs(shift_y) <= max_phase_y:
                phase_warp = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                aligned_gray = cv2.warpAffine(
                    current_gray, phase_warp, self.analysis_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                aligned = cv2.warpAffine(
                    current, phase_warp, self.analysis_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                motion_compensated = abs(shift_x) >= 0.25 or abs(shift_y) >= 0.25
        except cv2.error:
            pass
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
            affine_response, warp = cv2.findTransformECC(
                reference_gray.astype(np.float32) / 255.0,
                aligned_gray.astype(np.float32) / 255.0,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
            )
            linear = warp[:, :2]
            affine_scale_x = float(np.linalg.norm(linear[:, 0]))
            affine_scale_y = float(np.linalg.norm(linear[:, 1]))
            affine_shear = float(
                abs(np.dot(linear[:, 0], linear[:, 1]))
                / max(affine_scale_x * affine_scale_y, 1e-6)
            )
            affine_angle_degrees = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            max_residual_translation = min(self.analysis_size) * 0.12
            residual_translation = max(abs(float(warp[0, 2])), abs(float(warp[1, 2])))
            if (
                affine_response >= 0.85
                and abs(affine_angle_degrees) <= 8.0
                and residual_translation <= max_residual_translation
                and 0.92 <= affine_scale_x <= 1.08
                and 0.92 <= affine_scale_y <= 1.08
                and affine_shear <= 0.12
            ):
                aligned_gray = cv2.warpAffine(
                    aligned_gray,
                    warp,
                    self.analysis_size,
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aligned = cv2.warpAffine(
                    aligned,
                    warp,
                    self.analysis_size,
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                motion_compensated = motion_compensated or abs(affine_angle_degrees) >= 0.1
        except cv2.error:
            pass
        return aligned, aligned_gray, {
            "phase_shift_x": phase_shift_x,
            "phase_shift_y": phase_shift_y,
            "phase_response": float(phase_response),
            "affine_angle_degrees": affine_angle_degrees,
            "affine_response": float(affine_response),
            "affine_scale_x": affine_scale_x,
            "affine_scale_y": affine_scale_y,
            "affine_shear": affine_shear,
            "motion_compensated": motion_compensated,
        }

    @staticmethod
    def _crop_valid_region(old: Any, new: Any, old_gray: Any, new_gray: Any) -> tuple[Any, Any, Any, Any]:
        margin = max(4, min(old_gray.shape) // 16)
        crop = (slice(margin, -margin), slice(margin, -margin))
        return old[crop], new[crop], old_gray[crop], new_gray[crop]

    @staticmethod
    def _ssim(old_gray: Any, new_gray: Any, cv2: Any, np: Any) -> float:
        old = old_gray.astype(np.float32)
        new = new_gray.astype(np.float32)
        mu_old, mu_new = cv2.GaussianBlur(old, (7, 7), 1.5), cv2.GaussianBlur(new, (7, 7), 1.5)
        sigma_old = cv2.GaussianBlur(old * old, (7, 7), 1.5) - mu_old * mu_old
        sigma_new = cv2.GaussianBlur(new * new, (7, 7), 1.5) - mu_new * mu_new
        covariance = cv2.GaussianBlur(old * new, (7, 7), 1.5) - mu_old * mu_new
        c1, c2 = 6.5025, 58.5225
        numerator = (2 * mu_old * mu_new + c1) * (2 * covariance + c2)
        denominator = (mu_old * mu_old + mu_new * mu_new + c1) * (sigma_old + sigma_new + c2)
        return float(np.clip(np.mean(numerator / np.maximum(denominator, 1e-6)), 0.0, 1.0))

    @staticmethod
    def _flow_metrics(old_gray: Any, new_gray: Any, cv2: Any, np: Any) -> tuple[float, float]:
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        vectors = flow.reshape(-1, 2)
        magnitudes = np.linalg.norm(vectors, axis=1)
        median = np.median(vectors, axis=0)
        residual = np.linalg.norm(vectors - median, axis=1)
        mean_magnitude = float(np.mean(residual))
        coherence = float(np.clip(1.0 - np.mean(residual) / (np.mean(magnitudes) + 1e-6), 0.0, 1.0))
        return mean_magnitude, coherence

    def accept(self, path: Path) -> None:
        self._reference = path
        self._streak = 0


def visual_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def make_candidate(session_id: str, image_path: Path, reference: Path | None, tracks: list[TrackState], metrics: SceneMetrics, trigger_type: str) -> CandidateEvent:
    yolo_changes = tuple({"track_id": t.track_id, "category": t.category, "confidence": t.confidence, "bounding_box": t.bbox, "continuous_frames": t.continuous_frames} for t in tracks)
    signature_input = f"{trigger_type}|{','.join(str(t.track_id) for t in tracks)}|{visual_fingerprint(image_path)}"
    signature = hashlib.sha256(signature_input.encode()).hexdigest()[:32]
    return CandidateEvent(
        str(uuid.uuid4()),
        session_id,
        time.time(),
        image_path,
        reference,
        yolo_changes,
        {
            "phash_distance": metrics.phash_distance,
            "hsv_distance": metrics.hsv_distance,
            "edge_distance": metrics.edge_distance,
            "grid_distance": metrics.grid_distance,
            "ssim_similarity": metrics.ssim_similarity,
            "flow_magnitude": metrics.flow_magnitude,
            "flow_coherence": metrics.flow_coherence,
            "phase_shift_x": metrics.phase_shift_x,
            "phase_shift_y": metrics.phase_shift_y,
            "phase_response": metrics.phase_response,
            "affine_angle_degrees": metrics.affine_angle_degrees,
            "affine_response": metrics.affine_response,
            "affine_scale_x": metrics.affine_scale_x,
            "affine_scale_y": metrics.affine_scale_y,
            "affine_shear": metrics.affine_shear,
            "motion_compensated": metrics.motion_compensated,
            "processing_backend": metrics.processing_backend,
            "changed": metrics.changed,
            "confirmed_frames": metrics.confirmed_frames,
        },
        trigger_type,
        signature,
    )
