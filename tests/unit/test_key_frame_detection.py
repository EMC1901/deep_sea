from __future__ import annotations

import time
from pathlib import Path

import pytest

from deep_sea_explorer.services.candidate_queue import PerSessionEventQueue
from deep_sea_explorer.services.key_frame_detection import (
    ByteTrackState,
    CandidateEvent,
    Detection,
    SceneChangeDetector,
    SurveyEventEvaluation,
)


def test_tracker_keeps_one_id_for_a_moving_target() -> None:
    tracker = ByteTrackState(confirm_frames=2)
    first, _ = tracker.update([Detection("fish", 0.9, (0, 0, 10, 10))], now=1)
    second, _ = tracker.update([Detection("fish", 0.9, (1, 0, 11, 10))], now=2)
    assert first[0].track_id == second[0].track_id == 1
    assert second[0].continuous_frames == 2


def test_event_queue_replaces_pending_candidate(tmp_path: Path) -> None:
    started = []
    release = False

    def evaluate(candidate: CandidateEvent) -> SurveyEventEvaluation:
        while not release:
            time.sleep(0.005)
        return SurveyEventEvaluation(True, "new_element", True, (), candidate.signature, 0.9)

    def complete(candidate: CandidateEvent, evaluation: SurveyEventEvaluation) -> None:
        started.append(candidate.signature)

    queue = PerSessionEventQueue(evaluate, complete)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    def candidate(signature: str) -> CandidateEvent:
        return CandidateEvent(signature, "s", 1, image, None, (), {}, "scene", signature)
    assert queue.submit(candidate("one")) == "started"
    assert queue.submit(candidate("two")) == "pending"
    release = True
    deadline = time.time() + 2
    while len(started) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert started == ["one", "two"]
    queue.shutdown()


def _seabed_frame(tmp_path: Path, name: str, *, shift: tuple[float, float] = (0, 0), angle: float = 0) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(7)
    image = rng.integers(20, 120, size=(180, 320, 3), dtype=np.uint8)
    cv2.circle(image, (80, 80), 26, (220, 180, 100), -1)
    cv2.rectangle(image, (190, 42), (275, 130), (50, 160, 200), -1)
    cv2.line(image, (10, 155), (305, 12), (240, 240, 240), 3)
    matrix = cv2.getRotationMatrix2D((160, 90), angle, 1.0)
    matrix[:, 2] += shift
    image = cv2.warpAffine(image, matrix, (320, 180), borderMode=cv2.BORDER_REPLICATE)
    path = tmp_path / name
    assert cv2.imwrite(str(path), image)
    return path


def test_scene_detector_aligns_translation_before_comparing(tmp_path: Path) -> None:
    reference = _seabed_frame(tmp_path, "reference.jpg")
    shifted = _seabed_frame(tmp_path, "shifted.jpg", shift=(9, -5))
    detector = SceneChangeDetector(threshold=0.12, confirm_frames=1, gpu_enabled=False)

    detector.compare(reference)
    metrics = detector.compare(shifted)

    assert metrics.processing_backend == "cpu"
    assert metrics.motion_compensated is True
    assert abs(metrics.phase_shift_x) > 1 or abs(metrics.phase_shift_y) > 1
    # JPEG resampling and replicated borders lower raw SSIM slightly even after
    # successful alignment; the combined score remains below the change threshold.
    assert metrics.ssim_similarity > 0.65
    assert metrics.changed is False


def test_scene_detector_aligns_small_rotation_and_reports_affine_metrics(tmp_path: Path) -> None:
    reference = _seabed_frame(tmp_path, "reference.jpg")
    rotated = _seabed_frame(tmp_path, "rotated.jpg", angle=3)
    detector = SceneChangeDetector(threshold=0.12, confirm_frames=1, gpu_enabled=False)

    detector.compare(reference)
    metrics = detector.compare(rotated)

    assert metrics.affine_response > 0.8
    assert abs(metrics.affine_angle_degrees) > 0.5
    assert 0.92 <= metrics.affine_scale_x <= 1.08
    assert 0.92 <= metrics.affine_scale_y <= 1.08
    assert metrics.affine_shear <= 0.12
    assert metrics.ssim_similarity > 0.75
    assert metrics.changed is False


def test_scene_detector_uses_ssim_and_residual_flow_for_a_local_change(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")

    reference = _seabed_frame(tmp_path, "reference.jpg")
    changed = cv2.imread(str(reference))
    assert changed is not None
    cv2.circle(changed, (150, 105), 55, (255, 255, 255), -1)
    current = tmp_path / "changed.jpg"
    assert cv2.imwrite(str(current), changed)
    detector = SceneChangeDetector(threshold=0.12, confirm_frames=1, gpu_enabled=False)

    detector.compare(reference)
    metrics = detector.compare(current)

    assert metrics.ssim_similarity < 0.85
    assert metrics.flow_magnitude > 0.05
    assert metrics.flow_coherence < 0.95
    assert metrics.changed is True
