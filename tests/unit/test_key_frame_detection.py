from __future__ import annotations

import time
from pathlib import Path

from deep_sea_explorer.services.candidate_queue import PerSessionEventQueue
from deep_sea_explorer.services.key_frame_detection import (
    ByteTrackState,
    CandidateEvent,
    Detection,
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
