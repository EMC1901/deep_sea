from __future__ import annotations

import threading
from pathlib import Path

import pytest

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_local_container
from deep_sea_explorer.domain.models import ModelHealth
from deep_sea_explorer.infrastructure.models.local.adapters import QwenAdapter, _capture_decision
from deep_sea_explorer.infrastructure.models.local.errors import (
    GpuOutOfMemory,
    InferenceQueueFull,
    ModelOutputInvalid,
)
from deep_sea_explorer.infrastructure.models.local.gateways import (
    DisabledImageGateway,
    LocalEmbeddingGateway,
    LocalImageGateway,
    LocalVisionGateway,
)
from deep_sea_explorer.infrastructure.models.local.runtime import (
    InferenceCoordinator,
    LocalModelRuntime,
)


class RecordingAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.loads = 0
        self.unloads = 0
        self.ready = False

    def load(self) -> None:
        self.loads += 1
        self.ready = True

    def unload(self) -> None:
        self.unloads += 1
        self.ready = False

    def health(self) -> ModelHealth:
        return ModelHealth(self.ready, "ready" if self.ready else "not_loaded")


def test_video_frame_reader_duplicates_a_single_frame_for_qwen(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "single-frame.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 32))
    assert writer.isOpened()
    writer.write(np.full((32, 32, 3), 127, dtype=np.uint8))
    writer.release()

    frames = QwenAdapter._video_frames(video)

    assert frames.shape[0] == 2
    assert np.array_equal(frames[0], frames[1])


def test_runtime_reuses_current_model_and_unloads_before_switching() -> None:
    runtime = LocalModelRuntime(InferenceCoordinator())
    qwen, image = RecordingAdapter("qwen"), RecordingAdapter("image")

    assert runtime.invoke(qwen, lambda: "first") == "first"
    assert runtime.invoke(qwen, lambda: "second") == "second"
    assert runtime.invoke(image, lambda: "image") == "image"

    assert (qwen.loads, qwen.unloads) == (1, 1)
    assert (image.loads, image.unloads) == (1, 0)
    assert runtime.health(image) == ModelHealth(True, "ready")


def test_runtime_releases_gpu_slot_after_an_exception() -> None:
    coordinator = InferenceCoordinator()
    runtime = LocalModelRuntime(coordinator)
    adapter = RecordingAdapter("qwen")

    with pytest.raises(RuntimeError, match="boom"):
        runtime.invoke(adapter, lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert coordinator.active == 0
    assert runtime.invoke(adapter, lambda: "recovered") == "recovered"


def test_runtime_unloads_the_resident_model_after_gpu_oom() -> None:
    runtime = LocalModelRuntime(InferenceCoordinator())
    adapter = RecordingAdapter("qwen")

    with pytest.raises(GpuOutOfMemory):
        runtime.invoke(
            adapter,
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
        )

    assert (adapter.loads, adapter.unloads) == (1, 1)
    assert runtime.invoke(adapter, lambda: "recovered") == "recovered"
    assert adapter.loads == 2


def test_runtime_holds_its_gpu_slot_until_a_stream_is_consumed() -> None:
    coordinator = InferenceCoordinator(max_concurrent=1, max_queue=0)
    runtime = LocalModelRuntime(coordinator)
    adapter = RecordingAdapter("qwen")

    stream = runtime.stream(adapter, lambda: iter(("first", "second")))
    assert next(stream) == "first"
    assert coordinator.active == 1
    assert list(stream) == ["second"]
    assert coordinator.active == 0


def test_coordinator_rejects_work_when_queue_is_full() -> None:
    coordinator = InferenceCoordinator(max_concurrent=1, max_queue=0, queue_timeout_seconds=5)
    started, release = threading.Event(), threading.Event()
    worker = threading.Thread(
        target=lambda: coordinator.execute(lambda: (started.set(), release.wait(timeout=5))),
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=2)

    with pytest.raises(InferenceQueueFull):
        coordinator.execute(lambda: None)

    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert coordinator.active == 0


def test_frame_decision_requires_contract_shaped_json() -> None:
    decision = _capture_decision(
        '{"is_deepsea": true, "is_typical": false, "category": "bio", "description": "fish", '
        '"organisms": [{"name": "fish", "count": 2}], "env_features": []}'
    )
    assert decision.organisms[0].name == "fish"

    with pytest.raises(ModelOutputInvalid):
        _capture_decision("not json")


def test_frame_decision_normalizes_string_items_from_qwen() -> None:
    decision = _capture_decision(
        '{"is_deepsea": "true", "is_typical": 1, "category": "ENV", '
        '"description": "rocky seafloor", "organisms": [], '
        '"env_features": ["rock", {"name": "low light", "count": "2"}]}'
    )

    assert decision.is_deepsea is True
    assert decision.is_typical is True
    assert decision.env_features[0].name == "rock"
    assert decision.env_features[0].count == 1
    assert decision.env_features[1].name == "low light"
    assert decision.env_features[1].count == 2


def test_frame_decision_still_rejects_unusable_count_items() -> None:
    with pytest.raises(ModelOutputInvalid):
        _capture_decision(
            '{"is_deepsea": true, "is_typical": true, "category": "env", '
            '"description": "seafloor", "organisms": [], "env_features": [42]}'
        )


def test_report_summary_requests_chinese_and_removes_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = QwenAdapter("/models/qwen")
    captured: dict[str, object] = {}

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int,
    ) -> str:
        captured["content"] = content
        captured["max_new_tokens"] = max_new_tokens
        return "**任务总结**\n- 发现深海生物，并建议继续观测。"

    monkeypatch.setattr(adapter, "_generate", generate)

    summary = adapter.summarize_report({"memos": [{"text": "发现生物"}]})

    prompt = str(captured["content"])
    assert "必须使用简体中文" in prompt
    assert "不要标题、列表、编号、Markdown" in prompt
    assert captured["max_new_tokens"] == 512
    assert summary == "任务总结 发现深海生物，并建议继续观测。"


def test_video_description_keeps_chinese_with_an_original_proper_noun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = QwenAdapter("/models/qwen")
    calls: list[list[dict[str, object]]] = []
    monkeypatch.setattr(adapter, "_video_frames", lambda _: ["frame"])

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
    ) -> str:
        calls.append(content)
        assert max_new_tokens == 256
        return "画面中可见 Bathynomus giganteus，正在岩石底质附近缓慢移动。"

    monkeypatch.setattr(adapter, "_generate", generate)

    description = adapter.describe_video(Path("video.mp4"))

    assert description.startswith("画面中可见")
    assert "Bathynomus giganteus" in description
    assert len(calls) == 1
    assert "输出必须以中文为主" in str(calls[0])


def test_video_description_rewrites_an_english_first_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = QwenAdapter("/models/qwen")
    responses = iter(
        [
            "A deep-sea fish is swimming above a rocky seafloor.",
            "一条深海鱼正在岩石海床上方缓慢游动。",
        ]
    )
    prompts: list[str] = []
    monkeypatch.setattr(adapter, "_video_frames", lambda _: ["frame"])

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
    ) -> str:
        prompts.append(str(content))
        return next(responses)

    monkeypatch.setattr(adapter, "_generate", generate)

    assert adapter.describe_video(Path("video.mp4")) == "一条深海鱼正在岩石海床上方缓慢游动。"
    assert len(prompts) == 2
    assert "待改写内容" in prompts[1]


def test_video_description_rejects_output_that_remains_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = QwenAdapter("/models/qwen")
    monkeypatch.setattr(adapter, "_video_frames", lambda _: ["frame"])
    monkeypatch.setattr(
        adapter,
        "_generate",
        lambda *_, **__: "A deep-sea fish is swimming above rocks.",
    )

    with pytest.raises(ModelOutputInvalid, match="not Chinese"):
        adapter.describe_video(Path("video.mp4"))


def test_local_container_constructs_real_gateways_without_model_imports() -> None:
    settings = Settings(
        model_backend=ModelBackend.LOCAL,
        qwen_model_path="/models/qwen",
        image_model_path="/models/image",
        memo_embedding_model_path="/models/gte",
        rag_embedding_model_path="/models/minilm",
    )

    container = build_local_container(settings)

    assert isinstance(container.vision, LocalVisionGateway)
    assert isinstance(container.image, LocalImageGateway)
    assert isinstance(container.memo_embedding, LocalEmbeddingGateway)
    assert isinstance(container.rag_embedding, LocalEmbeddingGateway)


def test_local_container_can_disable_image_generation_without_an_image_model() -> None:
    settings = Settings(
        model_backend=ModelBackend.LOCAL,
        qwen_model_path="/models/qwen",
        image_generation_enabled=False,
        memo_embedding_model_path="/models/gte",
        rag_embedding_model_path="/models/minilm",
    )

    assert settings.validate_for_runtime() == []
    assert isinstance(build_local_container(settings).image, DisabledImageGateway)
