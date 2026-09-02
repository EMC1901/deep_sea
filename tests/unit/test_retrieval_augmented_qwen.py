from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from deep_sea_explorer.infrastructure.models.local.adapters import QwenAdapter


def _image(path: Path, color: str) -> None:
    Image.new("RGB", (12, 12), color=color).save(path)


def test_qwen_event_input_keeps_two_event_images_and_adds_labelled_examples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reference = tmp_path / "reference.jpg"
    candidate = tmp_path / "candidate.jpg"
    example = tmp_path / "example.jpg"
    _image(reference, "black")
    _image(candidate, "white")
    _image(example, "blue")
    adapter = QwenAdapter("unused")
    captured: list[dict[str, object]] = []

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
        direct_response: bool = False,
    ) -> str:
        assert max_new_tokens == 384
        assert direct_response is True
        captured.extend(content)
        return json.dumps(
            {
                "survey_value": False,
                "event_type": "none",
                "scene_changed": False,
                "new_elements": [],
                "description": "该场景包含稳定的深海底质。",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(adapter, "_generate", generate)
    evaluation = adapter.evaluate_survey_event(
        reference,
        candidate,
        {
            "trigger_type": "scene",
            "retrieval_context": [
                {
                    "image_path": str(example),
                    "similarity": 0.91,
                    "labels": {"catami": ["Biota > Fish"]},
                    "survey_labels": {"organism": ["Biota > Fish"]},
                }
            ],
        },
    )

    assert evaluation.event_type == "none"
    assert [item["type"] for item in captured].count("image") == 3
    example_text = next(
        str(item["text"])
        for item in captured
        if item["type"] == "text" and "检索到的相似标注范例" in str(item["text"])
    )
    assert "不能直接复制范例标签" in example_text
    assert "organism: Fish" in example_text
    prompt = str(captured[-1]["text"])
    assert "image_path" not in prompt
    assert "retrieval_context" not in prompt
    assert "输出前必须逐项检查 new_elements" in prompt


def test_qwen_repairs_only_an_invalid_survey_event_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = tmp_path / "candidate.jpg"
    _image(candidate, "white")
    adapter = QwenAdapter("unused")
    calls: list[tuple[list[dict[str, object]], int]] = []
    responses = iter(
        (
            '{"survey_value": true, "event_type": "new_element" '
            '"new_elements": [{"category": "Biota", "name": "fish", "is_new": true}], '
            '"description": "fish", "confidence": 0.9}',
            json.dumps(
                {
                    "survey_value": True,
                    "event_type": "new_element",
                    "scene_changed": True,
                    "new_elements": [{"category": "organism", "name": "fish", "is_new": True}],
                    "description": "The scene contains a fish.",
                    "confidence": 0.9,
                }
            ),
        )
    )

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
        direct_response: bool = False,
    ) -> str:
        assert direct_response is True
        calls.append((content, max_new_tokens))
        return next(responses)

    monkeypatch.setattr(adapter, "_generate", generate)
    evaluation = adapter.evaluate_survey_event(None, candidate, {"trigger_type": "scene"})

    assert evaluation.survey_value is True
    assert evaluation.new_elements == ({"category": "organism", "name": "fish", "is_new": True},)
    assert [max_new_tokens for _, max_new_tokens in calls] == [384, 384]
    assert len(calls[1][0]) == 1
    assert calls[1][0][0]["type"] == "text"
    assert "syntax and schema repair only" in str(calls[1][0][0]["text"])


def test_qwen_rejects_a_twice_malformed_event_without_failing_the_candidate_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = tmp_path / "candidate.jpg"
    _image(candidate, "white")
    adapter = QwenAdapter("unused")
    calls: list[tuple[list[dict[str, object]], int, bool]] = []

    def generate(
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
        direct_response: bool = False,
    ) -> str:
        calls.append((content, max_new_tokens, direct_response))
        return "not a JSON object"

    monkeypatch.setattr(adapter, "_generate", generate)

    evaluation = adapter.evaluate_survey_event(None, candidate, {"trigger_type": "scene"})

    assert evaluation.survey_value is False
    assert evaluation.event_type == "none"
    assert evaluation.new_elements == ()
    assert [max_new_tokens for _, max_new_tokens, _ in calls] == [384, 384]
    assert all(direct_response for _, _, direct_response in calls)


def test_qwen_requests_direct_mode_for_structured_survey_output() -> None:
    captured: dict[str, object] = {}

    class Batch(dict[str, object]):
        def to(self, _: str) -> "Batch":
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["template_args"] = kwargs
            return "prompt"

        def __call__(self, **kwargs):
            captured["processor_args"] = kwargs
            return Batch()

    QwenAdapter._inputs([], Processor(), direct_response=True)

    assert captured["template_args"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_qwen_structured_input_keeps_legacy_processors_compatible() -> None:
    class Batch(dict[str, object]):
        def to(self, _: str) -> "Batch":
            return self

    class LegacyProcessor:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
            assert messages == [{"role": "user", "content": []}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt"

        def __call__(self, **_: object) -> Batch:
            return Batch()

    assert QwenAdapter._inputs([], LegacyProcessor(), direct_response=True) == {}
