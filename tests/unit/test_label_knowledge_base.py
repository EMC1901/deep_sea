from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_sea_explorer.infrastructure.knowledge_base.label_knowledge_base import (
    CATEGORY_BIO,
    CATEGORY_GEOMORPHOLOGY,
    CATEGORY_SUBSTRATE,
    Annotation,
    LabelKnowledgeBase,
    PLACEHOLDERS,
    PromptTemplates,
    classify_label,
    clean_description,
    image_path_lookup,
    normalize_label,
    validate_description,
)
from deep_sea_explorer.infrastructure.knowledge_base.qwen_api import QwenApiError, QwenApiGenerator

PROMPT_FILE = Path(__file__).parents[2] / "src/deep_sea_explorer/resources/label_description_prompts.md"


def _write_image(path: Path, *, sharp: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.indices((96, 96)).sum(axis=0).astype("uint8") % 2 * 255 if sharp else np.full((96, 96), 127, dtype="uint8")
    assert cv2.imwrite(str(path), image)


def test_prompt_templates_keep_the_supplied_sections_verbatim() -> None:
    templates = PromptTemplates.from_file(PROMPT_FILE)
    for category, target in ((CATEGORY_BIO, "Biota > Sponges"), (CATEGORY_SUBSTRATE, "Substrate > Sand / mud"), (CATEGORY_GEOMORPHOLOGY, "No bedforms")):
        assert templates.render(category, target) == templates.templates[category].replace(PLACEHOLDERS[category], target)
        assert target in templates.render(category, target)


def test_normalization_and_three_category_classification() -> None:
    assert normalize_label("  Biota  >  Sponges ") == "Biota > Sponges"
    assert normalize_label("Biota > ") is None
    assert classify_label("Biota > Sponges") == CATEGORY_BIO
    assert classify_label("Substrate > Sand / mud") == CATEGORY_SUBSTRATE
    assert classify_label("No bedforms") == CATEGORY_GEOMORPHOLOGY
    assert classify_label("Anthropogenic > Tile") is None


def test_metadata_index_does_not_walk_the_image_tree(tmp_path: Path) -> None:
    metadata = tmp_path / "pool_meta.json"
    metadata.write_text(json.dumps([{"image": "site/a.jpg"}, {"image": "site/b.jpg"}, {"image": "other/b.png"}]), encoding="utf-8")
    lookup = image_path_lookup(metadata, tmp_path)
    assert lookup["a"] == "site/a.jpg"
    assert "b" not in lookup


def test_builder_samples_at_most_ten_and_selects_clear_low_label_count_representative(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "blur.jpg", sharp=False)
    _write_image(image_root / "many.jpg", sharp=True)
    _write_image(image_root / "single.jpg", sharp=True)
    annotations = [
        Annotation("blur.jpg", ("Biota > Sponges",), "a.json", 0),
        Annotation("many.jpg", ("Biota > Sponges", "Substrate > Sand / mud", "No bedforms"), "a.json", 1),
        Annotation("single.jpg", ("Biota > Sponges",), "a.json", 2),
        Annotation("many.jpg", ("Anthropogenic > Tile",), "a.json", 3),
    ]
    builder = LabelKnowledgeBase(tmp_path / "knowledge-base", PromptTemplates.from_file(PROMPT_FILE), blur_threshold=1.0)
    try:
        counts = builder.prepare(annotations, {"blur": "blur.jpg", "many": "many.jpg", "single": "single.jpg"}, image_root, sample_size=10)
        assert counts["selected"] == 3
        chosen = builder.db.execute("SELECT representative_image, annotation_label_count FROM labels WHERE canonical_label='Biota > Sponges'").fetchone()
        assert tuple(chosen) == ("single.jpg", 1)
        prompts_seen: list[str] = []
        result = builder.describe_pending(lambda image, prompt: (prompts_seen.append(prompt), "\u8be5\u5bf9\u8c61\u8868\u9762\u5448\u591a\u5b54\u72b6\uff0c\u5c40\u90e8\u53ef\u89c1\u4e0d\u89c4\u5219\u7eb9\u7406\u548c\u660e\u6697\u5dee\u5f02\u3002")[1], image_root)
        assert result == {"complete": 3}
        assert any("Biota > Sponges" in prompt for prompt in prompts_seen)
        builder.write_exports()
    finally:
        builder.close()
    universe = json.loads((tmp_path / "knowledge-base/label_universe.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "knowledge-base/build_report.json").read_text(encoding="utf-8"))
    assert len(universe["labels"][CATEGORY_BIO]) == 1
    assert universe["excluded_source_labels"][0]["canonical_label"] == "Anthropogenic > Tile"
    assert report["candidate_sample_size"] == "10"
    assert report["descriptions"]["complete"] == 3


def test_refresh_selection_supports_incremental_annotation_updates(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "bio.jpg", sharp=True)
    _write_image(image_root / "substrate.jpg", sharp=True)
    prompts = PromptTemplates.from_file(PROMPT_FILE)
    builder = LabelKnowledgeBase(tmp_path / "knowledge-base", prompts, blur_threshold=1.0)
    try:
        first = [Annotation("bio.jpg", ("Biota > Sponges",), "a.json", 0)]
        assert builder.prepare(first, {"bio": "bio.jpg"}, image_root)["bio"] == 1
        second = first + [Annotation("substrate.jpg", ("Substrate > Sand / mud",), "b.json", 0)]
        counts = builder.prepare(second, {"bio": "bio.jpg", "substrate": "substrate.jpg"}, image_root, refresh_selection=True)
        assert counts["bio"] == 1
        assert counts["substrate"] == 1
    finally:
        builder.close()


def test_failed_description_retries_without_rebuilding_selection(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "single.jpg", sharp=True)
    output = tmp_path / "knowledge-base"
    prompts = PromptTemplates.from_file(PROMPT_FILE)
    builder = LabelKnowledgeBase(output, prompts, blur_threshold=1.0)
    try:
        builder.prepare([Annotation("single.jpg", ("Biota > Sponges",), "a.json", 0)], {"single": "single.jpg"}, image_root)
        assert builder.describe_pending(lambda image, prompt: "\u58ee\u89c2\u3002", image_root) == {"failed": 1}
        failed = builder.db.execute("SELECT raw_response, last_error FROM labels WHERE canonical_label='Biota > Sponges'").fetchone()
        assert tuple(failed) == ("\u58ee\u89c2\u3002", "ValueError: description is empty or too short")
    finally:
        builder.close()
    resumed = LabelKnowledgeBase(output, prompts, blur_threshold=1.0)
    try:
        assert resumed.prepare([], {}, image_root)["bio"] == 1
        assert resumed.describe_pending(
            lambda image, prompt: "\u58ee\u89c2\u3002",
            image_root,
            retry_failed=True,
            retry_generator=lambda image, prompt: "\u5bf9\u8c61\u5448\u5b54\u72b6\u8868\u9762\uff0c\u7eb9\u7406\u4e0d\u89c4\u5219\uff0c\u5c40\u90e8\u660e\u6697\u5dee\u5f02\u6e05\u695a\u3002",
        ) == {"complete": 1}
    finally:
        resumed.close()


def test_recover_failed_description_from_cleanable_raw_response(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "single.jpg", sharp=True)
    builder = LabelKnowledgeBase(tmp_path / "knowledge-base", PromptTemplates.from_file(PROMPT_FILE), blur_threshold=1.0)
    try:
        builder.prepare([Annotation("single.jpg", ("Biota > Sponges",), "a.json", 0)], {"single": "single.jpg"}, image_root)
        builder.db.execute("UPDATE labels SET status='failed', raw_response=? WHERE canonical_label='Biota > Sponges'", ("**Visible granular surface with several irregular openings and clear tonal variation.**",))
        builder.db.commit()
        assert builder.recover_failed_from_raw() == 1
        status, description = builder.db.execute("SELECT status, description FROM labels WHERE canonical_label='Biota > Sponges'").fetchone()
        assert status == "complete"
        assert description == "Visible granular surface with several irregular openings and clear tonal variation."
    finally:
        builder.close()


def test_clean_description_removes_forbidden_sentences_and_markdown() -> None:
    raw = "**\u6574\u4f53\u5f62\u6001**\u5448\u5206\u679d\u72b6\u3002\u8be5\u7ed3\u6784\u590d\u6742\u800c\u9192\u76ee\u3002\u7531\u81ea\u7136\u5806\u79ef\u5f62\u6210\u3002\u8868\u9762\u53ef\u89c1\u5b54\u6d1e\u3002"
    cleaned = clean_description(raw, CATEGORY_GEOMORPHOLOGY)
    assert cleaned == "\u6574\u4f53\u5f62\u6001\u5448\u5206\u679d\u72b6\u3002\u8868\u9762\u53ef\u89c1\u5b54\u6d1e\u3002"
    assert "\u590d\u6742" not in cleaned
    assert "\u6210\u56e0" not in cleaned


def test_description_validation_rejects_prompt_prohibited_language() -> None:
    assert validate_description("\u8be5\u5bf9\u8c61\u5f62\u6001\u58ee\u89c2\uff0c\u7eb9\u7406\u6e05\u6670\u3002")[1] == "description contains a prohibited evaluative phrase"
    assert validate_description("\u7531\u81ea\u7136\u5806\u79ef\u5f62\u6210\u3002", CATEGORY_GEOMORPHOLOGY)[1] == "description contains a category-prohibited inference"


class _ApiResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _ApiSession:
    def __init__(self, response: _ApiResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, endpoint: str, **kwargs: object) -> _ApiResponse:
        self.calls.append({"endpoint": endpoint, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_api_generator_forwards_exact_prompt_and_uses_final_message_content(tmp_path: Path) -> None:
    image = tmp_path / "representative.jpg"
    image.write_bytes(b"jpeg-bytes")
    session = _ApiSession(_ApiResponse({"choices": [{"message": {"reasoning_content": "do not use", "content": "\u5bf9\u8c61\u8868\u9762\u53ef\u89c1\u591a\u4e2a\u5b54\u6d1e\uff0c\u8fb9\u7f18\u5f62\u6001\u4e0d\u89c4\u5219\uff0c\u5c40\u90e8\u7eb9\u7406\u53ef\u8fa8\u3002"}}]}))
    generator = QwenApiGenerator("https://example.test/api/v1/", "qwen-test", "secret-value", timeout=12.0, session=session)

    assert generator(image, "PROMPT-EXACT") == "\u5bf9\u8c61\u8868\u9762\u53ef\u89c1\u591a\u4e2a\u5b54\u6d1e\uff0c\u8fb9\u7f18\u5f62\u6001\u4e0d\u89c4\u5219\uff0c\u5c40\u90e8\u7eb9\u7406\u53ef\u8fa8\u3002"
    call = session.calls[0]
    assert call["endpoint"] == "https://example.test/api/v1/chat/completions"
    assert call["timeout"] == 12.0
    assert call["headers"] == {"Authorization": "Bearer secret-value", "Content-Type": "application/json"}
    payload = call["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen-test"
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "PROMPT-EXACT"}
    assert content[1]["image_url"]["url"].endswith("anBlZy1ieXRlcw==")


def test_api_generator_does_not_expose_secret_in_errors(tmp_path: Path) -> None:
    image = tmp_path / "representative.jpg"
    image.write_bytes(b"jpeg-bytes")
    generator = QwenApiGenerator("https://example.test", "qwen-test", "secret-value", session=_ApiSession(RuntimeError("network unavailable")))

    with pytest.raises(QwenApiError) as captured:
        generator(image, "prompt")

    assert "secret-value" not in str(captured.value)
    assert "network unavailable" not in str(captured.value)


def test_reset_descriptions_preserves_catalog_and_clears_old_responses(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "bio.jpg", sharp=True)
    _write_image(image_root / "bed.jpg", sharp=True)
    builder = LabelKnowledgeBase(tmp_path / "knowledge-base", PromptTemplates.from_file(PROMPT_FILE), blur_threshold=1.0)
    try:
        builder.prepare(
            [
                Annotation("bio.jpg", ("Biota > Sponges",), "a.json", 0),
                Annotation("bed.jpg", ("No bedforms",), "a.json", 1),
            ],
            {"bio": "bio.jpg", "bed": "bed.jpg"},
            image_root,
        )
        assert builder.describe_pending(lambda image, prompt: "\u5bf9\u8c61\u8868\u9762\u53ef\u89c1\u591a\u4e2a\u5b54\u6d1e\uff0c\u5c40\u90e8\u7eb9\u7406\u4e0d\u89c4\u5219\uff0c\u8fb9\u7f18\u8f6e\u5ed3\u6e05\u695a\u3002", image_root)["complete"] == 2
        assert builder.reset_descriptions("api:qwen3-vl-235b-a22b-thinking") == 2
        rows = builder.db.execute("SELECT canonical_label, category, description, raw_response, attempts, status FROM labels ORDER BY canonical_label").fetchall()
        assert [tuple(row) for row in rows] == [
            ("Biota > Sponges", CATEGORY_BIO, None, None, 0, "pending"),
            ("No bedforms", CATEGORY_GEOMORPHOLOGY, None, None, 0, "pending"),
        ]
        assert builder.catalog_counts() == {CATEGORY_BIO: 1, CATEGORY_SUBSTRATE: 0, CATEGORY_GEOMORPHOLOGY: 1, "unclassified": 0}
        assert builder._metadata("description_backend") == "api:qwen3-vl-235b-a22b-thinking"
    finally:
        builder.close()
