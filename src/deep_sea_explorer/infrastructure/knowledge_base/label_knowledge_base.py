"""Build a compact, resumable three-category label knowledge base offline."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORY_BIO = "bio"
CATEGORY_SUBSTRATE = "substrate"
CATEGORY_GEOMORPHOLOGY = "geomorphology"
INCLUDED_CATEGORIES = (CATEGORY_BIO, CATEGORY_SUBSTRATE, CATEGORY_GEOMORPHOLOGY)
SCHEMA_VERSION = 2

PLACEHOLDERS = {
    CATEGORY_BIO: "{{\u751f\u7269\u540d\u79f0}}",
    CATEGORY_SUBSTRATE: "{{\u5e95\u8d28\u540d\u79f0}}",
    CATEGORY_GEOMORPHOLOGY: "{{\u5730\u8c8c\u540d\u79f0}}",
}
EVALUATIVE_PHRASES = (
    "\u7f8e\u4e3d", "\u58ee\u89c2", "\u4e30\u5bcc", "\u9192\u76ee", "\u590d\u6742", "\u5c42\u6b21\u4e30\u5bcc",
    "\u5c42\u6b21\u660e\u663e", "\u5c42\u6b21\u5206\u660e", "\u9669\u5cfb", "\u955c\u5934\u4e2d\u53ef\u4ee5\u770b\u5230", "\u4ece\u753b\u9762\u6765\u770b",
)

CATEGORY_INFERENCE_PHRASES = {
    CATEGORY_BIO: ("\u5e74\u9f84", "\u6027\u522b", "\u884c\u4e3a", "\u98df\u6027", "\u751f\u6001\u529f\u80fd", "\u5065\u5eb7\u72b6\u6001", "\u771f\u5b9e\u4f53\u8272"),
    CATEGORY_SUBSTRATE: ("\u77ff\u7269\u7ec4\u6210", "\u5f62\u6210\u65f6\u4ee3", "\u6c89\u79ef\u901f\u7387", "\u5730\u8d28\u5e74\u4ee3", "\u5f62\u6210\u673a\u5236", "\u6210\u56e0"),
    CATEGORY_GEOMORPHOLOGY: ("\u5f62\u6210\u65f6\u4ee3", "\u6784\u9020\u80cc\u666f", "\u6c89\u79ef\u8fc7\u7a0b", "\u4fb5\u8680\u673a\u5236", "\u706b\u5c71\u6d3b\u52a8", "\u5f62\u6210\u673a\u5236", "\u6784\u9020\u6d3b\u52a8", "\u6784\u9020\u6027", "\u98ce\u8680", "\u6210\u56e0", "\u81ea\u7136\u5806\u79ef\u5f62\u6210"),
}


@dataclass(frozen=True, slots=True)
class Annotation:
    image_name: str
    labels: tuple[str, ...]
    source_file: str
    source_index: int


@dataclass(frozen=True, slots=True)
class Candidate:
    image_name: str
    relative_path: str
    source_file: str
    source_index: int
    annotation_label_count: int
    sharpness: float | None = None


@dataclass(slots=True)
class SampleBucket:
    random: random.Random
    seen: int = 0
    candidates: list[Candidate] | None = None

    def add(self, candidate: Candidate, limit: int) -> None:
        if self.candidates is None:
            self.candidates = []
        self.seen += 1
        if len(self.candidates) < limit:
            self.candidates.append(candidate)
            return
        index = self.random.randrange(self.seen)
        if index < limit:
            self.candidates[index] = candidate


class PromptTemplates:
    """The prompt resource is used verbatim except for its one target placeholder."""

    def __init__(self, source_path: Path, templates: dict[str, str]) -> None:
        self.source_path = source_path
        self.templates = templates
        self.sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    @classmethod
    def from_file(cls, source_path: Path) -> "PromptTemplates":
        text = source_path.read_text(encoding="utf-8")
        starts = list(re.finditer(r"(?m)^## [^\r\n]+", text))
        templates: dict[str, str] = {}
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            section = text[match.start():end]
            for category, placeholder in PLACEHOLDERS.items():
                if placeholder in section:
                    if category in templates:
                        raise ValueError(f"duplicate prompt section for {category}")
                    templates[category] = section
        missing = [category for category in INCLUDED_CATEGORIES if category not in templates]
        if missing:
            raise ValueError(f"prompt file misses category sections: {', '.join(missing)}")
        for category, template in templates.items():
            if template.count(PLACEHOLDERS[category]) != 1:
                raise ValueError(f"prompt section for {category} must have one target placeholder")
        return cls(source_path, templates)

    def render(self, category: str, label: str) -> str:
        return self.templates[category].replace(PLACEHOLDERS[category], label)


def normalize_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return None
    parts = [" ".join(part.split()) for part in text.split(">")]
    return " > ".join(parts) if all(parts) else None


def classify_label(label: str) -> str | None:
    root = label.split(" > ", 1)[0].casefold()
    if root == "biota":
        return CATEGORY_BIO
    if root == "substrate":
        return CATEGORY_SUBSTRATE
    if root in {"bedforms", "relief", "no bedforms"}:
        return CATEGORY_GEOMORPHOLOGY
    return None


def iter_annotations(annotation_root: Path) -> Iterator[Annotation]:
    for path in sorted(annotation_root.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            image, raw_labels = item.get("image"), item.get("labels")
            if not isinstance(image, str) or not isinstance(raw_labels, list):
                continue
            labels = tuple(sorted({label for value in raw_labels if (label := normalize_label(value))}))
            if labels:
                yield Annotation(Path(image).name, labels, path.name, index)


def image_path_lookup(metadata_path: Path, image_root: Path) -> dict[str, str]:
    """Read the existing retrieval metadata index rather than walking the image tree."""
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("image metadata index must be a JSON array")
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            continue
        relative = item["image"]
        key = Path(relative).stem.casefold()
        if key in result and result[key] != relative:
            ambiguous.add(key)
        else:
            result[key] = relative
    for key in ambiguous:
        result.pop(key, None)
    return result


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class LabelKnowledgeBase:
    """Final database only: labels, descriptions, representative image, and provenance."""

    def __init__(self, output_dir: Path, prompt_templates: PromptTemplates, *, blur_threshold: float = 35.0) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_dir / "knowledge_base.sqlite"
        self.prompts = prompt_templates
        self.blur_threshold = blur_threshold
        self.db = _connect(self.db_path)
        self._initialize()

    def close(self) -> None:
        self.db.close()

    def _initialize(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS labels (
                canonical_label TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT,
                representative_image TEXT,
                source_annotation TEXT,
                annotation_label_count INTEGER,
                sharpness REAL,
                status TEXT NOT NULL DEFAULT 'pending',
                raw_response TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS unclassified_labels (
                canonical_label TEXT PRIMARY KEY,
                occurrences INTEGER NOT NULL,
                example_source TEXT NOT NULL
            );
            """
        )
        self._set_metadata("schema_version", str(SCHEMA_VERSION))
        self._set_metadata("prompt_sha256", self.prompts.sha256)
        self.db.commit()

    def prepare(
        self,
        annotations: Iterable[Annotation],
        image_paths: dict[str, str],
        image_root: Path,
        *,
        sample_size: int = 10,
        random_seed: int = 20260814,
        refresh_selection: bool = False,
    ) -> dict[str, int]:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if self._has_metadata("selection_complete") and not refresh_selection:
            return self.catalog_counts()
        buckets: dict[str, SampleBucket] = {}
        occurrences: Counter[str] = Counter()
        unclassified: dict[str, tuple[int, str]] = {}
        for annotation in annotations:
            label_count = len(annotation.labels)
            image_key = Path(annotation.image_name).stem.casefold()
            relative_path = image_paths.get(image_key)
            for label in annotation.labels:
                category = classify_label(label)
                if category is None:
                    count, _ = unclassified.get(label, (0, f"{annotation.source_file}:{annotation.source_index}"))
                    unclassified[label] = (count + 1, f"{annotation.source_file}:{annotation.source_index}")
                    continue
                occurrences[label] += 1
                if relative_path is None:
                    continue
                bucket = buckets.get(label)
                if bucket is None:
                    seed_material = f"{random_seed}:{label}".encode("utf-8")
                    bucket = SampleBucket(random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")))
                    buckets[label] = bucket
                bucket.add(Candidate(annotation.image_name, relative_path, annotation.source_file, annotation.source_index, label_count), sample_size)
        selected = 0
        unavailable = 0
        for label in sorted(occurrences):
            category = classify_label(label)
            representative = self._best_representative(buckets.get(label), image_root)
            if representative is None:
                unavailable += 1
                self.db.execute(
                    """INSERT INTO labels(canonical_label, category, status, updated_at)
                       VALUES (?, ?, 'no_representative', ?)
                       ON CONFLICT(canonical_label) DO UPDATE SET category=excluded.category, status='no_representative', updated_at=excluded.updated_at""",
                    (label, category, _now()),
                )
                continue
            selected += 1
            status = self.db.execute("SELECT status, representative_image FROM labels WHERE canonical_label=?", (label,)).fetchone()
            preserve = status is not None and status["status"] == "complete" and status["representative_image"] == representative.relative_path
            self.db.execute(
                """INSERT INTO labels(canonical_label, category, representative_image, source_annotation, annotation_label_count, sharpness, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(canonical_label) DO UPDATE SET category=excluded.category, representative_image=excluded.representative_image,
                   source_annotation=excluded.source_annotation, annotation_label_count=excluded.annotation_label_count, sharpness=excluded.sharpness,
                   status=CASE WHEN labels.status='complete' AND labels.representative_image=excluded.representative_image THEN 'complete' ELSE 'pending' END,
                   description=CASE WHEN labels.status='complete' AND labels.representative_image=excluded.representative_image THEN labels.description ELSE NULL END,
                   raw_response=CASE WHEN labels.status='complete' AND labels.representative_image=excluded.representative_image THEN labels.raw_response ELSE NULL END,
                   attempts=CASE WHEN labels.status='complete' AND labels.representative_image=excluded.representative_image THEN labels.attempts ELSE 0 END,
                   last_error=NULL, updated_at=excluded.updated_at""",
                (label, category, representative.relative_path, f"{representative.source_file}:{representative.source_index}", representative.annotation_label_count, representative.sharpness, "complete" if preserve else "pending", _now()),
            )
        self.db.executemany(
            """INSERT INTO unclassified_labels(canonical_label, occurrences, example_source) VALUES (?, ?, ?)
               ON CONFLICT(canonical_label) DO UPDATE SET occurrences=excluded.occurrences, example_source=excluded.example_source""",
            [(label, count, source) for label, (count, source) in unclassified.items()],
        )
        self._set_metadata("selection_complete", _now())
        self._set_metadata("candidate_sample_size", str(sample_size))
        self._set_metadata("random_seed", str(random_seed))
        self._set_metadata("image_metadata_index", str(len(image_paths)))
        self.db.commit()
        self.write_exports()
        return {**self.catalog_counts(), "selected": selected, "no_representative": unavailable}

    def _best_representative(self, bucket: SampleBucket | None, image_root: Path) -> Candidate | None:
        if bucket is None or not bucket.candidates:
            return None
        try:
            import cv2
            import numpy as np
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("OpenCV and NumPy are required for representative-image selection") from error
        ranked: list[tuple[int, float, str, Candidate]] = []
        for candidate in bucket.candidates:
            path = image_root / candidate.relative_path
            try:
                data = np.fromfile(path, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    continue
                sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            except Exception:
                continue
            if sharpness >= self.blur_threshold:
                ranked.append((candidate.annotation_label_count, -sharpness, candidate.relative_path, Candidate(candidate.image_name, candidate.relative_path, candidate.source_file, candidate.source_index, candidate.annotation_label_count)))
        if not ranked:
            return None
        _, negative_sharpness, _, candidate = min(ranked)
        return Candidate(candidate.image_name, candidate.relative_path, candidate.source_file, candidate.source_index, candidate.annotation_label_count, -negative_sharpness)  # type: ignore[call-arg]

    def describe_pending(
        self,
        generator: Callable[[Path, str], str],
        image_root: Path,
        *,
        retry_failed: bool = False,
        retry_generator: Callable[[Path, str], str] | None = None,
        max_attempts: int = 3,
        limit: int | None = None,
    ) -> dict[str, int]:
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        rows = self.db.execute(
            f"SELECT canonical_label, category, representative_image, attempts FROM labels WHERE status IN ({','.join('?' for _ in statuses)}) AND attempts < ? ORDER BY canonical_label",
            (*statuses, max_attempts),
        ).fetchall()
        if limit is not None:
            rows = rows[:limit]
        result: Counter[str] = Counter()
        for row in rows:
            label, category = str(row["canonical_label"]), str(row["category"])
            attempts = int(row["attempts"]) + 1
            response: object = None
            try:
                active_generator = retry_generator if attempts > 1 and retry_generator is not None else generator
                response = active_generator(image_root / str(row["representative_image"]), self.prompts.render(category, label))
                description, error = validate_description(clean_description(response, category), category)
                if error:
                    raise ValueError(error)
                self.db.execute("UPDATE labels SET status='complete', description=?, raw_response=NULL, attempts=?, last_error=NULL, updated_at=? WHERE canonical_label=?", (description, attempts, _now(), label))
                result["complete"] += 1
            except Exception as error:
                raw_response = response if isinstance(response, str) else None
                self.db.execute("UPDATE labels SET status='failed', raw_response=?, attempts=?, last_error=?, updated_at=? WHERE canonical_label=?", (raw_response, attempts, f"{type(error).__name__}: {error}"[:1000], _now(), label))
                result["failed"] += 1
            self.db.commit()
        self.write_exports()
        return dict(result)

    def reset_descriptions(self, backend: str) -> int:
        """Rebuild descriptions from existing representatives without recataloging labels."""
        if not backend.strip():
            raise ValueError("description backend must not be empty")
        cursor = self.db.execute(
            """UPDATE labels
               SET status='pending', description=NULL, raw_response=NULL, attempts=0,
                   last_error=NULL, updated_at=?
               WHERE representative_image IS NOT NULL""",
            (_now(),),
        )
        self._set_metadata("description_backend", backend.strip())
        self._set_metadata("description_reset_at", _now())
        self.db.commit()
        self.write_exports()
        return cursor.rowcount

    def revalidate_completed(self) -> int:
        """Move historical outputs that violate their supplied category prompt back to failed."""
        invalid = 0
        for row in self.db.execute("SELECT canonical_label, category, description FROM labels WHERE status='complete'"):
            cleaned = clean_description(row["description"], row["category"])
            _, error = validate_description(cleaned, row["category"])
            if error:
                self.db.execute("UPDATE labels SET status='failed', last_error=?, updated_at=? WHERE canonical_label=?", (f"validation: {error}", _now(), row["canonical_label"]))
                invalid += 1
            elif cleaned != row["description"]:
                self.db.execute("UPDATE labels SET description=?, updated_at=? WHERE canonical_label=?", (cleaned, _now(), row["canonical_label"]))
        self.db.commit()
        if invalid:
            self.write_exports()
        return invalid

    def recover_failed_from_raw(self) -> int:
        """Promote recoverable Qwen outputs after conservative cleaning."""
        recovered = 0
        for row in self.db.execute("SELECT canonical_label, category, raw_response FROM labels WHERE status='failed' AND raw_response IS NOT NULL"):
            description = clean_description(row["raw_response"], row["category"])
            _, error = validate_description(description, row["category"])
            if error:
                continue
            self.db.execute(
                "UPDATE labels SET status='complete', description=?, raw_response=NULL, last_error=NULL, updated_at=? WHERE canonical_label=?",
                (description, _now(), row["canonical_label"]),
            )
            recovered += 1
        self.db.commit()
        if recovered:
            self.write_exports()
        return recovered


    def write_exports(self) -> None:
        labels = [dict(row) for row in self.db.execute("SELECT canonical_label, category, description, representative_image, source_annotation, annotation_label_count, sharpness, status, attempts, last_error, updated_at FROM labels ORDER BY category, canonical_label")]
        unclassified = [dict(row) for row in self.db.execute("SELECT * FROM unclassified_labels ORDER BY canonical_label")]
        grouped = {category: [row for row in labels if row["category"] == category] for category in INCLUDED_CATEGORIES}
        _atomic_json(self.output_dir / "label_universe.json", {"schema_version": SCHEMA_VERSION, "labels": grouped, "excluded_source_labels": unclassified})
        _atomic_json(self.output_dir / "category_index.json", {category: [row["canonical_label"] for row in rows] for category, rows in grouped.items()})
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now(),
            "prompt_source": str(self.prompts.source_path),
            "prompt_sha256": self.prompts.sha256,
            "blur_threshold": self.blur_threshold,
            "candidate_sample_size": self._metadata("candidate_sample_size"),
            "random_seed": self._metadata("random_seed"),
            "description_backend": self._metadata("description_backend"),
            "description_reset_at": self._metadata("description_reset_at"),
            "catalog": self.catalog_counts(),
            "descriptions": dict(Counter(row["status"] for row in labels)),
            "version": self._version(labels, unclassified),
        }
        _atomic_json(self.output_dir / "build_report.json", report)

    def catalog_counts(self) -> dict[str, int]:
        values = {category: 0 for category in INCLUDED_CATEGORIES}
        for row in self.db.execute("SELECT category, COUNT(*) AS total FROM labels GROUP BY category"):
            values[str(row["category"])] = int(row["total"])
        values["unclassified"] = int(self.db.execute("SELECT COUNT(*) FROM unclassified_labels").fetchone()[0])
        return values

    def _version(self, labels: list[dict[str, object]], unclassified: list[dict[str, object]]) -> str:
        payload = json.dumps({"labels": labels, "unclassified": unclassified, "prompt_sha256": self.prompts.sha256}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _has_metadata(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM metadata WHERE key=?", (key,)).fetchone() is not None

    def _metadata(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value))


def clean_description(response: object, category: str | None = None) -> str:
    """Remove only formatting and sentences containing explicitly forbidden claims."""
    if not isinstance(response, str):
        return ""
    text = response.replace("**", "").replace("__", "")
    text = re.sub(r"```(?:[A-Za-z0-9_-]+)?", "", text)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines).strip()
    forbidden = EVALUATIVE_PHRASES + CATEGORY_INFERENCE_PHRASES.get(category or "", ())
    sentences = re.split(r"(?<=[\u3002\uFF01\uFF1F\uFF1B])", text)
    kept: list[str] = []
    for sentence in sentences:
        if sentence and any(phrase in sentence for phrase in forbidden):
            continue
        kept.append(sentence)
    return "".join(kept).strip()


def validate_description(response: object, category: str | None = None) -> tuple[str, str | None]:
    if not isinstance(response, str):
        return "", "model response is not text"
    description = response.strip()
    if any(phrase in description for phrase in EVALUATIVE_PHRASES):
        return description, "description contains a prohibited evaluative phrase"
    if any(phrase in description for phrase in CATEGORY_INFERENCE_PHRASES.get(category or "", ())):
        return description, "description contains a category-prohibited inference"
    if len(description) < 20:
        return description, "description is empty or too short"
    if any(marker in description for marker in ("```", "\n#", "**")):
        return description, "description contains Markdown"
    return description, None
