#!/usr/bin/env python3
"""Construct a compact offline three-category label knowledge base."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from deep_sea_explorer.infrastructure.knowledge_base.label_knowledge_base import (
    LabelKnowledgeBase,
    PromptTemplates,
    image_path_lookup,
    iter_annotations,
)

DEFAULT_IMAGE_ROOT = Path("/sevenH/deepsea_vlm/data/raw/benthicnet/extracted/compiled_labelled_512pix")
DEFAULT_ANNOTATION_ROOT = Path("/sevenH/deepsea_vlm/data/raw/customer_annotations/captioning_labelled")
DEFAULT_IMAGE_METADATA_INDEX = Path("runtime/image-retrieval-index/pool_meta.json")
DEFAULT_MODEL_PATH = Path("/sevenH/deepsea_vlm/models/Qwen3-VL-2B-Instruct")
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "src/deep_sea_explorer/resources/label_description_prompts.md"
DEFAULT_OUTPUT = Path("runtime/label-knowledge-base")
DEFAULT_API_BASE_URL = "https://ws-xu25v8xnfaogu1kb.cn-beijing.maas.aliyuncs.com/api/v1"
DEFAULT_API_MODEL = "qwen3-vl-235b-a22b-thinking"


def qwen_generator(model_path: Path, adapter_path: Path | None = None) -> tuple[Callable[[Path, str], str], Callable[[Path, str], str], Callable[[], None]]:
    """Load Qwen once; retries vary decoding while the template text stays untouched."""
    from deep_sea_explorer.infrastructure.models.local.adapters import QwenAdapter

    adapter = QwenAdapter(str(model_path), str(adapter_path) if adapter_path else "")
    adapter.load()

    def generate(image_path: Path, prompt: str) -> str:
        return adapter.describe_knowledge_base_label(image_path, prompt)

    def retry_generate(image_path: Path, prompt: str) -> str:
        return adapter.describe_knowledge_base_label(image_path, prompt, retry_sample=True)

    return generate, retry_generate, adapter.unload


def api_generator(base_url: str, model: str, api_key_env: str, timeout: float) -> tuple[Callable[[Path, str], str], Callable[[Path, str], str], Callable[[], None]]:
    """Build an API-only generator; the key stays in process memory."""
    from deep_sea_explorer.infrastructure.knowledge_base.qwen_api import QwenApiGenerator

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"API key environment variable is not set: {api_key_env}")
    generator = QwenApiGenerator(base_url, model, api_key, timeout=timeout)
    return generator, generator, lambda: None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--image-metadata-index", type=Path, default=DEFAULT_IMAGE_METADATA_INDEX)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blur-threshold", type=float, default=35.0)
    parser.add_argument("--sample-size", type=int, default=10, help="Random candidate images retained per label (default: 10).")
    parser.add_argument("--random-seed", type=int, default=20260814)
    parser.add_argument("--refresh-selection", action="store_true", help="Rescan annotations and refresh deterministic representative-image selection for incremental dataset updates.")
    description_backend = parser.add_mutually_exclusive_group()
    description_backend.add_argument("--describe", action="store_true", help="Generate pending descriptions with local Qwen3-VL.")
    description_backend.add_argument("--api-describe", action="store_true", help="Generate pending descriptions with the configured remote Qwen API.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-workers", type=int, default=1, help="Concurrent remote API calls; SQLite checkpoint writes remain serial.")
    parser.add_argument("--reset-descriptions", action="store_true", help="Reset only existing representative-backed records before description generation; never refreshes label selection.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Limit Qwen descriptions for a controlled run.")
    args = parser.parse_args()
    if args.api_workers < 1:
        parser.error("--api-workers must be positive")

    for label, path in (("image root", args.image_root), ("annotation root", args.annotation_root), ("image metadata index", args.image_metadata_index), ("prompt file", args.prompt_file)):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    prompts = PromptTemplates.from_file(args.prompt_file)
    knowledge_base = LabelKnowledgeBase(args.output, prompts, blur_threshold=args.blur_threshold)
    try:
        lookup = image_path_lookup(args.image_metadata_index, args.image_root)
        result: dict[str, object] = {
            "catalog": knowledge_base.prepare(
                iter_annotations(args.annotation_root),
                lookup,
                args.image_root,
                sample_size=args.sample_size,
                random_seed=args.random_seed,
                refresh_selection=args.refresh_selection,
            ),
            "image_paths_read_from_existing_index": len(lookup),
        }
        if args.reset_descriptions:
            if not (args.describe or args.api_describe):
                parser.error("--reset-descriptions requires --describe or --api-describe")
            backend = f"api:{args.api_model}" if args.api_describe else "local:Qwen3-VL"
            result["reset_descriptions"] = knowledge_base.reset_descriptions(backend)
        else:
            result["revalidated_completed"] = knowledge_base.revalidate_completed()
            result["recovered_failed_from_raw"] = knowledge_base.recover_failed_from_raw()
        if args.describe:
            if not args.model_path.is_dir():
                parser.error(f"Qwen model path does not exist: {args.model_path}")
            generate, retry_generate, unload = qwen_generator(args.model_path, args.adapter_path)
            try:
                result["descriptions"] = knowledge_base.describe_pending(
                    generate,
                    args.image_root,
                    retry_failed=args.retry_failed,
                    retry_generator=retry_generate,
                    max_attempts=args.max_attempts,
                    limit=args.limit,
                )
            finally:
                unload()
        elif args.api_describe:
            try:
                generate, retry_generate, unload = api_generator(args.api_base_url, args.api_model, args.api_key_env, args.api_timeout)
            except ValueError as error:
                parser.error(str(error))
            try:
                result["descriptions"] = knowledge_base.describe_pending(
                    generate,
                    args.image_root,
                    retry_failed=args.retry_failed,
                    retry_generator=retry_generate,
                    max_attempts=args.max_attempts,
                    limit=args.limit,
                    workers=args.api_workers,
                )
            finally:
                unload()
        knowledge_base.write_exports()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        knowledge_base.close()


if __name__ == "__main__":
    raise SystemExit(main())
