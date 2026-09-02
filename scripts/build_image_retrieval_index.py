"""Create or validate the independent labelled deep-sea image retrieval index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make this repository tool runnable from a checkout before an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_sea_explorer.infrastructure.retrieval.dinov2 import DinoV2ImageEncoder  # noqa: E402
from deep_sea_explorer.infrastructure.retrieval.index_builder import (  # noqa: E402
    build_index,
    discover_gallery,
    verify_index_directory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="report safe image-to-caption matches without loading DINOv2")
    _add_source_arguments(scan)

    build = commands.add_parser("build", help="embed matched images and publish an index")
    _add_source_arguments(build)
    build.add_argument("--output-dir", type=Path, required=True, help="new runtime index directory")
    build.add_argument("--model-path", required=True, help="local DINOv2 model directory or identifier")
    build.add_argument("--device", default="auto", help="DINOv2 device: auto, cpu, cuda, or mps")
    build.add_argument("--batch-size", type=int, default=32, help="images per DINOv2 forward pass")
    build.add_argument("--limit", type=int, help="build only the first N sorted matches for a smoke check")
    build.add_argument("--overwrite", action="store_true", help="replace an existing output directory")

    verify = commands.add_parser("verify", help="validate a previously built runtime index")
    verify.add_argument("--index-dir", type=Path, required=True)
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    args = _parser().parse_args()
    if args.command == "verify":
        result = verify_index_directory(args.index_dir)
        _print({"index_dir": str(result.index_dir), "count": result.count, "dimension": result.dimension})
        return 0

    selection = discover_gallery(args.image_root, args.annotation_root)
    if args.command == "scan":
        _print(
            {
                "selected_images": len(selection.images),
                "selection": dict(selection.counters),
                "annotation_files": list(selection.annotation_files),
            }
        )
        return 0

    encoder = DinoV2ImageEncoder(args.model_path, device=args.device)
    result = build_index(
        selection,
        args.image_root,
        args.output_dir,
        encoder,
        model_name=args.model_path,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    _print(
        {
            "index_dir": str(result.index_dir),
            "count": result.count,
            "dimension": result.dimension,
            "selection": dict(result.selection_counters),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
