#!/usr/bin/env python3
"""Build the runtime CMECS hierarchy from the supplied bilingual JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalized(value: object) -> str:
    """Use the Chinese translation when available, otherwise retain source text."""
    if isinstance(value, list):
        choices = value[1:2] + value[:1]
        value = next((item for item in choices if isinstance(item, str) and item.strip()), "")
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def records(source: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(raw_records, list):
        raise ValueError(f"{source} has no records array")
    unique: set[tuple[str, ...]] = set()
    result: list[dict[str, str]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        item = {field.lower(): normalized(raw.get(field)) for field in fields}
        key = tuple(item[field.lower()] for field in fields)
        if not all(key) or key in unique:
            continue
        unique.add(key)
        result.append(item)
    return sorted(result, key=lambda item: tuple(item.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--biotic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    substrate = records(args.substrate, ("Component", "Class", "Subclass", "Group", "Definition"))
    biotic = records(args.biotic, ("Component", "Class", "Subclass", "Definition"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"version": 1, "substrate": substrate, "biotic": biotic}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"substrate": len(substrate), "biotic": len(biotic)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
