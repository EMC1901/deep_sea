"""Resumable, verified download for the approved Qwen3-VL-8B-Instruct revision."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

# huggingface_hub reads these settings while its modules import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, snapshot_download


REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_TARGET = Path("/sevenH/models/Qwen3-VL-8B-Instruct")
DEFAULT_PROXY = "http://127.0.0.1:7890"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(target: Path) -> None:
    """Verify each published file's size and, where available, SHA-256."""
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    failures: list[str] = []
    for sibling in info.siblings:
        if not sibling.rfilename:
            continue
        path = target / sibling.rfilename
        expected_size = sibling.size
        if not path.is_file():
            failures.append(f"missing {sibling.rfilename}")
            continue
        if expected_size is not None and path.stat().st_size != expected_size:
            failures.append(f"size mismatch {sibling.rfilename}")
            continue
        lfs = sibling.lfs
        expected_sha256 = (
            lfs.get("sha256")
            if isinstance(lfs, dict)
            else getattr(lfs, "sha256", None)
        )
        if expected_sha256 and _sha256(path) != expected_sha256:
            failures.append(f"sha256 mismatch {sibling.rfilename}")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(f"verified repository={REPOSITORY} revision={info.sha} files={len(info.siblings)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--retry-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.retry_seconds <= 0:
        parser.error("--retry-seconds must be positive")

    # Always use the server's Mihomo endpoint.  ``setdefault`` could retain a
    # stale inherited proxy and silently bypass the deployment route.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = args.proxy
    # The server proxy is stable with one resumable HTTP stream but not with
    # concurrent Xet requests.
    args.target.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            if args.verify_only:
                verify(args.target)
                return 0
            snapshot_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                local_dir=args.target,
                max_workers=1,
                etag_timeout=10,
            )
            verify(args.target)
            return 0
        except Exception as error:
            print(f"download retry after {type(error).__name__}: {error}", flush=True)
            time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
