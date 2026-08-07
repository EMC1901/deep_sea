from __future__ import annotations

import os
import time
from pathlib import Path

from smoke_common import DEFAULT_RESULTS_DIR, main, required_model_path


def validate_request(prompt: str, width: int, height: int) -> None:
    if not prompt.strip() or len(prompt) > 1_000:
        raise ValueError("prompt must contain 1 to 1000 characters")
    if width < 512 or height < 512 or width % 8 or height % 8:
        raise ValueError("width and height must be multiples of 8 and at least 512")


def smoke(result: dict[str, object]) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    validate_request("深海热液喷口旁的发光生物，科研插画", 512, 512)
    for invalid in (("", 512, 512), ("x" * 1_001, 512, 512), ("test", 510, 512)):
        try:
            validate_request(*invalid)
        except ValueError:
            continue
        raise RuntimeError("invalid image-generation request was accepted")

    model_path = required_model_path("IMAGE_MODEL_PATH")
    started = time.perf_counter()
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    result["load_seconds"] = round(time.perf_counter() - started, 3)
    result["precision"] = "float16"

    started = time.perf_counter()
    image = pipeline(
        prompt="深海热液喷口旁的发光生物，科研插画",
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(20260724),
    ).images[0]
    result["inference_seconds"] = round(time.perf_counter() - started, 3)

    output_dir = Path(os.environ.get("S7_RESULTS_DIR", DEFAULT_RESULTS_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke-image.jpg"
    image.save(output_path, format="JPEG", quality=85)
    if not output_path.read_bytes().startswith(b"\xff\xd8"):
        raise RuntimeError("generated output is not a JPEG")
    result["output_file"] = output_path.name
    result["seed"] = 20260724
    result["size"] = [512, 512]
    del pipeline


if __name__ == "__main__":
    main(smoke, "image")
