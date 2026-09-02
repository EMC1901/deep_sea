from __future__ import annotations

import time

from smoke_common import main, required_model_path


def generate(model, processor, content: list[dict[str, object]]) -> str:
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_kwargs: dict[str, object] = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
    }
    images = [item["image"] for item in content if item["type"] == "image"]
    videos = [item["video"] for item in content if item["type"] == "video"]
    if images:
        inputs_kwargs["images"] = images
    if videos:
        inputs_kwargs["videos"] = videos
    inputs = processor(**inputs_kwargs).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    input_length = inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0].strip()
    if not text or not text.encode("utf-8"):
        raise RuntimeError("model returned an empty or invalid UTF-8 response")
    return text


def smoke(result: dict[str, object]) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = required_model_path("QWEN_MODEL_PATH")
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    result["load_seconds"] = round(time.perf_counter() - started, 3)
    result["precision"] = "bfloat16"

    image = Image.new("RGB", (64, 64), color=(12, 54, 94))
    started = time.perf_counter()
    image_text = generate(
        model,
        processor,
        [{"type": "image", "image": image}, {"type": "text", "text": "请简要描述图像。"}],
    )
    result["image_inference_seconds"] = round(time.perf_counter() - started, 3)

    video = np.stack(
        [np.full((64, 64, 3), fill_value=value, dtype=np.uint8) for value in (20, 60, 100, 140)]
    )
    started = time.perf_counter()
    video_text = generate(
        model,
        processor,
        [{"type": "video", "video": video}, {"type": "text", "text": "视频画面发生了什么变化？"}],
    )
    result["video_inference_seconds"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    report_text = generate(
        model,
        processor,
        [{"type": "text", "text": "请将“发现深海生物活动”改写为一句简短报告摘要。"}],
    )
    result["report_inference_seconds"] = round(time.perf_counter() - started, 3)
    result["image_response_characters"] = len(image_text)
    result["video_response_characters"] = len(video_text)
    result["report_response_characters"] = len(report_text)
    del processor, model


if __name__ == "__main__":
    main(smoke, "qwen")
