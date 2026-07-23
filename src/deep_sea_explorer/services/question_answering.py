from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.models import StreamEvent


class QuestionAnsweringService:
    RAG_KEYWORDS = (
        "文档",
        "报告",
        "资料",
        "记录",
        "数据",
        "介绍",
        "说明",
        "什么",
        "多少",
        "如何",
        "哪些",
    )
    IMAGE_TRIGGERS = ("生成图片", "生成一张", "画", "create image", "generate image")

    def __init__(self, vision: object, image: object, rag: object, sessions: object) -> None:
        self.vision, self.image, self.rag, self.sessions = vision, image, rag, sessions

    def answer(self, session_id: str, question: str, video_path: Path) -> Iterator[StreamEvent]:
        state = self.sessions.get(session_id)
        state.is_answering = True
        try:
            lowered = question.lower()
            if any(trigger.lower() in lowered for trigger in self.IMAGE_TRIGGERS):
                prompt = question
                for trigger in self.IMAGE_TRIGGERS:
                    prompt = prompt.replace(trigger, "")
                yield StreamEvent(StreamEventType.CHUNK, text="正在为您构想图像...")
                image = self.image.generate(prompt.strip())
                yield StreamEvent(
                    StreamEventType.IMAGE,
                    content=base64.b64encode(image).decode("ascii"),
                    prompt=prompt.strip(),
                )
                return
            enriched = question
            if any(keyword in question for keyword in self.RAG_KEYWORDS):
                context = self.rag.context(question)
                if context:
                    enriched = (
                        f"基于以下文档资料和视频内容回答问题：\n{context}\n\n问题：{question}"
                    )
            yield from self.vision.answer(video_path, enriched)
        finally:
            state.is_answering = False
