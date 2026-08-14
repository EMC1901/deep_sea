from __future__ import annotations

import base64
import shutil
import uuid
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image as PillowImage

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_fake_container
from deep_sea_explorer.infrastructure.reports.reportlab_renderer import ReportLabRenderer


def test_fake_report_service_generates_a_readable_pdf() -> None:
    root = Path(".deep-sea-explorer-tmp-test")
    container = build_fake_container(Settings(model_backend=ModelBackend.FAKE, temp_dir=root))
    target = container.reports.generate({"memos": [], "chats": []})
    try:
        assert target.read_bytes().startswith(b"%PDF")
    finally:
        target.unlink(missing_ok=True)
        for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            directory.rmdir()
        root.rmdir()


def test_complete_report_contains_original_sections_images_and_page_numbers() -> None:
    root = Path(".deep-sea-explorer-tmp") / f"report-{uuid.uuid4().hex}"
    target = root / "complete-report.pdf"
    image_bytes = BytesIO()
    PillowImage.new("RGB", (320, 180), (10, 60, 90)).save(image_bytes, format="JPEG")
    image = "data:image/jpeg;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")
    material = {
        "meta": {
            "session_id": "mission-001",
            "time_range": "20:50:34 - 20:52:08",
        },
        "memos": [
            {"time": "20:50:40", "text": "画面显示海底岩石与缓慢移动的深海生物。"},
            {"time": "20:51:10", "text": "水体清晰，未发现异常扰动。"},
        ],
        "chats": [
            {"time": "20:51:20", "role": "user", "text": "当前发现了什么？"},
            {
                "time": "20:51:24",
                "role": "assistant",
                "text": "发现一处典型深海生物样本。",
                "image": image,
            },
        ],
        "bio_samples": [
            {
                "name": "深海鱼",
                "time": "20:51:00",
                "description": "一条深海鱼从岩石附近游过。",
                "image": image,
            }
        ],
        "substrate_samples": [
            {
                "name": "岩石底质",
                "time": "20:51:30",
                "description": "海床以岩石和少量沉积物为主。",
                "image": image,
            }
        ],
        "geomorphology_samples": [
            {
                "name": "平坦海床",
                "time": "20:51:35",
                "description": "岩石海床局部平坦，可见缓慢起伏。",
                "image": image,
            }
        ],
        "bio_stats": [{"name": "深海鱼", "count": 1}],
        "substrate_stats": [{"name": "岩石底质", "count": 1}],
        "geomorphology_stats": [{"name": "平坦海床", "count": 1}],
    }
    try:
        ReportLabRenderer().render(
            target,
            material,
            "本次任务完成了连续深海场景监测，记录到典型深海鱼和岩石底质，并完成问答交互。",
        )

        with fitz.open(target) as document:
            text = "\n".join(page.get_text() for page in document)
            image_count = sum(len(page.get_images(full=True)) for page in document)
            assert document.page_count >= 3

        for heading in (
            "深海探测任务综合报告",
            "报告生成时间",
            "任务时间范围",
            "任务会话",
            "一、智能任务总结",
            "二、生物探测结果",
            "三、底质探测结果",
            "四、地貌探测结果",
            "五、场景动态监测日志",
            "六、指挥官与系统交互记录",
            "附录：统计明细",
            "生物统计明细",
            "底质统计明细",
            "地貌统计明细",
        ):
            assert heading in text
        assert "20:50:34 - 20:52:08" in text
        assert "Page 1" in text
        # ReportLab reuses identical images as one PDF XObject on a page.
        assert image_count >= 3
    finally:
        shutil.rmtree(root, ignore_errors=True)
