from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


LOGGER = logging.getLogger(__name__)


class ReportLabRenderer:
    """Render the complete mission report preserved by the original application."""

    def __init__(self, font_path: str = "") -> None:
        self.font_name = self._register_font(font_path)

    def render(self, target: Path, material: dict[str, object], summary: str) -> Path:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            KeepTogether,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DeepSeaTitle",
            parent=styles["Title"],
            fontName=self.font_name,
            fontSize=22,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#123B65"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "DeepSeaHeading",
            parent=styles["Heading2"],
            fontName=self.font_name,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#087F8C"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        )
        subheading_style = ParagraphStyle(
            "DeepSeaSubheading",
            parent=styles["Heading3"],
            fontName=self.font_name,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#294C60"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
        body_style = ParagraphStyle(
            "DeepSeaBody",
            parent=styles["BodyText"],
            fontName=self.font_name,
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#172B3A"),
            wordWrap="CJK",
            spaceAfter=4,
        )
        table_style = ParagraphStyle(
            "DeepSeaTableText",
            parent=body_style,
            fontSize=9,
            leading=13,
            spaceAfter=0,
        )
        metadata_value_style = ParagraphStyle(
            "DeepSeaMetadataValue",
            parent=table_style,
            # Keep timestamps and IDs searchable even when Chinese content is
            # rendered by the built-in CID fallback font.
            fontName="Helvetica",
        )
        table_header_style = ParagraphStyle(
            "DeepSeaTableHeader",
            parent=table_style,
            textColor=colors.white,
            alignment=TA_CENTER,
        )

        doc = SimpleDocTemplate(
            str(target),
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=17 * mm,
            bottomMargin=18 * mm,
            title="深海探测任务综合报告",
            author="Deep Sea Explorer",
        )
        story: list[Any] = []
        bio_samples = self._records(material.get("bio_samples"))
        env_samples = self._records(material.get("env_samples"))
        bio_stats = self._records(material.get("bio_stats"))
        env_stats = self._records(material.get("env_stats"))
        memos = self._records(material.get("memos"))
        chats = self._records(material.get("chats"))
        raw_meta = material.get("meta")
        meta: dict[str, object] = raw_meta if isinstance(raw_meta, dict) else {}

        story.append(Paragraph("深海探测任务综合报告", title_style))
        header_data = [
            [
                self._paragraph("报告生成时间", table_style),
                self._paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), metadata_value_style),
            ],
            [
                self._paragraph("任务时间范围", table_style),
                self._paragraph(meta.get("time_range", "-"), metadata_value_style),
            ],
            [
                self._paragraph("任务会话", table_style),
                self._paragraph(meta.get("session_id", "-"), metadata_value_style),
            ],
        ]
        header = Table(header_data, colWidths=[32 * mm, 138 * mm], hAlign="CENTER")
        header.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F1F5")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9BB7C5")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C4D5DD")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([header, Spacer(1, 8)])

        story.append(Paragraph("一、智能任务总结", heading_style))
        story.append(self._paragraph(summary or "未生成任务摘要。", body_style))

        story.append(Paragraph("二、生物探测结果", heading_style))
        story.append(self._bar_chart(bio_stats, "生物统计（Top 10）"))
        story.append(Spacer(1, 6))
        self._append_samples(
            story,
            bio_samples,
            "生物样本",
            "未捕获到典型生物样本。",
            subheading_style,
            body_style,
            Image,
            Spacer,
            KeepTogether,
        )

        story.append(Paragraph("三、底质与环境探测结果", heading_style))
        story.append(self._bar_chart(env_stats, "底质/环境要素统计（Top 10）"))
        story.append(Spacer(1, 6))
        self._append_samples(
            story,
            env_samples,
            "环境样本",
            "未捕获到典型底质或环境样本。",
            subheading_style,
            body_style,
            Image,
            Spacer,
            KeepTogether,
        )

        story.append(Paragraph("四、场景动态监测日志", heading_style))
        if memos:
            memo_rows: list[list[Any]] = [
                [
                    self._paragraph("时间", table_header_style),
                    self._paragraph("监测内容", table_header_style),
                ]
            ]
            for memo in memos[:200]:
                memo_rows.append(
                    [
                        self._paragraph(str(memo.get("time", "-"))[:8], metadata_value_style),
                        self._paragraph(
                            memo.get("text") or memo.get("content") or "-",
                            table_style,
                        ),
                    ]
                )
            story.append(
                self._data_table(
                    LongTable,
                    TableStyle,
                    colors,
                    memo_rows,
                    [28 * mm, 142 * mm],
                    "#185FA5",
                )
            )
        else:
            story.append(self._paragraph("无场景动态监测记录。", body_style))

        story.append(Paragraph("五、指挥官与系统交互记录", heading_style))
        if chats:
            chat_rows: list[list[Any]] = [
                [
                    self._paragraph("时间/角色", table_header_style),
                    self._paragraph("交互内容", table_header_style),
                ]
            ]
            role_names = {
                "user": "指挥官",
                "ai": "AI 助手",
                "assistant": "AI 助手",
            }
            for chat in chats[:100]:
                role = str(chat.get("role", "未知"))
                role = role_names.get(role.lower(), role)
                identity = f"{str(chat.get('time', '-'))[:8]}\n{role}"
                left = self._paragraph(identity, table_style)
                right: list[Any] = [
                    self._paragraph(chat.get("text") or "（图片消息）", table_style)
                ]
                image = self._image_flowable(chat.get("image"), Image, 58 * mm, 38 * mm)
                if image is not None:
                    right.extend([Spacer(1, 4), image])
                chat_rows.append([left, right])
            story.append(
                self._data_table(
                    LongTable,
                    TableStyle,
                    colors,
                    chat_rows,
                    [32 * mm, 138 * mm],
                    "#087F8C",
                )
            )
        else:
            story.append(self._paragraph("无指挥官与系统交互记录。", body_style))

        story.extend([PageBreak(), Paragraph("附录：统计明细", heading_style)])
        story.append(Paragraph("A. 生物统计明细", subheading_style))
        story.append(
            self._stats_table(
                bio_stats,
                LongTable,
                TableStyle,
                colors,
                table_header_style,
                table_style,
            )
        )
        story.append(Spacer(1, 10))
        story.append(Paragraph("B. 底质/环境统计明细", subheading_style))
        story.append(
            self._stats_table(
                env_stats,
                LongTable,
                TableStyle,
                colors,
                table_header_style,
                table_style,
            )
        )

        def page_chrome(canvas: Any, current_doc: Any) -> None:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#C4D5DD"))
            canvas.line(15 * mm, 13 * mm, A4[0] - 15 * mm, 13 * mm)
            canvas.setFillColor(colors.HexColor("#526D7A"))
            canvas.setFont(self.font_name, 8)
            canvas.drawString(15 * mm, 8 * mm, "Deep Sea Explorer")
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(
                A4[0] - 15 * mm,
                8 * mm,
                f"Page {current_doc.page}",
            )
            canvas.restoreState()

        doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
        return target

    def _register_font(self, configured_path: str) -> str:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont

        font_name = "DeepSeaCJK"
        if font_name in pdfmetrics.getRegisteredFontNames():
            return font_name
        candidates = [
            configured_path,
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for value in candidates:
            if not value:
                continue
            path = Path(value)
            if not path.is_file():
                continue
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(path)))
                LOGGER.info("report font loaded path=%s", path)
                return font_name
            except Exception as error:
                LOGGER.warning(
                    "report font rejected path=%s error_type=%s",
                    path,
                    type(error).__name__,
                )
        # Linux installations commonly provide CJK fonts as .ttc collections,
        # which ReportLab's TTFont loader cannot read. Its built-in CID font
        # keeps Chinese titles, labels, and timestamps visible in that case.
        cid_font_name = "STSong-Light"
        if cid_font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont(cid_font_name))
        LOGGER.warning("report CJK font unavailable; using built-in CID font")
        return cid_font_name

    @staticmethod
    def _records(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _clean_text(value: object) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\x00", "")
        text = re.sub(r"[\u200b\u200e\u200f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _paragraph(self, value: object, style: Any) -> Any:
        from reportlab.platypus import Paragraph

        text = escape(self._clean_text(value)).replace("\n", "<br/>")
        return Paragraph(text or "-", style)

    def _append_samples(
        self,
        story: list[Any],
        samples: list[dict[str, object]],
        default_name: str,
        empty_message: str,
        subheading_style: Any,
        body_style: Any,
        image_class: Any,
        spacer_class: Any,
        keep_together_class: Any,
    ) -> None:
        from reportlab.lib.units import mm

        if not samples:
            story.append(self._paragraph(empty_message, body_style))
            return
        for index, sample in enumerate(samples[:12], 1):
            name = sample.get("name") or f"{default_name}{index}"
            block: list[Any] = [self._paragraph(f"{index}. {name}", subheading_style)]
            if sample.get("time"):
                block.append(self._paragraph(f"时间：{sample['time']}", body_style))
            if sample.get("description"):
                block.append(self._paragraph(sample["description"], body_style))
            image = self._image_flowable(sample.get("image"), image_class, 82 * mm, 52 * mm)
            if image is not None:
                block.extend([spacer_class(1, 3), image])
            block.append(spacer_class(1, 7))
            story.append(keep_together_class(block))

    @staticmethod
    def _image_flowable(
        value: object,
        image_class: Any,
        max_width: float,
        max_height: float,
    ) -> Any | None:
        if not isinstance(value, str) or len(value) < 100 or len(value) > 20_000_000:
            return None
        try:
            from PIL import Image as PillowImage

            encoded = value.split(",", 1)[1] if "," in value else value
            source = BytesIO(base64.b64decode(encoded))
            normalized = BytesIO()
            with PillowImage.open(source) as pillow_image:
                pillow_image.load()
                pillow_image.convert("RGB").save(normalized, format="JPEG", quality=90)
            normalized.seek(0)
            image = image_class(normalized)
            image._restrictSize(max_width, max_height)
            return image
        except Exception:
            return None

    def _bar_chart(self, stats: list[dict[str, object]], title: str) -> Any:
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.lib import colors
        from reportlab.lib.units import mm

        rows = sorted(stats, key=self._count, reverse=True)[:10]
        width = 170 * mm
        height = max(28 * mm, (18 + max(len(rows), 1) * 15))
        chart = Drawing(width, height)
        chart.add(
            String(
                0,
                height - 11,
                title,
                fontName=self.font_name,
                fontSize=10,
                fillColor=colors.HexColor("#294C60"),
            )
        )
        if not rows:
            chart.add(
                String(
                    0,
                    height / 2,
                    "（无统计数据）",
                    fontName=self.font_name,
                    fontSize=9,
                    fillColor=colors.grey,
                )
            )
            return chart
        maximum = max(self._count(item) for item in rows) or 1
        label_width = 46 * mm
        bar_width = width - label_width - 14 * mm
        top = height - 27
        for index, item in enumerate(rows):
            y = top - index * 15
            name = self._clean_text(item.get("name", "-"))[:20]
            count = self._count(item)
            chart.add(
                String(
                    0,
                    y,
                    name,
                    fontName=self.font_name,
                    fontSize=8,
                    fillColor=colors.HexColor("#172B3A"),
                )
            )
            chart.add(
                Rect(
                    label_width,
                    y - 2,
                    bar_width,
                    7,
                    strokeColor=colors.HexColor("#C4D5DD"),
                    fillColor=colors.HexColor("#E8F1F5"),
                )
            )
            chart.add(
                Rect(
                    label_width,
                    y - 2,
                    max(1, bar_width * count / maximum),
                    7,
                    strokeColor=None,
                    fillColor=colors.HexColor("#2C7FB8"),
                )
            )
            chart.add(
                String(
                    label_width + bar_width + 5,
                    y,
                    str(count),
                    fontName=self.font_name,
                    fontSize=8,
                    fillColor=colors.HexColor("#123B65"),
                )
            )
        return chart

    def _stats_table(
        self,
        stats: list[dict[str, object]],
        table_class: Any,
        table_style_class: Any,
        colors: Any,
        header_style: Any,
        text_style: Any,
    ) -> Any:
        from reportlab.lib.units import mm

        if not stats:
            return self._paragraph("无统计数据。", text_style)
        rows: list[list[Any]] = [
            [self._paragraph("类别", header_style), self._paragraph("数量", header_style)]
        ]
        for item in sorted(stats, key=self._count, reverse=True)[:50]:
            rows.append(
                [
                    self._paragraph(item.get("name", "-"), text_style),
                    self._paragraph(self._count(item), text_style),
                ]
            )
        return self._data_table(
            table_class,
            table_style_class,
            colors,
            rows,
            [135 * mm, 35 * mm],
            "#185FA5",
        )

    @staticmethod
    def _data_table(
        table_class: Any,
        table_style_class: Any,
        colors: Any,
        rows: list[list[Any]],
        widths: list[float],
        header_color: str,
    ) -> Any:
        table = table_class(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            table_style_class(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9BB7C5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F3F7F9")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    @staticmethod
    def _count(item: dict[str, object]) -> int:
        try:
            value = item.get("count", 0)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return 0
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
