"""历史报告实现，仅供迁移对照；当前实现位于 src/deep_sea_explorer/。"""

import os
import re
import base64
import logging
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PlatypusImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import torch

logger = logging.getLogger(__name__)



class ReportGenerator:
    def __init__(self, font_path: str = None):
        self.font_name = "CombinedFont"
        self._init_font(font_path)

    def _init_font(self, font_path: str = None):
        """
        乱码根因通常来自 TTC + TTFont 的兼容性问题：
        - 优先找 TTF/OTF（最稳）
        - TTC 放最后（兜底）
        """
        possible_fonts = [
            font_path,
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "./simhei.ttf"
        ]

        selected = next((f for f in possible_fonts if f and os.path.exists(f)), None)

        if selected:
            try:
                pdfmetrics.registerFont(TTFont(self.font_name, selected))
                logger.info(f"✅ 已加载字体: {selected}")
                return
            except Exception as e:
                logger.error(f"字体注册失败: {e}")

        self.font_name = "Helvetica"
        logger.error("❌ 未找到可用中文字体，PDF 可能乱码")

    def _process_image(self, b64_str):
        if not b64_str or len(b64_str) < 100:
            return None
        try:
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            return BytesIO(base64.b64decode(b64_str))
        except:
            return None

    def _sanitize_text(self, s: str) -> str:
        """避免控制字符/异常符号导致渲染问题"""
        if not s:
            return ""
        s = s.replace("\x00", "")
        s = re.sub(r"[\u200b\u200e\u200f]", "", s)  # 零宽字符
        s = re.sub(r"[ \t]+", " ", s)
        # 如果出现大量 !!!!! 之类也做个软清理（可选）
        s = re.sub(r"([!！]){10,}", "！", s)
        return s.strip()

    def _get_styles(self):
        styles = getSampleStyleSheet()

        title_s = ParagraphStyle(
            "C_Title", parent=styles["Title"], fontName=self.font_name,
            fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=10,
            textColor=colors.HexColor("#1e3a8a")
        )

        heading_s = ParagraphStyle(
            "C_Head", parent=styles["Heading2"], fontName=self.font_name,
            fontSize=15, leading=18, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#0f766e")
        )

        sub_heading_s = ParagraphStyle(
            "C_SubHead", parent=styles["Normal"], fontName=self.font_name,
            fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
            textColor=colors.HexColor("#334155")
        )

        body_s = ParagraphStyle(
            "C_Body", parent=styles["Normal"], fontName=self.font_name,
            fontSize=10, leading=16, wordWrap="CJK", spaceAfter=4
        )

        table_t = ParagraphStyle("T_Text", parent=body_s, fontSize=9, leading=12, spaceAfter=0)
        table_h = ParagraphStyle("T_Head", parent=table_t, textColor=colors.white, alignment=TA_CENTER)

        number_s = ParagraphStyle(
            "Num_Text", parent=body_s, fontName=self.font_name,
            fontSize=10, leading=12, alignment=TA_LEFT
        )
        return title_s, heading_s, sub_heading_s, body_s, table_t, table_h, number_s

    def _join_limit(self, parts, max_chars: int):
        """把多段文本拼接，但限制总字符数，避免 prompt 超长导致模型输出异常"""
        out = []
        total = 0
        for p in parts:
            if not p:
                continue
            p = self._sanitize_text(str(p))
            if not p:
                continue
            need = len(p) + 1
            if total + need > max_chars:
                remain = max_chars - total
                if remain > 20:
                    out.append(p[:remain] + "…")
                break
            out.append(p)
            total += need
        return "\n".join(out)

    # =========================
    # LLM 总结（核心修改部分）
    # =========================
    def generate_summary_with_llm(self, report_payload: dict, analyzer_instance):
        if not getattr(analyzer_instance, "model", None):
            return "模型未加载。"

        # 1. 暂停后台监测，独占 GPU 资源
        analyzer_instance.is_processing_user_question = True

        memos = report_payload.get("memos") or []
        chats = report_payload.get("chats") or []
        meta = report_payload.get("meta") or {}
        bio_stats = report_payload.get("bio_stats") or []
        env_stats = report_payload.get("env_stats") or []
        bio_samples = report_payload.get("bio_samples") or []
        env_samples = report_payload.get("env_samples") or []

        # --- 组织信息 ---
        memo_lines = []
        for m in memos:
            if isinstance(m, dict):
                t = m.get("time") or m.get("timestamp") or ""
                txt = m.get("text") or m.get("content") or ""
                memo_lines.append(f"[{t}] {txt}")
        memo_txt = self._join_limit(memo_lines, max_chars=3000)

        chat_lines = []
        for c in chats:
            if isinstance(c, dict):
                t = c.get("time") or ""
                role = c.get("role", "-")
                txt = c.get("text") or ""
                has_img = "（含图片）" if c.get("image") else ""
                chat_lines.append(f"[{t}] {role}: {txt}{has_img}")
        chat_txt = self._join_limit(chat_lines, max_chars=3000)

        bio_lines = []
        for i, s in enumerate(bio_samples, 1):
            if isinstance(s, dict):
                name = s.get("name", f"生物样本{i}")
                t = s.get("time", "")
                desc = s.get("description", "")
                bio_lines.append(f"{i}. {name} | {t} | {desc}")
        bio_samples_txt = self._join_limit(bio_lines, max_chars=2000)

        env_lines = []
        for i, s in enumerate(env_samples, 1):
            if isinstance(s, dict):
                name = s.get("name", f"底质样本{i}")
                t = s.get("time", "")
                desc = s.get("description", "")
                env_lines.append(f"{i}. {name} | {t} | {desc}")
        env_samples_txt = self._join_limit(env_lines, max_chars=2000)

        def stats_to_lines(stats, title):
            lines = [title]
            for s in stats:
                if isinstance(s, dict):
                    lines.append(f"- {s.get('name','-')}: {s.get('count',0)}")
            return lines

        bio_stats_txt = self._join_limit(stats_to_lines(bio_stats, "生物统计："), max_chars=2000)
        env_stats_txt = self._join_limit(stats_to_lines(env_stats, "底质/环境统计："), max_chars=2000)

        time_range = meta.get("time_range", "-")
        prompt = f"""
    你是“深海探测专用AI系统”，专门分析深海视频与相关文档资料，具备丰富的海洋生物学、地质学和深海探测技术知识。
    请基于以下材料，输出一段《智能任务总结摘要》。

    硬性要求：
    - 只输出一段中文总结（不要标题、不要分点、不要编号、不要Markdown）
    - 不要出现“1.”“2.”之类结构化符号
    - 500字以内
    - 内容必须覆盖：生物探测要点、底质/环境要点、过程监测与异常、指挥官-系统交互情况，并给出可执行建议或下一步动作（1-2条，写进同一段里）

    任务时间范围：{time_range}

    【生物探测结果-统计】
    {bio_stats_txt}

    【生物探测结果-样本记录】
    {bio_samples_txt}

    【底质/环境探测结果-统计】
    {env_stats_txt}

    【底质/环境探测结果-样本记录】
    {env_samples_txt}

    【场景监测日志（memos）】
    {memo_txt}

    【交互问答记录（chats）】
    {chat_txt}
    """.strip()

        try:
            prompt = self._sanitize_text(prompt)
            msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            inputs = analyzer_instance.processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(analyzer_instance.model.device)

            # 2. 核心：加锁并清理碎片，防止总结被后台进程打断
            with analyzer_instance.inference_lock:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    ids = analyzer_instance.model.generate(
                        **inputs,
                        max_new_tokens=512,  # 适度增加 token，防止中途截断
                        do_sample=True,      # 启用采样，增加生成稳定性
                        temperature=0.7,
                        repetition_penalty=1.1,
                        top_p=0.9
                    )

            out = analyzer_instance.processor.batch_decode(
                ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            out = self._sanitize_text(out)
            logger.info(f"🧠 LLM summary generated (len={len(out)})")

            # 强制输出长度保护
            if len(out) > 450:
                out = out[:450].rstrip("，。；、") + "。"

            return out

        except Exception as e:
            logger.exception("自动摘要生成失败")
            return "自动摘要生成失败。"
        finally:
            # 3. 恢复后台监测线程
            analyzer_instance.is_processing_user_question = False

    def _build_bar_chart(self, stats, title, width=170*mm, height=60*mm, topn=10):
        stats = stats or []
        stats = sorted(stats, key=lambda x: x.get("count", 0), reverse=True)[:topn]
        d = Drawing(width, height)
        d.add(String(0, height-12, title, fontName=self.font_name, fontSize=10, fillColor=colors.HexColor("#334155")))

        if not stats:
            d.add(String(0, height/2, "（无统计数据）", fontName=self.font_name, fontSize=9, fillColor=colors.grey))
            return d

        maxv = max([s.get("count", 0) for s in stats] + [1])
        left = 0
        top = height - 22
        row_h = 10
        bar_h = 6
        label_w = 45*mm
        bar_w = width - label_w - 12

        for i, s in enumerate(stats):
            y = top - i * row_h
            name = str(s.get("name", "-"))
            val = int(s.get("count", 0))
            d.add(String(left, y, name[:18], fontName=self.font_name, fontSize=8, fillColor=colors.HexColor("#0f172a")))
            d.add(Rect(left + label_w, y-2, bar_w, bar_h, strokeColor=colors.lightgrey, fillColor=colors.HexColor("#f1f5f9")))
            fill_w = max(1, bar_w * (val / maxv))
            d.add(Rect(left + label_w, y-2, fill_w, bar_h, strokeColor=None, fillColor=colors.HexColor("#60a5fa")))
            d.add(String(left + label_w + bar_w + 4, y, str(val), fontName=self.font_name, fontSize=8, fillColor=colors.HexColor("#1e3a8a")))

        return d

    def create_pdf(self, filename, report_payload: dict, summary_text: str):
        # 1. 获取数据包
        bio_samples = report_payload.get("bio_samples") or []
        env_samples = report_payload.get("env_samples") or []
        bio_stats = report_payload.get("bio_stats") or []
        env_stats = report_payload.get("env_stats") or []
        memos = report_payload.get("memos") or []
        chats = report_payload.get("chats") or []
        meta = report_payload.get("meta") or {}

        # 2. 初始化文档
        doc = SimpleDocTemplate(
            filename, pagesize=A4,
            rightMargin=15*mm, leftMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm
        )
        title_s, heading_s, sub_s, body_s, table_t, table_h, number_s = self._get_styles()
        story = []

        # --- 页眉与标题 ---
        story.append(Paragraph("深海探测任务综合报告", title_s))
        gen_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_range = meta.get("time_range", "-")
        header_data = [
            [Paragraph("报告生成时间：", body_s), Paragraph(gen_time_str, number_s)],
            [Paragraph("任务时间范围：", body_s), Paragraph(self._sanitize_text(time_range), number_s)],
        ]
        header_table = Table(header_data, colWidths=[30*mm, 140*mm], hAlign='CENTER')
        header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
        story.append(header_table)
        story.append(Spacer(1, 10))

        # --- 一、摘要 ---
        story.append(Paragraph("一、摘要", heading_s))
        story.append(Paragraph(self._sanitize_text(summary_text) or "（摘要为空）", body_s))
        story.append(Spacer(1, 10))

        # --- 二、生物探测结果 ---
        story.append(Paragraph("二、生物探测结果", heading_s))
        story.append(self._build_bar_chart(bio_stats, "生物统计（Top）"))
        story.append(Spacer(1, 6))

        if bio_samples:
            for idx, s in enumerate(bio_samples[:12], 1):
                # 直接获取前端过滤后的精准名称
                name = self._sanitize_text(s.get("name", f"生物样本{idx}"))
                time = self._sanitize_text(s.get("time", ""))
                desc = self._sanitize_text(s.get("description", ""))
                
                story.append(Paragraph(f"{idx}. {name}", sub_s))
                if time: story.append(Paragraph(f"时间：{time}", body_s))
                if desc: story.append(Paragraph(desc, body_s))
                
                img_io = self._process_image(s.get("image"))
                if img_io:
                    story.append(Spacer(1, 4))
                    story.append(PlatypusImage(img_io, width=80*mm, height=45*mm))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("（无生物样本捕获）", body_s))

        # --- 三、底质探测结果 ---
        story.append(Paragraph("三、底质探测结果", heading_s))
        story.append(self._build_bar_chart(env_stats, "底质/环境要素统计（Top）"))
        story.append(Spacer(1, 6))

        if env_samples:
            for idx, s in enumerate(env_samples[:12], 1):
                name = self._sanitize_text(s.get("name", f"环境样本{idx}"))
                time = self._sanitize_text(s.get("time", ""))
                desc = self._sanitize_text(s.get("description", ""))
                
                story.append(Paragraph(f"{idx}. {name}", sub_s))
                if time: story.append(Paragraph(f"时间：{time}", body_s))
                if desc: story.append(Paragraph(desc, body_s))
                
                img_io = self._process_image(s.get("image"))
                if img_io:
                    story.append(Spacer(1, 4))
                    story.append(PlatypusImage(img_io, width=80*mm, height=45*mm))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("（无底质/环境样本捕获）", body_s))

        # --- 四、场景监测日志 ---
        story.append(Paragraph("四、场景监测日志", heading_s))
        if memos:
            data = [[Paragraph("时间", table_h), Paragraph("监测内容", table_h)]]
            for m in memos:
                data.append([Paragraph(str(m.get('time', '-'))[:8], number_s), Paragraph(self._sanitize_text(m.get('text', '')), table_t)])
            t = Table(data, colWidths=[30*mm, 150*mm], repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'TOP'), ('ROWBACKGROUNDS', (1,0), (-1,-1), [colors.white, colors.HexColor('#f1f5f9')]), ('padding', (0,0), (-1,-1), 6)]))
            story.append(t)
        # --- 五、交互问答记录 (新增部分) ---
        story.append(Paragraph("五、交互问答记录", heading_s))
        if chats:
            chat_data = [[Paragraph("时间/角色", table_h), Paragraph("交互内容", table_h)]]
            for c in chats:
                # 处理角色和时间
                role_map = {"user": "指挥官", "ai": "AI助手"}
                role = role_map.get(c.get('role', '').lower(), c.get('role', '未知'))
                time_str = str(c.get('time', '-'))[:8]
                role_cell_text = f"{time_str}<br/><b>{role}</b>"
                
                # 处理内容，包括文本和可能的图片
                content_text = self._sanitize_text(c.get('text', ''))
                
                # 构建内容单元格元素列表
                content_elements = [Paragraph(content_text, table_t)]
                
                # 检查是否有图片
                img_io = self._process_image(c.get('image'))
                if img_io:
                    content_elements.append(Spacer(1, 4))
                    # 聊天记录中的图片稍微小一点
                    content_elements.append(PlatypusImage(img_io, width=60*mm, height=34*mm))
                
                chat_data.append([
                    Paragraph(role_cell_text, table_t),
                    content_elements 
                ])
                
            chat_table = Table(chat_data, colWidths=[30*mm, 150*mm], repeatRows=1)
            chat_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f766e')), # 使用不同的表头颜色区分
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ROWBACKGROUNDS', (1,0), (-1,-1), [colors.white, colors.HexColor('#f0fdfa')]),
                ('padding', (0,0), (-1,-1), 6)
            ]))
            story.append(chat_table)
        else:
            story.append(Paragraph("（无交互问答记录）", body_s))
        # --- 附录 ---
        story.append(PageBreak())
        story.append(Paragraph("附录：统计明细", heading_s))
        story.append(Paragraph("A. 生物统计明细", sub_s))
        story.append(self._stats_table(bio_stats, table_h, table_t))
        story.append(Spacer(1, 8))
        story.append(Paragraph("B. 底质/环境统计明细", sub_s))
        story.append(self._stats_table(env_stats, table_h, table_t))

        doc.build(story)
        return filename

    def _stats_table(self, stats, table_h, table_t):
        stats = stats or []
        if not stats: return Paragraph("（无统计数据）", table_t)
        data = [[Paragraph("类别", table_h), Paragraph("数量", table_h)]]
        for s in sorted(stats, key=lambda x: x.get("count", 0), reverse=True)[:50]:
            data.append([
                Paragraph(self._sanitize_text(str(s.get("name", "-"))), table_t),
                Paragraph(str(s.get("count", 0)), table_t)
            ])
        t = Table(data, colWidths=[120*mm, 30*mm], repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f1f5f9')]),
            ('padding', (0,0), (-1,-1), 6),
        ]))
        return t


