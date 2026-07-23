from __future__ import annotations

from pathlib import Path


class ReportLabRenderer:
    def render(self, target: Path, material: dict[str, object], summary: str) -> Path:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen.canvas import Canvas

        canvas = Canvas(str(target), pagesize=A4)
        canvas.setTitle("Deep Sea Explorer Report")
        text = canvas.beginText(40, 800)
        text.textLine("Deep Sea Explorer Mission Report")
        text.textLine(summary[:450])
        canvas.drawText(text)
        canvas.save()
        return target
