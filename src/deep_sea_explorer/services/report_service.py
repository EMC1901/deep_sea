from pathlib import Path


class ReportService:
    def __init__(self, vision: object, renderer: object, files: object) -> None:
        self.vision, self.renderer, self.files = vision, renderer, files

    def generate(self, material: dict[str, object]) -> Path:
        summary = self.vision.summarize_report(material)
        return self.renderer.render(self.files.report_path(), material, summary)
