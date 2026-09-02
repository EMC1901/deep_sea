from pathlib import Path

from deep_sea_explorer.domain.report_material import compact_report_material
from deep_sea_explorer.ports.file_store import FileStore
from deep_sea_explorer.ports.model_gateway import VisionModelGateway
from deep_sea_explorer.ports.report_renderer import ReportRenderer


class ReportService:
    def __init__(
        self,
        vision: VisionModelGateway,
        renderer: ReportRenderer,
        files: FileStore,
    ) -> None:
        self.vision, self.renderer, self.files = vision, renderer, files

    def generate(self, material: dict[str, object]) -> Path:
        summary = self.vision.summarize_report(compact_report_material(material))
        return self.renderer.render(self.files.report_path(), material, summary)
