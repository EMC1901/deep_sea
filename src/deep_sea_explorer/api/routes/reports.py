from flask import Blueprint, after_this_request, current_app, request, send_file

bp = Blueprint("reports", __name__)


@bp.post("/generate_report")
def generate_report():
    container = current_app.extensions["container"]
    target = container.reports.generate(request.get_json(silent=True) or {})

    @after_this_request
    def cleanup(response):
        target.unlink(missing_ok=True)
        return response

    return send_file(
        target, as_attachment=True, download_name="Report.pdf", mimetype="application/pdf"
    )
