from flask import Blueprint, current_app, request

from deep_sea_explorer.domain.exceptions import ValidationError

bp = Blueprint("rag", __name__)


@bp.post("/rag/upload")
def upload():
    container = current_app.extensions["container"]
    file = request.files.get("file")
    if file is None:
        raise ValidationError("No file provided")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files are supported")
    target = container.files.upload_path(file.filename)
    file.save(target)
    if not container.rag.add_pdf(target, file.filename):
        raise ValidationError("Failed to process PDF")
    container.rag.build_index()
    return {
        "status": "success",
        "message": "document added",
        "total_documents": len(container.rag.documents),
    }


@bp.post("/rag/search")
def search():
    container = current_app.extensions["container"]
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("No query provided")
    return {
        "results": [
            {
                "content": item.content,
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "score": item.score,
            }
            for item in container.rag.search(query.strip())
        ]
    }
