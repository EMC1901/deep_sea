from flask import Blueprint, current_app, jsonify


bp = Blueprint("health", __name__)


@bp.get("/health")
def health():
    container = current_app.extensions["container"]
    model = container.vision.health()
    retrieval = container.image_retrieval.health()
    return jsonify(
        {
            "status": "ok",
            "process": "ok",
            "model": "loaded" if model.ready else "loading",
            "model_backend": container.settings.model_backend.value,
            "model_detail": model.detail,
            "rag": "loaded" if container.rag.index else "no_documents",
            "documents": len(container.rag.documents),
            "worker": "running" if container.worker.last_success_monotonic else "idle",
            "image_retrieval": {
                "enabled": retrieval.enabled,
                "ready": retrieval.ready,
                "detail": retrieval.detail,
                "index_size": retrieval.index_size,
                "embedding_dimension": retrieval.embedding_dimension,
            },
        }
    )
