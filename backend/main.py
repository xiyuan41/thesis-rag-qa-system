from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, PDF_PATH, settings
from .llm import LLMClient, LLMError, build_evidence_answer
from .pdf_ingest import PDFIngestor
from .retriever import Retriever
from .schemas import IngestResponse, QueryRequest, QueryResponse, StatusResponse

app = FastAPI(title="论文内容查询智能体", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/resources", StaticFiles(directory=DATA_DIR), name="resources")
retriever = Retriever()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    manifest = DATA_DIR / "manifest.json"
    manifest_exists = manifest.exists()
    manifest_valid = False
    manifest_data: dict[str, Any] = {}
    if manifest_exists:
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest_data = loaded
                manifest_valid = isinstance(loaded.get("chunks"), list)
        except (OSError, json.JSONDecodeError):
            pass
    pages = manifest_data.get("pages", 0)
    details: dict[str, Any] = {
        "images": manifest_data.get("images", 0),
        "tables": manifest_data.get("tables", 0),
        "warnings": manifest_data.get("warnings", []),
        "manifest_exists": manifest_exists,
        "manifest_valid": manifest_valid,
        "index_error": getattr(retriever, "load_error", None),
        "invalid_chunks": getattr(retriever, "invalid_chunk_count", 0),
        "retrieval_degraded": getattr(retriever, "retrieval_degraded", False),
        "retrieval_error": getattr(retriever, "retrieval_error", None),
    }
    if not PDF_PATH.exists():
        readiness = "source_missing"
    elif getattr(retriever, "load_error", None) and not retriever.indexed:
        readiness = "manifest_error"
    elif retriever.indexed:
        readiness = "ready"
    else:
        readiness = "not_indexed"
    details["readiness"] = readiness
    details["retrieval_mode"] = retriever.mode
    details["answer_configured"] = bool(settings.chat_api_key and settings.chat_model)
    details["answer_mode"] = "llm" if details["answer_configured"] else "local_evidence"
    return StatusResponse(
        pdf=PDF_PATH.name,
        indexed=retriever.indexed,
        chunks=len(retriever.documents),
        pages=pages,
        details=details,
    )


def do_ingest(force: bool = False) -> dict[str, Any]:
    result = PDFIngestor().ingest(force=force)
    retriever.reload()
    return result


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(force: bool = False) -> IngestResponse:
    try:
        result = do_ingest(force)
        return IngestResponse(
            indexed=len(result.get("chunks", [])),
            pages=result.get("pages", 0),
            images=result.get("images", 0),
            tables=result.get("tables", 0),
            warnings=result.get("warnings", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _resource_url(value: str | None, kind: str) -> str | None:
    if not value or kind not in {"images", "tables"}:
        return None
    resource_dir = (DATA_DIR / kind).resolve()
    candidate = (resource_dir / Path(value).name).resolve()
    if candidate.parent != resource_dir or not candidate.is_file():
        return None
    return f"/resources/{kind}/{candidate.name}"


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not retriever.indexed:
        raise HTTPException(status_code=409, detail="尚未建立索引，请先调用 /api/ingest。")
    contexts = retriever.search(request.question, request.top_k, request.page)
    if not contexts:
        raise HTTPException(status_code=404, detail="没有找到相关论文内容，请换一种问法。")
    if not settings.chat_api_key or not settings.chat_model:
        answer = build_evidence_answer(request.question, contexts)
    else:
        try:
            answer = LLMClient().answer(request.question, contexts)
        except LLMError as exc:
            raise HTTPException(
                status_code=503,
                detail="在线模型服务调用失败，请检查模型配置或服务状态。",
            ) from exc

    citations = []
    for item in contexts:
        citation = dict(item)
        citation["chunk_id"] = citation.pop("id", citation.get("chunk_id", ""))
        citation["image_url"] = _resource_url(citation.get("image_url"), "images")
        citation["table_url"] = _resource_url(citation.get("table_url"), "tables")
        citations.append(citation)
    return QueryResponse(answer=answer, citations=citations, retrieval_count=len(contexts))
