from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    page: int | None = Field(default=None, ge=1)


class Citation(BaseModel):
    chunk_id: str
    page: int
    source_type: str
    text: str
    image_url: str | None = None
    table_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_count: int


class IngestResponse(BaseModel):
    indexed: int
    pages: int
    images: int
    tables: int
    warnings: list[str] = []


class StatusResponse(BaseModel):
    pdf: str
    indexed: bool
    chunks: int
    pages: int
    details: dict[str, Any] = {}
