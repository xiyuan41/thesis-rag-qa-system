from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .chunking import classify_section_role
from .config import DATA_DIR, settings


class Retriever:
    """Hybrid LangChain retriever with an optional persistent Chroma branch."""

    def __init__(self):
        self.manifest_path = DATA_DIR / "manifest.json"
        self.documents: list[Document] = []
        self.bm25: BM25Retriever | None = None
        self.vector_store: Any = None
        self.mode = "none"
        self.load_error: str | None = None
        self.manifest_valid = False
        self.invalid_chunk_count = 0
        self.retrieval_degraded = False
        self.retrieval_error: str | None = None
        if not hasattr(self, "_vector_sync_path"):
            self._vector_sync_path = self.manifest_path.parent / "chroma_sync.json"
        self.reload()

    def reload(self) -> None:
        if not self.manifest_path.exists():
            self.manifest_valid = False
            self.load_error = "manifest_missing"
            self.invalid_chunk_count = 0
            return

        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
                raise ValueError("manifest_invalid_structure")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.manifest_valid = False
            self.load_error = "manifest_invalid"
            self.invalid_chunk_count = 0
            return

        documents: list[Document] = []
        seen: set[str] = set()
        invalid_chunk_count = 0
        for chunk in data["chunks"]:
            if not isinstance(chunk, dict):
                invalid_chunk_count += 1
                continue
            chunk_id = str(chunk.get("id", "")).strip()
            page = chunk.get("page")
            text = chunk.get("text")
            if (
                not chunk_id
                or chunk_id in seen
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(page, int)
                or isinstance(page, bool)
                or page < 1
            ):
                invalid_chunk_count += 1
                continue
            inferred_role = chunk.get("section_role")
            if not inferred_role and chunk.get("source_type", "text") == "text":
                inferred_role = classify_section_role(text)
            seen.add(chunk_id)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "page": page,
                        "source_type": chunk.get("source_type", "text"),
                        "image_url": chunk.get("image_url"),
                        "table_url": chunk.get("table_url"),
                        "section_role": inferred_role,
                    },
                )
            )

        bm25 = BM25Retriever.from_documents(documents) if documents else None
        if bm25:
            bm25.k = settings.top_k
        vector_store = None
        mode = "bm25" if bm25 else "none"
        retrieval_degraded = False
        retrieval_error = None
        if documents and settings.enable_vector and settings.embedding_model:
            try:
                from langchain_chroma import Chroma
                from langchain_huggingface import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={"device": settings.embedding_device},
                    encode_kwargs={"normalize_embeddings": True},
                    cache_folder=settings.embedding_cache_dir or None,
                )
                vector_store = Chroma(
                    collection_name=settings.vector_collection,
                    embedding_function=embeddings,
                    persist_directory=str(self.manifest_path.parent / "chroma"),
                )
                self._sync_vector_store(vector_store, documents)
                mode = "hybrid"
            except Exception as exc:
                # Keep BM25 available if the local embedding dependency or model fails.
                vector_store = None
                retrieval_degraded = True
                retrieval_error = type(exc).__name__
        elif settings.enable_vector and settings.embedding_model:
            retrieval_degraded = True
            retrieval_error = "empty_index"

        self.documents = documents
        self.bm25 = bm25
        self.vector_store = vector_store
        self.mode = mode
        self.manifest_valid = True
        self.load_error = None
        self.invalid_chunk_count = invalid_chunk_count
        self.retrieval_degraded = retrieval_degraded
        self.retrieval_error = retrieval_error

    def _embedding_fingerprint(self) -> str:
        payload = f"{settings.embedding_model}|{settings.embedding_device}|{settings.embedding_cache_dir}|{settings.vector_collection}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _sync_vector_store(self, vector_store: Any, documents: list[Document]) -> None:
        """Synchronize Chroma by stable IDs without resetting the collection."""
        current = {
            doc.metadata["chunk_id"]: hashlib.sha256(
                (doc.page_content + "\n" + json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True)).encode("utf-8")
            ).hexdigest()
            for doc in documents
        }
        fingerprint = self._embedding_fingerprint()
        previous: dict[str, Any] = {}
        try:
            previous = json.loads(self._vector_sync_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

        if previous.get("embedding_fingerprint") != fingerprint:
            try:
                vector_store.reset_collection()
            except AttributeError:
                pass
            previous = {}

        old_chunks = previous.get("chunks", {})
        changed_ids = [chunk_id for chunk_id, digest in current.items() if old_chunks.get(chunk_id) != digest]
        removed_ids = [chunk_id for chunk_id in old_chunks if chunk_id not in current]
        if removed_ids:
            vector_store.delete(ids=removed_ids)
        if changed_ids:
            by_id = {doc.metadata["chunk_id"]: doc for doc in documents}
            vector_store.add_documents(
                [by_id[chunk_id] for chunk_id in changed_ids],
                ids=changed_ids,
            )
        self._vector_sync_path.write_text(
            json.dumps({"embedding_fingerprint": fingerprint, "chunks": current}, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _rrf_order(*ranked_lists: list[Document], constant: int = 60) -> list[Document]:
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                chunk_id = doc.metadata["chunk_id"]
                docs[chunk_id] = doc
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
        return [
            docs[chunk_id]
            for chunk_id in sorted(
                scores,
                key=lambda item: (-scores[item], docs[item].metadata.get("page", 0), item),
            )
        ]

    @property
    def indexed(self) -> bool:
        return bool(self.documents)

    def search(self, question: str, top_k: int | None = None, page: int | None = None) -> list[dict[str, Any]]:
        if not self.bm25:
            return []
        requested_k = top_k if top_k is not None else settings.top_k
        try:
            k = max(1, min(int(requested_k), 20))
        except (TypeError, ValueError):
            k = settings.top_k
        self.bm25.k = len(self.documents)
        bm25_documents = self.bm25.invoke(question)
        if page is not None:
            bm25_documents = [doc for doc in bm25_documents if doc.metadata.get("page") == page]

        vector_documents: list[Document] = []
        if self.vector_store is not None:
            try:
                vector_documents = self.vector_store.similarity_search(
                    question, k=min(len(self.documents), max(k * 4, 20))
                )
                if page is not None:
                    vector_documents = [doc for doc in vector_documents if doc.metadata.get("page") == page]
            except Exception as exc:
                self.vector_store = None
                self.mode = "bm25"
                self.retrieval_degraded = True
                self.retrieval_error = type(exc).__name__

        documents = self._rrf_order(bm25_documents, vector_documents)
        explicit_section = any(term in question for term in ("目录", "目次", "摘要", "致谢", "谢辞", "参考文献", "引用文献"))
        apply_role_boost = page is None and not explicit_section
        ranked = []
        for original_rank, doc in enumerate(documents):
            role = doc.metadata.get("section_role")
            adjustment = {
                "abstract": -2.0,
                "body": -1.0,
                "toc": 3.0,
                "references": 3.0,
                "acknowledgements": 4.0,
                "front_matter": 2.0,
            }.get(role, 0.0) if apply_role_boost else 0.0
            ranked.append((original_rank + adjustment, original_rank, doc))
        ranked.sort(key=lambda item: (item[0], item[1], item[2].metadata.get("page", 0), item[2].metadata.get("chunk_id", "")))
        documents = [item[2] for item in ranked]

        return [
            {
                "id": doc.metadata["chunk_id"],
                "text": doc.page_content,
                **doc.metadata,
                "score": None,
            }
            for doc in documents[:k]
        ]
