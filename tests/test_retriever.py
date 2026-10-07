import json
from unittest.mock import Mock

from langchain_core.documents import Document

from backend.retriever import Retriever


def test_broad_query_reranks_content_but_explicit_section_and_page_are_preserved(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "chunks": [
                    {"id": "ack", "text": "致谢 论文研究感谢老师", "page": 97, "source_type": "text"},
                    {"id": "body", "text": "研究主题是家禽养殖安防系统设计", "page": 20, "source_type": "text"},
                    {"id": "abs", "text": "摘要 本研究设计家禽养殖系统", "page": 7, "source_type": "text"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = manifest
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()

    broad = retriever.search("论文研究主题", top_k=3)
    assert broad[0]["id"] == "abs"
    assert broad[1]["id"] == "body"
    assert broad[0]["section_role"] == "abstract"
    assert broad[2]["id"] == "ack"
    assert retriever.search("致谢", top_k=1)[0]["id"] == "ack"
    assert retriever.search("论文研究主题", top_k=1, page=97)[0]["id"] == "ack"


def test_legacy_manifest_is_classified_without_changing_citation_metadata(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"chunks": [{"id": "old", "text": "致谢 感谢指导老师", "page": 9, "source_type": "text", "image_url": "images/p.png", "table_url": None}]}),
        encoding="utf-8",
    )
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = manifest
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()
    result = retriever.search("论文主题", top_k=1)[0]
    assert result["id"] == "old"
    assert result["page"] == 9
    assert result["source_type"] == "text"
    assert result["image_url"] == "images/p.png"
    assert result["section_role"] == "acknowledgements"


def test_retriever_filters_page_before_top_k(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "chunks": [
                    {"id": "text-p1-t0", "text": "猫舍管理", "page": 1},
                    {"id": "text-p2-t0", "text": "鸡舍管理", "page": 2},
                    {"id": "text-p2-t1", "text": "鸡舍安全管理", "page": 2},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = manifest
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()

    results = retriever.search("管理", top_k=2, page=2)

    assert len(results) == 2
    assert {item["page"] for item in results} == {2}


def test_retriever_skips_duplicate_and_malformed_chunks(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "chunks": [
                    {"id": "same", "text": "有效内容", "page": 1},
                    {"id": "same", "text": "重复内容", "page": 1},
                    {"id": "bad", "text": "缺页"},
                ]
            }
        ),
        encoding="utf-8",
    )
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = manifest
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()

    assert retriever.documents[0].metadata["chunk_id"] == "same"
    assert retriever.invalid_chunk_count == 2
    assert retriever.manifest_valid is True


def test_corrupt_manifest_is_reported_without_discarding_loaded_index(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"chunks": [', encoding="utf-8")
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = manifest
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()
    assert retriever.load_error == "manifest_invalid"
    assert retriever.manifest_valid is False

    manifest.write_text(
        json.dumps({"chunks": [{"id": "new", "text": "新内容", "page": 2}]}),
        encoding="utf-8",
    )
    retriever.reload()
    previous_documents = retriever.documents
    manifest.write_text("not json", encoding="utf-8")
    retriever.reload()

    assert retriever.documents is previous_documents
    assert retriever.documents[0].metadata["chunk_id"] == "new"
    assert retriever.load_error == "manifest_invalid"




def _doc(chunk_id: str, page: int = 1) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id, "page": page})


def test_rrf_merges_and_deduplicates_bm25_and_vector_results():
    bm25 = [_doc("shared"), _doc("keyword")]
    vector = [_doc("shared"), _doc("semantic")]

    ranked = Retriever._rrf_order(bm25, vector)

    assert [doc.metadata["chunk_id"] for doc in ranked] == ["shared", "keyword", "semantic"]


def test_hybrid_search_uses_rrf_and_page_filter():
    retriever = Retriever.__new__(Retriever)
    retriever.documents = [_doc("bm25"), _doc("vector"), _doc("other", page=2)]
    retriever.bm25 = Mock()
    retriever.bm25.invoke.return_value = retriever.documents
    retriever.vector_store = Mock()
    retriever.vector_store.similarity_search.return_value = [_doc("vector"), _doc("other", page=2)]
    retriever.mode = "hybrid"
    retriever.retrieval_degraded = False
    retriever.retrieval_error = None

    results = retriever.search("question", top_k=2, page=1)

    assert [item["id"] for item in results] == ["vector", "bm25"]
    retriever.vector_store.similarity_search.assert_called_once()


def test_vector_search_failure_keeps_bm25_results():
    retriever = Retriever.__new__(Retriever)
    retriever.documents = [_doc("bm25")]
    retriever.bm25 = Mock()
    retriever.bm25.invoke.return_value = retriever.documents
    retriever.vector_store = Mock()
    retriever.vector_store.similarity_search.side_effect = RuntimeError("offline")
    retriever.mode = "hybrid"
    retriever.retrieval_degraded = False
    retriever.retrieval_error = None

    results = retriever.search("question", top_k=1)

    assert [item["id"] for item in results] == ["bm25"]
    assert retriever.mode == "bm25"
    assert retriever.retrieval_degraded is True
    assert retriever.retrieval_error == "RuntimeError"


def test_vector_sync_only_updates_changed_chunks_and_removes_deleted(tmp_path):
    retriever = Retriever.__new__(Retriever)
    retriever._vector_sync_path = tmp_path / "sync.json"
    retriever._embedding_fingerprint = lambda: "model-v1"
    retriever._vector_sync_path.write_text(
        json.dumps({"embedding_fingerprint": "model-v1", "chunks": {"same": "old", "deleted": "x"}}),
        encoding="utf-8",
    )
    store = Mock()
    documents = [_doc("same")]
    retriever._sync_vector_store(store, documents)

    store.reset_collection.assert_not_called()
    store.delete.assert_called_once_with(ids=["deleted"])
    store.add_documents.assert_called_once()
    assert json.loads(retriever._vector_sync_path.read_text(encoding="utf-8"))["chunks"].keys() == {"same"}
    retriever = Retriever.__new__(Retriever)
    retriever.manifest_path = tmp_path / "missing.json"
    retriever.documents = []
    retriever.bm25 = None
    retriever.vector_store = None
    retriever.mode = "none"
    retriever.reload()

    assert retriever.load_error == "manifest_missing"
    assert retriever.indexed is False
