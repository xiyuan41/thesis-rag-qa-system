from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_shape():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert {"pdf", "indexed", "chunks", "pages"}.issubset(response.json())
    assert response.json()["details"]["answer_mode"] in {"local_evidence", "llm"}
    assert "readiness" in response.json()["details"]
    assert "manifest_exists" in response.json()["details"]


def test_query_rejects_empty_index(monkeypatch):
    class EmptyRetriever:
        indexed = False

    monkeypatch.setattr("backend.main.retriever", EmptyRetriever())
    response = client.post("/api/query", json={"question": "系统架构"})
    assert response.status_code == 409


def test_resource_url_rejects_missing_and_traversal():
    from backend.main import _resource_url

    assert _resource_url("../manifest.json", "images") is None
    assert _resource_url("missing.png", "images") is None
    assert _resource_url("anything", "other") is None


def test_query_returns_local_evidence_without_model(monkeypatch):
    class FakeRetriever:
        indexed = True

        def search(self, question, top_k=None, page=None):
            return [
                {
                    "id": "text-p3-t0",
                    "text": "架构证据",
                    "page": 3,
                    "source_type": "text",
                    "image_url": None,
                    "table_url": None,
                    "score": None,
                }
            ]

    class UnexpectedLLMClient:
        def __init__(self):
            raise AssertionError("本地证据模式不应实例化 LLM 客户端")

    monkeypatch.setattr("backend.main.retriever", FakeRetriever())
    monkeypatch.setattr("backend.main.settings.openai_api_key", "")
    monkeypatch.setattr("backend.main.settings.dashscope_api_key", "")
    monkeypatch.setattr("backend.main.settings.chat_model", "")
    monkeypatch.setattr("backend.main.LLMClient", UnexpectedLLMClient)

    response = client.post("/api/query", json={"question": "系统架构"})

    assert response.status_code == 200
    body = response.json()
    assert "当前未配置在线语言模型" in body["answer"]
    assert "[p.3]（text）" in body["answer"]
    assert body["retrieval_count"] == 1
    assert body["citations"][0]["chunk_id"] == "text-p3-t0"


def test_query_returns_503_when_configured_model_fails(monkeypatch):
    class FakeRetriever:
        indexed = True

        def search(self, question, top_k=None, page=None):
            return [{
                "id": "text-p3-t0",
                "text": "架构证据",
                "page": 3,
                "source_type": "text",
                "image_url": None,
                "table_url": None,
            }]

    class BrokenLLMClient:
        def answer(self, question, contexts):
            from backend.llm import LLMError
            raise LLMError("服务不可用")

    monkeypatch.setattr("backend.main.retriever", FakeRetriever())
    monkeypatch.setattr("backend.main.settings.openai_api_key", "test-key")
    monkeypatch.setattr("backend.main.settings.chat_model", "test-model")
    monkeypatch.setattr("backend.main.LLMClient", BrokenLLMClient)

    response = client.post("/api/query", json={"question": "系统架构"})

    assert response.status_code == 503
    assert "在线模型服务调用失败" in response.json()["detail"]


def test_query_returns_normalized_citations(monkeypatch):
    class FakeRetriever:
        indexed = True

        def search(self, question, top_k=None, page=None):
            assert question == "系统架构"
            assert top_k == 2
            assert page == 3
            return [
                {
                    "id": "text-p3-t0",
                    "text": "架构证据",
                    "page": 3,
                    "source_type": "text",
                    "image_url": None,
                    "table_url": None,
                    "score": None,
                }
            ]

    monkeypatch.setattr("backend.main.retriever", FakeRetriever())

    class FakeLLMClient:
        def answer(self, question, contexts):
            return "答案 [p.3]"

    monkeypatch.setattr("backend.main.settings.openai_api_key", "test-key")
    monkeypatch.setattr("backend.main.settings.chat_model", "test-model")
    monkeypatch.setattr("backend.main.LLMClient", FakeLLMClient)

    response = client.post(
        "/api/query", json={"question": "系统架构", "top_k": 2, "page": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_count"] == 1
    assert body["citations"][0]["chunk_id"] == "text-p3-t0"
    assert body["answer"] == "答案 [p.3]"
