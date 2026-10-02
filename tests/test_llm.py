from copy import deepcopy
from unittest.mock import Mock

from backend.llm import LLMClient, build_evidence_answer


def test_build_evidence_answer_lists_verbatim_page_evidence():
    contexts = [
        {"page": 3, "source_type": "text", "text": "系统由三个模块组成。"},
        {"page": 5, "source_type": "table", "text": "| 模块 | 功能 |"},
    ]
    original = deepcopy(contexts)

    answer = build_evidence_answer("系统架构如何设计？", contexts)

    assert "当前未配置在线语言模型" in answer
    assert "不是生成式结论" in answer
    assert "[p.3]（text）" in answer
    assert "[p.5]（table）" in answer
    assert "系统由三个模块组成。" in answer
    assert contexts == original


def test_build_evidence_answer_handles_empty_contexts():
    assert build_evidence_answer("问题", []) == "当前没有检索到可用的论文证据。"


def test_llm_client_uses_configured_dashscope(monkeypatch):
    from backend import llm

    constructor = Mock()
    fake_chat = Mock()
    fake_chat.invoke.return_value.content = "测试回答"
    constructor.return_value = fake_chat
    monkeypatch.setattr(llm, "ChatOpenAI", constructor)
    prompt = Mock()
    chain = Mock()
    chain.invoke.return_value.content = "测试回答"
    prompt.__or__ = Mock(return_value=chain)
    monkeypatch.setattr(llm, "PROMPT", prompt)
    monkeypatch.setattr(llm.settings, "openai_api_key", "dash-key")
    monkeypatch.setattr(llm.settings, "dashscope_api_key", "dash-key")
    monkeypatch.setattr(llm.settings, "openai_base_url", "https://maas.qianwenaiapi.com/compatible-mode/v1")
    monkeypatch.setattr(llm.settings, "chat_model", "qwen3.8-omni-flash")

    client = LLMClient()
    answer = client.answer("问题", [{"text": "证据", "page": 1}])

    assert answer == "测试回答"
    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["model"] == "qwen3.8-omni-flash"
    assert kwargs["api_key"] == "dash-key"
    assert kwargs["base_url"] == "https://maas.qianwenaiapi.com/compatible-mode/v1"
