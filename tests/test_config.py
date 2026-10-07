from backend.config import Settings


def test_dashscope_defaults_and_key_fallback(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    settings.dashscope_api_key = "dash-key"

    assert settings.openai_base_url == "https://maas.qianwenaiapi.com/compatible-mode/v1"
    assert settings.chat_model == "qwen3.8-omni-flash"
    assert settings.chat_api_key == "dash-key"


def test_embedding_defaults_to_local_cpu_and_is_separate_from_chat_keys():
    settings = Settings(_env_file=None, dashscope_api_key="chat-only")
    assert settings.embedding_model == ""
    assert settings.embedding_device == "cpu"
    assert settings.embedding_cache_dir == ""
    assert settings.chat_api_key == "chat-only"


def test_explicit_openai_key_takes_precedence():
    settings = Settings(
        _env_file=None,
        openai_api_key="openai-key",
        dashscope_api_key="dash-key",
    )
    assert settings.chat_api_key == "openai-key"
