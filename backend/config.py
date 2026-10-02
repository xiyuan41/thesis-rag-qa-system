from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "基于计算机图像识别的家禽养殖及安防系统的设计与实现 (2).pdf"
DATA_DIR = PROJECT_ROOT / "app_data"


class Settings(BaseSettings):
    # Keep OPENAI_API_KEY for generic OpenAI-compatible services. DashScope is
    # an explicit chat-only fallback and is not reused for embeddings.
    openai_api_key: str = ""
    dashscope_api_key: str = ""
    openai_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    chat_model: str = "qwen3.8-omni-flash"
    embedding_model: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    frontend_api_url: str = "http://127.0.0.1:8000"
    top_k: int = 6
    chunk_size: int = 1200
    chunk_overlap: int = 180
    enable_ocr: bool = False
    enable_vector: bool = True
    vector_collection: str = "paper_chunks"
    resource_token: str = ""

    @property
    def chat_api_key(self) -> str:
        return self.openai_api_key or self.dashscope_api_key

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

for directory in (DATA_DIR, DATA_DIR / "images", DATA_DIR / "tables", DATA_DIR / "chroma"):
    directory.mkdir(parents=True, exist_ok=True)
