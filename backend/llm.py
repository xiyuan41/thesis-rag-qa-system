from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import settings

SYSTEM_PROMPT = """你是论文内容查询与解释助手。只能依据提供的论文证据回答，使用中文。
回答时区分“论文原文证据”和“你的解释”，不得把证据中没有的信息当作事实。
每个关键结论后标注页码，如 [p.3]。如果证据不足，明确说“论文提供的证据不足”，不要编造。
对表格、图片和算法先描述论文中可见内容，再给出通俗解释。"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "论文证据：\n{context}\n\n问题：{question}"),
])


class LLMError(RuntimeError):
    pass


def build_evidence_answer(question: str, contexts: list[dict[str, Any]]) -> str:
    """Return a deterministic, non-generative answer for offline use."""
    if not contexts:
        return "当前没有检索到可用的论文证据。"
    lines = [
        "当前未配置在线语言模型，以下内容仅为检索到的论文原文证据，不是生成式结论。",
        f"问题：{question}",
        "",
        "检索证据：",
    ]
    for index, item in enumerate(contexts, start=1):
        page = item.get("page", "?")
        source_type = item.get("source_type", "text")
        text = str(item.get("text", "")).strip()
        if text:
            lines.append(f"{index}. [p.{page}]（{source_type}）\n{text}")
    return "\n\n".join(lines)


class LLMClient:
    def __init__(self):
        if not settings.chat_api_key or not settings.chat_model:
            raise LLMError("未配置聊天模型密钥或 CHAT_MODEL，请填写 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。")
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            base_url=settings.openai_base_url,
            temperature=0.2,
            timeout=120,
            max_retries=2,
        )

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> str:
        documents = [
            Document(
                page_content=item["text"],
                metadata={"page": item["page"], "source_type": item.get("source_type", "text")},
            )
            for item in contexts
        ]
        context = "\n\n".join(
            f"[p.{document.metadata['page']}]（{document.metadata['source_type']}）\n{document.page_content}"
            for document in documents
        )
        try:
            chain = PROMPT | self.llm
            response = chain.invoke({"context": context, "question": question})
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as exc:
            raise LLMError(f"调用 OpenAI-compatible 服务失败: {exc}") from exc
