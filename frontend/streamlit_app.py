from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("FRONTEND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="论文内容查询智能体", page_icon="📚", layout="wide")
st.title("📚 论文内容查询与解释智能体")
st.caption("基于 RAG 检索论文文字、图片与表格相关证据，并给出带页码引用的解释")


def request_error_message(response: requests.Response, action: str) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    messages = {
        409: "索引尚未建立，请先点击侧栏的“重新建立索引”。",
        404: "没有找到相关证据，请换一种问法或取消页码筛选。",
        503: "在线模型服务暂时不可用，请检查模型配置；检索到的证据仍可用于本地查看。",
    }
    return messages.get(response.status_code, f"{action}失败（HTTP {response.status_code}）：{detail}")


def show_citation(citation: dict[str, Any]) -> None:
    st.markdown(f"**第 {citation['page']} 页 · {citation['source_type']}**")
    st.write(citation["text"])
    if citation.get("image_url"):
        st.image(f"{API_URL}{citation['image_url']}", caption=f"第 {citation['page']} 页图片")
    if citation.get("table_url"):
        table_url = f"{API_URL}{citation['table_url']}"
        st.markdown(f"[打开表格 Markdown]({table_url})")


with st.sidebar:
    st.header("索引管理")
    try:
        status = requests.get(f"{API_URL}/api/status", timeout=10).json()
        details = status.get("details", {})
        st.write(f"文档：{status.get('pdf', '未发现')}")
        st.write(f"页数：{status.get('pages', 0)} · 文本块：{status.get('chunks', 0)}")
        st.write(f"检索模式：{details.get('retrieval_mode', 'unknown')}")
        st.write(f"回答模式：{details.get('answer_mode', 'unknown')}")
        readiness = details.get("readiness", "unknown")
        readiness_labels = {
            "ready": "可查询",
            "not_indexed": "尚未建立索引",
            "manifest_error": "索引加载异常",
            "source_missing": "论文 PDF 不存在",
        }
        st.write(f"索引状态：{readiness_labels.get(readiness, readiness)}")
        if readiness == "manifest_error":
            st.warning("索引文件无法读取，请点击“重新建立索引”。")
        elif readiness == "source_missing":
            st.error("未找到论文 PDF，请检查项目目录。")
        if details.get("retrieval_degraded"):
            st.warning("向量检索不可用，当前已降级为 BM25。")
        st.write(
            f"资源：图片 {details.get('images', 0)} · 表格 {details.get('tables', 0)}"
        )
        if details.get("warnings"):
            with st.expander("处理提示"):
                for warning in details["warnings"]:
                    st.warning(warning)
    except (requests.RequestException, ValueError) as exc:
        st.error(f"无法连接 FastAPI，请先启动后端：{exc}")
        status = {}
    if st.button("重新建立索引", use_container_width=True):
        with st.spinner("正在解析论文，请稍候…"):
            try:
                response = requests.post(
                    f"{API_URL}/api/ingest", params={"force": True}, timeout=300
                )
                if response.status_code >= 400:
                    raise RuntimeError(request_error_message(response, "索引"))
                st.success(f"索引完成：{response.json()['indexed']} 个文本块")
                st.rerun()
            except (requests.RequestException, ValueError) as exc:
                st.error(f"索引失败：{exc}")
    top_k = st.slider("召回证据数量", 1, 12, 6)
    page_filter = st.number_input("限定页码（0 表示全部）", min_value=0, value=0, step=1)

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander(f"查看 {len(message['citations'])} 条论文证据"):
                for citation in message["citations"]:
                    show_citation(citation)

question = st.chat_input("例如：请解释论文中系统架构的设计思路")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在检索论文并生成解释…"):
            try:
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "page": page_filter or None,
                    },
                    timeout=180,
                )
                data = response.json()
                if response.status_code >= 400:
                    raise RuntimeError(request_error_message(response, "查询"))
                st.markdown(data["answer"])
                with st.expander(f"查看 {len(data['citations'])} 条论文证据"):
                    for citation in data["citations"]:
                        show_citation(citation)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "citations": data["citations"],
                    }
                )
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                error = f"查询失败：{exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
