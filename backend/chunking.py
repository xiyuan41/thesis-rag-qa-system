from __future__ import annotations

import re
from typing import Any


def classify_section_role(text: str) -> str:
    """Infer a page-level section role from its heading and layout cues."""
    lines = [re.sub(r"\s+", "", line).strip("：: ") for line in text.splitlines() if line.strip()]
    opening = "".join(lines[:5])
    normalized = re.sub(r"\s+", "", text)

    if re.search(r"^(致谢|谢辞|鸣谢)(?:$|[：:]|[0-9一二三四五六七八九十])", opening) or re.search(r"^(致谢|谢辞|鸣谢)\s", text):
        return "acknowledgements"
    if re.search(r"^(参考文献|引用文献|主要参考文献)(?:$|[：:]|[0-9一二三四五六七八九十])", opening) or re.search(r"^(参考文献|引用文献|主要参考文献)\s", text):
        return "references"
    if re.search(r"^(目录|目次)(?:$|[：:]|[0-9一二三四五六七八九十])", opening) or re.search(r"^(目录|目次)\s", text):
        return "toc"
    if re.search(r"^(摘要|摘\s*要)(?:$|[：:]|[0-9一二三四五六七八九十IVXivx])", opening) or re.search(r"^摘要\s", text):
        return "abstract"
    if re.search(r"^(毕业设计原创性声明|原创性声明|版权使用授权书|授权书)", opening):
        return "front_matter"
    if re.search(r"(?:\.{2,}|…{2,}|·{2,})\s*\d{1,3}", normalized) and len(re.findall(r"(?:\.{2,}|…{2,}|·{2,})\s*\d{1,3}", normalized)) >= 2:
        return "toc"
    return "body"


def split_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind("；", start, end), text.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def build_chunks(pages: list[dict[str, Any]], size: int, overlap: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in pages:
        source_type = page.get("source_type", "text")
        source_key = page.get("source_key", source_type)
        for index, text in enumerate(split_text(page["text"], size, overlap)):
            result.append({
                "id": f"{source_key}-p{page['page']}-t{index}",
                "text": text,
                "page": page["page"],
                "source_type": source_type,
                "image_url": page.get("image_url"),
                "table_url": page.get("table_url"),
                **({"section_role": page["section_role"]} if page.get("section_role") else {}),
            })
    return result
