import pytest

from backend.chunking import build_chunks, classify_section_role, split_text


def test_section_role_classification_uses_heading_and_layout():
    assert classify_section_role("致谢 83\n致 谢\n感谢老师的指导") == "acknowledgements"
    assert classify_section_role("摘要 I\n摘 要\n本研究设计了一个系统") == "abstract"
    assert classify_section_role("目录\n第一章 绪论......1\n第二章 设计......8") == "toc"
    assert classify_section_role("正文提到参考文献格式和致谢安排，但本节讨论系统设计") == "body"


def test_build_chunks_passes_section_role():
    chunks = build_chunks([{"page": 7, "text": "摘要内容", "section_role": "abstract"}], 100, 10)
    assert chunks[0]["section_role"] == "abstract"


def test_chunk_ids_include_source_and_are_stable():
    pages = [
        {"page": 2, "text": "正文内容", "source_type": "text"},
        {"page": 2, "text": "表格内容", "source_type": "table", "source_key": "table-1"},
        {"page": 2, "text": "图片文字", "source_type": "image_ocr", "source_key": "ocr"},
    ]
    first = build_chunks(pages, size=100, overlap=10)
    second = build_chunks(pages, size=100, overlap=10)
    assert [chunk["id"] for chunk in first] == [chunk["id"] for chunk in second]
    assert len({chunk["id"] for chunk in first}) == 3
    assert {chunk["id"] for chunk in first} == {
        "text-p2-t0", "table-1-p2-t0", "ocr-p2-t0"
    }


def test_build_chunks_empty_text():
    assert build_chunks([{"page": 1, "text": ""}], 100, 10) == []


    text = "第一段。" * 100
    chunks = split_text(text, size=60, overlap=10)
    assert len(chunks) > 1
    assert "第一段" in "".join(chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("text", size=10, overlap=10)
