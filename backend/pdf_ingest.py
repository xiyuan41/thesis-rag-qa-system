from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import fitz

from .chunking import build_chunks, classify_section_role
from .config import DATA_DIR, PDF_PATH, settings


class PDFIngestor:
    def __init__(self, pdf_path: Path = PDF_PATH):
        self.pdf_path = pdf_path
        self.images_dir = DATA_DIR / "images"
        self.tables_dir = DATA_DIR / "tables"
        self.manifest_path = DATA_DIR / "manifest.json"

    def ingest(self, force: bool = False) -> dict[str, Any]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"论文 PDF 不存在: {self.pdf_path}")

        digest = hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()
        old: dict[str, Any] | None = None
        if self.manifest_path.exists():
            try:
                old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old = None
            if not force and old and old.get("sha256") == digest and old.get("chunks"):
                generated = old.get("generated_images", []) + old.get("generated_tables", [])
                if all((DATA_DIR / relative_path).is_file() for relative_path in generated):
                    return old

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        pages: list[dict[str, Any]] = []
        warnings: list[str] = []
        image_count = 0

        with fitz.open(self.pdf_path) as document:
            repeated: dict[str, int] = {}
            raw_pages: list[dict[str, Any]] = []
            for page_number, page in enumerate(document, start=1):
                blocks = page.get_text("blocks")
                text_parts: list[str] = []
                for block in blocks:
                    value = block[4].strip()
                    if value:
                        text_parts.append(value)
                        normalized = re.sub(r"\s+", " ", value)
                        repeated[normalized] = repeated.get(normalized, 0) + 1

                raw_pages.append(
                    {
                        "page": page_number,
                        "text": "\n".join(text_parts),
                    }
                )

                for image_index, image in enumerate(page.get_images(full=True)):
                    try:
                        image_data = document.extract_image(image[0])
                        suffix = image_data.get("ext", "png")
                        image_name = f"page-{page_number}-image-{image_index}.{suffix}"
                        image_path = self.images_dir / image_name
                        image_path.write_bytes(image_data["image"])
                        image_count += 1
                    except Exception as exc:
                        warnings.append(f"第 {page_number} 页图片提取失败: {exc}")

            repeated_headers = {
                key for key, count in repeated.items() if count >= 3 and len(key) < 150
            }
            for raw in raw_pages:
                lines = [line.strip() for line in raw["text"].splitlines() if line.strip()]
                retained = [
                    line
                    for line in lines
                    if re.sub(r"\s+", " ", line) not in repeated_headers
                ]
                page_text = "\n".join(retained) or raw["text"]
                page_images = sorted(self.images_dir.glob(f"page-{raw['page']}-image-*"))
                pages.append(
                    {
                        "page": raw["page"],
                        "text": page_text,
                        "source_type": "text",
                        "source_key": "text",
                        "section_role": classify_section_role(page_text),
                        "image_url": str(page_images[0].relative_to(DATA_DIR)) if page_images else None,
                    }
                )

        chunks = build_chunks(pages, settings.chunk_size, settings.chunk_overlap)
        image_by_page = {page["page"]: page.get("image_url") for page in pages}
        for chunk in chunks:
            chunk["image_url"] = image_by_page.get(chunk["page"])

        tables = self._extract_tables(warnings)
        for table in tables:
            chunks.extend(build_chunks([table], settings.chunk_size, settings.chunk_overlap))

        if settings.enable_ocr:
            self._add_ocr_chunks(pages, chunks, warnings)

        generated_images = {
            page["image_url"]
            for page in pages
            if page.get("image_url")
        }
        generated_tables = {
            table["table_url"]
            for table in tables
            if table.get("table_url")
        }
        current_resources = generated_images | generated_tables
        previous_resources = set(old.get("generated_images", []) + old.get("generated_tables", [])) if old else set()
        for relative_path in previous_resources - current_resources:
            candidate = (DATA_DIR / relative_path).resolve()
            if candidate.parent in {self.images_dir.resolve(), self.tables_dir.resolve()}:
                candidate.unlink(missing_ok=True)

        manifest = {
            "sha256": digest,
            "pdf": self.pdf_path.name,
            "pages": len(pages),
            "images": image_count,
            "tables": len(tables),
            "generated_images": sorted(generated_images),
            "generated_tables": sorted(generated_tables),
            "chunks": chunks,
            "warnings": warnings,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="manifest-", suffix=".json", dir=self.manifest_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                json.dump(manifest, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, self.manifest_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return manifest

    def _extract_tables(self, warnings: list[str]) -> list[dict[str, Any]]:
        try:
            import pdfplumber
        except ImportError:
            warnings.append("未安装 pdfplumber，表格暂按页面文本检索。")
            return []

        tables: list[dict[str, Any]] = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    for index, table in enumerate(page.extract_tables() or [], start=1):
                        rows = [
                            [str(cell or "").replace("|", "\\|") for cell in row]
                            for row in table
                            if row
                        ]
                        if len(rows) < 2 or not rows[0]:
                            continue
                        width = len(rows[0])
                        rows = [row + [""] * (width - len(row)) for row in rows]
                        markdown = "\n".join(
                            "| " + " | ".join(row[:width]) + " |" for row in rows
                        )
                        markdown += "\n| " + " | ".join(["---"] * width) + " |"
                        path = self.tables_dir / f"page-{page_number}-table-{index}.md"
                        path.write_text(markdown, encoding="utf-8")
                        tables.append(
                            {
                                "page": page_number,
                                "text": f"表格（第 {page_number} 页）\n{markdown}",
                                "source_type": "table",
                                "source_key": f"table-{index}",
                                "table_url": str(path.relative_to(DATA_DIR)),
                            }
                        )
        except Exception as exc:
            warnings.append(f"表格解析失败，已保留页面文本: {exc}")
        return tables

    def _add_ocr_chunks(
        self,
        pages: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            warnings.append("已开启 OCR，但未安装 rapidocr-onnxruntime；跳过 OCR。")
            return

        try:
            ocr = RapidOCR()
            with fitz.open(self.pdf_path) as document:
                image_by_page = {page["page"]: page.get("image_url") for page in pages}
                for page_number, page in enumerate(document, start=1):
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    result, _ = ocr(pixmap.tobytes("png"))
                    text = "\n".join(
                        str(item[1]) for item in (result or []) if len(item) > 1
                    )
                    if text.strip():
                        ocr_chunks = build_chunks(
                            [
                                {
                                    "page": page_number,
                                    "text": text,
                                    "source_type": "image_ocr",
                                    "source_key": "ocr",
                                    "image_url": image_by_page.get(page_number),
                                }
                            ],
                            settings.chunk_size,
                            settings.chunk_overlap,
                        )
                        chunks.extend(ocr_chunks)
        except Exception as exc:
            warnings.append(f"OCR 执行失败，已保留原始文本: {exc}")
