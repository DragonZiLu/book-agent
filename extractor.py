"""
文本提取器：支持 PDF/EPUB/URL，章节分割 + 容错降级
PDF → pypdf；EPUB → ebooklib；URL → requests 下载后走 PDF 流程
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """单个章节"""
    index: int
    title: str
    content: str
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            # 粗略估算：中文约 1.5 字符/token，英文约 4 字符/token
            # 取折中 ~2.5 字符/token
            self.token_count = max(1, len(self.content) // 3)


@dataclass
class ExtractionResult:
    """提取结果"""
    chapters: List[Chapter]
    warnings: List[str] = field(default_factory=list)


class BookExtractor:
    """书籍文本提取器，支持 PDF/EPUB/URL，含章节分割与容错"""

    # 章节标题匹配模式（按优先级排列）
    CHAPTER_PATTERNS = [
        # "第N章" / "第N节" / "第十四章"
        re.compile(r"(第[一二三四五六七八九十百千\d]+[章节])"),
        # "Chapter N" / "Chapter One" (英文数字)
        re.compile(r"(Chapter\s+(\d+|[A-Z][a-z]+))", re.IGNORECASE),
        # "CHAPTER FOURTEEN" (全大写)
        re.compile(r"(CHAPTER\s+(\d+|[A-Z]+))"),
        # "Part N" / "Section N"
        re.compile(r"((Part|Section)\s+(\d+|[IVX]+))", re.IGNORECASE),
        # Markdown 风格标题 "## Chapter N"
        re.compile(r"(#+\s+.+)"),
    ]

    TOP_LEVEL_HEADINGS = {
        "第",
        "Chapter",
        "CHAPTER",
        "Part",
        "Section",
    }

    def extract(self, source: str) -> ExtractionResult:
        """
        从文件路径或 URL 提取文本并分割为章节。
        source: 本地文件路径（.pdf/.epub）或 HTTP URL
        """
        source_lower = source.lower()
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            return self._extract_url(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {source}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".epub":
            return self._extract_epub(path)
        elif suffix == ".txt":
            return self._extract_txt(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，支持 .pdf / .epub / .txt")

    def _extract_url(self, url: str) -> ExtractionResult:
        """下载 URL 内容后提取（优先尝试 PDF，失败走纯文本）"""
        import tempfile

        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        suffix = ".pdf" if "pdf" in content_type else ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
            tmp_path = Path(f.name)

        try:
            return self.extract(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        """从 PDF 提取文本"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf>=4.0.0")

        reader = PdfReader(str(path))
        full_text_parts: List[str] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        if not full_text.strip():
            logger.warning(
                f"PDF 提取为空，可能需要 OCR 或尝试 pdfminer.six。"
                f"页码数: {len(reader.pages)}"
            )

        chapters = self._split_chapters(full_text)
        warnings = self._validate_chapters(chapters)
        return ExtractionResult(chapters=chapters, warnings=warnings)

    def _extract_epub(self, path: Path) -> ExtractionResult:
        """从 EPUB 提取文本"""
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("请安装 ebooklib + beautifulsoup4")

        book = epub.read_epub(str(path))
        full_text_parts: List[str] = []

        for item in book.get_items_of_type(9):  # ITEM_DOCUMENT = 9
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)
        chapters = self._split_chapters(full_text)
        warnings = self._validate_chapters(chapters)
        return ExtractionResult(chapters=chapters, warnings=warnings)

    def _extract_txt(self, path: Path) -> ExtractionResult:
        """从纯文本文件提取"""
        full_text = path.read_text(encoding="utf-8")
        chapters = self._split_chapters(full_text)
        warnings = self._validate_chapters(chapters)
        return ExtractionResult(chapters=chapters, warnings=warnings)

    def _split_chapters(self, text: str) -> List[Chapter]:
        """
        优先按标题匹配分割章节；
        失败时降级为滑动窗口分块。
        """
        lines = text.split("\n")
        # 找出所有可能的章节标题行
        heading_indices: List[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            matched_title = self._match_heading(stripped)
            if matched_title:
                heading_indices.append((i, matched_title))

        if not heading_indices:
            logger.info("未检测到章节标题，降级为滑动分块")
            return self._sliding_window_split(text)

        chapters: List[Chapter] = []
        # 处理第一个标题前的内容（前言）
        if heading_indices[0][0] > 0:
            preamble = "\n".join(lines[: heading_indices[0][0]]).strip()
            if preamble:
                chapters.append(Chapter(
                    index=0, title="前言",
                    content=preamble[:from_config("MAX_CHAPTER_CHARS", 12000)],
                ))

        for idx, (start, title) in enumerate(heading_indices):
            end = heading_indices[idx + 1][0] if idx + 1 < len(heading_indices) else len(lines)
            content = "\n".join(lines[start:end]).strip()
            if content:
                max_chars = from_config("MAX_CHAPTER_CHARS", 12000)
                truncated = content[:max_chars]
                chapters.append(Chapter(
                    index=idx + 1,
                    title=title,
                    content=truncated,
                ))

        return chapters

    def _match_heading(self, line: str) -> Optional[str]:
        """检查一行文本是否为章节标题，是则返回标题文本"""
        # 过滤太长的行（不太可能是标题）
        if len(line) > 200:
            return None

        for pattern in self.CHAPTER_PATTERNS:
            m = pattern.search(line)
            if m:
                # 确保匹配在行首附近（标题不太可能出现在很后面）
                if m.start() < 50:
                    return m.group(0)
        return None

    def _sliding_window_split(self, text: str) -> List[Chapter]:
        """
        滑动窗口分块（token 级降级方案）。
        每块大约 SLIDING_WINDOW_SIZE tokens，重叠 SLIDING_WINDOW_OVERLAP tokens。
        """
        window_size = from_config("SLIDING_WINDOW_SIZE", 3000)
        overlap = from_config("SLIDING_WINDOW_OVERLAP", 200)

        # 估算字符数（~3 字符/token）
        chunk_chars = window_size * 3
        overlap_chars = overlap * 3
        step = chunk_chars - overlap_chars

        if step <= 0:
            step = chunk_chars

        chapters: List[Chapter] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chapters.append(Chapter(
                    index=idx + 1,
                    title=f"Block {idx + 1}",
                    content=chunk,
                ))
            start += step
            idx += 1

        return chapters

    def _validate_chapters(self, chapters: List[Chapter]) -> List[str]:
        """章节质量告警"""
        warnings: List[str] = []

        if len(chapters) <= 1:
            warnings.append(
                f"⚠️ 只有 {len(chapters)} 个章节，可能章节分割失败，已降级为分块模式"
            )
            return warnings

        token_counts = [ch.token_count for ch in chapters]
        avg_tokens = sum(token_counts) / len(token_counts)

        for ch in chapters:
            if ch.token_count < 200:
                warnings.append(
                    f"⚠️ 章节过短: 第{ch.index}章「{ch.title}」仅 {ch.token_count} tokens"
                )
            if avg_tokens > 0 and ch.token_count > avg_tokens * 5:
                warnings.append(
                    f"⚠️ 章节过大: 第{ch.index}章「{ch.title}」"
                    f" {ch.token_count} tokens (均值 {avg_tokens:.0f})"
                )

        return warnings


def from_config(key: str, default):
    """从 config 模块读取配置值，避免循环导入"""
    try:
        import config
        return getattr(config, key, default)
    except ImportError:
        return default
