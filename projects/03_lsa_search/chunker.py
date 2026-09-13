"""Markdown 分块器：按标题层级切块，标题路径并入正文（标题是强检索信号）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.M)


@dataclass
class Chunk:
    doc: str          # 来源文档名
    section: str      # 标题路径（如 项目五 · 检验一）
    text: str

    @property
    def label(self) -> str:
        return f"{self.doc} · {self.section}"


def chunk_markdown(doc_name: str, text: str, min_chars: int = 40,
                   max_chars: int = 1200) -> list[Chunk]:
    """按 1-4 级标题切块；块的正文前拼上标题路径；超长块按段落二切。"""
    text = re.sub(r"<[^>]+>", "", text)                    # 去 HTML 标签（badge 等）
    lines = text.splitlines(keepends=True)

    # 扫描标题行，记录 (行号, 级别, 标题)
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    doc_title = headings[0][2].lstrip("#").strip() if headings else doc_name
    chunks: list[Chunk] = []

    for idx, (line_no, level, title) in enumerate(headings):
        end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "".join(lines[line_no + 1:end_line]).strip()
        if not body:
            continue
        # 层级路径：沿更高级标题向上回溯
        path = [title]
        for prev_level, prev_title in [(lv, t) for _, lv, t in headings[:idx]][::-1]:
            if prev_level < level:
                path.insert(0, prev_title)
                level = prev_level
                if prev_level == 1:
                    break
        section = " · ".join(path)
        full = f"{section}\n{body}"
        for piece in _split_long(full, max_chars):
            if len(piece) >= min_chars:
                chunks.append(Chunk(doc=doc_name, section=section, text=piece))

    # 标题前的前言
    pre_end = headings[0][0] if headings else len(lines)
    intro = "".join(lines[:pre_end]).strip()
    if len(intro) >= min_chars:
        chunks.insert(0, Chunk(doc=doc_name, section=f"{doc_title} · 简介",
                               text=f"{doc_title}\n{intro}"))
    return chunks or _fallback(doc_name, text, min_chars)


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"(\n\s*\n)", text):
        if len(buf) + len(para) <= max_chars or not buf:
            buf += para
        else:
            parts.append(buf.strip())
            buf = para
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _fallback(doc_name: str, text: str, min_chars: int) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_chars]
    return [Chunk(doc=doc_name, section="（全文）", text=p) for p in paras]


def chunk_corpus(docs: dict[str, str]) -> list[Chunk]:
    out: list[Chunk] = []
    for name, text in docs.items():
        out.extend(chunk_markdown(name, text))
    return out
