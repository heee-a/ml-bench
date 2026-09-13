"""BM25 检索器：从零实现（Okapi BM25），中文 jieba 分词 + 英文小写切分。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import jieba

from chunker import Chunk, chunk_corpus

jieba.setLogLevel(60)  # 静音构建日志


def tokenize(text: str) -> list[str]:
    """中文按 jieba 切，连续英文/数字整块保留，全部小写。"""
    tokens: list[str] = []
    for seg in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if seg.isascii():
            tokens.append(seg)
        else:
            tokens.extend(t for t in jieba.cut(seg) if len(t.strip()) >= 2)
    return tokens


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    _doc_tokens: list[list[str]] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)
    _avgdl: float = 0.0

    @classmethod
    def from_corpus_dir(cls, corpus_dir: str) -> "BM25Index":
        from pathlib import Path

        docs = {p.name: p.read_text(encoding="utf-8")
                for p in sorted(Path(corpus_dir).glob("*.md"))}
        return cls.from_docs(docs)

    @classmethod
    def from_docs(cls, docs: dict[str, str]) -> "BM25Index":
        idx = cls(chunks=chunk_corpus(docs))
        idx._doc_tokens = [tokenize(c.text) for c in idx.chunks]
        idx._avgdl = (sum(len(t) for t in idx._doc_tokens) / len(idx._doc_tokens)
                      if idx._doc_tokens else 0.0)
        for tokens in idx._doc_tokens:
            for term in set(tokens):
                idx._df[term] = idx._df.get(term, 0) + 1
        return idx

    def _idf(self, term: str) -> float:
        n = len(self._doc_tokens)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], index: int) -> float:
        score, doc = 0.0, self._doc_tokens[index]
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc)
        for qt in query_tokens:
            f = tf.get(qt, 0)
            if not f:
                continue
            score += (self._idf(qt) * f * (self.k1 + 1)
                      / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)))
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q_tokens = tokenize(query)
        scored = []
        for i in range(len(self.chunks)):
            s = self.score(q_tokens, i)
            if s <= 0:
                continue
            chunk = self.chunks[i]
            # 标题命中加权：查询词出现在标题路径里，说明块主题对齐
            title_tokens = set(tokenize(chunk.section))
            matched = sum(1 for t in set(q_tokens) if t in title_tokens)
            if matched:
                s *= 1 + 0.25 * matched
            scored.append((chunk, s))
        scored.sort(key=lambda t: -t[1])
        return scored[:top_k]
