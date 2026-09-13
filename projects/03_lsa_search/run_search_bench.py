"""检索方法对照评测：BM25 vs TF-IDF+SVD（LSA）vs 混合，同一评测集。

语料与评测用例来自 ai-lab 的 RAG 项目（本账号作品集的真实 README）。
LSA 用 scikit-learn：TfidfVectorizer（jieba 分词）-> TruncatedSVD 降维 ->
L2 归一化，查询投影到同一语义空间后做余弦相似度。

运行: python run_search_bench.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bm25 import BM25Index, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"

CASES = [
    ("哪个项目做了假设检验？用的是什么检验方法？", ["05_stats_inference"], "假设检验"),
    ("怎么防止标签泄漏？", ["01_ml_pipeline"], "泄漏"),
    ("FastAPI 的配置怎么外置？数据库路径用什么环境变量？", ["dev-portfolio-software"], "TASK_DB_PATH"),
    ("GitHub 上头部开源仓库最多用什么语言？", ["02_github_top"], "Python"),
    ("LRU 缓存怎么实现？", ["dev-portfolio-algorithms"], "LRU"),
    ("ReAct 智能体的循环是什么样？", ["dev-portfolio-ai-agents"], "Observation"),
    ("哪个城市四季如春？怎么验证的？", ["01_weather_cities", "05_stats_inference"], "昆明"),
    ("中国人均GDP 增长了多少倍？", ["03_world_indicators"], "13.4"),
    ("零钱兑换的最少硬币数用什么算法？", ["dev-portfolio-algorithms"], "零钱"),
    ("文本挖掘里中文项目的高频词有哪些？", ["06_text_mining"], "爬虫"),
    ("SQL 里怎么算同比增速？", ["04_sql_analysis"], "LAG"),
    ("时序预测怎么保证不偷看未来数据？", ["07_forecast"], "回测"),
]

N_COMPONENTS = 48


def build_lsa(docs: dict[str, str]):
    texts = []
    from chunker import chunk_corpus

    chunks = chunk_corpus(docs)
    for c in chunks:
        texts.append(f"{c.section}\n{c.text}")
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                 token_pattern=None)
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(N_COMPONENTS, tfidf.shape[1] - 1),
                       random_state=42)
    lsa = Normalizer().fit_transform(svd.fit_transform(tfidf))
    return chunks, vectorizer, svd, lsa


def lsa_search(query: str, vectorizer, svd, lsa, chunks, top_k: int = 3):
    qv = vectorizer.transform([query])
    q = Normalizer().transform(svd.transform(qv))[0]
    scores = lsa @ q
    order = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def evaluate(name: str, search_fn) -> dict:
    recall_hits, rr = 0, 0.0
    for q, doc_subs, kw in CASES:
        hits = search_fn(q, top_k=3)
        rank = None
        for i, (chunk, _s) in enumerate(hits, 1):
            if any(sub in chunk.doc for sub in doc_subs) and \
                    kw.lower() in (chunk.section + chunk.text).lower():
                rank = i
                break
        rr += 1 / rank if rank else 0.0
        recall_hits += rank is not None
        print(f"  [{('hit@' + str(rank)) if rank else 'MISS':>6}] {q[:22]}…")
    out = {"recall@3": recall_hits / len(CASES), "mrr": round(rr / len(CASES), 3)}
    print(f"  => {name}: recall@3={out['recall@3']:.0%}  MRR={out['mrr']}\n")
    return out


def main() -> None:
    docs = {p.name: p.read_text(encoding="utf-8")
            for p in sorted(CORPUS.glob("*.md"))}
    bm25 = BM25Index.from_docs(docs)
    chunks, vectorizer, svd, lsa = build_lsa(docs)
    print(f"语料: {len(bm25.chunks)} 块（BM25 与 LSA 共用分块）\n")

    print("== BM25 ==")
    r_bm25 = evaluate("BM25", lambda q, top_k: bm25.search(q, top_k=top_k))

    print("== TF-IDF + SVD (LSA) ==")
    r_lsa = evaluate("LSA", lambda q, top_k: lsa_search(q, vectorizer, svd, lsa,
                                                        chunks, top_k))

    print("== Hybrid (BM25 + LSA 分数归一化后 0.5/0.5) ==")

    def hybrid_search(q: str, top_k: int = 3):
        h1 = bm25.search(q, top_k=20)
        h2 = lsa_search(q, vectorizer, svd, lsa, chunks, top_k=20)
        score: dict = {}
        for lst, w in ((h1, 0.5), (h2, 0.5)):
            if not lst:
                continue
            vals = np.array([s for _, s in lst])
            norm = minmax(vals)
            for (chunk, _s), nv in zip(lst, norm):
                score[id(chunk)] = score.get(id(chunk), 0.0) + w * nv
        ranked = sorted(
            ((chunk, sc) for chunk, sc in
             [(c, score.get(id(c), 0)) for c, _s in h1 + h2]),
            key=lambda t: -t[1])
        seen, out = set(), []
        for chunk, sc in ranked:
            key = (chunk.doc, chunk.section, chunk.text[:50])
            if key in seen:
                continue
            seen.add(key)
            out.append((chunk, sc))
        return out[:top_k]

    r_hybrid = evaluate("Hybrid", hybrid_search)

    summary = pd.DataFrame([
        {"method": "BM25", **r_bm25},
        {"method": "LSA(TF-IDF+SVD)", **r_lsa},
        {"method": "Hybrid", **r_hybrid},
    ])
    summary.to_csv(ROOT / "reports" / "search_bench.csv", index=False,
                   encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import pandas as pd

    main()
