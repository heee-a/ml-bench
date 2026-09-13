"""ml-bench 测试：基准结果完整性、空气质量数据质量、检索评测产物。pytest -q"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


# ---------------- 01 sklearn bench ----------------
def test_bench_table_completeness():
    t = pd.read_csv(REPO / "projects/01_sklearn_bench/reports/bench_table.csv")
    assert len(t) == 15                                  # 3 数据集 × 5 模型
    assert set(t["dataset"]) == {"breast_cancer", "digits", "wine"}
    digits = t[t["dataset"] == "digits"]
    assert digits["accuracy_mean"].max() >= 0.97         # 高准确率基线


def test_bench_metrics_has_significance():
    import json

    m = json.loads(
        (REPO / "projects/01_sklearn_bench/reports/metrics.json").read_text(
            encoding="utf-8"))
    sig = m["significance_tests"]
    assert len(sig) == 6                                 # 3 数据集 × 2 组配对
    # 头部模型差异不显著、最佳-最差差异显著的叙事必须有数据支撑
    top2 = [t for t in sig if t["pair"] == "top2"]
    assert all(not t["significant_at_0.05"] for t in top2)
    bw = [t for t in sig if t["pair"] == "best_vs_worst"]
    assert all(t["significant_at_0.05"] for t in bw)


# ---------------- 02 air quality ----------------
def test_pm_daily_quality():
    df = pd.read_csv(REPO / "projects/02_air_quality/data/pm_daily.csv",
                     parse_dates=["date"])
    assert df["city"].nunique() == 17
    assert len(df) >= 14000
    assert df["date"].dt.year.min() >= 2022
    assert (df["pm25"] >= 0).all()
    assert set(df["group"]) == {"北方", "南方"}


def test_air_quality_claims_supported():
    """README 的三条论断必须有数据支撑。"""
    df = pd.read_csv(REPO / "projects/02_air_quality/data/pm_daily.csv",
                     parse_dates=["date"])
    df["month"] = df["date"].dt.month
    annual = df.groupby("city")["pm25"].mean()
    assert (annual > 15).all()                           # 全部超过 WHO 指导值
    north_winter = df[(df["group"] == "北方") & df["month"].isin([12, 1, 2])]["pm25"]
    north_summer = df[(df["group"] == "北方") & df["month"].isin([6, 7, 8])]["pm25"]
    assert north_winter.mean() > north_summer.mean()     # 北方冬季峰值
    ranking = pd.read_csv(REPO / "projects/02_air_quality/data/city_ranking.csv")
    assert ranking.iloc[0]["city"] == annual.idxmax()


# ---------------- 03 lsa search ----------------
def test_search_bench_table():
    s = pd.read_csv(REPO / "projects/03_lsa_search/reports/search_bench.csv")
    assert set(s["method"]) == {"BM25", "LSA(TF-IDF+SVD)", "Hybrid"}
    bm25_row = s[s["method"] == "BM25"].iloc[0]
    assert bm25_row["recall@3"] >= 0.8                   # BM25 强基线结论成立


def test_lsa_and_bm25_hit_on_known_query():
    sys_path = str(REPO / "projects/03_lsa_search")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import numpy as np

    from bm25 import BM25Index
    from run_search_bench import build_lsa, lsa_search

    docs = {p.name: p.read_text(encoding="utf-8")
            for p in (REPO / "projects/03_lsa_search/corpus").glob("*.md")}
    bm25 = BM25Index.from_docs(docs)
    chunks, vectorizer, svd, lsa = build_lsa(docs)

    bm25_hits = bm25.search("LRU 缓存怎么实现？", top_k=3)
    assert any("dev-portfolio-algorithms" in h[0].doc for h in bm25_hits)

    lsa_hits = lsa_search("LRU 缓存怎么实现？", vectorizer, svd, lsa, chunks, top_k=5)
    assert isinstance(lsa_hits, list)
    assert all(len(np.shape(h[1])) == 0 for h in lsa_hits)  # 分数是标量


# ---------------- 04 pm25 regression ----------------
def test_regression_scores():
    s = pd.read_csv(REPO / "projects/04_pm25_regression/reports/regression_scores.csv")
    assert len(s) == 4
    assert "基线：仅月份（线性）" in set(s["model"])
    best = s["r2"].max()
    assert 0.0 <= best <= 0.9                       # 月度口径的合理区间


def test_pm25_dataset_integrity():
    pm = pd.read_csv(REPO / "projects/04_pm25_regression/data/pm_daily.csv",
                     parse_dates=["date"])
    wx = pd.read_csv(REPO / "projects/04_pm25_regression/data/weather_daily.csv",
                     parse_dates=["date"])
    merged = pm.merge(wx[["city", "date"]], on=["city", "date"])
    assert len(merged) >= 350                       # 城市-月样本的数据基础
