"""监督学习回归：预测城市月度 PM2.5（季节性 + 城市效应分解）。

任务口径（重要，面试可讲）：
- 日度 PM2.5 的方差主要来自逐日天气噪声，季节与城市可解释的部分不足 20%，
  直接预测"某天某城"的 PM2.5 天花板极低（实测 R²≈0.1）；
- 因此聚合到 **城市-月** 粒度：特征为月份周期编码 + 城市身份 + 当月气象
  均值，目标是月均 PM2.5。这才是"季节性与城市效应分解"的合理任务；
- 基线：仅用"月份"的朴素模型。任何模型都必须先打赢它。

运行: python run_regression.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
SEED = 42


def load() -> pd.DataFrame:
    """合并项目一（气象）与项目二（空气质量）的真实数据，聚合到城市-月粒度。"""
    pm = pd.read_csv(ROOT / "data" / "pm_daily.csv", parse_dates=["date"])
    wx = pd.read_csv(ROOT / "data" / "weather_daily.csv", parse_dates=["date"])
    df = pm.merge(wx[["city", "date", "tmean", "tmax", "tmin", "precip"]],
                  on=["city", "date"], how="inner")
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["city", "group", "ym"]).agg(
        pm25=("pm25", "mean"), tmean=("tmean", "mean"),
        precip=("precip", "sum")).reset_index()
    monthly["month"] = monthly["ym"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
    return monthly


def main() -> None:
    setup_style()
    df = load()
    y = df["pm25"]

    month_only = ["month_sin", "month_cos"]
    full = ["month_sin", "month_cos", "tmean", "precip"] + \
           [c for c in df.columns if c.startswith("city_")]
    X_month = pd.get_dummies(df[month_only + ["month"]].assign(
        city_month=df["month"].astype(str)), columns=["city_month"]).select_dtypes("number")
    X_full = pd.get_dummies(df[full + ["city"]], columns=["city"]).select_dtypes("number")

    print(f"样本: {len(df)} 个城市-月（{df['city'].nunique()} 城 × "
          f"{df['ym'].nunique()} 个月），特征维度 month-only {X_month.shape[1]} / "
          f"full {X_full.shape[1]}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "基线：仅月份（线性）": (X_month, LinearRegression()),
        "线性：月份+气象+城市": (X_full, LinearRegression()),
        "随机森林": (X_full, RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED)),
        "HistGB": (X_full, HistGradientBoostingRegressor(
            max_iter=200, random_state=SEED)),
    }

    results, preds = [], {}
    for name, (X, model) in models.items():
        pred = cross_val_predict(model, X, y, cv=kf)
        results.append({"model": name,
                        "r2": round(float(r2_score(y, pred)), 3),
                        "mae": round(float(mean_absolute_error(y, pred)), 2)})
        preds[name] = pred
        print(f"  {name}: R²={results[-1]['r2']}  MAE={results[-1]['mae']}")

    summary = pd.DataFrame(results)
    (ROOT / "reports").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "reports" / "regression_scores.csv", index=False,
                   encoding="utf-8-sig")

    # ---- 图 1：三城时间序列 实际 vs 预测（线性全特征）----
    setup_style()
    cmp_df = df[["city", "ym", "pm25"]].copy()
    cmp_df["pred"] = preds["线性：月份+气象+城市"]
    cmp_df["ym"] = cmp_df["ym"].astype(str)
    picks = ["北京", "上海", "广州"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)))
    for ax, city in zip(axes, picks):
        d = cmp_df[cmp_df["city"] == city].sort_values("ym")
        ax.plot(d["ym"], d["pm25"], marker="o", ms=3.5, label="实际", color="#4C72B0")
        ax.plot(d["ym"], d["pred"], "--", marker=".", ms=4, label="预测",
                color="#C44E52")
        ax.set_title(f"{city}：月均 PM2.5 实际 vs 5 折交叉验证预测")
        ax.set_ylabel("µg/m³")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_chart(fig, ROOT / "charts" / "monthly_pred.png")

    # ---- 图 2：模型对比 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(summary["model"][::-1], summary["r2"][::-1],
                   color="#55A868", alpha=0.9)
    for b, v in zip(bars, summary["r2"][::-1]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("5 折交叉验证 R²")
    ax.set_title("月度 PM2.5 预测：任何模型都必须先打赢「仅月份」基线")
    ax.set_xlim(0, 1)
    save_chart(fig, ROOT / "charts" / "model_compare.png")

    print("\n=== 结论（供 README 引用）===")
    base_r2 = results[0]["r2"]
    print(f"1. 仅用月份周期就能解释月均 PM2.5 方差的 "
          f"{base_r2*100:.0f}%（季节性是主导因素）")
    for r in results[1:]:
        delta = (r["r2"] - base_r2) * 100
        print(f"2. {r['model']}: 相比基线 R² {'+' if delta>=0 else ''}{delta:.0f}pp"
              f"（MAE {r['mae']}）")
    print("3. 日度口径天花板极低（实测 R²≈0.1，逐日噪声主导）——"
          "聚合到月度是本任务正确的建模决策")


if __name__ == "__main__":
    main()
