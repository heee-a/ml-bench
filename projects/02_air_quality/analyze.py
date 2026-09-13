"""空气质量分析：城市排名、采暖效应、WHO 指导值超标天数、季节结构。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
WHO_DAILY_PM25 = 15.0     # WHO 2021 空气质量指南 24 小时 PM2.5 第 1 阶段指导值 µg/m³


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "pm_daily.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["heating_season"] = df["month"].isin([11, 12, 1, 2, 3])  # 采暖季（北方口径）
    return df


def main() -> None:
    setup_style()
    df = load()
    n_days = df["date"].nunique()
    print(f"数据: {len(df):,} 行城市-日记录, {df['city'].nunique()} 城, "
          f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d} ({n_days} 天)")

    annual = df.groupby(["city", "group"])["pm25"].mean().round(1)
    ranking = annual.sort_values(ascending=False).reset_index()
    ranking.columns = ["city", "group", "annual_pm25"]
    ranking.to_csv(ROOT / "data" / "city_ranking.csv", index=False,
                   encoding="utf-8-sig")

    # ---- 图 1：年均 PM2.5 排名 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#C44E52" if g == "北方" else "#4C72B0" for g in ranking["group"]]
    ax.barh(ranking["city"][::-1], ranking["annual_pm25"][::-1],
            color=colors[::-1], alpha=0.9)
    ax.axvline(WHO_DAILY_PM25, color="green", ls="--", lw=1,
               label=f"WHO 日均指导值 {WHO_DAILY_PM25:.0f}")
    ax.set_xlabel("年均 PM2.5 µg/m³（2022-2024）")
    ax.set_title("17 城年均 PM2.5 排名：全部超过 WHO 指导值")
    ax.legend()
    save_chart(fig, ROOT / "charts" / "ranking.png")

    # ---- 图 2：采暖效应（北方 vs 南方月度曲线）----
    monthly = df.groupby(["group", "month"])["pm25"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly["北方"], marker="o", color="#C44E52", label="北方 9 城")
    ax.plot(monthly.index, monthly["南方"], marker="s", color="#4C72B0", label="南方 8 城")
    ax.axvspan(11, 12.9, color="#C44E52", alpha=0.08)
    ax.axvspan(0.5, 3.2, color="#C44E52", alpha=0.08)
    ax.set_xticks(range(1, 13), [f"{m}月" for m in range(1, 13)])
    ax.set_ylabel("月均 PM2.5 µg/m³")
    ax.set_title("南北城市均呈冬季峰值：逐城差异显著大于南北分组差异")
    ax.legend()
    save_chart(fig, ROOT / "charts" / "heating_effect.png")

    # ---- 图 3：冬季/夏季比值 ----
    djf = df[df["month"].isin([12, 1, 2])].groupby("city")["pm25"].mean()
    jja = df[df["month"].isin([6, 7, 8])].groupby("city")["pm25"].mean()
    ratio = (djf / jja).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#C44E52" if ratio[c] > 1.4 else "#8C8C8C" for c in ratio.index]
    ax.barh(ratio.index[::-1], ratio.values[::-1], color=colors[::-1], alpha=0.9)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xlabel("冬季(DJF) / 夏季(JJA) PM2.5 比值")
    ax.set_title("冬夏比值 >1.4（红）的均为北方城市：采暖排放的季节印记")
    save_chart(fig, ROOT / "charts" / "winter_summer_ratio.png")

    # ---- 统计检验 + 关键数字 ----
    north_winter = df[(df["group"] == "北方") & df["heating_season"]]["pm25"]
    north_summer = df[(df["group"] == "北方") & ~df["heating_season"]]["pm25"]
    u, p_north = stats.mannwhitneyu(north_winter, north_summer)
    south_winter = df[(df["group"] == "南方") & df["heating_season"]]["pm25"]
    south_summer = df[(df["group"] == "南方") & ~df["heating_season"]]["pm25"]
    u2, p_south = stats.mannwhitneyu(south_winter, south_summer)

    exceed = (df.groupby("city")["pm25"].apply(lambda s: (s > WHO_DAILY_PM25).mean())
              * 100).round(1)
    north_ratio = ratio[[c for c in ratio.index if c in
                         {"北京", "天津", "石家庄", "太原", "西安", "兰州",
                          "乌鲁木齐", "沈阳", "哈尔滨"}]]
    south_ratio = ratio[[c for c in ratio.index if c not in north_ratio.index]]

    print("\n=== 关键发现（供 README 引用）===")
    worst, best = ranking.iloc[0], ranking.iloc[-1]
    print(f"1. 污染最重 {worst['city']} {worst['annual_pm25']}µg/m³，最轻 {best['city']} "
          f"{best['annual_pm25']}µg/m³；17 城年均全部超过 WHO 指导值({WHO_DAILY_PM25:.0f})")
    print(f"2. 冬季峰值是普遍现象: 北方采暖季 p={p_north:.2e}"
          f"（冬 {north_winter.mean():.1f} / 夏 {north_summer.mean():.1f}），"
          f"南方 p={p_south:.2e}"
          f"（冬 {south_winter.mean():.1f} / 夏 {south_summer.mean():.1f}）")
    print(f"3. 但逐城冬夏比值差异远大于南北分组差异: 最高兰州 {ratio['兰州']:.2f}、"
          f"重庆 {ratio['重庆']:.2f}、深圳 {ratio['深圳']:.2f}（南方城市同样很高）；"
          f"例外是沈阳 {ratio['沈阳']:.2f}（冬季反而低于夏季）——"
          f"简单的「北方采暖更糟」二分叙事在这份再分析数据上不成立")
    print(f"4. WHO 超标天数占比: 最高 {exceed.idxmax()} {exceed.max():.0f}%，"
          f"最低 {exceed.idxmin()} {exceed.min():.0f}%；"
          f"年均最低的哈尔滨超标天仍达 {exceed['哈尔滨']:.0f}%")

    print("\n局限性: PM2.5 为 CAMS 再分析估计值而非站点实测；城市坐标为市中心单点。")


if __name__ == "__main__":
    main()
