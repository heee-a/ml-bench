"""采集 17 个城市 2022-2024 年逐小时 PM2.5/PM10（Open-Meteo Air Quality API）。

数据源: CAMS 全球大气再分析（欧洲中期天气预报中心），免费无需 Key。
采集协议: 3s 限速 + 429 退避 61s + 磁盘缓存（复用 datap.fetching.Fetcher）。
产出: data/pm_daily.csv（city, date, pm25, pm10 的日均值长表）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from datap.fetching import Fetcher, ensure_dir

API = "https://air-quality-api.open-meteo.com/v1/air-quality"
START, END = "2022-01-01", "2024-12-31"

# 北方（冬季集中供暖）与南方分组，用于采暖效应分析
NORTH = ["北京", "天津", "石家庄", "太原", "西安", "兰州", "乌鲁木齐", "沈阳", "哈尔滨"]
SOUTH = ["上海", "南京", "杭州", "武汉", "成都", "重庆", "广州", "深圳"]
CITIES = {c: xy for c, xy in {
    "北京": (39.90, 116.41), "天津": (39.13, 117.20), "石家庄": (38.04, 114.51),
    "太原": (37.87, 112.55), "西安": (34.34, 108.94), "兰州": (36.06, 103.83),
    "乌鲁木齐": (43.83, 87.62), "沈阳": (41.80, 123.43), "哈尔滨": (45.80, 126.53),
    "上海": (31.23, 121.47), "南京": (32.06, 118.80), "杭州": (30.27, 120.16),
    "武汉": (30.59, 114.31), "成都": (30.57, 104.07), "重庆": (29.56, 106.55),
    "广州": (23.13, 113.26), "深圳": (22.54, 114.06)}.items()}
GROUP = {c: ("北方" if c in NORTH else "南方") for c in CITIES}


def fetch_city(f: Fetcher, city: str, lat: float, lon: float) -> pd.DataFrame:
    data = f.get_json(API, params={
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10", "start_date": START, "end_date": END,
        "timezone": "Asia/Shanghai"})
    h = data["hourly"]
    df = pd.DataFrame({"city": city, "time": pd.to_datetime(h["time"]),
                       "pm25": h["pm2_5"], "pm10": h["pm10"]})
    return df


def main() -> None:
    root = ensure_dir(Path(__file__).parent / "data")
    f = Fetcher(cache_dir=root / ".cache", min_interval=3.0, max_retries=6)
    frames = []
    for i, (city, (lat, lon)) in enumerate(CITIES.items(), 1):
        df = fetch_city(f, city, lat, lon)
        frames.append(df)
        valid = df["pm25"].notna().mean() * 100
        print(f"[{i:>2}/{len(CITIES)}] {city}({GROUP[city]}): {len(df):,} 小时记录, "
              f"PM2.5 有效率 {valid:.1f}%")
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = raw["time"].dt.date
    daily = (raw.groupby(["city", "date"])
             .agg(pm25=("pm25", "mean"), pm10=("pm10", "mean"),
                  hours=("pm25", "count")).reset_index())
    daily = daily[daily["hours"] >= 12]           # 当日有效小时数过半才保留
    daily["group"] = daily["city"].map(GROUP)
    daily = daily.drop(columns="hours")
    daily.to_csv(root / "pm_daily.csv", index=False, encoding="utf-8-sig")
    print(f"\n日均值 {len(daily):,} 行 -> {root / 'pm_daily.csv'}")


if __name__ == "__main__":
    main()
