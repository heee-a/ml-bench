# data-portfolio · 数据采集与分析作品集

[![CI](https://github.com/heee-a/data-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/heee-a/data-portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

面向数据分析岗位的实战作品集：**七个**完整的小项目，覆盖采集、SQL、统计推断、
文本挖掘、时序预测、可视化。每个项目都走完 **数据 → 清洗 → 分析 → 结论** 全链路。
所有数据、图表、结论均由仓库内脚本从公开数据源真实采集生成，可一键复现。

## 作品速览

| ![六城月均温](projects/01_weather_cities/charts/monthly_temp.png) | ![Preston曲线](projects/03_world_indicators/charts/income_life.png) |
|---|---|
| [01 城市气象](projects/01_weather_cities/)：南北冬差 37℃ vs 夏差 13℃ | [03 世界指标](projects/03_world_indicators/)：收入-寿命 Preston 曲线 |
| ![语言构成](projects/02_github_top/charts/language.png) | ![四季如春](projects/05_stats_inference/charts/test3_kunming.png) |
| [02 GitHub 画像](projects/02_github_top/)：Python 占 23%，AI 潮创诞生峰值 | [05 统计推断](projects/05_stats_inference/)：「四季如春」的统计验证 |

## 项目一览

| # | 项目 | 维度 | 数据源 | 面试考点 |
|---|---|---|---|---|
| 01 | [全国主要城市气象分析](projects/01_weather_cities/) | API 采集 · 时序/地理分析 | Open-Meteo | 限速与退避、缺失率、口径设计 |
| 02 | [GitHub 头部仓库画像](projects/02_github_top/) | API 采集 · 分页/分布分析 | GitHub Search API | 分页上限处理、幂律分布、幸存者偏差 |
| 03 | [世界发展指标](projects/03_world_indicators/) | API 采集 · 面板数据 | World Bank | 长宽转换、对数变换的理由 |
| 04 | [SQL 分析](projects/04_sql_analysis/) | **SQL** | SQLite（复用03数据） | 窗口函数/CTE/条件聚合，10 个业务问题 |
| 05 | [统计推断](projects/05_stats_inference/) | **假设检验** | 复用01数据 | t/Mann-Whitney/Levene、效应量、Bootstrap CI、负结果报告 |
| 06 | [文本挖掘](projects/06_text_mining/) | **NLP 轻量** | 复用02数据 | jieba 分词、停用词、二连词、口径诚实 |
| 07 | [时序预测](projects/07_forecast/) | **预测建模** | 复用01数据 | 回测切分、朴素基线纪律、MAPE 缺陷实录 |

每个项目的目录里都有独立的 README（数据来源、清洗口径、结论、复现命令），
以及 `collect.py` / `analyze.py`（或 build_db.py + run_queries.py 等对应脚本）。

## 工程实践（面试可展开讲）

- **礼貌采集基座** [datap/fetching.py](datap/fetching.py)：
  - 磁盘缓存——同 URL 只请求一次，重跑分析零请求；
  - 限速 + 指数退避 + `Retry-After` 尊重——项目一中真实触发过 Open-Meteo
    每分钟限流（共享出口 IP），自动等待 61s 后恢复，全程无人值守；
  - 项目二踩到 GitHub 搜索 API「最多返回 1000 条」的硬上限（第 11 页 422），
    采集器按业务边界优雅终止并在 README 中如实记录；
- **数据质量**：每个项目 README 报告缺失率与分析口径；仓库附带数据质量
  回归测试（`tests/`），防止脏数据被当成结论；
- **诚实呈现**：6 年温度趋势明确标注"样本太短，勿过度解读"；GitHub 数据
  明确标注"头部幸存者画像"；统计项目报告了不显著的负结果；预测项目
  实录了 MAPE 在零附近的爆炸——负结果与坑位记录是作品集的一部分。

## 快速开始

```bash
pip install -e .
# 任选一个项目
cd projects/01_weather_cities
python collect.py   # 首次真实采集（含限速等待，有缓存后秒级）
python analyze.py   # 出图表与结论
```

## 目录结构

```
data-portfolio/
├── datap/                  # 共享库：限速/重试/缓存采集器 + 图表样式
├── projects/
│   ├── 01_weather_cities/  # 天气：collect.py + analyze.py + README + data/ + charts/
│   ├── 02_github_top/
│   └── 03_world_indicators/
└── tests/                  # 数据质量回归测试 + 采集器缓存单测
```

## 数据与版权说明

三个数据源均为公开 API，允许程序化访问；本仓库以研究学习为目的采集，
遵守其服务条款与限速要求。采集日期：2026-09（数据快照随仓库提供，
重跑 `collect.py` 可刷新）。

## License

[MIT](LICENSE)
