# ml-bench · 机器学习基准与真实数据分析

[![CI](https://github.com/heee-a/ml-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/heee-a/ml-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

三个以「真实数据 + 可验证指标」为核心的项目。设计原则：**结论必须有统计
或实验证据，预设叙事被数据推翻时如实报告**。

## 项目一览

### 🏆 [01 sklearn_bench 经典基准](projects/01_sklearn_bench/)
3 个真实数据集（乳腺癌/手写数字/葡萄酒）× 5 模型 × 重复分层交叉验证。
最高准确率 **98.3%**（digits，SVM-RBF；留出集 98.0%）。配对 Wilcoxon 检验
证明：头部模型之间差异不显著（p>0.2），最佳与最差之间显著（p<0.01）。
→ [README](projects/01_sklearn_bench/README.md)

### 🌫 [02_air_quality 空气质量分析](projects/02_air_quality/)
17 城 × 3 年逐小时 PM2.5/PM10（CAMS 再分析，14,977 城市-日）。冬季峰值
在南北都显著（Mann-Whitney p<1e-76），但逐城冬夏比值差异远大于南北分组
差异——「北方采暖更糟」的二分叙事被数据推翻，过程完整保留。
→ [README](projects/02_air_quality/README.md)

### 📈 [04_pm25_regression 监督学习回归](projects/04_pm25_regression/)
**任务**：合并气象与空气质量采集数据，预测城市月度 PM2.5（377 个城市-月）。
- 两次如实的口径修正：日度口径天花板实测（R²≈0.1）→ 聚合月度；
- 朴素基线纪律：仅月份特征即 R²=0.128，随机森林 +10pp 至 0.225；
- TimeSeriesSplit 滚动回测、周期编码、特征重要性。
→ [README](projects/04_pm25_regression/README.md)

### 🔍 [03_lsa_search 检索方法对照](projects/03_lsa_search/)
同一语料、同一评测集下：BM25（83%）> 混合（75%）> LSA（67%）。
小语料 + 关键词型查询场景下，BM25 仍是不可撼动的基线——
用评测说话，不迷信"向量检索更高级"。
→ [README](projects/03_lsa_search/README.md)

### 🎲 [05_ab_simulation A/B 测试方法论](projects/05_ab_simulation/)
样本量公式 × 蒙特卡洛互证（公式 80% vs 实测 80.5%）× 双比例 z 检验全流程
× A/A 诚实性检查（假阳性 5.3%≈α）× 功效曲线。模拟数据明确标注，方法工业级。
→ [README](projects/05_ab_simulation/README.md)

## 快速开始

```bash
pip install -e .
python projects/01_sklearn_bench/run_bench.py          # 约 1 分钟
python projects/02_air_quality/collect.py && python projects/02_air_quality/analyze.py
python projects/03_lsa_search/run_search_bench.py
```

## 共享基建

`datap/fetching.py`：磁盘缓存 + 限速 + 429 退避 + 指数重试的采集器
（与 data-portfolio 同源），本仓库所有 API 采集复用。

## 声明

- sklearn 数据集与 CAMS 再分析数据均为公开数据；PM2.5 为再分析估计值，
  绝对量级有系统偏差（README 已标注），结论以趋势与相对比较为主；
- 基准准确率由仓库脚本在 `random_state=42` 下实测生成。
License: [MIT](LICENSE)
