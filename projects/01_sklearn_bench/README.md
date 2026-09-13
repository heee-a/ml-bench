# 经典数据集机器学习基准：3 数据集 × 5 模型 × 10 次交叉验证

在三个 scikit-learn 内置的真实数据集上，用统一协议对比五种主流模型的准确率，
并做**配对显著性检验**。所有数字由 `run_bench.py` 实跑生成（`random_state=42`）。

## 结果

| 数据集 | 规模 | 最佳模型 | CV 准确率 | 留出集 |
|---|---|---|---|---|
| breast_cancer | 569 × 30，2 类 | SVM-RBF | **97.6%** ± 0.7% | — |
| digits | 1797 × 64，10 类 | SVM-RBF | **98.3%** ± 0.6% | 98.0% |
| wine | 178 × 13，3 类 | LogReg | **98.9%** ± 3.5% | — |

## 两个有统计依据的结论

1. **头部模型之间差异不显著**：三个数据集上前两名的 Wilcoxon 配对检验
   p 值分别为 0.83 / 0.21 / 0.63——"SVM 比 LogReg 好 0.1%"这种说法没有
   统计证据，选哪个都对；
2. **最佳与最差之间差异显著**（p = 0.008 / 0.002 / 0.008）：GBDT 在这三个
   小型表格数据集上 consistently 垫底（95.4%/96.3%/93.3%）——树模型
   "默认更强"的印象在经典小数据集上不成立，线性与核方法仍是最强基线。

## 协议（为什么可信）

- `RepeatedStratifiedKFold`：5 折 × 2 次重复 = 每个模型 10 个配对分数，
  降低单次划分的运气成分；
- 全部模型使用**同一批折**——这是配对检验（Wilcoxon 符号秩）的前提；
- 标准化在 Pipeline 内部完成：只在训练折上 fit，杜绝折外信息泄漏；
- digits 另做 25% 留出集验证：SVM-RBF **98.00%**（混淆矩阵见
  [charts/digits_confusion.png](charts/digits_confusion.png)）。

## 复现

```bash
python run_bench.py   # 约 1 分钟
```

产物：[reports/bench_table.csv](reports/bench_table.csv)、
[reports/metrics.json](reports/metrics.json)、
[charts/bench_boxplot.png](charts/bench_boxplot.png)。
