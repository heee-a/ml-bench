"""经典数据集机器学习基准：3 个真实数据集 × 5 种模型 × 重复分层交叉验证。

严谨协议（面试可讲）：
- RepeatedStratifiedKFold(5 折 × 2 次)：既保持类别比例，又重复采样降低运气成分；
- 全部模型跑在同一批折上（配对比较的前提）；
- 每对模型做 Wilcoxon 符号秩检验，报告 p 值——"谁更好"要有统计证据；
- 预处理放进 Pipeline 内部，杜绝折外信息泄漏（标准化必须只在训练折上 fit）。

运行: python run_bench.py
产出: reports/bench_table.csv, reports/metrics.json, charts/*.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
SEED = 42


def scaled(model) -> Pipeline:
    """基于距离/方差的模型需要标准化；标准化放在 Pipeline 内防泄漏。"""
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


MODELS = {
    "LogReg": scaled(LogisticRegression(max_iter=3000)),
    "SVM-RBF": scaled(SVC()),
    "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "GBDT": GradientBoostingClassifier(random_state=SEED),
}

DATASETS = {
    "breast_cancer": load_breast_cancer,   # 569 样本 × 30 特征，2 类
    "digits": load_digits,                 # 1797 样本 × 64 特征，10 类
    "wine": load_wine,                     # 178 样本 × 13 特征，3 类
}


def run() -> pd.DataFrame:
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    rows = []
    fold_scores: dict[str, dict[str, np.ndarray]] = {}
    for ds_name, loader in DATASETS.items():
        data = loader()
        X, y = data.data, data.target
        fold_scores[ds_name] = {}
        for model_name, model in MODELS.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
            fold_scores[ds_name][model_name] = scores
            rows.append({"dataset": ds_name, "model": model_name,
                         "accuracy_mean": scores.mean().round(4),
                         "accuracy_std": scores.std().round(4),
                         "n_samples": len(y), "n_features": X.shape[1],
                         "n_classes": len(np.unique(y))})
        print(f"[{ds_name}] " +
              "  ".join(f"{m}={fold_scores[ds_name][m].mean():.4f}" for m in MODELS))
    return pd.DataFrame(rows), fold_scores


def significance_tests(fold_scores) -> list[dict]:
    """每个数据集两组配对 Wilcoxon：前两名之间、最佳与最差之间。"""
    tests = []
    for ds_name, per_model in fold_scores.items():
        ranked = sorted(per_model, key=lambda m: -per_model[m].mean())
        pairs = [("top2", ranked[0], ranked[1]), ("best_vs_worst", ranked[0], ranked[-1])]
        for tag, a, b in pairs:
            stat, p = stats.wilcoxon(per_model[a], per_model[b])
            tests.append({"dataset": ds_name, "pair": tag, "best": a, "other": b,
                          "diff_mean": round(float(per_model[a].mean()
                                                    - per_model[b].mean()), 4),
                          "wilcoxon_p": round(float(p), 4),
                          "significant_at_0.05": bool(p < 0.05)})
    return tests


def main() -> None:
    table, fold_scores = run()
    table.to_csv(ROOT / "reports" / "bench_table.csv", index=False,
                 encoding="utf-8-sig")

    tests = significance_tests(fold_scores)
    metrics = {
        "protocol": "RepeatedStratifiedKFold(5 splits x 2 repeats), random_state=42",
        "results": table.to_dict(orient="records"),
        "significance_tests": tests,
    }
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 显著性检验（Wilcoxon 配对）===")
    for t in tests:
        verdict = "差异显著" if t["significant_at_0.05"] else "差异不显著"
        print(f"  [{t['dataset']} | {t['pair']}] {t['best']} vs {t['other']}: "
              f"+{t['diff_mean']:.4f}, p={t['wilcoxon_p']} -> {verdict}")

    # ---- 图：各数据集模型准确率箱线 ----
    from datap.plotstyle import save_chart, setup_style

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (ds_name, per_model) in zip(axes, fold_scores.items()):
        data = [per_model[m] for m in MODELS]
        ax.boxplot(data, tick_labels=list(MODELS))
        ax.set_title(f"{ds_name}")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.85, 1.005)
    axes[0].set_ylabel("CV 准确率")
    fig.suptitle("3 数据集 × 5 模型 × 10 次交叉验证的准确率分布", y=1.02)
    save_chart(fig, ROOT / "charts" / "bench_boxplot.png")

    # ---- 图：digits 最佳模型混淆矩阵 ----
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target,
        random_state=SEED)
    best_digits = max(
        ((m, fold_scores["digits"][m].mean()) for m in MODELS),
        key=lambda t: t[1])[0]
    model = MODELS[best_digits].fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax,
                                          cmap="Blues", colorbar=False)
    ax.set_title(f"digits 留出集：{best_digits} 准确率 {acc:.4f}")
    save_chart(fig, ROOT / "charts" / "digits_confusion.png")
    print(f"\ndigits 留出集验证（{best_digits}）: {acc:.4f}")
    print("产物: reports/bench_table.csv, reports/metrics.json, charts/*.png")


if __name__ == "__main__":
    main()
