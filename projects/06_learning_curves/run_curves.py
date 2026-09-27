"""学习曲线：样本量对模型准确率的影响（digits 与 breast_cancer 实测）。

回答面试高频问题"需要多少数据"：
- 每个训练集规模用分层子采样 × 5 个随机种子，评估固定留出集；
- 模型：LogReg / SVM-RBF / 随机森林（与 ml-bench/01 同口径，Pipeline 内标准化）；
- 产出：学习曲线图 + "到达最优准确率 99% 所需样本量"的量化结论。

运行: python run_curves.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, load_digits

from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
SEED = 42
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5


def models() -> dict:
    return {
        "LogReg": Pipeline([("s", StandardScaler()),
                            ("c", LogisticRegression(max_iter=3000))]),
        "SVM-RBF": Pipeline([("s", StandardScaler()), ("c", SVC())]),
        "KNN": Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(5))]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=SEED),
    }


def stratified_subsample(X, y, frac: float, seed: int):
    if frac >= 1.0:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=frac,
                              stratify=y, random_state=seed)
    return X[idx], y[idx]


def run_dataset(name: str, loader) -> dict:
    data = loader()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target,
        random_state=SEED)
    out = {"dataset": name, "train_full": len(y_tr), "test": len(y_te),
           "curves": {}}
    print(f"\n=== {name}（训练集 {len(y_tr)}，留出集 {len(y_te)}）===")
    for m_name, model in models().items():
        xs, means, stds = [], [], []
        for frac in FRACTIONS:
            accs = []
            for seed in range(SEED, SEED + N_SEEDS):
                Xs, ys = stratified_subsample(X_tr, y_tr, frac, seed)
                accs.append(float(model.fit(Xs, ys).score(X_te, y_te)))
            xs.append(int(round(len(y_tr) * frac)))
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
        out["curves"][m_name] = {
            "sizes": xs,
            "mean": [round(m, 4) for m in means],
            "std": [round(s, 4) for s in stds],
        }
        print(f"  {m_name:<13}" + "  ".join(f"{m:.3f}" for m in means))
    return out


def main() -> None:
    setup_style()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)

    results = [run_dataset("digits", load_digits),
               run_dataset("breast_cancer", load_breast_cancer)]

    (ROOT / "reports" / "curves.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"LogReg": "#4C72B0", "SVM-RBF": "#C44E52",
              "KNN": "#55A868", "RandomForest": "#DD8452"}
    findings = {}
    for ax, res in zip(axes, results):
        for m_name, curve in res["curves"].items():
            means = np.array(curve["mean"])
            stds = np.array(curve["std"])
            ax.plot(curve["sizes"], means, marker="o", ms=4, label=m_name,
                    color=colors[m_name])
            ax.fill_between(curve["sizes"], means - stds, means + stds,
                            alpha=0.12, color=colors[m_name])
            # "到达最终准确率 99% 所需样本量"
            final = means[-1]
            reached = next((s for s, m in zip(curve["sizes"], means)
                            if m >= final * 0.99), curve["sizes"][-1])
            findings.setdefault(res["dataset"], {})[m_name] = {
                "final_accuracy": round(float(final), 4),
                "samples_to_99pct_of_final": reached,
            }
        ax.set_xscale("log")
        ax.set_title(f"{res['dataset']}（5 种子均值 ± 标准差）")
        ax.set_xlabel("训练样本量（log）")
        ax.set_ylim(0.8, 1.01)
    axes[0].set_ylabel("留出集准确率")
    axes[0].legend(fontsize=9)
    fig.suptitle("学习曲线：同样本量下模型间差距在数据充足时才收敛", y=1.02)
    save_chart(fig, ROOT / "charts" / "learning_curves.png")

    print("\n=== 量化结论（到达最终准确率 99% 所需样本量）===")
    for ds, per_model in findings.items():
        print(f"  [{ds}]")
        for m, v in per_model.items():
            print(f"    {m}: {v['samples_to_99pct_of_final']} 样本 -> "
                  f"{v['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
