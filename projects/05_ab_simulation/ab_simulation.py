"""A/B 测试方法论完整实现与蒙特卡洛验证（模拟数据，方法真实）。

内容（面试统计题的全覆盖）：
1. 样本量计算：给定基线转化率、MDE、α、power，求每组所需样本量；
2. 蒙特卡洛验证：按计算出的样本量模拟 2000 次实验，实测「检验功效」
   是否达到设定值（理论 vs 模拟互证）；
3. 双比例 z 检验全流程：检验统计量、p 值、效应量（相对提升）、置信区间；
4. 诚实性检查：A/A 检验（无差异时假阳性率应 ≈ α）。

运行: python ab_simulation.py
产出: charts/*.png, reports/simulation.json
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
SEED = 42
rng = np.random.default_rng(SEED)


def sample_size_per_group(p_baseline: float, mde: float,
                          alpha: float = 0.05, power: float = 0.8) -> int:
    """双比例检验的每组样本量（正态近似公式）。

    p1 = p_baseline, p2 = p_baseline * (1 + mde)：
    n = (z_{α/2}·√(2·p̄·(1-p̄)) + z_power·√(p1(1-p1)+p2(1-p2)))² / (p2-p1)²
    """
    p2 = p_baseline * (1 + mde)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_baseline + p2) / 2
    n = (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
         + z_b * math.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))) ** 2 \
        / (p2 - p_baseline) ** 2
    return math.ceil(n)


def two_proportion_z_test(x_a: int, n_a: int, x_b: int, n_b: int) -> dict:
    """双比例 z 检验（合并方差的原假设口径）。"""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci = (p_b - p_a - 1.96 * se_unpooled, p_b - p_a + 1.96 * se_unpooled)
    return {"z": round(z, 3), "p": round(p_value, 5),
            "uplift_abs": round(p_b - p_a, 5),
            "uplift_rel": round((p_b - p_a) / p_a, 4),
            "ci95": tuple(round(v, 5) for v in ci)}


def run_experiment(p_a: float, uplift: float, n_per_group: int) -> tuple[int, int]:
    x_a = int(rng.binomial(n_per_group, p_a))
    x_b = int(rng.binomial(n_per_group, p_a * (1 + uplift)))
    return x_a, x_b


def main() -> None:
    setup_style()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    p0, mde, alpha, power = 0.10, 0.05, 0.05, 0.80   # 基线转化率 10%，相对提升 5%

    # ---- 1) 样本量计算 ----
    n_needed = sample_size_per_group(p0, mde, alpha, power)
    print(f"基线转化率 {p0:.0%}，MDE {mde:.0%}（相对），α={alpha}，power={power}")
    print(f"-> 每组需要样本量 n = {n_needed:,}")

    # ---- 2) 蒙特卡洛验证功效：按 n 跑 2000 次有差异实验 ----
    n_sims, rejects = 2000, 0
    for _ in range(n_sims):
        x_a, x_b = run_experiment(p0, mde, n_needed)
        if two_proportion_z_test(x_a, n_needed, x_b, n_needed)["p"] < alpha:
            rejects += 1
    measured_power = rejects / n_sims
    print(f"蒙特卡洛实测功效: {measured_power:.1%}（目标 {power:.0%}，"
          f"{n_sims} 次模拟）——公式与模拟互证")

    # ---- 3) 一次完整实验的展示报告 ----
    x_a, x_b = run_experiment(p0, mde, n_needed)
    result = two_proportion_z_test(x_a, n_needed, x_b, n_needed)
    print(f"\n示例实验: 对照 {x_a}/{n_needed} ({x_a/n_needed:.2%}) vs "
          f"实验组 {x_b}/{n_needed} ({x_b/n_needed:.2%})")
    print(f"z={result['z']}, p={result['p']}, 相对提升 {result['uplift_rel']:.1%}, "
          f"95% CI {result['ci95']}")
    print("-> " + ("拒绝 H0：改动显著提升转化" if result["p"] < alpha
                  else "不能拒绝 H0：差异不显著"))

    # ---- 4) A/A 诚实性检查：无差异时假阳性率应 ≈ α ----
    n_aa, false_positives = 2000, 0
    for _ in range(n_aa):
        x_a, x_b = run_experiment(p0, 0.0, n_needed)
        if two_proportion_z_test(x_a, n_needed, x_b, n_needed)["p"] < alpha:
            false_positives += 1
    aa_rate = false_positives / n_aa
    print(f"\nA/A 检验（{n_aa} 次无差异实验）: 假阳性率 {aa_rate:.1%}"
          f"（理论 α={alpha:.0%}）——未超过 α 的 1.5 倍即健康")

    # ---- 5) 功效曲线：样本量 vs 实测功效 ----
    sizes = [int(n_needed * f) for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    curve = []
    for n in sizes:
        rej = 0
        for _ in range(400):
            x_a, x_b = run_experiment(p0, mde, n)
            if two_proportion_z_test(x_a, n, x_b, n)["p"] < alpha:
                rej += 1
        curve.append((n, rej / 400))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([s for s, _ in curve], [p for _, p in curve], marker="o",
            color="#4C72B0")
    ax.axhline(power, color="#C44E52", ls="--", label=f"目标功效 {power:.0%}")
    ax.axvline(n_needed, color="gray", ls=":", label=f"公式样本量 {n_needed:,}")
    ax.set_xscale("log")
    ax.set_xlabel("每组样本量")
    ax.set_ylabel("实测功效")
    ax.set_title(f"功效曲线（蒙特卡洛，MDE={mde:.0%}）：样本不足时检验无力")
    ax.legend()
    save_chart(fig, ROOT / "charts" / "power_curve.png")

    (ROOT / "reports" / "simulation.json").write_text(
        json.dumps({"params": {"p_baseline": p0, "mde": mde, "alpha": alpha,
                               "power": power},
                    "sample_size_formula": n_needed,
                    "measured_power": round(measured_power, 4),
                    "aa_false_positive_rate": round(aa_rate, 4),
                    "example_experiment": result},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: reports/simulation.json, charts/power_curve.png")


if __name__ == "__main__":
    main()
