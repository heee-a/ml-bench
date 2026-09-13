# 项目二 · GitHub 头部开源仓库画像

采集 GitHub 全站星标最高的 **1000 个仓库**（stars ≥ 3 万，按星标排序拉满搜索 API
的上限），回答：头部开源世界用什么语言写？什么时候诞生？长什么样？

![语言构成](charts/language.png)

## 数据来源与采集

| 项 | 说明 |
|---|---|
| 数据源 | GitHub Search API（未认证，搜索限速 10 次/分钟） |
| 范围 | `stars:>30000` 按星标降序，共 1196 个中取可分页的前 **1000** 个 |
| 采集方式 | `collect.py`：分页拉满 + 7s 强制间隔 + 429/Retry-After 退避 + 磁盘缓存 |
| 工程细节 | GitHub 搜索 API 硬上限 1000 条（第 11 页返回 422），采集器优雅终止并记录边界 |

## 清洗与口径

- 搜索接口返回的 `language` 可能为 null，统一归为"其他/未标注"；
- topics 为多值字段，按 `|` 拆分后统计词频；
- "头部"定义：stars ≥ 3 万。**这是幸存者画像，不代表全体开源项目**。

## 核心发现

1. **Python 统治头部开源**：229 个（23%），TypeScript 172、JavaScript 107、Go 75；
   前三语言合计 **51%**；
2. **诞生峰值在 2023 年**（87 个，历史最高）——AI 项目让新仓库快速冲进头部（topic 里
   `ai` 出现 99 次）；2013-2018 是上一轮爆发期（占 40%）；
3. **星标是幂律分布**：中位数 4.9 万，头部 `codecrafters-io/build-your-own-x`
   54.7 万（中位数的 11 倍），log-log 下近似直线；
4. 8% 的头部仓库**没有 license**；
5. 高频 topic：javascript / python / **ai** / hacktoberfest / react。

## 图表

| 图表 | 说明 |
|---|---|
| [language.png](charts/language.png) | 语言构成 Top10 |
| [created_year.png](charts/created_year.png) | 诞生年代分布 + 星标中位数 |
| [star_power_law.png](charts/star_power_law.png) | 星标幂律（log-log） |
| [topics.png](charts/topics.png) | 高频 topic Top15 |

## 复现

```bash
python collect.py   # 约 1-2 分钟（受搜索限速），之后走缓存
python analyze.py
```
