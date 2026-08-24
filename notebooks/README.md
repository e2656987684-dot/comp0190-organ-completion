# notebooks

| notebook | 干什么 | 什么时候用 |
|---|---|---|
| **[`MSN_train_skullfix.ipynb`](MSN_train_skullfix.ipynb)** | **发起一次训练** + 本轮自检 + 存档 | 每次要跑训练。**只改第 1 节控制面板** |
| **[`MSN_compare_runs.ipynb`](MSN_compare_runs.ipynb)** | **判读结果**：指标词典、同轮次表、配对检验、主表、可视化、对照组 | 训练跑完之后 |
| [`MSN_surface_quality.ipynb`](MSN_surface_quality.ipynb) | mesh 重建、密度诊断、有符号偏差着色 | 想看**表面质量**而不是数字时 |
| [`MSN_baseline_pretrained.ipynb`](MSN_baseline_pretrained.ipynb) | 作者发布的 `MSN_weights3.h5` 在本项目数据上推理 | 对照组，已跑完，基本不用再动 |
| [`explore_skull.ipynb`](explore_skull.ipynb) | 最早的数据探索 + 已废弃的 `.ply` 批转换 | 只作历史参考，数据管线以 `src/data/prepare_skullfix.py` 为准 |
| [`demo/`](demo/) | 原作者的 vendor demo（推理 / 训练），**已被上面几个取代** | 只在查证"原实现到底怎么写的"时打开 |

## 两条硬规则

**1. 训练和评估不能在同一个 kernel 里。** 训练子进程要 15.5 GiB / 24 GiB；
`MSN_compare_runs` / `MSN_surface_quality` 一旦建了模型就占住显存。
**跑训练前先 Restart Kernel。**

**2. 改过 `src/eval/report.py` 或 `src/eval/mesh_viz.py` 之后**，notebook 里要
`importlib.reload(rp)` 或重启 kernel，否则拿到的是缓存的旧模块。

## 一次完整的实验流程

```
MSN_train_skullfix  第 1 节改 RUN_NAME / FROM_RUN
                    → 第 2 节预检 → 第 3 节训练（35~60 分钟）
                    → 第 4 节自检（停止原因、LR 降了几次、退火了没）
                    → 第 5 节存档到 experiments_log/
                            ↓
                    Restart Kernel
                            ↓
MSN_compare_runs    第 1 节把新 run 加进 RUNS
                    → 第 5 节同轮次表（去掉"跑得久占便宜"）
                    → 第 6 节主表 → 第 7 节配对检验
                    → 对着第 3.2 节的判决清单打勾
                            ↓
                    experiments_log/README.md 加一行 + devlog.md 追加一节 + git commit
```

⚠️ **重复实验一定要用 `FROM_RUN`**（它照抄那个 run 的全部 18 个超参），不要手抄 flag。
