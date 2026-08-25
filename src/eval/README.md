# src/eval — 评测与分析

**约定：每个可执行脚本都在这张表里写清楚「怎么跑 / 结果在哪看 / 写不写文件 / 要不要 GPU」。**
新增脚本必须同时加一行，否则以后没人知道那个数字是从哪来的
（粗糙度 GT 0.736 / pred 0.760 和注意力坍缩那两组数就是这么变成不可复现的）。

所有命令都在仓库根目录下跑，用 `comp0190-msn` 环境：

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python
```

## 可执行脚本

| 脚本 | 怎么跑 | 结果在哪看 | 写文件吗 | GPU | 耗时 |
|---|---|---|---|---|---|
| **`fold_text_branch.py`** | `$PY src/eval/fold_text_branch.py` | **只打印到终端** | ❌ 不写 | ✅ 建两个 187M 模型 | ~40 秒 |
| **`eval_pretrained_baseline.py`** | `$PY src/eval/eval_pretrained_baseline.py` | 终端 + CSV | ✅ 写 `experiments_log/pretrained_baseline/eval_val20_x5.csv`（**不动**旧的 `eval_val20.csv`） | ✅ 296.9M（含 BERT） | 十几分钟 |
| `make_report_figures.py` | `$PY src/eval/make_report_figures.py` | `reports/figures/*.png` | ✅ 覆盖写 | ✅ | 几分钟 |
| `make_report_deck.py` | `$PY src/eval/make_report_deck.py` | `reports/progress_report.pptx` | ✅ 覆盖写 | ❌ | 秒级 |
| `make_progress_deck.py` | `$PY src/eval/make_progress_deck.py` | `reports/progress_report_2.pptx` | ✅ 覆盖写 | ❌ | 秒级 |

⚠️ **需要 GPU 的脚本跑之前，先把 notebook 的 kernel Restart** —— 一个 187M 模型就占 15.5/24 GiB，
kernel 占着显存时这些脚本会 OOM。

⚠️ `make_report_figures.py` 的 `RUNS` **目前指向已裁剪权重的三个 run**（`baseline_es20` /
`dcd_l2` / `dcd_w3`），照现状跑会失败。第一次汇报的图已用 `git add -f` 入库，
要复用这个脚本得先把 `RUNS` 换成还有权重的 run —— 见脚本顶部的说明。

## 模块（不单独跑，被 notebook `import`）

| 模块 | 做什么 |
|---|---|
| `report.py` | `Run` / `load_runs` / `eval_runs`（逐颅骨指标）/ `epoch_matched`（同轮次）/ `paired_stats`（配对检验）/ 各种 `fig_*` |
| `mesh_viz.py` | 点云 → mesh 重建、`surface_stats`（扎堆率/间距 CV/有符号偏差）、诊断图 |

⚠️ 改完这两个模块，notebook 里要 `importlib.reload(rp)` 或重启 kernel，否则拿到的是缓存的旧模块。

## 「结果只打印、不写文件」什么时候可以接受

只有当**脚本本身在仓库里、且结果确定性可复现**时才行 —— 那样任何人重跑都能拿到
逐位相同的数字，终端输出丢了也无所谓。`fold_text_branch.py` 属于这一类（推理走固定
种子的 stateless 采样，实测复算偏差 0.00e+00）。

**训练产出的数字不适用这条**：训练在 GPU 上不可复现，所以逐颅骨指标必须冻进
`experiments_log/eval_all_runs.csv`，删权重之前也必须先冻。
