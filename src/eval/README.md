# src/eval — 评测与分析

**约定：每个可执行脚本都在这张表里写清楚「怎么跑 / 结果在哪看 / 写不写文件 / 要不要 GPU」。**
新增脚本必须同时加一行，否则以后没人知道那个数字是从哪来的
（粗糙度 GT 0.736 / pred 0.760 和注意力坍缩那两组数就是这么变成不可复现的）。
⚠️ **这两笔都已在 2026-08-26 补上，而且都发现旧值有错** —— 连同采样地板，本项目**三个**临时脚本数字**全部**在正式测量后被更正。这条约定不是洁癖。

所有命令都在仓库根目录下跑，用 `comp0190-msn` 环境：

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python
```

## 可执行脚本

| 脚本 | 怎么跑 | 结果在哪看 | 写文件吗 | GPU | 耗时 |
|---|---|---|---|---|---|
| **`fold_text_branch.py`** | `$PY src/eval/fold_text_branch.py` | **只打印到终端** | ❌ 不写 | ✅ 建两个 187M 模型 | ~40 秒 |
| **`attention_collapse.py`** | `$PY src/eval/attention_collapse.py`<br>（`--runs <run> ...` 指定，`--n` 改颅骨数） | 终端 + CSV | ✅ **合并写** `experiments_log/attention_collapse.csv` | ✅ 每套架构建一个 187M 模型 | 约 3~5 分钟（默认 3 个 run） |
| **`roughness.py`** | `$PY src/eval/roughness.py`（`--runs` / `--n`） | 终端 + CSV | ✅ **合并写** `experiments_log/roughness.csv` | ✅ 建一个 187M 模型 | 约 1 分钟 |
| **`sampling_floor.py`** | `$PY src/eval/sampling_floor.py` | 终端 + CSV | ✅ 写 `experiments_log/sampling_floor.csv` | ❌ 纯 CPU | 约 3~4 分钟（100 颗） |
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
| `mesh_viz.py` | 点云 → mesh 重建、`surface_stats`（扎堆率/间距 CV/有符号偏差/**粗糙度**）、`local_roughness`（⚠️ 读它的 docstring：这个度量被骨壳厚度污染，只有 GT-vs-pred 的**差值**有意义）、诊断图 |

⚠️ 改完这两个模块，notebook 里要 `importlib.reload(rp)` 或重启 kernel，否则拿到的是缓存的旧模块。

## 「结果只打印、不写文件」什么时候可以接受

只有当**脚本本身在仓库里、且结果确定性可复现**时才行 —— 那样任何人重跑都能拿到
逐位相同的数字，终端输出丢了也无所谓。`fold_text_branch.py` 属于这一类（推理走固定
种子的 stateless 采样，实测复算偏差 0.00e+00）。

**训练产出的数字不适用这条**：训练在 GPU 上不可复现，所以逐颅骨指标必须冻进
`experiments_log/eval_all_runs.csv`，删权重之前也必须先冻。


## ⚠️ k 折之后要重跑什么

**现在所有的结果都是暂时的。** 定稿时要跑 k 折，届时大部分数字会变。规矩是：
**凡是消费当前结果的东西，代码必须留在仓库里、并且留好输入口**，不能把数字抄进文档就算完。
下面这张表就是那个清单。

| 产物 | k 折后 | 输入口 / 怎么重跑 |
|---|---|---|
| `experiments_log/sampling_floor.csv` | ❌ **不用重跑** | 只依赖数据和点数，与划分/权重无关。这是唯一一个 fold-independent 的量，所以它默认跑全部 100 颗而不是某一折的 20 颗 |
| `experiments_log/eval_all_runs.csv` | ✅ **必须** | `report.eval_runs(REPO, runs)` —— 把折的 run 传进去即可 |
| `experiments_log/surface_quality.csv` | ✅ **必须** | `MSN_surface_quality.ipynb` 的 `MODELS`（⚠️ 存档 cell 是合并写，不会丢旧行） |
| `experiments_log/pretrained_baseline/eval_val20_x5.csv` | ✅ **每折各一次** | `eval_pretrained_baseline.py --split-from <fold run> --out <per-fold csv>`。基线必须在**和它对比的模型同一批颅骨**上评 |
| `experiments_log/attention_collapse.csv` | ✅ **每个最终模型各一次** | `attention_collapse.py --runs <各折的 run>`。坍缩是**一组权重**的性质，与划分无关，所以结论几乎不会变；但论文引的是最终模型那一份，得从那份读。⚠️ 三套以上架构一次跑会撞显存（TF 不归还显存），分几次跑即可 —— CSV 是合并写的 |
| `experiments_log/roughness.csv` | ⚠️ **只在论文引用这个比较时才需要** | `roughness.py --runs <最终模型>`。GT 那一侧只依赖数据、与划分无关；预测那一侧取决于引哪份权重 |
| `fold_text_branch.py` 打印的数字 | ⚠️ 建议重跑 | `--run <最终模型>`。代数结论不会变，但 bias 范数和「46%」是那份权重特有的 |
| `experiments_log/README.md` 里的对照表 | ✅ **必须手工更新** | 表里每个数字都来自上面那些 CSV |
| `reports/figures/*.png`、`*.pptx` | ✅ **必须** | `make_report_figures.py` 的 `RUNS`（⚠️ 目前还指向已裁剪的 run） |

### 两个 k 折会撞上的坑（现在就知道，省得到时候查）

1. **`report.paired_stats` 在 k 折下不适用。** 它靠"所有 run 评的是同一批 20 颗颅骨"来逐颗配对；
   各折的验证集不同，配不起来。k 折要的是**折间**均值 ± 标准差，是另一套聚合逻辑，得新写。
   `MSN_compare_runs.ipynb` 第 1 节那条 `assert` 会先把你拦下来（这是故意的）。
2. **`report.epoch_matched` 仍然可用**，而且更需要 —— 各折的停止轮数同样会不一样。
