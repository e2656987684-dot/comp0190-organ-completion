# notebooks

| notebook | 干什么 | 什么时候用 |
|---|---|---|
| **[`MSN_train_skullfix.ipynb`](MSN_train_skullfix.ipynb)** | ⭐ **训练**（2026-09-08 重写）。2×2 设计、20 个折怎么跑出来的、逐折自检 + 曲线 | 想复现训练、或核对那 20 折干不干净时。⚠️ 第 4 节起**不用 GPU 也能跑**（记录在 git 里） |
| **[`MSN_inference.ipynb`](MSN_inference.ipynb)** | **推理**（2026-09-08 新建）。加载某一折的权重，把残缺颅骨补全 | 想直接看模型输出什么时。要 GPU + 权重，一颗约 1 秒 |
| **[`MSN_compare_runs.ipynb`](MSN_compare_runs.ipynb)** | **判读结果**：指标词典、同轮次表、配对检验、主表、可视化、对照组 | 训练跑完之后 |
| [`MSN_surface_quality.ipynb`](MSN_surface_quality.ipynb) | mesh 重建、密度诊断、有符号偏差着色 | 想看**表面质量**而不是数字时 |
| **[`MSN_baseline_pretrained.ipynb`](MSN_baseline_pretrained.ipynb)** | ⭐ **对照评估**（2026-09-08 重写）。作者发布的权重 vs 本项目 k 折模型；含权重加载自检 | 想知道「专精训练到底买到了多少」时。⚠️ 数字由 `eval_pretrained_baseline.py` 产出，这里只读表 |
| **[`explore_skull.ipynb`](explore_skull.ipynb)** | ⭐ **看数据**（2026-09-07 重写）。原始 nrrd 体数据 + `prepare_skullfix.py` 产出的点云缓存，含缺损区真值 | 想搞清楚「喂进模型的到底是什么」时。**只读，不写任何文件** |
| [`upstream_msn/`](upstream_msn/) | 上游 MedShapeNet 项目的两个 notebook（推理 / 训练）。⚠️ **不是原样未改**：本项目往推理那个里接过自己的数据管线 | 只在查证"上游到底怎么写的"时打开。⚠️ 逐字原版看 `src/models/msn_demo_arch.py` |

## 跑 k 折训练时（`MSN_train_skullfix.ipynb`）

每个训练单元格调的是 `run_kfold.py`，不是 `train_skullfix.py`。多出来的五件事，
每一件都对应一种真实踩过的失败：

| | 为什么 |
|---|---|
| 撞 `--epochs` 上限就中止 | 那种 run 还在下降时被截断，**不可引用**。裸循环会接着跑，几小时后才发现 |
| 每轮开跑前查磁盘 | 20 个权重要 14.3 GB；中途撑爆看起来像训练 bug |
| 已完成的自动跳过 | 断了就重跑同一个单元格，不用判断 |
| 每轮跑完自动自检 + 存档 | 否则要手做 20 次，凌晨三点那次一定跳过 |
| 软警告只记不停 | LR 降太少、末段抖、val/train 偏高都是「读数当心」，不是坏 run |

三个会**静默**毁掉整轮的陷阱（不会报错，只会让结果不可比）：

- **不要用 `--from-run`** —— 它照抄老 run 的超参，包括 `n_folds=0`，会把一折悄悄变成单次划分
- **`--dcd-lambda 2` 必须显式写** —— 默认是 1，而现有两个 DCD 格用的是 2
- **run 名必须是 `<配置>_f<折号>`** —— `report.fold_frame` 拿名字和 `run.json` 里的折号互校，
  这是 20 个手写名字唯一的防线

⚠️ 训练跑着的时候**不要改 `src/models/` 和 `data/`**：每一折是新的子进程，
改动会落到下一折上，五折就不是同一个实验了。

---

## 四条硬规则

（第 3、4 条太长，单独成节放在下面。）

**1. 训练和评估不能在同一个 kernel 里。** 训练子进程要 15.5 GiB / 24 GiB；
`MSN_compare_runs` / `MSN_surface_quality` 一旦建了模型就占住显存。
**跑训练前先 Restart Kernel。**

**2. 改过 `src/eval/report.py` 或 `src/eval/mesh_viz.py` 之后**，notebook 里要
`importlib.reload(rp)` 或重启 kernel，否则拿到的是缓存的旧模块。

## 硬规则 4：这些 notebook 必须**从第 1 节开始按顺序跑**

不是洁癖，是结构决定的：**第 1 节把 `src/eval` / `src/models` 加进 `sys.path`**
（这几个模块不在包路径上），**第 2 节跑推理产生 `preds`**。跳过它们直接点后面的 cell，
拿到的是 `ModuleNotFoundError: No module named 'mesh_preview'` ——
**那不是缺依赖，是没按顺序跑。**

VSCode 里用顶部的 **Run All**，或者先跑第 1、2 节。
`MSN_surface_quality.ipynb` 第 4 节之后的几个 cell 已经加了前置检查，
冷启动时会直接告诉你缺的是哪个变量、该先跑哪一节。

⚠️ **编辑器里的黄色波浪线是另一回事**：Pylance 不执行 `sys.path.insert`，所以会把
`mesh_viz` / `report` / `mesh_preview` / `msn_skullfix` 标成"无法解析"。
`.vscode/settings.json` 里的 `python.analysis.extraPaths` 已经把这个消掉了。

---

## 硬规则 3：大图不要连输出一起提交

`.ipynb` 里的图**不是链接，是整张图塞在文件里**（plotly 是一大坨 JSON）。重跑一次输出
整个变一遍，git 存不了差异，只能整份再存一遍。实测代价：

```
explore_skull.ipynb        一张 marching-cubes 图 = 72.8 MB，提交过 6 次 ≈ 425 MB
MSN_surface_quality.ipynb  13~32 MB，提交过 13 次
.git 因此涨到 1.8 GB —— 每次 clone 都要下这些
```

**约定**：要长期留的图走 `reports/`，不走 `.ipynb`。notebook 里的图是「跑完看一眼」，
提交前清掉：

```bash
$PY -c "
import json,sys
p=sys.argv[1]; nb=json.load(open(p))
for c in nb['cells']: c['outputs']=[]; c['execution_count']=None
json.dump(nb,open(p,'w'),ensure_ascii=False,indent=1); open(p,'a').write('\n')" notebooks/<名字>.ipynb
```

### ⛔ 但清之前必须看一眼里面画的是谁

**有些图再也生成不出来。** 训练在 GPU 上不可逐位复现，权重删了就没了 ——
`MSN_surface_quality.ipynb` 的 cell 10/11 里就有 `baseline` 的密度图和诊断图，
而它的权重 2026-08-24 已删。**那两张和 `surface_quality.csv` 里 `baseline` 那一行是同一性质的东西，
清掉等于永久销毁。**

判断规则：**图里出现的 run，权重还在不在 `experiments/`？**

```bash
ls experiments/msn_skullfix/          # 还有权重的 run
$PY -c "
import json,sys
nb=json.load(open(sys.argv[1]))
for i,c in enumerate(nb['cells']):
    for o in c.get('outputs',[]):
        d=o.get('data',{}).get('application/vnd.plotly.v1+json')
        if d:
            lay=d.get('layout',{}); t=lay.get('title',{})
            print(i, (t.get('text') if isinstance(t,dict) else t),
                  [a.get('text') for a in lay.get('annotations',[]) if a.get('text')])" notebooks/<名字>.ipynb
```

⚠️ 已经提交过的旧版本**不会因为清了当前文件而消失** —— git 只往里加。真要把 `.git`
缩回去得重写全部历史（`git filter-repo`）再强制推送，那是论文交完之后再考虑的事。

---

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

⚠️ **重复实验一定要用 `FROM_RUN`**（它照抄那个 run 的全部 19 个超参，见
`train_skullfix._REPLAY`），不要手抄 flag。
