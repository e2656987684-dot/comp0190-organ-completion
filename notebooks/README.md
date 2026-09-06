# notebooks

| notebook | 干什么 | 什么时候用 |
|---|---|---|
| **[`MSN_train_skullfix.ipynb`](MSN_train_skullfix.ipynb)** | **发起一次训练** + 本轮自检 + 存档 | 每次要跑训练。**只改第 1 节控制面板** |
| **[`MSN_compare_runs.ipynb`](MSN_compare_runs.ipynb)** | **判读结果**：指标词典、同轮次表、配对检验、主表、可视化、对照组 | 训练跑完之后 |
| [`MSN_surface_quality.ipynb`](MSN_surface_quality.ipynb) | mesh 重建、密度诊断、有符号偏差着色 | 想看**表面质量**而不是数字时 |
| [`MSN_baseline_pretrained.ipynb`](MSN_baseline_pretrained.ipynb) | 作者发布的 `MSN_weights3.h5` 在本项目数据上推理 | 对照组，已跑完，基本不用再动 |
| [`explore_skull.ipynb`](explore_skull.ipynb) | 最早的数据探索 + 已废弃的 `.ply` 批转换 | 只作历史参考，数据管线以 `src/data/prepare_skullfix.py` 为准 |
| [`demo/`](demo/) | 原作者的 vendor demo（推理 / 训练），**已被上面几个取代** | 只在查证"原实现到底怎么写的"时打开 |

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
