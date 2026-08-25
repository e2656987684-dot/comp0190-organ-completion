# CLAUDE.md — 给每一次新会话看的

COMP0190 硕士项目：**颅骨点云补全**（SkullFix，100 对，80 训 / 20 验）。
模型是 MedShapeNet Foundation Model（MSN = PCT + BERT 文本条件，187M 参数）的重写版。
**当前阶段：收尾**——逐条清 TODO、把结论固化成可复现的脚本、准备写论文。

**这个文件只做路由，不复述内容。** 具体的数字和结论都在下面指的那些文件里，
它们随时在更新，而这个文件不会。

---

## 开始干活之前，按这个顺序读

| 顺序 | 文件 | 读多少 |
|---|---|---|
| 1 | `TODO.md` **最后一节** | **只读最后一节**。前面都是历史快照，最后一节永远是当前有效清单 |
| 2 | `devlog.md` **最后 5~8 条日期条目** | ⚠️ **不要全读**（近 3000 行）。倒着往前读到你需要的地方为止 |
| 3 | `experiments_log/README.md` | 全读。有效性分界、噪声判据、采样地板、各 run 的定性都在这 |
| 4 | `src/eval/README.md` | 全读。每个脚本怎么跑、输出到哪、**k 折后要不要重跑** |
| 5 | `notebooks/README.md` | 训练与判读两个 notebook 的分工 |

**当前各 run 的数字，跑这条命令看**（比任何写死的表都可靠）：

```bash
cd /root/comp0190-organ-completion && /root/miniconda3/envs/comp0190-msn/bin/python -c "
import pandas as pd
df = pd.read_csv('experiments_log/eval_all_runs.csv')
print(df.groupby('run', sort=False)[['CD_t_mm','defect_cov_mm','clump_%']].mean().round(3).to_string())"
```

---

## 工作约定（用户定的，别自作主张）

**训练和分析脚本都由用户自己跑。** 我负责改代码、写脚本、判读结果、落盘记录。

**我给命令时必须一并说明**（缺一不可）：

- 怎么跑（完整命令，用 `/root/miniconda3/envs/comp0190-msn/bin/python`）
- **结果在哪看**：只打印？还是写文件？写哪个文件？
- **要不要 GPU**（要的话提醒先 Restart notebook 的 kernel，否则 OOM）
- 大概多久
- **k 折之后要不要重跑**

**写新脚本时**：放进 `src/eval/`，同时在 `src/eval/README.md` 的表里加一行。
不加就等于又造了一个"临时脚本数字"（见下面的硬规矩）。

---

## 硬规矩 —— 每一条都是踩过坑换来的

### 记录

1. **`devlog.md` 只追加，不改历史。** 结论变了就写**新条目**说明更正，旧条目原样保留
   （它记录的是"当时怎么想的"）。
2. **`TODO.md` 追加式快照**，每次在文件末尾另起一节、重列完整清单，**最后一节有效**。
   ⚠️ 例外：**回答清单上的开放问题**（而非完成一项工作）可以就地改最后一节 ——
   2026-08-25 破过一次例，理由记在 devlog 里。
3. **不可复现的数字不能进论文。** 项目里已经出过三次：粗糙度（0.736/0.760）、
   注意力坍缩、采样地板。**最后一次证明旧数字还是错的**（4.43 → 实测 4.619）。
   → **不入库的数字不但没法复核，还可能一直是错的。**

### 实验

4. **重复实验必须用 `--from-run`**，不要手抄 flag。它照抄那个 run 记录的全部 20 个超参。
5. **加新的训练 flag，必须同时**：① 写进 `run.json` 的 meta ② 加进 `train_skullfix._REPLAY`。
   只做①会警告；两个都忘 → `--from-run` **静默漏掉它**，重复实验悄悄变成另一个实验。
6. **加改变网络拓扑的开关，必须加进 `report.Run.arch_key`。** 不加会**静默**把旧权重
   灌进新拓扑（`load_weights` 在拓扑维度上不设防，形状对得上就不报错）。
7. **训练在 GPU 上不可逐位复现**（没开 `enable_op_determinism`）。所以：
   **删权重之前必须先把逐颅骨指标冻进 `experiments_log/eval_all_runs.csv`。**
8. **推理是确定性的**（验证时走固定种子的 stateless 采样，实测复算偏差 0.00e+00）。
   训练不可复现、评估可复现，这两件事在论文里要分开写。

### 会静默出错的地方（都发生过）

9. `--run-name` 撞名 → 曾覆盖掉 `cd_only` 的 checkpoint，而 `run.json` 只在训练结束时写，
   所以记录和权重指向两次不同的训练且不报错。**已加保护**，要覆盖得显式 `--overwrite`。
10. `ReduceLROnPlateau` 的 patience **必须小于** EarlyStopping 的，否则衰减永不触发 ——
    这个冲突作废了本项目最初四轮。现在 `--lr-patience` 是显式参数并在启动时检查。
11. `ReduceLROnPlateau` 的 `min_delta` 默认 1e-4 是**绝对阈值**，换个损失量级就失效。本项目设 0。
12. `CSVLogger` 在**第 1 轮**就固定列名 → 新增的 per-epoch 指标必须在 epoch 0 就有值。
13. `model(x)` 逐个调用**每次泄漏 0.29 GiB**，用 `model.predict(..., batch_size=1)`。
14. **notebook 里覆盖写 CSV 会毁掉算不出来的行**（`surface_quality.csv` 出过）。一律**合并写**。
15. 改过 `report.py` / `mesh_viz.py` 之后，notebook 要 `importlib.reload`，否则拿到旧模块。
    ⚠️ 只 reload 这两个纯 numpy/plotly 模块；`msn_skullfix` 定义 Keras 层，热重载会出怪事。
16. **CD_t 有两个口径，不要混引**：`run.json` 的 `best_val_cd_t_mm`（训练期、数据集平均 scale）
    比 `report.py` 的逐颅骨口径系统性低约 0.09mm。**论文引 `eval_all_runs.csv`**。

---

## 判读结果的规矩

**主指标是 `defect_cov_mm`（缺损区覆盖）。** 只有约 6.4% 的 GT 点在缺损区，其余是输入里
已给、模型只需复现的表面 —— 全点云指标主要在量"抄得像不像"。实测佐证：2×2 消融在缺损
覆盖上四格全部显著，在全点云 CD_t 上**全部不显著**。

**一个改动算"成立"，要同时满足四条**（详见 `MSN_compare_runs.ipynb` 第 3 节）：

- [ ] 同轮次表上仍领先（`report.epoch_matched`）—— 不是靠多跑几十轮
- [ ] 配对检验主指标 `p_wilcoxon < 0.002` 或改善 ≥17/20（`report.paired_stats`）
- [ ] 有同配置重复，且两次差异 < 声称的效应
- [ ] 方向与机制解释一致

**三把尺子互不替代**：① 同配置重跑（训练随机性）② 逐颅骨配对（换一批颅骨还成不成立）
③ 末段 epoch 抖动。⚠️ **配对检验能证明"两个模型不同"，不能证明"这个配置更好"** ——
同配置训练两次本身就能产生跨颅骨一致的差异（`tie_qk` 那对就是）。

⚠️ **`lr_fix_only` 之前的四轮**（`baseline_es20` / `dcd_w3` / `dcd_l2` / `rep05_void`）
**是⛔错误性实验**，不得作为基线或对照，权重也已删除。详见 `experiments_log/README.md` 开头。

---

## ⚠️ 现在所有结果都是暂时的

**定稿前要跑 k 折**（4 配置 × 5 折 ≈ 18~20h），届时大部分数字会变。所以：

> **凡是消费当前结果的东西，代码必须留在仓库里、并且留好输入口** ——
> 不能把数字抄进文档就算完。

**哪些要重跑、从哪个口子重跑，见 `src/eval/README.md` 的「k 折之后要重跑什么」一节。**

⚠️ 两个 k 折会撞上的坑：`report.paired_stats` **在 k 折下不适用**（各折验证集不同，
配不起来，得新写折间聚合）；`MSN_compare_runs.ipynb` 第 1 节那条 assert 会先拦下你。

⚠️ **k 折是所有改进做完之后的终审，不是现在做的事**（用户明确过）。
重复实验（跑两次）是现阶段分辨运气的工具，**不是 k 折的替代**。

---

## 论文范围（2026-08-21 用户拍板，详见 devlog）

**进论文**：注意力坍缩自查 · 移植审计 + `tie_qk` 的受控验证 · 训练配置与原实现的差异表 ·
采样地板与跨域不可比性 · 缺损区限定评测 + DCD 2×2 · 文本分支的折叠证明。

**不进论文**：自己引入并修好的错误（EarlyStopping/LR 冲突、缺损区掩码定义错误、
repulsion 未无量纲化）· 逐点交叉注意力的阴性结果。
**原则：自己写错的错误留在 devlog 作为工作记录，不进论文。**

⚠️ **不能写 "zero-shot"** —— 已查证那套预训练权重训练时**见过颅骨数据**。

---

## 环境

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python     # conda 环境 comp0190-msn
```

- **单卡 RTX 4090 24GB**。一个 187M 模型占 15.5 GiB → **notebook kernel 和脚本抢显存**，
  跑 GPU 脚本前先 Restart kernel。
- 训练约 **10.6 s/epoch**，近期各轮 220~410 轮 = **40~70 分钟**。
- `/root` 是临时盘（重部署会清空），`/workspace` 是网络盘。
  **代码靠 git，产物和数据靠 `bash sync_workspace.sh backup`**，两者互补，都要做。
- `experiments/`（权重，5.6G）和 `data/` 都 gitignored；`experiments_log/`（run.json +
  history.csv + 各种 CSV）**跟踪进 git**。

## 仓库地图

```
src/data/prepare_skullfix.py     raw nrrd -> 对齐点云对 -> data/cache/*.npz（唯一数据路径）
src/models/msn_skullfix.py       重写版网络 + 损失 + 指标（训练本项目模型用）
src/models/msn_demo_arch.py      原 demo 架构逐字复制（只用来跑作者的预训练权重）
                                 ⚠️ 两者权重不兼容，别混
src/models/train_skullfix.py     训练 CLI
src/eval/report.py               Run / eval_runs / epoch_matched / paired_stats / fig_*
src/eval/mesh_viz.py             mesh 重建 + surface_stats（⚠️ 重建只能看，不能算指标）
src/eval/*.py                    各个一次性分析脚本，见 src/eval/README.md
notebooks/MSN_train_skullfix.ipynb   发起训练（只改第 1 节控制面板）
notebooks/MSN_compare_runs.ipynb     判读结果（指标词典、判决标准都在里面）
```
