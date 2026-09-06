# CLAUDE.md — 给每一次新会话看的

COMP0190 硕士项目：**颅骨点云补全**（SkullFix，100 对，80 训 / 20 验，单卡 4090）。
模型是 MedShapeNet Foundation Model（MSN = PCT + BERT 文本条件，187M 参数）的重写版。
**当前阶段：收尾**——逐条清 TODO、把结论固化成可复现的脚本、准备写论文。

> ⏳ **2026-09-06 起：k 折正在跑**（`cd_only` 五折进行中，见 [`KFOLD.md`](KFOLD.md)）。
> ⚠️ **训练跑着的时候 `src/models/` 和 `data/` 不能动**（下一折是新子进程，会用改后的代码
> → 五折不可比），**要 GPU 的脚本也不能跑**。`src/eval/` 和所有文档随便动 ——
> 训练完全不碰它们。清单见 KFOLD.md「训练跑着的时候，哪些文件能动」。

**这个文件只做路由和立规矩，不复述内容。** 具体数字都在它指的那些文件里，
那些文件随时在变，而这个文件不会。

---

## 一、这个项目是怎么走到今天的

读状态之前先看这段，有了脉络再看细节会快很多。**这一节是历史，不会过期。**

| 阶段 | 干了什么 | 留下的关键认识 |
|---|---|---|
| **数据** | 修了两个正确性 bug：配对错位（complete/defective 各自归一化 → 落到不同坐标系）、各向异性体素间距 | 数据只走 `prepare_skullfix.py` 这一条路 |
| **跑得动** | 距离矩阵从 tile 改成 `\|a\|²−2a·b+\|b\|²`，187M 模型在单卡上从跑不动变成 372ms/步 | 不用缩模型 |
| **发现真问题** | 原以为"预测表面不够平滑"，实测被推翻——GT 和预测的局部粗糙度基本持平；**真正确凿的差距是点密度**（扎堆 11.2% vs GT 0.0%） | 前提要先量再信 |
| **损失** | DCD 超参扫（后作废）→ **repulsion loss 生效**（扎堆 13.6% → 1.3%）→ **DCD 消融证明可以去掉** | **最优配置 = CD + repulsion**，全项目最硬的一条 |
| **评测协议** | 发现全点云指标 93.6% 的权重在"复现输入"上 → 建**缺损区限定指标**；掩码定义错过一次，已修正 | **主指标 = 缺损区覆盖** |
| **⛔ 有效性重划分** | 查证发现 `ReduceLROnPlateau(patience=40) > EarlyStopping(20)` 是本项目自己写的配置错误，导致衰减从未触发 → **最初四轮整档作废** | 自己的错误不进论文 |
| **架构侧尝试** | 逐点交叉注意力 ❌（注意力从未启动）· `--no-text` ✅（现象成立）· Q/K 初始绑定 ❌（重复实验推翻） | 80 个样本学不出注意力选择性 |
| **判读方法学** | 三把尺子分开、同轮次表、逐颅骨配对检验；发现"0.004mm 噪声底线"是撞大运 | 见下面「判读规矩」 |
| **收尾（进行中）** | 仓库整理（19G→5.6G）· TODO ⚙️ 清空 · 折叠验证 · 基线重评 · 采样地板正式入库 | 剩下的见 TODO 最后一节 |
| **临时脚本数字正式化** | 采样地板 · 注意力坍缩 · 粗糙度 —— **三个全部在正式测量后被更正**（不只是"不可复现"，是**真的错了**） | 不入库的数字可能一直是错的 |
| **表示的能力边界** | 点到面指标（地板两法互验 2.310 vs 2.300）· 法向不可估（定向翻转≈50%）→ **Poisson 取消** | "离临床多远"是**分辨率**问题：6144 点 ≈4mm，AutoImplant 是 0.45mm 体素 |
| **⭐ 缺损区换真值** | 数据集自带的 `implant/` 从没被用过；5mm 距离规则实测 precision 0.79 / recall 0.81 → **改用真值** | **数量对 ≠ 集合对**；三条结论全部存活且两条变强 |

---

## 二、读状态：两档模式，自己选

### 全面模式（默认，实测约 90k token — 上下文窗口的 45%）

用户希望新会话尽量了解全貌。按这个顺序读：

| 顺序 | 文件 | 读多少 | 约 token |
|---|---|---|---|
| 1 | `TODO.md` **最后一节** | 全读（当前有效清单） | 3.8k |
| 1.5 | `RUNBOOK.md` | 全读（所有终端命令，按任务排） | 1.2k |
| 1.6 | `KFOLD.md` | ⭐ **k 折进行中就先看这个** —— 20 条命令 + 进度表 + 每条的自检 | 2k |
| 2 | `devlog.md` | **全读**（54 条日期条目，倒着读更快进入状态）<br>⚠️ 先看文件头那条「devlog 里一切都是待定的」 | **~80k** |
| 3 | `experiments_log/README.md` | 全读（有效性分界、噪声判据、采样地板、各 run 定性） | 7.6k |
| 4 | `src/eval/README.md` | 全读（脚本怎么跑 + **k 折后要重跑什么**） | 2k |
| 5 | `notebooks/README.md` | 全读（两个 notebook 的分工） | 1k |
| 6 | `README.md` | 全读（对外的项目描述、范围决策） | 2k |
| 7 | 四个模块 docstring | `msn_skullfix.py`(76行) `train_skullfix.py`(85行) `mesh_viz.py`(45行) `report.py` 的「三把尺子」注释块 | 4k |
| 8 | `MSN_compare_runs.ipynb` 的 markdown | 指标词典 + 判决标准（干活时最常查的） | 4k |

### 轻量模式（约 20k token）

**这份 CLAUDE.md 自己就占 5.3k**，是两档模式共同的底。

**如果上下文吃紧、或者发现回答开始不准，改用这个**：第 1、3、4 项全读，
devlog 只读**最近 8 条**。上面的「项目脉络」已经覆盖了更早的内容。

### 定位 devlog 的技巧（不用手工维护目录）

```bash
grep -n "^## " devlog.md          # 54 条的日期 + 标题 + 行号，永远是最新的
grep -n "⭐\|⚠️" devlog.md | head -40   # 标了星号和警告的条目，重要性排序
```

标题里的 ⭐ 越多越重要；⚠️ 表示那条包含**对先前结论的更正**。

### 当前各 run 的数字，跑这条命令看

比任何写死的表都可靠：

```bash
cd /root/comp0190-organ-completion && /root/miniconda3/envs/comp0190-msn/bin/python -c "
import pandas as pd
df = pd.read_csv('experiments_log/eval_all_runs.csv')
df = df[df.defect_def == 'implant']      # ⚠️ 排掉 5mm 旧口径的两行
print(df.groupby('run', sort=False)[['CD_t_mm','defect_cov_mm','clump_%']].mean().round(3).to_string())"
```

---

## 三、工作约定（用户定的，别自作主张）

**训练和分析脚本都由用户自己跑。** 我负责改代码、写脚本、判读结果、落盘记录。

**我给命令时必须一并说明**（缺一不可）：

- 完整命令（用 `/root/miniconda3/envs/comp0190-msn/bin/python`）
- **结果在哪看**：只打印？还是写文件？写哪个文件？
- **要不要 GPU**（要的话提醒先 Restart notebook 的 kernel，否则 OOM）
- 大概多久
- **k 折之后要不要重跑**

**写新脚本时**：放进 `src/eval/`，同时在 `src/eval/README.md` 的表里加一行；
**如果它是用户会反复敲的命令，再往 `RUNBOOK.md` 里加一条**。
不加就等于又造了一个"临时脚本数字"（见下面第 3 条）。

⚠️ **给用户看的东西，先确认他那一侧打得开。** 远程 pod + VSCode：PNG 能直接看，
几十 MB 的 HTML 看不了。**修好一个底层能力 ≠ 用户拿到了它 —— 交付的单位是
用户实际会跑的那条命令。**（2026-09-06 连栽两次换来的。）

**给用户的回答**：用中文；结论先行；数字要带出处；不确定就说不确定。

---

## 四、硬规矩 —— 每一条都是踩过坑换来的

### 记录

1. **`devlog.md` 只追加，不改历史。** 结论变了就写**新条目**说明更正，旧条目原样保留
   （它记录的是"当时怎么想的"，那本身有信息量）。
2. **`TODO.md` 追加式快照**，每次在文件末尾另起一节、重列完整清单，**最后一节有效**。
   ⚠️ 例外：**回答清单上的开放问题**（而非完成一项工作）可以就地改最后一节 ——
   2026-08-25 破过一次例，理由记在 devlog 里。
3. **不可复现的数字不能进论文。** 项目里已经出过三次：粗糙度（0.736/0.760）、
   注意力坍缩、采样地板。**最后一次证明旧数字还是错的**（4.43 → 实测 4.619）。
   → **不入库的数字不但没法复核，还可能一直是错的。**

### 实验

4. **重复实验必须用 `--from-run`**，不要手抄 flag。它照抄那个 run 记录的全部 20 个超参，
   并在你手动覆盖某项时警告"这不再是严格重复"。
5. **加新的训练 flag，必须同时**：① 写进 `run.json` 的 meta ② 加进 `train_skullfix._REPLAY`。
   只做①会警告；两个都忘 → `--from-run` **静默漏掉它**，重复实验悄悄变成另一个实验。
6. **加改变网络拓扑的开关，必须加进 `report.Run.arch_key`。** 不加会**静默**把旧权重
   灌进新拓扑（`load_weights` 在拓扑维度上不设防，权重形状对得上就不报错）。
7. **训练在 GPU 上不可逐位复现**（没开 `enable_op_determinism`，实测同 seed 从第 1 轮就分岔）。
   所以：**删权重之前必须先把逐颅骨指标冻进 `experiments_log/eval_all_runs.csv`。**
8. **推理是确定性的**（验证走固定种子的 stateless 采样，实测复算偏差 0.00e+00）。
   **训练不可复现、评估可复现**，这两件事在论文里要分开写。
9. **诊断指标绝不参与模型选择。** `--defect-every` 记的缺损区覆盖只写进 history，
   不驱动早停或 checkpoint —— 用报告结果的同一个指标、在同一批 20 颗颅骨上挑最好看的一轮，
   会让报告值乐观偏置。**用 `val_loss` 选、用缺损覆盖报，这个安排本身是对的，论文值得写一句。**

### 会静默出错的地方（都真的发生过）

10. **`--run-name` 撞名** → 曾覆盖掉 `cd_only` 的 checkpoint，而 `run.json` 只在训练结束时写，
    于是记录和权重指向两次不同的训练且不报错。**已加保护**，要覆盖得显式 `--overwrite`。
11. **`ReduceLROnPlateau` 的 patience 必须小于 EarlyStopping 的**，否则衰减永不触发 ——
    这个冲突作废了最初四轮。现在 `--lr-patience` 是显式参数并在启动时检查。
12. **`ReduceLROnPlateau` 的 `min_delta` 默认 1e-4 是绝对阈值**，换个损失量级就失效
    （`cd_dcd` 在 1.0 附近能过线，纯 `cd` 在 0.07 附近就全被判为无改善）。本项目设 0。
13. **`CSVLogger` 在第 1 轮就固定列名** → 新增的 per-epoch 指标必须在 epoch 0 就有值，
    否则整列被丢掉。
14. **`model(x)` 逐个调用每次泄漏 0.29 GiB**，用 `model.predict(..., batch_size=1)`。
15. **notebook 里覆盖写 CSV 会毁掉算不出来的行**（`surface_quality.csv` 出过；
    `eval_all_runs.csv` 里 `baseline`/`dcd_l2` 两行的权重已删、再也算不出来）。**一律合并写。**
16. **改过 `report.py` / `mesh_viz.py` 之后 notebook 要 `importlib.reload`**，否则拿到旧模块。
    ⚠️ 只 reload 这两个纯 numpy/plotly 模块；`msn_skullfix` 定义 Keras 层，热重载会出怪事。
17. **CD_t 有两个口径，不要混引**：`run.json` 的 `best_val_cd_t_mm`（训练期、数据集平均 scale、
    全程最优）比 `report.py` 的逐颅骨口径系统性**低约 0.09mm**。**论文引 `eval_all_runs.csv`。**
18. **编辑器会把移动过的文件写回老路径**（`git mv` 之后 VS Code 重存了一份旧版）。
19. ⚠️⚠️ **读任何带 `id` 的 CSV 一律 `pd.read_csv(..., dtype={"id": str})`。**
    颅骨编号带前导零（`'083'`），不指定就被读成整数 `83`，而 **`str(83)` 是 `'83'` 不是 `'083'`
    ——`astype(str)` 顶不上它**。以 id 为键的合并会**静默落空**，重跑一次数据就变成两份。
    **本周同一个陷阱出现五次**，最险的一次两组行「与掩码无关的列」逐位相同，看表的人根本
    察觉不到有两份数据。**根上的修法是让它从来不变成整数，而不是事后补救。**
20. **`Run.label` 取自目录名，可能与 CSV 里的历史标签不同**（`lr_fix_only` vs `lr_fix`）。
    以 run 名为键的合并同样会静默落空 —— `recompute_eval_all.py` 因此加了
    `EXPECTED_LEGACY` 白名单，任何本该被重算却留在旧口径的 run 都直接中止。

---

## 五、判读规矩

**主指标是 `defect_cov_mm`（缺损区覆盖）。** 只有约 6.2% 的 GT 点在缺损区，其余是输入里
已给、模型只需复现的表面 —— 全点云指标主要在量"抄得像不像"。实测佐证：2×2 消融在缺损
覆盖上四格**全部显著**（95% CI 全部不跨零，p ≤ 0.002），在全点云 CD_t 上**四格 CI 全部跨零**。

⚠️ **后半句必须带判据**：CD_t 上最强的那格（+repulsion，无 DCD）是 18/20、`p_wilcoxon = 0.0017`
—— **过了 0.002 线**，只是区间仍跨零。简写成"全部不显著"会被问穿。

⚠️ **缺损区自 2026-08-28 起由数据集自带的植入物真值定义**（`experiments_log/defect_mask_labels.npz`，
由 `make_defect_labels.py` 生成），**不再是距离规则**。`report.DEFECT_MM = 5.0` 只剩**预测侧**的容差。
**⛔ 换口径前后的 `defect_*` 数字不可混引** —— `eval_all_runs.csv` 有 `defect_def` 列标明每行的口径
（只有权重已删的 `baseline` / `dcd_l2` 仍是 `5mm_legacy`，而它们本来就不进论文）。

⚠️ **`defect_prec_mm` 可以被糊弄**（不往洞里放点就好看），必须和 `defect_n_pred` 一起看。
真实案例：预训练基线的缺损精度只比本工作差 **1.13×**，但它只往洞里放了 **30 个点**
（GT 真实缺损 380，本工作放 391）—— **它几乎一个点都没放进去**。

**一个改动算"成立"，要同时满足四条**（详见 `MSN_compare_runs.ipynb` 第 3 节）：

- [ ] 同轮次表上仍领先（`report.epoch_matched`）—— 不是靠多跑几十轮
- [ ] 配对检验主指标 `p_wilcoxon < 0.002` 或改善 ≥17/20（`report.paired_stats`）
- [ ] 有同配置重复，且两次差异 < 声称的效应
- [ ] 方向与机制解释一致

**三把尺子互不替代**：① 同配置重跑（训练随机性）② 逐颅骨配对（换一批颅骨还成不成立）
③ 末段 epoch 抖动（报告值本身抖多少）。

⚠️ **配对检验能证明"两个模型不同"，不能证明"这个配置更好"** —— 同配置训练两次本身
就能产生跨颅骨一致的差异（`tie_qk` 那对就是活例子）。

⚠️ **同配置重跑的差异通常 ≲0.005mm，但当一个 run 在另一个停下的位置上仍在下降时可达 0.15mm。**
读小差异前先看 `epoch_matched` 的末几列有没有走平。

⚠️ **`lr_fix_only` 之前的四轮**（`baseline_es20` / `dcd_w3` / `dcd_l2` / `rep05_void`）
**是 ⛔ 错误性实验**，不得作为基线或对照，权重也已删除。方法比较的基线用 `lr_fix_only`。

---

## 六、⚠️ 现在所有结果都是暂时的

**定稿前要跑 k 折**（4 配置 × 5 折）。⏳ **2026-09-06 已开跑**，届时大部分数字会变。

⚠️ **早停 patience 2026-09-06 从 30 改回 20**（`train_skullfix` 的默认值）。
8/25 那个「patience=20 会把五个 run 擦边掐死、抽签值 0.15mm」的发现**没有被推翻**，
改回来是因为代价：patience=30 下第一个 run 599 轮仍在创新低、撞 600 报废，20 折要 28h，
而可用时间是 20h；patience=20 约 12h（实测前三折 250~275 轮）。
⭐ 两条理由让它可接受：① 5 折标准误 0.09mm 的估算本就基于**含这个抽签**的 0.15mm 训练方差；
② **零个历史 run 用过 30**（三条证据见 devlog 2026-09-06 第五条），所以不产生孤儿数据。

所以用户定了一条规矩：

> **凡是消费当前结果的东西，代码必须留在仓库里、并且留好输入口** ——
> 不能把数字抄进文档就算完，还要注明"k 折之后要不要重跑"。

**哪些要重跑、从哪个口子重跑，见 `src/eval/README.md` 的「k 折之后要重跑什么」一节。**

⚠️ k 折会撞上的坑：`report.paired_stats` **在 k 折下不适用**（各折验证集不同，配不起来）。
替代品 `fold_frame` / `fold_summary` / `fold_paired` **已于 2026-08-29 入库**，用之前先知道三条：
⚠️ **`--run-name` 必须写成 `<config>_f<fold>`**（折号以 `run.json` 为准、拿名字核对，对不上直接报错）；
⚠️ **k=5 时符号检验 p 的下限是 0.0625，够不到项目的 p<0.002**，判据换成「k 折同向 且 |delta| > 2×SE」；
⚠️ 池化 100 颗只是尺子②，它的 p 对"这个配置更好"偏松。
`MSN_compare_runs.ipynb` 第 1 节那条 assert **仍会先拦下你**——notebook 的 k 折读表一节还没写。

⚠️ **k 折是所有改进做完之后的终审，不是现在做的事**（用户明确过）。
重复实验（同配置跑两次）是现阶段分辨运气的工具，**不是 k 折的替代**。

---

## 七、论文范围（2026-08-21 用户拍板，详见 devlog）

**进论文**：注意力坍缩自查 · 移植审计 + `tie_qk` 的受控验证 · 训练配置与原实现的差异表 ·
采样地板与跨域不可比性 · 缺损区限定评测 + DCD 2×2 · 文本分支的折叠证明。

**不进论文**：自己引入并修好的错误（EarlyStopping/LR 冲突、缺损区掩码定义错误、
repulsion 未无量纲化）· 逐点交叉注意力的阴性结果。

> **原则：自己写错的错误留在 devlog 作为工作记录，不进论文。**
> 但「与参考实现的差异 + 受控验证」是**正面材料**，要写（`tie_qk` 属于后者）。

⚠️ **不能写 "zero-shot"** —— 已查证那套预训练权重训练时**见过颅骨数据**。
而且可能存在训练/测试污染（MedShapeNet 部分源自 AutoImplant = SkullFix 的来源），
方向是**让本工作的领先被低估**，论文要主动写明。

---

## 八、环境

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python     # conda 环境 comp0190-msn
```

- **单卡 RTX 4090 24GB**。一个 187M 模型占 15.5 GiB → **notebook kernel 和脚本抢显存**，
  跑 GPU 脚本前先 Restart kernel。
- 训练约 **10.6 s/epoch**，近期各轮 220~410 轮 = **40~70 分钟**。
- `/root` 是临时盘（重部署会清空），`/workspace` 是网络盘。
  **代码靠 git，产物和数据靠 `bash sync_workspace.sh backup`**，两者互补，都要做。
  （`/root/.claude` 是指向 `/workspace/.claude-config` 的软链接，所以对话记录本身是安全的。）
- `experiments/`（权重 5.6G）和 `data/` 都 gitignored；
  `experiments_log/`（run.json + history.csv + 各种 CSV）**跟踪进 git**。
- ⚠️ **出图（`fig.write_image`）依赖一个无头 Chrome + `libnss3`/`libnspr4`**，
  装在 `/root`（临时盘）→ **重部署后要重跑 `setup_env.sh` 第 5 节**，否则论文图一张都出不来。

## 九、仓库地图

```
src/data/prepare_skullfix.py     raw nrrd -> 对齐点云对 -> data/cache/*.npz（唯一数据路径）
src/models/msn_skullfix.py       重写版网络 + 损失 + 指标（训练本项目模型用）
src/models/msn_demo_arch.py      原 demo 架构逐字复制（只用来跑作者的预训练权重）
                                 ⚠️ 两者权重不兼容，别混；加载失败是静默的
src/models/train_skullfix.py     训练 CLI（--from-run / --overwrite / --defect-every ...）
src/models/run_kfold.py          ⭐ k 折驱动：20 个顺序跑，可断点续，撞上限/磁盘不足即中止
src/eval/report.py               Run / eval_runs / epoch_matched / paired_stats / fig_*
                                 k 折折间聚合：fold_frame / fold_summary / fold_paired
src/eval/mesh_viz.py             mesh 重建 + surface_stats（⚠️ 重建只能看，不能算指标）
src/eval/fold_text_branch.py     文本分支折叠验证（证明它等于 4 个偏置向量）
src/eval/sampling_floor.py       采样地板（k 折后不用重跑）
src/eval/make_defect_labels.py   ⭐ 缺损区真值标签（100 颗已生成入 git，k 折不用重跑）
src/eval/defect_mask_audit.py    审计 5mm 规则 vs implant 真值（打标签的唯一来源 label_one）
src/eval/recompute_eval_all.py   换口径后重算主表（带「与掩码无关的列必须不变」的断言）
src/eval/point_to_surface.py     点到面指标（地板互验；⚠️ signed_deviation 已判过时）
src/eval/normal_quality.py       法向可估性闸门（⛔ 不通过 → Poisson 取消）
src/eval/roughness.py            粗糙度（⚠️ 旧值 0.736/0.760 符号是反的，已作废）
src/eval/eval_pretrained_baseline.py  预训练基线重评（k 折后每折各跑一次）
src/eval/mesh_preview.py         看一眼 mesh（默认出 PNG；--truth 加原始体数据两格）
                                 ⚠️ 只能看不能算指标；不产出入库数字
RUNBOOK.md                       所有终端命令，按「我现在要干嘛」排
KFOLD.md                         ⭐ k 折执行清单：20 条命令 + 自检 + 进度表
notebooks/MSN_train_skullfix.ipynb    发起训练（只改第 1 节控制面板）
notebooks/MSN_compare_runs.ipynb      判读结果（指标词典、判决标准都在里面）
notebooks/MSN_surface_quality.ipynb   mesh 可视化 + 密度诊断
```
