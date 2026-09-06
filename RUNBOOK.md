# RUNBOOK — 所有要在终端敲的命令，按「我现在要干嘛」排

这份是**任务索引**。按文件查（每个脚本写不写文件、要不要 GPU、耗时）看
[`src/eval/README.md`](src/eval/README.md)；规矩和判读口径看
[`CLAUDE.md`](CLAUDE.md)。**同一个命令只在这里写一次，改了要同步。**

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python
```

> ⚠️ **需要 GPU 的命令跑之前先 Restart notebook 的 kernel。** 一个 187M 模型占
> 15.5 / 24 GiB，kernel 占着显存时脚本直接 OOM。下面标了 🎮 的都要。
>
> ⚠️ **能在 notebook 里看的就别在终端看。** 终端适合"算"，notebook 适合"读"——
> 读 CSV、出表、配对检验、看图，全都该在 `MSN_compare_runs.ipynb` 里。
> 终端只在两种情况下必需：① 训练（几十分钟，kernel 断了就没了）② 抢显存的脚本。

---

## 0. 换了机器 / 重部署之后

```bash
bash setup_env.sh                     # conda 环境 + 权重软链 + 出图依赖
bash sync_workspace.sh restore        # 从 /workspace 取回 data/ 和 experiments/
```

⚠️ `/root` 是临时盘。**代码靠 git，产物和数据靠 `sync_workspace.sh`，两者互补。**
⚠️ `setup_env.sh` 第 5 节装的无头 Chrome 也在 `/root` —— 不重跑就一张论文图都出不来。

备份（跑完重要实验就做一次）：

```bash
bash sync_workspace.sh backup
```

---

## 1. 跑训练

单次实验 → 用 [`notebooks/MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb)
第 1 节控制面板，它会把命令打印出来。重复实验**必须** `--from-run`，别手抄 flag。

### k 折（4 配置 × 5 折 = 20 个，约 17~20 小时）🎮

**一个模型跑完它的 5 折，你手动切下一个**：

```bash
tmux new -s kfold
$PY src/models/run_kfold.py --dry-run cd_only   # 看计划
$PY src/models/run_kfold.py cd_only             # 5 折，约 4~5 小时；跑完打印小结
$PY src/models/run_kfold.py lr_fix_only         # 看完结果再切下一个
$PY src/models/run_kfold.py rep_w05
$PY src/models/run_kfold.py cd_rep05_full
```

撞 `--epochs` 上限或磁盘不足会**中止**；每轮自动自检+存档；断了重跑同一条就续上。
⚠️ 按模型走的代价：**第二个模型跑完之前没有可比对象**。想按折走用 `--all`。

**手动一条条跑（含 notebook 控制面板写法）全在 [`KFOLD.md`](KFOLD.md)** —— 带编号、标了每条是 2×2 里的哪一格、
验证哪 20 颗颅骨，还有每跑完一个的自检与存档命令、进度表、已知的坑。
⚠️ **命令只在那份文件里写一份**，这里不重复。

开跑前三条：在 `tmux` 里跑（断线不丢）· Restart notebook 的 kernel（显存）·
`bash sync_workspace.sh backup`（`/root` 是临时盘）。

---

## 2. 判读结果 → **在 notebook 里**

[`notebooks/MSN_compare_runs.ipynb`](notebooks/MSN_compare_runs.ipynb)：第 1 节加 run →
第 5 节同轮次表 → 第 6 节主表 → 第 7 节配对检验 → 对着第 3 节的判决清单打勾。

想在终端快速扫一眼各 run 的数字：

```bash
$PY -c "
import pandas as pd
df = pd.read_csv('experiments_log/eval_all_runs.csv', dtype={'id': str})
df = df[df.defect_def == 'implant']      # ⚠️ 排掉 5mm 旧口径的两行
print(df.groupby('run', sort=False)[['CD_t_mm','defect_cov_mm','clump_%']].mean().round(3).to_string())"
```

⚠️ k 折之后 `paired_stats` 不适用，改用 `fold_frame` / `fold_summary` / `fold_paired`。

---

## 3. 看 mesh（三维图）

```bash
$PY src/eval/mesh_preview.py --skull 070 --truth              # 🎮 出 PNG，约 1 分钟
$PY src/eval/mesh_preview.py --skull 070 --truth --html       # 再多写一份可旋转的
```

结果在 `reports/preview/<run>_<skull>.png`（gitignored，约 1 MB，编辑器直接打开）。

**想自己转角度**：用 [`notebooks/MSN_surface_quality.ipynb`](notebooks/MSN_surface_quality.ipynb)
第 4.1~4.3 节 —— 4.1 四格内联、4.2 拖动并把角度存成具名视角、4.3 出高分辨率 PNG。

⚠️ 挑颅骨：`cd_rep05_full` 在 20 颗上的主指标排名 —— 好 `031/000/070`、中位 `004/033`、
⚠️ 离群 `053`（6.769，是均值的两倍多）。**只看最好的那颗会高估。**

---

## 4. 重算主表（换口径、或 k 折之后）🎮

```bash
$PY src/eval/recompute_eval_all.py            # 约 15 分钟，合并写 eval_all_runs.csv
```

带「与掩码无关的列必须逐位不变」的断言。⚠️ **整表重算走它，不要用 notebook 第 6.1 节**
（那个只负责归档）。

---

## 5. 一次性研究脚本（结论已定，一般不用再跑）

| 命令 | 产出 | GPU | 耗时 | k 折后 |
|---|---|---|---|---|
| `$PY src/eval/sampling_floor.py` | `sampling_floor.csv` | ❌ | 3~4 分 | ❌ 不用 |
| `$PY src/eval/defect_mask_audit.py` | `defect_mask.csv` | ❌ | 7 分 | ❌ 不用 |
| `$PY src/eval/normal_quality.py` | `normal_quality.csv` | ❌ | 2~4 分 | ❌ 不用 |
| `$PY src/eval/make_defect_labels.py` | `defect_mask_labels.npz` | ❌ | 35 分 | ❌ 已备齐 100 颗 |
| `$PY src/eval/attention_collapse.py` | `attention_collapse.csv` | 🎮 | 3~5 分 | ✅ 每个最终模型 |
| `$PY src/eval/point_to_surface.py` | `p2s.csv` | 🎮 | 3~5 分 | ✅ 必须 |
| `$PY src/eval/roughness.py` | `roughness.csv` | 🎮 | 1 分 | ⚠️ 只在引用时 |
| `$PY src/eval/fold_text_branch.py` | **只打印** | 🎮 | 40 秒 | ⚠️ 建议 |
| `$PY src/eval/eval_pretrained_baseline.py` | `eval_val20_x5.csv` | 🎮 | 十几分 | ✅ 每折各一次 |

完整的「k 折之后要重跑什么」在 [`src/eval/README.md`](src/eval/README.md) 末尾。

---

## 6. 出论文图 🎮

```bash
$PY src/eval/make_report_figures.py           # -> reports/figures/*.png
```

⚠️ 需要无头 Chrome（`setup_env.sh` 第 5 节）。**3D 用 PNG，2D 曲线用 SVG/PDF**
（矢量，放大不糊）。`RUNS` 现在指向干净 2×2 —— **哪几个 run 讲故事是叙事选择，写论文时随便改。**

---

## 7. 提交之前

```bash
git status --short
```

⚠️ **notebook 的大图输出别提交**（硬规则 3，见 [`notebooks/README.md`](notebooks/README.md)）。
⛔ 清之前先看图里画的是谁 —— 权重已删的 run（`baseline` / `dcd_l2`）的图**再也生成不出来**。
