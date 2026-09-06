# k 折训练执行清单（4 配置 × 5 折 = 20 个）

**一个一个跑，跑完一个勾一个。** 每一条都写明了：**这是哪个模型**、
**notebook 第 1 节要改成什么**、以及等价的终端命令。

---

## 两种跑法：自动 / 手动

### ⭐ A. 自动（推荐）—— 一条命令跑完 20 个

```bash
tmux new -s kfold                              # 20 小时，断线不丢
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python

$PY src/models/run_kfold.py --dry-run           # 先看它打算干什么
$PY src/models/run_kfold.py --folds 0           # ⭐ 先只跑 fold 0 的四格（约 4 小时）
$PY src/models/run_kfold.py                     # 对完账再放开跑剩下 16 个
```

⚠️ **建议分两步**：fold 0 就是现在这 20 颗验证颅骨，跑完能和已知的单折数字对账 ——
这是整轮里**唯一一次**能验证"跑法没错"的机会。对完再投入剩下 16 小时。

它比裸的 `for` 循环多做五件事，每一件都对应一个真实的失败方式：

| | 为什么 |
|---|---|
| **撞 `--epochs` 上限就中止** | 那种 run 是**不可引用**的（还在下降时被截断）。裸循环会接着跑，你 16 小时后才发现 |
| **每轮开跑前查磁盘** | 20 个权重要 14 GB，现有 19.5 GB。中途撑爆会让后面的 run 死得像训练 bug |
| **已完成的自动跳过** | 断了、崩了，**重跑同一条命令就续上**，而且不会和 `guard_out_dir` 打架 |
| **每轮跑完自动自检 + 存档** | 否则这件事要手做 20 次，凌晨三点那次一定会跳过 |
| **软警告只记不停** | LR 衰减 <5 次、末 30 轮 std ≥0.02、val/train ≥1.25 都是"读数当心"，不是坏 run。全部汇总在末尾。想严格就加 `--strict` |

日志按轮写进 `experiments/kfold_logs/<run>.log`。
⚠️ 四格配置以这个脚本为准，`--list` 打印出来可以和下面的手动清单对照
（**已机器核对过两边完全一致**）。

### B. 手动一条条 —— 下面 20 条

想在 notebook 里跑、或者只补跑其中某一个时用。**两种跑法产出的 run 完全一样**，
可以混着来（自动脚本会跳过你手动跑完的）。

---

## ⬅️ 要改的就是这三行

训练从 [`notebooks/MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb)
**第 1 节「控制面板」**（第一个 code cell）发起，**每次只改这三个变量**：

```python
RUN_NAME    = "cd_only_f0"     # 这一轮叫什么 -> experiments/msn_skullfix/<这个名字>/
FROM_RUN    = ""               # ⛔ k 折一律留空
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "0", "--loss", "cd"]   # 决定"这是哪个模型"
```

改完往下依次跑：**第 2 节预检 → 第 3 节训练 → 第 4 节自检 → 第 5 节存档**。
下面 20 条把这三行给你写好了，直接复制粘贴。

⚠️ **`FROM_RUN` 必须留空。** 它会照抄老 run 的 `patience=20` / `epochs=500` /
`n_folds=0`，等于把 8/25 修好的早停口径退回去，k 折就废了。

---

## 四个模型分别是什么

**唯一在变的只有两件事**：DCD 有没有、repulsion 有没有。这就是那个 2×2。

| | 模型 | `EXTRA_FLAGS` 里决定它的部分 | 单折实测扎堆率 |
|---|---|---|---|
| 左下 | **`cd_only`** | `"--loss", "cd"` | 13.60% |
| 左上 | **`lr_fix_only`** | `"--loss", "cd_dcd", "--dcd-lambda", "2"` | 5.61% |
| 右上 | **`rep_w05`** | `"--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"` | 1.36% |
| 右下 ⭐ | **`cd_rep05_full`** | `"--loss", "cd", "--repulsion-weight", "0.5"` | 1.24% |

```
                    无 repulsion          有 repulsion
    有 DCD          lr_fix_only            rep_w05
    无 DCD          cd_only            **cd_rep05_full** ⭐ 最优
```

⚠️ **`--dcd-lambda 2` 必须显式写**（默认是 1，而现有两个 DCD 格用的是 2）。
**这是个会影响论文的决定，要在开跑前定**：沿用 λ=2，k 折才和现有那个干净 2×2 是
同一个实验；改成 λ=1 则 DCD 格等于原实现、论文叙述更干净，但没人在修好学习率之后
跑过 λ=1，没有参照可对账。**跑到一半改就废了。**

---

## 开跑前（只做一次）

- [ ] **Restart notebook 的 kernel** —— 一个 187M 模型占 15.5 / 24 GiB
- [ ] `nvidia-smi --query-gpu=memory.used --format=csv` → 应该接近 0
- [ ] `df -h /root` → 现在 19 G 空闲，**20 个权重约需 14 G**。够但不宽裕
- [ ] `bash sync_workspace.sh backup` —— `/root` 是临时盘

⚠️ **用终端跑的话强烈建议开 tmux**（`tmux new -s kfold`），20 小时里 VSCode 一断线
就丢。用 notebook 跑则要保持 kernel 活着。

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python
```

---

## fold 0
⭐ **这一折就是现在这 20 颗验证颅骨**（实测 `val_ids` 与单折完全相同），
所以头四个跑完可以直接和现有单折数字对账，当烟雾测试。
⚠️ 但不会相等：早停 patience 从 20 变成 30，且训练不可逐位复现。
**差 0.05mm 量级正常，差 0.3mm 要停下来查。**
### 1/20 · `cd_only_f0`

**模型**：CD 单独（**不加 DCD、不加 repulsion**） — 2×2 的左下格
**这一折验证**：`000 004 010 012 018 022 030 031 033 039 044 045 053 070 073 076 077 080 083 090`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_only_f0"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "0", "--loss", "cd"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_only_f0 --n-folds 5 --fold 0 --loss cd
```
</details>
### 2/20 · `lr_fix_only_f0`

**模型**：CD + **DCD**(λ=2) — 2×2 的左上格（比上一条**多了 DCD**）
**这一折验证**：`000 004 010 012 018 022 030 031 033 039 044 045 053 070 073 076 077 080 083 090`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "lr_fix_only_f0"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "0",
               "--loss", "cd_dcd", "--dcd-lambda", "2"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name lr_fix_only_f0 --n-folds 5 --fold 0 --loss cd_dcd --dcd-lambda 2
```
</details>
### 3/20 · `rep_w05_f0`

**模型**：CD + **DCD**(λ=2) + **repulsion**(0.5) — 2×2 的右上格（比上一条**多了 repulsion**）
**这一折验证**：`000 004 010 012 018 022 030 031 033 039 044 045 053 070 073 076 077 080 083 090`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "rep_w05_f0"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "0",
               "--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name rep_w05_f0 --n-folds 5 --fold 0 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
```
</details>
### 4/20 · `cd_rep05_full_f0`

**模型**：CD + **repulsion**(0.5)（**不加 DCD**） — 2×2 的右下格 ⭐ **目前最优配置**（比上一条**去掉 DCD**）
**这一折验证**：`000 004 010 012 018 022 030 031 033 039 044 045 053 070 073 076 077 080 083 090`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_rep05_full_f0"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "0",
               "--loss", "cd", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_rep05_full_f0 --n-folds 5 --fold 0 --loss cd --repulsion-weight 0.5
```
</details>

---

## fold 1
### 5/20 · `cd_only_f1`

**模型**：CD 单独（**不加 DCD、不加 repulsion**） — 2×2 的左下格
**这一折验证**：`005 009 011 015 016 026 028 035 040 042 047 055 065 066 069 072 085 088 093 096`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_only_f1"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "1", "--loss", "cd"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_only_f1 --n-folds 5 --fold 1 --loss cd
```
</details>
### 6/20 · `lr_fix_only_f1`

**模型**：CD + **DCD**(λ=2) — 2×2 的左上格（比上一条**多了 DCD**）
**这一折验证**：`005 009 011 015 016 026 028 035 040 042 047 055 065 066 069 072 085 088 093 096`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "lr_fix_only_f1"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "1",
               "--loss", "cd_dcd", "--dcd-lambda", "2"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name lr_fix_only_f1 --n-folds 5 --fold 1 --loss cd_dcd --dcd-lambda 2
```
</details>
### 7/20 · `rep_w05_f1`

**模型**：CD + **DCD**(λ=2) + **repulsion**(0.5) — 2×2 的右上格（比上一条**多了 repulsion**）
**这一折验证**：`005 009 011 015 016 026 028 035 040 042 047 055 065 066 069 072 085 088 093 096`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "rep_w05_f1"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "1",
               "--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name rep_w05_f1 --n-folds 5 --fold 1 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
```
</details>
### 8/20 · `cd_rep05_full_f1`

**模型**：CD + **repulsion**(0.5)（**不加 DCD**） — 2×2 的右下格 ⭐ **目前最优配置**（比上一条**去掉 DCD**）
**这一折验证**：`005 009 011 015 016 026 028 035 040 042 047 055 065 066 069 072 085 088 093 096`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_rep05_full_f1"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "1",
               "--loss", "cd", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_rep05_full_f1 --n-folds 5 --fold 1 --loss cd --repulsion-weight 0.5
```
</details>

---

## fold 2
### 9/20 · `cd_only_f2`

**模型**：CD 单独（**不加 DCD、不加 repulsion**） — 2×2 的左下格
**这一折验证**：`003 006 007 008 013 017 019 024 025 027 034 036 038 049 062 064 078 081 089 095`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_only_f2"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "2", "--loss", "cd"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_only_f2 --n-folds 5 --fold 2 --loss cd
```
</details>
### 10/20 · `lr_fix_only_f2`

**模型**：CD + **DCD**(λ=2) — 2×2 的左上格（比上一条**多了 DCD**）
**这一折验证**：`003 006 007 008 013 017 019 024 025 027 034 036 038 049 062 064 078 081 089 095`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "lr_fix_only_f2"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "2",
               "--loss", "cd_dcd", "--dcd-lambda", "2"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name lr_fix_only_f2 --n-folds 5 --fold 2 --loss cd_dcd --dcd-lambda 2
```
</details>
### 11/20 · `rep_w05_f2`

**模型**：CD + **DCD**(λ=2) + **repulsion**(0.5) — 2×2 的右上格（比上一条**多了 repulsion**）
**这一折验证**：`003 006 007 008 013 017 019 024 025 027 034 036 038 049 062 064 078 081 089 095`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "rep_w05_f2"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "2",
               "--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name rep_w05_f2 --n-folds 5 --fold 2 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
```
</details>
### 12/20 · `cd_rep05_full_f2`

**模型**：CD + **repulsion**(0.5)（**不加 DCD**） — 2×2 的右下格 ⭐ **目前最优配置**（比上一条**去掉 DCD**）
**这一折验证**：`003 006 007 008 013 017 019 024 025 027 034 036 038 049 062 064 078 081 089 095`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_rep05_full_f2"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "2",
               "--loss", "cd", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_rep05_full_f2 --n-folds 5 --fold 2 --loss cd --repulsion-weight 0.5
```
</details>
> 📍 **跑到这里检查一次磁盘**：`df -h /root`。12 个权重约 8.6 G。
> 剩不到 5 G 的话先回收：⛔ 已否决的 `pp_attn` / `tie_qk` / `tie_qk_r2` 权重共 2.1 G，
> 指标早已冻进 `eval_all_runs.csv`。**但删权重是单向的**，要不要删你自己定。

---

## fold 3
### 13/20 · `cd_only_f3`

**模型**：CD 单独（**不加 DCD、不加 repulsion**） — 2×2 的左下格
**这一折验证**：`032 041 043 046 048 050 054 056 057 058 059 061 067 068 075 079 094 097 098 099`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_only_f3"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "3", "--loss", "cd"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_only_f3 --n-folds 5 --fold 3 --loss cd
```
</details>
### 14/20 · `lr_fix_only_f3`

**模型**：CD + **DCD**(λ=2) — 2×2 的左上格（比上一条**多了 DCD**）
**这一折验证**：`032 041 043 046 048 050 054 056 057 058 059 061 067 068 075 079 094 097 098 099`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "lr_fix_only_f3"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "3",
               "--loss", "cd_dcd", "--dcd-lambda", "2"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name lr_fix_only_f3 --n-folds 5 --fold 3 --loss cd_dcd --dcd-lambda 2
```
</details>
### 15/20 · `rep_w05_f3`

**模型**：CD + **DCD**(λ=2) + **repulsion**(0.5) — 2×2 的右上格（比上一条**多了 repulsion**）
**这一折验证**：`032 041 043 046 048 050 054 056 057 058 059 061 067 068 075 079 094 097 098 099`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "rep_w05_f3"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "3",
               "--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name rep_w05_f3 --n-folds 5 --fold 3 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
```
</details>
### 16/20 · `cd_rep05_full_f3`

**模型**：CD + **repulsion**(0.5)（**不加 DCD**） — 2×2 的右下格 ⭐ **目前最优配置**（比上一条**去掉 DCD**）
**这一折验证**：`032 041 043 046 048 050 054 056 057 058 059 061 067 068 075 079 094 097 098 099`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_rep05_full_f3"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "3",
               "--loss", "cd", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_rep05_full_f3 --n-folds 5 --fold 3 --loss cd --repulsion-weight 0.5
```
</details>

---

## fold 4
### 17/20 · `cd_only_f4`

**模型**：CD 单独（**不加 DCD、不加 repulsion**） — 2×2 的左下格
**这一折验证**：`001 002 014 020 021 023 029 037 051 052 060 063 071 074 082 084 086 087 091 092`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_only_f4"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "4", "--loss", "cd"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_only_f4 --n-folds 5 --fold 4 --loss cd
```
</details>
### 18/20 · `lr_fix_only_f4`

**模型**：CD + **DCD**(λ=2) — 2×2 的左上格（比上一条**多了 DCD**）
**这一折验证**：`001 002 014 020 021 023 029 037 051 052 060 063 071 074 082 084 086 087 091 092`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "lr_fix_only_f4"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "4",
               "--loss", "cd_dcd", "--dcd-lambda", "2"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name lr_fix_only_f4 --n-folds 5 --fold 4 --loss cd_dcd --dcd-lambda 2
```
</details>
### 19/20 · `rep_w05_f4`

**模型**：CD + **DCD**(λ=2) + **repulsion**(0.5) — 2×2 的右上格（比上一条**多了 repulsion**）
**这一折验证**：`001 002 014 020 021 023 029 037 051 052 060 063 071 074 082 084 086 087 091 092`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "rep_w05_f4"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "4",
               "--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name rep_w05_f4 --n-folds 5 --fold 4 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
```
</details>
### 20/20 · `cd_rep05_full_f4`

**模型**：CD + **repulsion**(0.5)（**不加 DCD**） — 2×2 的右下格 ⭐ **目前最优配置**（比上一条**去掉 DCD**）
**这一折验证**：`001 002 014 020 021 023 029 037 051 052 060 063 071 074 082 084 086 087 091 092`

notebook [`MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) **第 1 节改成这三行**，
然后依次跑第 2（预检）→ 3（训练）→ 4（自检）→ 5（存档）：

```python
RUN_NAME    = "cd_rep05_full_f4"
FROM_RUN    = ""          # ⛔ k 折一律留空，别用 --from-run
EXTRA_FLAGS = ["--n-folds", "5", "--fold", "4",
               "--loss", "cd", "--repulsion-weight", "0.5"]
```

<details><summary>或者用终端（等价，二选一）</summary>

```bash
$PY src/models/train_skullfix.py --run-name cd_rep05_full_f4 --n-folds 5 --fold 4 --loss cd --repulsion-weight 0.5
```
</details>

---

## 每跑完一个：自检 + 存档

**用 notebook 的话**：第 4 节自检、第 5 节存档，都不用改任何东西，直接跑。

**用终端的话**：

```bash
RUN=cd_only_f0        # ← 改成刚跑完的那个

# ① 自检：三个数，任何一个不对就别存档
$PY -c "
import json, pandas as pd
m = json.load(open(f'experiments/msn_skullfix/$RUN/run.json'))
h = pd.read_csv(f'experiments/msn_skullfix/$RUN/history.csv')
n, best = len(h), int(h['val_loss'].idxmin()) + 1
drops = sum(1 for i in range(1, n) if h['lr'][i] < h['lr'][i-1] - 1e-12)
std = (h['val_cd_t_metric'] * m['scale_mm']).tail(30).std()
stop = ('✅ EarlyStopping' if n - best == m['early_stop_patience']
        else '❌ 撞 --epochs 上限被截断，作废重跑' if n >= m['epochs'] else '⚠️ 墙钟掐停')
print(f'{stop}   轮数 {n}（最优第 {best}）')
print(f'LR 衰减 {drops} 次  {\"✅\" if drops >= 5 else \"❌ 太少，检查配置\"}')
print(f'末 30 轮 std {std:.4f} mm  {\"✅ 已退火\" if std < 0.02 else \"❌ 没退火好\"}')
print(f'best val CD_t {m[\"best_val_cd_t_mm\"]:.3f} mm   ← run.json 口径，别和主表混引')
print(f'fold {m[\"fold\"]}/{m[\"n_folds\"]}  验证 {len(m[\"val_ids\"])} 颗')"

# ② 存档（记录进 git，权重不进）
mkdir -p experiments_log/$RUN && cp experiments/msn_skullfix/$RUN/{run.json,history.csv} experiments_log/$RUN/
```

---

## 进度记录（跑完填一行）

| # | run | 模型 | 轮数 | 停止原因 | LR 降 | 末30轮 std | best val CD_t | 存档 |
|---|---|---|---|---|---|---|---|---|
| 1 | `cd_only_f0` | CD | | | | | | |
| 2 | `lr_fix_only_f0` | CD+DCD | | | | | | |
| 3 | `rep_w05_f0` | CD+DCD+rep | | | | | | |
| 4 | `cd_rep05_full_f0` | CD+rep ⭐ | | | | | | |
| 5 | `cd_only_f1` | CD | | | | | | |
| 6 | `lr_fix_only_f1` | CD+DCD | | | | | | |
| 7 | `rep_w05_f1` | CD+DCD+rep | | | | | | |
| 8 | `cd_rep05_full_f1` | CD+rep ⭐ | | | | | | |
| 9 | `cd_only_f2` | CD | | | | | | |
| 10 | `lr_fix_only_f2` | CD+DCD | | | | | | |
| 11 | `rep_w05_f2` | CD+DCD+rep | | | | | | |
| 12 | `cd_rep05_full_f2` | CD+rep ⭐ | | | | | | |
| 13 | `cd_only_f3` | CD | | | | | | |
| 14 | `lr_fix_only_f3` | CD+DCD | | | | | | |
| 15 | `rep_w05_f3` | CD+DCD+rep | | | | | | |
| 16 | `cd_rep05_full_f3` | CD+rep ⭐ | | | | | | |
| 17 | `cd_only_f4` | CD | | | | | | |
| 18 | `lr_fix_only_f4` | CD+DCD | | | | | | |
| 19 | `rep_w05_f4` | CD+DCD+rep | | | | | | |
| 20 | `cd_rep05_full_f4` | CD+rep ⭐ | | | | | | |

---

## 20 个全部跑完之后

1. **算主表**（🎮 约 25~30 分钟）—— `report.eval_runs` + `fold_frame` / `fold_summary` /
   `fold_paired`，用法见 [`src/eval/README.md`](src/eval/README.md)
2. **基线每折各评一次**：`eval_pretrained_baseline.py --split-from <fold run>`
   —— 基线必须在**和它对比的模型同一批颅骨**上评
3. `p2s.csv` / `attention_collapse.csv` 按最终模型重跑
4. `bash sync_workspace.sh backup`

⚠️ `MSN_compare_runs.ipynb` 第 1 节那条 assert 会拦下你（各折验证集不同，故意的）——
notebook 的 k 折读表一节还没写，先用脚本读。

---

## 已知的坑（都是踩过的）

| | |
|---|---|
| ⛔ **`FROM_RUN` 必须留空** | 它会照抄老 run 的 `patience=20` / `epochs=500` / `n_folds=0` |
| ⚠️ **`--dcd-lambda 2` 必须显式写** | 默认是 1，而现有两个 DCD 格用的是 2 |
| ⚠️ **run 名必须是 `<模型>_f<折号>`** | `report.fold_frame` 拿它和 `run.json` 的 `fold` 互校，对不上直接报错 |
| ⚠️ **撞 `--epochs` 上限就作废** | 默认 600，现有最长 411 轮，应该够。自检里那条 ❌ 出现就重跑 |
| ⚠️ **别指望分辨 0.1mm 以下** | 5 折均值标准误约 0.09mm，2×2 效应 0.20~0.24mm，t≈2，勉强够 |
| ⚠️ **卡在边缘时补救是关键格每折跑两次** | 不是加折数 —— 折数只减划分方差，减不了训练方差 |
| ⚠️ **`RUN_NAME` 撞名会被拒绝启动** | 这是保护。真要覆盖得显式加 `"--overwrite"` |
