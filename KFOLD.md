# k 折训练执行清单（4 配置 × 5 折 = 20 个）

**一个一个跑，跑完一个勾一个。** 每条都标了它是 2×2 里的哪一格、验证哪 20 颗颅骨。

```bash
cd /root/comp0190-organ-completion
PY=/root/miniconda3/envs/comp0190-msn/bin/python
```

> ⚠️ **强烈建议在 tmux 里跑**，否则 VSCode 一断线，跑了 40 分钟的训练就没了：
> ```bash
> tmux new -s kfold        # 新建；断线后用 tmux attach -t kfold 回来
> ```
> 或者用 `nohup`：`nohup $PY ... > /tmp/f0_cd_only.log 2>&1 &`，然后 `tail -f` 看进度。

---

## 开跑前（只做一次）

- [ ] **Restart notebook 的 kernel** —— 一个 187M 模型占 15.5 / 24 GiB，kernel 占着显存脚本会 OOM
- [ ] `nvidia-smi --query-gpu=memory.used --format=csv` → 应该接近 0
- [ ] `df -h /root` → 现在 19 G 空闲，**20 个权重约需 14 G**。够但不宽裕
- [ ] `bash sync_workspace.sh backup` —— `/root` 是临时盘，开跑前先备份一次

**⚠️ 决定：`--dcd-lambda 2`。** 现有那个干净 2×2 的两个 DCD 格用的就是 λ=2，
沿用它 k 折才是同一个实验。（代价：论文要脚注一句「DCD 项按 λ=2，与原实现的 λ=1 不同」。
想改成 λ=1 就把下面所有 `--dcd-lambda 2` 删掉 —— **但要在开跑前决定，跑到一半改就废了**。）

---

## 每跑完一个，做两件事

**① 自检**（三个数，任何一个不对就别存档）：

```bash
RUN=cd_only_f0        # ← 改成刚跑完的那个
$PY -c "
import json, pandas as pd, sys
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
```

**② 存档**（记录进 git，权重不进）：

```bash
mkdir -p experiments_log/$RUN && cp experiments/msn_skullfix/$RUN/{run.json,history.csv} experiments_log/$RUN/
```

（完整版自检在 [`notebooks/MSN_train_skullfix.ipynb`](notebooks/MSN_train_skullfix.ipynb) 第 4 节；
上面是终端简版，只看最要紧的三个数。）

---

## fold 0 — 验证集 `000 004 010 012 018 022 030 031 033 039 044 045 053 070 073 076 077 080 083 090`

⭐ **这一折就是现在这 20 颗验证颅骨**（实测 `val_ids` 与单折完全相同），
所以头四个跑完可以直接和现有单折数字对账。⚠️ 但不会相等：早停 patience 从 20 变成 30，
而且训练不可逐位复现。差 0.05mm 量级正常，差 0.3mm 要停下来查。

- [ ] **1/20 · `cd_only_f0`** — 2×2：CD 单独（无 DCD 无 rep）
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_only_f0 --n-folds 5 --fold 0 --loss cd
  ```
- [ ] **2/20 · `lr_fix_only_f0`** — 2×2：CD + DCD
  ```bash
  $PY src/models/train_skullfix.py --run-name lr_fix_only_f0 --n-folds 5 --fold 0 --loss cd_dcd --dcd-lambda 2
  ```
- [ ] **3/20 · `rep_w05_f0`** — 2×2：CD + DCD + repulsion
  ```bash
  $PY src/models/train_skullfix.py --run-name rep_w05_f0 --n-folds 5 --fold 0 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
  ```
- [ ] **4/20 · `cd_rep05_full_f0`** — 2×2：CD + repulsion ⭐ **目前最优配置**
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_rep05_full_f0 --n-folds 5 --fold 0 --loss cd --repulsion-weight 0.5
  ```

---

## fold 1 — 验证集 `005 009 011 015 016 026 028 035 040 042 047 055 065 066 069 072 085 088 093 096`

- [ ] **5/20 · `cd_only_f1`** — CD 单独
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_only_f1 --n-folds 5 --fold 1 --loss cd
  ```
- [ ] **6/20 · `lr_fix_only_f1`** — CD + DCD
  ```bash
  $PY src/models/train_skullfix.py --run-name lr_fix_only_f1 --n-folds 5 --fold 1 --loss cd_dcd --dcd-lambda 2
  ```
- [ ] **7/20 · `rep_w05_f1`** — CD + DCD + repulsion
  ```bash
  $PY src/models/train_skullfix.py --run-name rep_w05_f1 --n-folds 5 --fold 1 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
  ```
- [ ] **8/20 · `cd_rep05_full_f1`** — CD + repulsion ⭐
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_rep05_full_f1 --n-folds 5 --fold 1 --loss cd --repulsion-weight 0.5
  ```

---

## fold 2 — 验证集 `003 006 007 008 013 017 019 024 025 027 034 036 038 049 062 064 078 081 089 095`

- [ ] **9/20 · `cd_only_f2`** — CD 单独
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_only_f2 --n-folds 5 --fold 2 --loss cd
  ```
- [ ] **10/20 · `lr_fix_only_f2`** — CD + DCD
  ```bash
  $PY src/models/train_skullfix.py --run-name lr_fix_only_f2 --n-folds 5 --fold 2 --loss cd_dcd --dcd-lambda 2
  ```
- [ ] **11/20 · `rep_w05_f2`** — CD + DCD + repulsion
  ```bash
  $PY src/models/train_skullfix.py --run-name rep_w05_f2 --n-folds 5 --fold 2 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
  ```
- [ ] **12/20 · `cd_rep05_full_f2`** — CD + repulsion ⭐
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_rep05_full_f2 --n-folds 5 --fold 2 --loss cd --repulsion-weight 0.5
  ```

> 📍 **跑到这里检查一次磁盘**：`df -h /root`。12 个权重约 8.6 G。
> 剩不到 5 G 的话先回收：⛔ 已否决的 `pp_attn` / `tie_qk` / `tie_qk_r2` 权重共 2.1 G，
> 指标早已冻进 `eval_all_runs.csv`。**但删权重是单向的**，要删你自己决定。

---

## fold 3 — 验证集 `032 041 043 046 048 050 054 056 057 058 059 061 067 068 075 079 094 097 098 099`

- [ ] **13/20 · `cd_only_f3`** — CD 单独
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_only_f3 --n-folds 5 --fold 3 --loss cd
  ```
- [ ] **14/20 · `lr_fix_only_f3`** — CD + DCD
  ```bash
  $PY src/models/train_skullfix.py --run-name lr_fix_only_f3 --n-folds 5 --fold 3 --loss cd_dcd --dcd-lambda 2
  ```
- [ ] **15/20 · `rep_w05_f3`** — CD + DCD + repulsion
  ```bash
  $PY src/models/train_skullfix.py --run-name rep_w05_f3 --n-folds 5 --fold 3 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
  ```
- [ ] **16/20 · `cd_rep05_full_f3`** — CD + repulsion ⭐
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_rep05_full_f3 --n-folds 5 --fold 3 --loss cd --repulsion-weight 0.5
  ```

---

## fold 4 — 验证集 `001 002 014 020 021 023 029 037 051 052 060 063 071 074 082 084 086 087 091 092`

- [ ] **17/20 · `cd_only_f4`** — CD 单独
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_only_f4 --n-folds 5 --fold 4 --loss cd
  ```
- [ ] **18/20 · `lr_fix_only_f4`** — CD + DCD
  ```bash
  $PY src/models/train_skullfix.py --run-name lr_fix_only_f4 --n-folds 5 --fold 4 --loss cd_dcd --dcd-lambda 2
  ```
- [ ] **19/20 · `rep_w05_f4`** — CD + DCD + repulsion
  ```bash
  $PY src/models/train_skullfix.py --run-name rep_w05_f4 --n-folds 5 --fold 4 --loss cd_dcd --dcd-lambda 2 --repulsion-weight 0.5
  ```
- [ ] **20/20 · `cd_rep05_full_f4`** — CD + repulsion ⭐
  ```bash
  $PY src/models/train_skullfix.py --run-name cd_rep05_full_f4 --n-folds 5 --fold 4 --loss cd --repulsion-weight 0.5
  ```

---

## 进度记录（跑完填一行）

| # | run | 轮数 | 停止原因 | LR 降 | 末30轮 std | best val CD_t | 存档了 |
|---|---|---|---|---|---|---|---|
| 1 | `cd_only_f0` | | | | | | |
| 2 | `lr_fix_only_f0` | | | | | | |
| 3 | `rep_w05_f0` | | | | | | |
| 4 | `cd_rep05_full_f0` | | | | | | |
| 5 | `cd_only_f1` | | | | | | |
| 6 | `lr_fix_only_f1` | | | | | | |
| 7 | `rep_w05_f1` | | | | | | |
| 8 | `cd_rep05_full_f1` | | | | | | |
| 9 | `cd_only_f2` | | | | | | |
| 10 | `lr_fix_only_f2` | | | | | | |
| 11 | `rep_w05_f2` | | | | | | |
| 12 | `cd_rep05_full_f2` | | | | | | |
| 13 | `cd_only_f3` | | | | | | |
| 14 | `lr_fix_only_f3` | | | | | | |
| 15 | `rep_w05_f3` | | | | | | |
| 16 | `cd_rep05_full_f3` | | | | | | |
| 17 | `cd_only_f4` | | | | | | |
| 18 | `lr_fix_only_f4` | | | | | | |
| 19 | `rep_w05_f4` | | | | | | |
| 20 | `cd_rep05_full_f4` | | | | | | |

---

## 20 个全部跑完之后

1. **算主表**（🎮 约 25~30 分钟，20 个 run × 20 颗）—— 见 `report.eval_runs` +
   `fold_frame` / `fold_summary` / `fold_paired`，用法在
   [`src/eval/README.md`](src/eval/README.md) 的「k 折之后要重跑什么」
2. **基线每折各评一次**：`eval_pretrained_baseline.py --split-from <fold run>` ——
   基线必须在**和它对比的模型同一批颅骨**上评
3. `p2s.csv` / `attention_collapse.csv` 按最终模型重跑
4. `bash sync_workspace.sh backup`

⚠️ **`MSN_compare_runs.ipynb` 第 1 节那条 assert 会拦下你**（各折验证集不同，故意的）——
notebook 的 k 折读表一节还没写，先用脚本读。

---

## 已知的坑（都是踩过的）

| | |
|---|---|
| ⛔ **别用 `--from-run`** | 它会照抄老 run 的 `patience=20` / `epochs=500` / `n_folds=0`，等于把 8/25 修好的早停口径退回去 |
| ⚠️ **`--dcd-lambda 2` 必须显式传** | 默认是 1，而现有两个 DCD 格用的是 2 |
| ⚠️ **run 名必须是 `<config>_f<fold>`** | `report.fold_frame` 拿它和 `run.json` 的 `fold` 互校，对不上直接报错 |
| ⚠️ **撞 `--epochs` 上限就作废** | 默认 600，现有最长 411 轮，应该够。自检里那条 ❌ 出现就重跑 |
| ⚠️ **别指望分辨 0.1mm 以下** | 5 折均值的标准误约 0.09mm，而 2×2 的效应是 0.20~0.24mm，t≈2，勉强够 |
| ⚠️ **卡在边缘时补救是关键格每折跑两次** | 不是加折数 —— 折数只减划分方差，减不了训练方差 |
| ⚠️ **`--run-name` 撞名会被拒绝启动** | 这是保护。真要覆盖得显式 `--overwrite` |
