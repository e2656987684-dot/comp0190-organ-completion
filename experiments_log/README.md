# 实验记录 (experiments_log)

`experiments/` 整个目录在 `.gitignore` 里（单个 `.h5` 权重 750 MB，不该进 git）。
但 `run.json`（超参 + 划分 + 最终指标）和 `history.csv`（逐 epoch 曲线）都很小，
值得跟代码一起进版本控制 —— 权重丢了还能重训，实验记录丢了就没法复现对比表了。

约定：每次值得留档的训练，在这里建一个和 `--run-name` 同名的子目录，
只放 `run.json` + `history.csv`。

| run | 说明 | val CD_t | epochs |
|---|---|---:|---:|
| `baseline_es20` | 100 对数据 + EarlyStopping(patience=20) 首次完整跑通。<br>单次 80/20 划分（seed 42），未做 k-fold，无正则化。 | 7.076 mm | 133（最优在 98） |
| `dcd_w3` | `--dcd-weight 3`。密度改善（扎堆 12.6→9.7%）但精度变差。<br>**弃用** —— 兑换率太差，weight 保持默认 1。 | 7.406 mm | 77 |
| `dcd_l2` | `--dcd-lambda 2`（weight 留 1）。密度没动，但四项精度指标同向改善。 | 6.945 mm | 149（最优在 129） |
| `drop01_rejected` | `--dropout 0.1`。❌ **否决**：模型本不过拟合，val CD_t 劣化 2 倍。<br>相关代码已删除。 | 14.36 mm | 92 |
| `rep05_void` | ⚠️ **无效实验，勿引用**。当时 repulsion 未做无量纲化，<br>`weight 0.5` 只占总梯度 0.0095%，等于没开。<br>其价值在于意外测出**运行间方差约 0.49mm**。 | 7.435 mm | 114 |
| **`lr_fix_only`** | **对照组**：只修好学习率衰减，不加 repulsion。<br>证明 CD_t 的增益（6.945 → 6.317）**全部来自学习率修复**。 | **6.317 mm** | 279（最优在 277） |
| **`rep_w05`** | `--repulsion-weight 0.5`（无量纲化后）+ 修好的学习率衰减。<br>相对 `lr_fix_only` 的净效果：**扎堆 5.2% → 0.9%、CV 0.294 → 0.229，<br>CD_t 无可测变化**。repulsion 是密度组件，不是精度组件。 | 6.326 mm | 222（最优在 220） |
| `cd_rep05_truncated` | ⚠️ **作废，勿引用**。撞 `--epochs 300` 上限被截断，<br>最优值落在最后一轮，说明还在收敛中。 | 6.332 mm | 300（截断） |
| **`cd_rep05_full`** | **DCD 消融**：`--loss cd --repulsion-weight 0.5`。<br>去掉 DCD 后 CD_t / HD95 反而更好，F1 与扎堆率持平。 | **6.267 mm** | 256（最优在 236） |
| `cd_rep05_r2` | 与上一行**完全相同的配置，重复一次**。CD_t 只差 0.004mm，<br>证明修好学习率衰减后结果高度可复现。 | 6.274 mm | 249（最优在 229） |
| `cd_only` | **2×2 的第四格**：`--loss cd`，无 DCD 无 repulsion。<br>除 HD95 外每项都最差，扎堆率 13.6% 甚至高于 baseline。<br>证明 DCD 单独仍有效（13.6% → 5.6%），只是远弱于 repulsion。 | 6.353 mm | 305（最优在 285） |
| `pretrained_baseline` | 对照组：作者发布的 `MSN_weights3.h5` 直接推理，<br>不在颅骨上训练。由 `notebooks/MSN_baseline_pretrained.ipynb` 产出。 | 见 CSV | — |

⚠️ **噪声底线随学习率衰减是否生效而变，差两个数量级 —— 读任何差异前先确认属于哪一档。**

| | LR 衰减 | 末 30 轮 CD_t 的 std | 同配置重复的 CD_t 差异 |
|---|---:|---:|---:|
| 前四轮（baseline / dcd_w3 / dcd_l2 / rep05_void） | **0 次** | ~0.25mm | — |
| 修复之后（lr_fix 起） | 6~9 次 | **~0.003~0.010mm** | **0.004mm** |

恒定学习率下验证曲线在平台上来回弹，而指标取「全程最优的一轮」——
那本质是在随机游走上取最小值。**先前采信的「运行间方差 0.49mm」是这个伪影，
不是方法的固有噪声**（详见 devlog 2026-08-07）。

所以：`baseline → dcd_l2` 那 0.13mm 仍按噪声处理（它们都在无衰减的一档）；
但 `rep_w05 → cd_rep05_full` 那 0.05mm **是可分辨的**，因为重复实测只差 0.004mm。

密度指标（扎堆率、间距 CV）不受这个问题困扰 —— 12.1% → 0.9% 远超任何方差解释，
是目前**唯一一个可以确定性下结论**的改进。

⚠️ **前四个 run（`baseline` / `dcd_w3` / `dcd_l2` / `rep05_void`）都跑在
"学习率衰减从未生效"的次优配置下**（`ReduceLROnPlateau` 的 patience 大于早停的，
从来没触发过）。它们彼此之间仍可比，但整体基线被拖累了；
`dcd_l2` 相对 `baseline` 那 0.13mm 应按噪声处理。修复后的结果见后两行。

对应代码状态见 git tag：`baseline-7.08mm`、`rep-6.33mm`。

`surface_quality.csv` 存的是密度/表面偏差指标（8 颗验证颅骨均值），
由 `notebooks/MSN_surface_quality.ipynb` 产出，和上表的 CD_t 是两套互补的指标。

两者评估的是**同一批 20 颗验证颅骨**、同一套指标定义（`msn_skullfix.calc_cd`）、
同一份已对齐的 `.npz` 数据，所以可以直接对比。注意这个对照的性质是
"预训练权重直接应用于颅骨" vs "在颅骨上从零训练"，不是同任务下两个方法的较量。

## 复现说明

`run.json` 里存了 `seed` / `train_ids` / `val_ids`，只要 seed 和 `--val-frac`
不变，划分就是可复现的 —— 后续改动请沿用同一组验证颅骨，否则指标不可比。

对应的代码版本见 git tag：

```bash
git show baseline-7.08mm          # 看这次用的是哪个 commit
git diff baseline-7.08mm          # 看当前代码相对这个基线改了什么
```
