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
| `pretrained_baseline` | 对照组：作者发布的 `MSN_weights3.h5` 直接推理，<br>不在颅骨上训练。由 `notebooks/MSN_baseline_pretrained.ipynb` 产出。 | 见 CSV | — |

⚠️ **单折的 CD_t 差异要谨慎读。** 逐 epoch 抖动 std ~0.24mm，而 `rep05_void` 意外
测出**运行间方差约 0.49mm**（同配置不同随机轨迹）。所以 `baseline → dcd_l2` 那 0.13mm
基本是噪声；`dcd_l2 → rep_w05` 那 0.62mm 超出了方差，但仍需 k-fold 才能写进论文。

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
