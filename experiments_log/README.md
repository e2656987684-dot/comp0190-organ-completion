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
| `dcd_l2` | `--dcd-lambda 2`（weight 留 1）。密度没动，但四项精度指标同向改善。<br>**当前最好**，采纳为新默认。 | **6.945 mm** | 149（最优在 129） |
| `pretrained_baseline` | 对照组：作者发布的 `MSN_weights3.h5` 直接推理，<br>不在颅骨上训练。由 `notebooks/MSN_baseline_pretrained.ipynb` 产出。 | 见 CSV | — |

⚠️ **CD_t 的差距（~0.13mm）小于逐 epoch 抖动（std ~0.24mm）**，单折结论不足以定性，
需要 k-fold 才能写进论文。详细的噪声核查见 `devlog.md` 的 2026-08-06 条目。

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
