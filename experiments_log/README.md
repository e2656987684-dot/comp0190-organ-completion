# 实验记录 (experiments_log)

`experiments/` 整个目录在 `.gitignore` 里（单个 `.h5` 权重 750 MB，不该进 git）。
但 `run.json`（超参 + 划分 + 最终指标）和 `history.csv`（逐 epoch 曲线）都很小，
值得跟代码一起进版本控制 —— 权重丢了还能重训，实验记录丢了就没法复现对比表了。

约定：每次值得留档的训练，在这里建一个和 `--run-name` 同名的子目录，
只放 `run.json` + `history.csv`。

| run | 说明 | best val CD_t | epochs |
|---|---|---:|---:|
| `baseline_es20` | 100 对数据 + EarlyStopping(patience=20) 首次完整跑通。<br>单次 80/20 划分（seed 42），未做 k-fold，无正则化。 | **7.08 mm** | 133（最优在 112） |

## 复现说明

`run.json` 里存了 `seed` / `train_ids` / `val_ids`，只要 seed 和 `--val-frac`
不变，划分就是可复现的 —— 后续改动请沿用同一组验证颅骨，否则指标不可比。

对应的代码版本见 git tag：

```bash
git show baseline-7.08mm          # 看这次用的是哪个 commit
git diff baseline-7.08mm          # 看当前代码相对这个基线改了什么
```
