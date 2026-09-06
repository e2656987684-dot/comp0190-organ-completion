# 3D Organ Shape Completion (COMP0190)

Geometry-aware deep learning for 3D cranial shape completion, evaluated with a
defect-region-restricted protocol.

- **Student:** Qi Jinyu   **Supervisor:** Petru Manescu
- **Target organ:** cranial / skull, using SkullFix's own defective/complete pairs
- **Baseline:** the released MedShapeNet Foundation Model weights, run on this
  project's aligned data. ⚠️ *Whether a voxel baseline is also trained is still
  undecided* -- see `src/corruption/` and the note below.

**Scope changes (2026-08-25), both previously flagged in TODO:**

- ❌ **"self-built corrupted dataset" is dropped.** SkullFix ships its own
  defective/complete pairs and those are what every result here uses;
  `src/corruption/` was never implemented and the claim was never true of the
  work. Removed from the title rather than left as an aspiration.
- ⏸ **"voxel baseline" is still open.** No voxel code exists. If it does get
  built, feed its binary output through `prepare_skullfix.py`'s pipeline
  (marching cubes -> dense sample -> FPS to 6144) so both sides share the
  sampling floor and stay comparable.

## Repo structure

Only the paths marked ● actually hold code today; the rest are placeholders
from the initial skeleton, kept because the project plan still points at them.

```
src/data/
  ● prepare_skullfix.py    raw nrrd -> aligned point-cloud pairs -> data/cache/*.npz
                           (this is THE data path; the .ply files under data/ are
                            an older, superseded route -- see "Two data routes")
src/models/
  ● msn_skullfix.py        the reworked MSN network + losses/metrics. Used for
                           TRAINING THIS PROJECT'S OWN MODEL.
  ● train_skullfix.py      training CLI (early stopping, k-fold, --run-name)
  ● msn_demo_arch.py       the published demo architecture, lifted verbatim.
                           Used ONLY to run the author's pretrained weights.
                           ⚠ the two model modules are NOT weight-compatible
                             with each other -- see the note in that file.
src/corruption/            (placeholder) self-built corruption operators
src/eval/                  (placeholder) Chamfer / Hausdorff / MSD / Dice
configs/                   (placeholder) config files + fixed seeds

notebooks/                         see notebooks/README.md for the map and the
                                   two hard rules (kernel/VRAM, module reload)
  ● MSN_train_skullfix.ipynb       RUN a training. Only its section 1 changes
                                   between runs; repeats use --from-run
  ● MSN_compare_runs.ipynb         JUDGE the results: metric glossary, epoch-
                                   matched table, paired tests, main tables
  ● MSN_surface_quality.ipynb      mesh reconstruction + density diagnostics
  ● MSN_baseline_pretrained.ipynb  BASELINE: author's weights on aligned data
  ● explore_skull.ipynb            first-look + the older .ply batch conversion
  demo/
    MSN_model_inference_demo.ipynb  original vendor demo (superseded -- its
                                    eval re-breaks pair alignment, see the
                                    baseline notebook's intro for the measurement)
    MSN_model_training_Demo.ipynb   original vendor demo (superseded)
    progress_report/                slides / figures for the progress report

data/                      git-ignored. raw nrrd + data/cache/*.npz
msn_downloads/             git-ignored. MSN_weights3.h5 (author's pretrained weights)
experiments/               git-ignored. training artifacts, one dir per run
experiments_log/           TRACKED. run.json + history.csv only (small), so the
                           numbers survive even though the weights do not.
```

### Two data routes (why there are two, and which to use)

| route | produced by | contains | use it? |
|---|---|---|---|
| `data/cache/*.npz` | `src/data/prepare_skullfix.py` | aligned pairs, 4096 in / 6144 gt, **`scale_mm`** | **yes** — this is what training and the baseline both read |
| `data/.../point_clouds/*.ply` | `explore_skull.ipynb` | aligned pairs, 8192 pts, no `scale_mm` | superseded; kept for the visual exploration cells |

Both apply the same two fixes (pair alignment + anisotropic voxel spacing), but
only the `.npz` route carries `scale_mm`, without which metrics cannot be
converted to millimetres and are therefore not comparable across runs.

### Where the numbers live

`experiments_log/<run-name>/` holds `run.json` (hyper-parameters, the exact
train/val id split, final metrics) and `history.csv` (per-epoch curves) for each
run worth keeping. Git tags mark the matching code state, e.g.
`git diff baseline-7.08mm` shows what changed since that result was produced.

## Where things are tracked

| 文件 | 放什么 |
|---|---|
| **[`RUNBOOK.md`](RUNBOOK.md)** | **所有要在终端敲的命令，按「我现在要干嘛」排。** 记不住命令时看这个 |
| **[`CLAUDE.md`](CLAUDE.md)** | **给 AI 助手（和新来的人）的入口。** 按什么顺序读状态、工作约定、踩过的坑、判读规矩。Claude Code 每次新会话自动加载它 |
| **[`TODO.md`](TODO.md)** | **接下来做什么。** 追加式：每完成一项另起一节写日期、重列完整清单，**最后一节永远是当前有效的** |
| [`devlog.md`](devlog.md) | 做过什么、为什么、量到了什么。按日期追加 |
| [`experiments_log/`](experiments_log/) | 每次训练的 `run.json` + `history.csv`，以及各 run 的对照表 |

## Setup
On a fresh GPU pod, `bash setup_env.sh` sets up a conda environment
(`comp0190-msn`, see comments in that file) covering everything the current
code needs: `notebooks/explore_skull.ipynb`, `src/data/prepare_skullfix.py`,
and the MSN demo notebooks under `notebooks/demo/`. Dependencies are pinned in
`requirements-msn.txt`.

## Data
Data is NOT stored in this repo (see .gitignore). It lives on Google Drive / UCL HPC.
