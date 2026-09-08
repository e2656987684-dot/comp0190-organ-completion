# 3D Organ Shape Completion (COMP0190)

Geometry-aware deep learning for 3D cranial shape completion, evaluated with a
defect-region-restricted protocol.

- **Student:** Qi Jinyu   **Supervisor:** Petru Manescu
- **Target organ:** cranial / skull, using SkullFix's own defective/complete pairs
- **Baseline:** the released MedShapeNet Foundation Model weights, run on this
  project's aligned data. ⚠️ *Whether a voxel baseline is also trained is still
  undecided* -- see the note below.

**Scope changes (2026-08-25), both previously flagged in TODO:**

- ❌ **"self-built corrupted dataset" is dropped.** SkullFix ships its own
  defective/complete pairs and those are what every result here uses;
  `src/corruption/` was never implemented and the claim was never true of the
  work. Both the directory and the claim are gone rather than left as an aspiration.
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
src/eval/                  (placeholder) Chamfer / Hausdorff / MSD / Dice

notebooks/                         see notebooks/README.md for the map and the
                                   two hard rules (kernel/VRAM, module reload)
  ● MSN_train_skullfix.ipynb       RUN a training. Only its section 1 changes
                                   between runs; repeats use --from-run
  ● MSN_eval_metrics.ipynb         JUDGE the results: metric glossary, epoch-
                                   matched table, paired tests, main tables
  ● MSN_eval_surface.ipynb      mesh reconstruction + density diagnostics
  ● MSN_baseline_pretrained.ipynb  BASELINE: author's weights on aligned data
  ● explore_skull.ipynb            first-look + the older .ply batch conversion
  upstream_msn/                     the released implementation, kept as the
                                    provenance of msn_demo_arch.py. NOT verbatim
                                    upstream: dfae1cf rewired 2687 lines of the
                                    inference notebook into this project's data
                                    pipeline, which is where its known-broken
                                    pair alignment comes from. Read them; the
                                    verbatim reference is src/models/msn_demo_arch.py
    MSN_model_inference_demo.ipynb
    MSN_model_training_Demo.ipynb
    skullfix_eval_results.csv       what that notebook measured, cited by
                                    msn_skullfix.py for the distance convention

reports/                   TRACKED. reports/figures/, the paper figures. Also
                           reports/preview/, which is git-ignored scratch
data/                      git-ignored. raw nrrd + data/cache/*.npz
msn_downloads/             git-ignored. MSN_weights3.h5 (author's pretrained weights)
experiments/               git-ignored. training artifacts, one dir per run
experiments_log/           TRACKED. run.json + history.csv only (small), so the
                           numbers survive even though the weights do not.
```

### Where configuration lives

There is no `configs/` directory, deliberately. Configuration sits in three
places, and each one is where it is for a reason:

| what | where | why there and not in a config file |
|---|---|---|
| training hyper-parameters (27 fields: loss, lr, patience, repulsion weight, the exact id split, ...) | `experiments_log/<run>/run.json` — one per run, tracked in git | so "which settings produced this number" is still answerable months later. A checked-in config file drifts away from the artifacts it produced; a `run.json` cannot, because it is written by the run |
| network architecture (layer widths, point counts, `dec_seed`, ...) | `MSNConfig` in `src/models/msn_skullfix.py`, via `paper()` / `small()` | it has to be code: `report.Run.arch_key` reads it to decide whether a checkpoint is topology-compatible, and `load_weights` will silently accept a mismatched topology if nothing checks |
| the k-fold experiment grid (the 2x2) | `CONFIGS` in `src/models/run_kfold.py`, printed by `--list`, mirrored in [`KFOLD.md`](KFOLD.md) | one source of truth. A second copy in YAML is a second thing that can drift |

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
| **[`KFOLD.md`](KFOLD.md)** | ⭐ **k 折执行清单**：20 条训练命令、每条的自检与存档、进度表 |
| **[`CLAUDE.md`](CLAUDE.md)** | **给 AI 助手（和新来的人）的入口。** 按什么顺序读状态、工作约定、踩过的坑、判读规矩。Claude Code 每次新会话自动加载它 |
| **[`TODO.md`](TODO.md)** | **接下来做什么。** 追加式：每完成一项另起一节写日期、重列完整清单，**最后一节永远是当前有效的** |
| [`devlog.md`](devlog.md) | 做过什么、为什么、量到了什么。按日期追加 |
| [`experiments_log/`](experiments_log/) | 每次训练的 `run.json` + `history.csv`，以及各 run 的对照表 |

## Setup
On a fresh GPU pod, `bash setup_env.sh` sets up a conda environment
(`comp0190-msn`, see comments in that file) covering everything the current
code needs: `notebooks/explore_skull.ipynb`, `src/data/prepare_skullfix.py`,
and the upstream MSN notebooks under `notebooks/upstream_msn/`. Dependencies are pinned in
`requirements-msn.txt`.

## Data

`data/` is entirely gitignored, so a fresh clone has no data. Two downloads:

**1. SkullFix** — [Figshare 14161307](https://figshare.com/articles/dataset/SkullFix_-_MICCAI_AutoImplant_2020_Challenge_Dataset/14161307).
Extract under `data/` (`14161307` is the article id, and `src/data/paths.py`
expects that name):

```
data/14161307/SkullFix/
  training_set/{complete_skull,defective_skull,implant}/000.nrrd ... 099.nrrd
  test_set_give_participants/            (100 volumes, unused here)
  additional_test_set_for_participants/  (10 volumes, unused here)
```

⚠️ Every result here comes from `training_set` only — those 100 triplets are what
gets split 80/20 or into 5 folds. `implant/` is the defect-region ground truth.

Cite both, as the dataset's own readme asks:

> J. Li and J. Egger. *SkullFix — MICCAI AutoImplant 2020 Challenge Dataset.* Figshare, 2021.
>
> O. Kodym, J. Li, et al. *SkullBreak / SkullFix.* Data in Brief 106902, 2021.
> <https://doi.org/10.1016/j.dib.2021.106902>

⚠️ SkullFix derives from CQ500 (CC BY-NC-SA 4.0). No volumes are redistributed
here; check the Figshare terms before redistributing anything derived from them.

**2. Pretrained MSN weights** — `bash setup_env.sh` fetches
`msn_downloads/MSN_weights3.h5` (1.2 GB). Only needed for the baseline comparison.

### Building the cache

Training and evaluation read one 12 MB `.npz`, never the raw nrrd:

```bash
python src/data/prepare_skullfix.py --n-samples 0 --n-dense 16384 \
    --n-in 4096 --n-out 6144 --workers 8
```

CPU only, ~5 min. `--workers 8` is a measured optimum; the job is
memory-bandwidth bound and gets slower past 8. Output: `ids` (3-char strings, so
leading zeros survive), `inputs` (100, 4096, 3), `gt` (100, 6144, 3), `scale_mm`.
The step is bit-for-bit reproducible, so a lost cache rebuilds identically.

`data/cache/bert_skull.npy` is the frozen BERT embedding of "skull", written on
the first training run — one class means it is a constant for every sample.

### Keeping a working copy

Data, weights and `experiments/` are all gitignored, so git alone will not save
you: `bash sync_workspace.sh backup` mirrors them to `/workspace`, `restore`
brings them back on a fresh pod.
