# Cranial shape completion from point clouds

Geometry-aware deep learning for 3D cranial shape completion, evaluated with a
defect-region-restricted protocol.

A reimplementation of the MedShapeNet Foundation Model (a Point Cloud Transformer
with a frozen text-conditioning branch, 187M parameters) reworked so it trains on
one 24 GB GPU, trained from scratch on 100 SkullFix defective/complete pairs, and
evaluated on the region a completion model is actually asked to invent rather
than on the whole cloud.

MSc project, COMP0190. Student: Qi Jinyu. Supervisor: Petru Manescu.

## What this found

Results are 5-fold cross-validation over all 100 skulls, four loss
configurations, reported as the mean of the five fold means. The main metric is
**defect coverage**: for every ground-truth point inside the missing region, the
distance to the nearest predicted point.

| loss | defect coverage (mm) | whole-cloud CD_t (mm) | clumping (%) |
|---|---|---|---|
| CD | 3.438 ± 0.061 | 6.443 ± 0.169 | 14.54 ± 3.84 |
| CD + DCD | 3.298 ± 0.137 | 6.411 ± 0.165 | 5.81 ± 1.60 |
| CD + DCD + repulsion | 3.245 ± 0.141 | 6.367 ± 0.123 | 1.48 ± 0.58 |
| **CD + repulsion** | **3.230 ± 0.075** | 6.335 ± 0.171 | 1.86 ± 0.35 |

**An explicit repulsion term solves point clumping, and DCD then has nothing left
to add.** Repulsion takes clumping from 14.5% to 1.9% of points, agreeing in
every fold. Adding DCD on top of it has no measurable effect at all (+0.016 mm,
one fold out of five, interval covering zero), so the density-aware term can be
dropped and the simpler two-term loss used. Showing a term can be removed is a
stronger result than stacking another one on.

**Restricting the metric to the defect is what makes that visible.** Only about
6% of a ground-truth cloud lies in the missing region; the rest is surface the
model was handed and merely has to copy. Of the four edges of the ablation, the
defect-region metric resolves three and the whole-cloud Chamfer distance resolves
one — and the two it misses are the two the defect region resolves most strongly.
The same protocol puts the released upstream weights 3.3x behind this model on
defect coverage while whole-cloud Chamfer shows only 1.7x.

**Most of the reported error is the representation, not the model.** Two
independent samplings of one surface, with no model involved, differ by 4.62 mm
CD_t — about 73% of the reported figure. Related: surface normals cannot be
recovered from these clouds at all (orientation flips on roughly half the points,
because a skull is a 5-7 mm shell sampled at ~4 mm), so normal-based surface
reconstruction is not available at this output resolution. Both are measurements,
and both are in `src/eval/`.

Every number above is recomputable from this repository: per-skull metrics are
tracked in `experiments_log/eval_all_runs.csv`, and reading them needs no GPU, no
checkpoints and no raw data.

## Layout

```
src/data/     the one path raw data takes: nrrd volumes -> aligned point-cloud
              pairs -> a single cached .npz that training and evaluation read
src/models/   the network, the training CLI, the cross-validation driver, and a
              verbatim copy of the upstream architecture for the baseline
src/eval/     two importable modules and thirteen single-purpose scripts, each
              producing one table under experiments_log/
notebooks/    six notebooks in pipeline order: explore, train, infer, and three
              for evaluation
experiments_log/   tracked: hyper-parameters, per-epoch curves and per-skull
                   metrics for every run. The checkpoints are not tracked, so
                   these are what make the results checkable
reports/      figures for the write-up
```

Each directory has its own README with the detail: [`src/models/`](src/models/README.md),
[`src/eval/`](src/eval/README.md), [`notebooks/`](notebooks/README.md).

Two things are git-ignored: `experiments/` and `msn_downloads/`, which hold the
checkpoints and the upstream released weights, and `data/14161307/`, the raw
SkullFix volumes. None of them are redistributed here.

Everything *derived* from the volumes is tracked, and it is deliberately enough
to check the work without them: the point-cloud cache the model actually reads
(`data/cache/`, 12 MB), the per-point defect labels, the per-skull metrics, and
`experiments_log/preds_fold0.npz` — fold 0's predictions for all four loss
configurations over its twenty validation skulls, 5.5 MB, which reproduce the
frozen `defect_cov_mm` exactly. With those, every figure showing a completion
redraws on a laptop with no GPU and no checkpoints.

### Where configuration lives

There is no `configs/` directory, deliberately:

| what | where | why there |
|---|---|---|
| training hyper-parameters, including the exact id split | `experiments_log/<run>/run.json`, one per run | so "which settings produced this number" stays answerable. A checked-in config file drifts away from the artifacts it produced; a file written *by* the run cannot |
| network architecture | `MSNConfig` in `src/models/msn_skullfix.py` | it has to be code: the evaluation reads it to decide whether a checkpoint is topology-compatible, and loading weights will silently accept a mismatched topology if nothing checks |
| the cross-validation grid | `CONFIGS` in `src/models/run_kfold.py` | one source of truth, printed by `--list` |

## Setup

```bash
bash setup_env.sh      # conda environment, pinned in requirements-msn.txt
```

It also fetches the released MSN weights (1.2 GB), which are needed only for the
baseline comparison.

## Data

The raw volumes are git-ignored, so a fresh clone has none of them. It does have
`data/cache/`, which is what training and evaluation read — the volumes are only
needed to rebuild that cache or to render a raw-voxel panel.

Download **SkullFix** from
[Figshare 14161307](https://figshare.com/articles/dataset/SkullFix_-_MICCAI_AutoImplant_2020_Challenge_Dataset/14161307)
and extract it under `data/` keeping the article id as the directory name:

```
data/14161307/SkullFix/training_set/{complete_skull,defective_skull,implant}/000.nrrd ...
```

Every result here comes from `training_set` only — those 100 triplets are what
gets split into five folds. `implant/` is the defect-region ground truth, and it
is what the main metric is defined against.

Cite both, as the dataset's own readme asks:

> J. Li and J. Egger. *SkullFix — MICCAI AutoImplant 2020 Challenge Dataset.* Figshare, 2021.
>
> O. Kodym, J. Li, et al. *SkullBreak / SkullFix.* Data in Brief 106902, 2021.
> <https://doi.org/10.1016/j.dib.2021.106902>

SkullFix derives from CQ500 (CC BY-NC-SA 4.0). No volumes are redistributed here;
check the Figshare terms before redistributing anything derived from them.

### Building the cache

Training and evaluation read one 12 MB `.npz`, never the raw volumes:

```bash
python src/data/prepare_skullfix.py --n-samples 0 --n-dense 16384 \
    --n-in 4096 --n-out 6144 --workers 8
```

CPU only, a few minutes. `--workers 8` is a measured optimum: the job is
memory-bandwidth bound and gets slower beyond it. The step is bit-for-bit
reproducible, so a lost cache rebuilds identically.

## Running it

```bash
python src/models/run_kfold.py --list          # the four configurations
python src/models/run_kfold.py cd_rep05_full   # its five folds, ~4-5 h
python src/eval/recompute_eval_all.py          # per-skull metrics for every run
```

Then read the results in `notebooks/MSN_eval_metrics.ipynb`, which needs no GPU:
it reads the frozen metrics rather than recomputing them.

The training driver aborts rather than producing a run that cannot be quoted — it
stops if a run hits the epoch ceiling while still improving, or if the disk would
fill part way through — and skips folds that already finished, so re-running the
same command resumes.

## Scope

The defective/complete pairs are the dataset's own; no corruption is synthesised
here. A voxel baseline was considered and not built: comparing against one would
require pushing its output through the same sampling pipeline so that both sides
share the same floor, and the comparison would then discard exactly the
resolution advantage that makes voxels interesting.

## Repository notes

`KFOLD.md` is the cross-validation run list; `devlog.md` and `TODO.md` are the
working log, kept in the language they were written in.
