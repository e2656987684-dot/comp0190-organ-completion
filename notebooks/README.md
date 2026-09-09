# notebooks

Six notebooks, in the order a model is built. Preprocessing is not a notebook —
it is `src/data/prepare_skullfix.py`, the only path raw data takes into this
project.

| step | notebook | what it does | what it needs |
|---|---|---|---|
| explore | [`explore_skull.ipynb`](explore_skull.ipynb) | The raw nrrd volumes and the cached point clouds side by side: what actually goes into the model, and where the defect is. Read-only, writes nothing | `data/` |
| train | [`MSN_train_skullfix.ipynb`](MSN_train_skullfix.ipynb) | The 2x2 loss ablation as 5-fold cross-validation — 4 configurations x 5 folds, one training cell each — then a self-check over all 20 runs and their curves | GPU + `data/` for sections 1-3. **Sections 4-5 need nothing**: they read the records tracked in `experiments_log/` |
| infer | [`MSN_inference.ipynb`](MSN_inference.ipynb) | Loads one fold's checkpoint and completes a defective skull: 4096 points in, 6144 out | GPU + that fold's checkpoint. About a second per skull |
| evaluate | [`MSN_eval_metrics.ipynb`](MSN_eval_metrics.ipynb) | The results. Fold means of the 2x2, the four edges of the ablation, defect region against whole cloud, and whether anything won by training longer | **Nothing for sections 1-8** — they read the frozen per-skull metrics in `experiments_log/eval_all_runs.csv`. Section 9 optionally recomputes one fold and needs a GPU |
| evaluate | [`MSN_baseline_pretrained.ipynb`](MSN_baseline_pretrained.ipynb) | The comparison this project reports against: the original authors' released weights on this data, scored fold by fold | GPU + `msn_downloads/MSN_weights3.h5` |
| evaluate | [`MSN_eval_surface.ipynb`](MSN_eval_surface.ipynb) | What a completion looks like: meshes, the four-panel figure showing the cost of the point representation, and the density diagnostic a mesh cannot show | GPU + checkpoints + the raw volumes |
| — | [`upstream_msn/`](upstream_msn/) | The two notebooks from the upstream MedShapeNet project, kept for reference. Warning: they are **not** unmodified — this project wired its own data pipeline into the inference one. The verbatim copy of the upstream architecture is `src/models/msn_demo_arch.py` | — |

A typical pass: train, restart the kernel, run
`python src/eval/recompute_eval_all.py` to write the per-skull metrics, then read
them in `MSN_eval_metrics.ipynb`. Every terminal command is in
[`RUNBOOK.md`](../RUNBOOK.md).

## How to run them

**Start at section 1 and go down.** Section 1 puts `src/` on `sys.path` and
section 2 produces the state later cells use, so clicking straight into a later
cell raises `ModuleNotFoundError`. That means *not run in order*, not a missing
dependency. Use Run All, or run sections 1 and 2 first.

**Restart the kernel before training, and before any GPU script under
`src/eval/`.** One 187M model holds 15.5 of the 24 GiB on the card, so a kernel
that has already built one will make the next thing you start run out of memory.

**After editing `src/eval/report.py` or `src/eval/mesh_viz.py`, reload it**
(`importlib.reload`) or restart the kernel — otherwise the notebook keeps the
cached module and you get the previous version's numbers and figures. Only those
two: `msn_skullfix` defines Keras layers, and hot-reloading it makes models built
earlier fail against the new classes.

**Editor warnings are a separate thing.** Pylance does not execute
`sys.path.insert`, so it marks `report`, `mesh_viz` and `msn_skullfix` as
unresolved; `.vscode/settings.json` sets `python.analysis.extraPaths` to silence
that.

## Do not commit figure outputs

A figure in a `.ipynb` is not a link — the whole thing is embedded, as a large
blob of JSON for plotly. Re-running changes all of it, so git cannot store a diff
and keeps another full copy. One 3D figure can be tens of megabytes, and every
clone downloads all of them.

Long-lived figures belong in `reports/`. Clear notebook outputs before committing:

```bash
python -c "
import json,sys
p=sys.argv[1]; nb=json.load(open(p))
for c in nb['cells']: c['outputs']=[]; c['execution_count']=None
json.dump(nb,open(p,'w'),ensure_ascii=False,indent=1); open(p,'a').write('\n')" notebooks/<name>.ipynb
```

Warning: check which runs a stored figure shows before clearing it. Training is
not bit-reproducible on a GPU, so a figure of a run whose checkpoint has been
deleted cannot be regenerated.

## What this reimplementation changed, against the upstream demo

The network topology is unchanged — LBR blocks, offset self-attention, the
cross-attention decoder, copy-and-mapping upsampling. These five differences are
what made it trainable on one GPU and what makes its numbers mean anything. They
were written when the rewrite was first committed and are reproduced here because
the notebook that used to hold them was deleted.

**A. Pair alignment — a correctness bug.** The baseline in
`explore_skull.ipynb` normalises the complete and the defective cloud
*independently*, each by its own centroid and maximum radius. Cropping a defect
moves the centroid and shrinks that radius, so the two clouds describing one
skull land in different frames. Measured on skull 000: centroid off by 7.55
voxels, 3.6% of the radius; scale off by 2.8%; the ground-truth-to-input nearest
distance inflated by 32%. Fixed by deriving **one** similarity transform from the
defective cloud alone — the only shape available at inference — and applying it
to both.

**B. The distance matrix exhausted the GPU.** The demo's `distance_matrix` tiles
both clouds into `(B, N, M, 3)`. At the demo's own settings, batch 8 and 6144
points, that single tensor is about 12 GB on the forward pass and is kept for the
backward one — which is why the upstream inference notebook fell back to the CPU.
Replaced with `|a|² − 2a·b + |b|²`, which only materialises `(B, N, M)`. The full
paper architecture, 187.5M parameters, then trains on one RTX 4090 at 372 ms per
step in 15.5 GiB. The model did not need shrinking.

**C. DCD cannot bootstrap from random initialisation.** DCD is bounded in [0, 2]
and both of its factors vanish when the prediction is far from the target.
Measured at this model's initialisation: mean nearest-neighbour distance 2.79, so
`exp(-2.79) = 0.067`, while all 6144 ground-truth points collapse onto **six**
distinct predicted points, giving a density weight near 1/1970. Their product
pins the loss at 1.9995 against a bound of 2.0, and sweeping the learning rate
over 1e-7, 1e-4, 3e-4 and 1e-3 for 40 steps moves it by less than 0.03. The
default became `cd_dcd`: Chamfer is unbounded and pulls the shape into place,
DCD refines once it is.

**D. Inference was not deterministic.** The demo's `UniformSampler` draws
centroids with a stateful RNG, so the same model on the same input differed by
1.03 between two calls. Sampling stays random during training, where it is free
augmentation, and validation uses fixed-seed stateless sampling, so repeated
evaluation is bit-identical and the metrics are reproducible.

**E. Training configuration.**

- Learning rate 1e-7 to 3e-4 with a 100-step warm-up. 1e-7 under Adam is three
  orders of magnitude below normal.
- Batch 8 to 4; 8 runs out of memory on 24 GB. There is no BatchNorm anywhere —
  `LBR` is Dense + ReLU — so a smaller batch only adds gradient noise.
- `validation_split=0.1` replaced by an explicit split on skull id. Keras takes
  the tail *before* shuffling, so as soon as one skull yields more than one
  partial cloud (the demo's `preprocess_data` yields two) sibling samples
  straddle the boundary and leak.
- The frozen BERT output is precomputed. It is `trainable=False` and there is a
  single class, so it is a constant; recomputing 110M parameters every step is
  waste.
- Checkpoints are written as `best.h5`, not `best.weights.h5`. The latter selects
  the newer Keras format, which stores Adam's moments even under
  `save_weights_only=True` — 187.5M parameters become 562M stored values, 2.25 GB
  — and rewrites the file on every improvement. The older format stores weights
  only, 750 MB.
- Voxel spacing. The nrrd header carries an anisotropic and sheared
  `space directions` matrix (0.451 / 0.446 / 0.625 mm), and the demo runs
  marching cubes in index space, stretching the skull by about 39% along z. The
  transform is applied before anything else and the normalisation radius stored
  per sample as `scale_mm`, so any normalised metric converts to millimetres.

A and the voxel-spacing item are also documented in
`src/data/prepare_skullfix.py`, which implements them.


## Elsewhere

- Terminal commands, ordered by task: [`RUNBOOK.md`](../RUNBOOK.md)
- What each analysis script writes, and whether it needs a GPU:
  [`src/eval/README.md`](../src/eval/README.md)
- The k-fold run list: [`KFOLD.md`](../KFOLD.md)
- Environment setup: `bash setup_env.sh`
