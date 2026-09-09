# src/models

| file | what it is |
|---|---|
| `msn_skullfix.py` | The rewritten network, its losses and its metrics. This is what gets trained. |
| `train_skullfix.py` | The training CLI: one run, one configuration, one fold. |
| `run_kfold.py` | Drives a configuration's five folds, with the guards a bare loop does not have. |
| `msn_demo_arch.py` | The upstream architecture, copied verbatim. Only used to run the authors' released weights. |

Warning: the two architectures are **not weight-compatible**, and loading the
wrong checkpoint fails *silently* — `load_weights(by_name=True,
skip_mismatch=True)` returns without error while matching 3 of 32 weight groups,
leaving most of the network randomly initialised. The cause is layer nesting, not
shape. Load strictly and check the tensors changed.

This file holds the reasoning behind decisions that look arbitrary in the code,
so the code itself stays readable.

## What the rewrite changed inside the model

Same layer topology as the published demo. Three changes are mathematically
identical and only faster:

1. `distance_matrix` uses `|a|^2 - 2a.b + |b|^2` instead of tiling both clouds
   into `(B, N, M, 3)`, so it materialises only `(B, N, M)` — about 10x less
   memory. Values keep the demo's convention: Euclidean, **not** squared.
2. `calc_dcd` is a batched `bincount` rather than a `tf.map_fn` over the batch.
3. The frozen BERT branch is precomputed — one class, so its output is a constant.

Four change behaviour, and each is a fix:

4. `UniformSampler` drew centroids **with replacement**, so they contained
   duplicates and the groups around them were degenerate. Distinct indices now;
   `sampler="original"` restores the demo.
5. The demo's encoder sampled as many stage-1 centroids as there are input
   points, i.e. no downsampling while still paying the O(N^2) kNN.
   **Correction (2026-08-25):** this was once described as "the configs here
   always downsample", which is false for `paper()` — it keeps
   `sg1_sample = n_in` on purpose, faithfully reproducing the published
   behaviour, so its stage-1 sampler only permutes. Only `small()` downsamples
   there.
6. `eye_seed` was random during training and zeros at inference. Zeros everywhere.
7. The final layer is float32, so mixed precision cannot corrupt coordinates.

The data-side differences (pair alignment, voxel spacing) are in
[`notebooks/README.md`](../../notebooks/README.md).

## Sizing

`paper()` is the published architecture — 4096 in, 6144 out, 187M parameters — and
trains at 372 ms/step, 15.5 GiB, batch 4 on one RTX 4090. Batch 8 still runs out
of memory at 24 GB. `small()` is a 9.4M variant (2048 in, 1536 out, 25 ms/step,
0.9 GiB) for debugging the pipeline.

## Losses

**`--dcd-weight` is a gradient-share knob.** DCD saturates, so the loss value
badly overstates it: at weight 1 it is 92.6% of the loss VALUE but 36% of the
gradient. Measured shares of the gradient — 1: 36%, 2: 53%, 3: 63%, 5: 74%,
10: 85%, 20: 92%. Past ~10 Chamfer barely votes; 1-5 is the useful range.

**`--dcd-lambda` has never been validly measured here.** By construction the
exponent targets clumping specifically, but the only run that tested it sits in
the voided pre-learning-rate-fix tier, and the explanation once built on it was
withdrawn untested.

**`--repulsion-weight` is dimensionless on purpose.** The shortfall is divided by
`r0` before squaring, giving a per-pair fraction in [0, 1] rather than a squared
length — that is what makes a weight of 0.5 mean what it looks like (gradient
norm ~0.51 against Chamfer's 0.72). It is also the only term that can push two
predicted points apart: DCD's density factor comes from an `argmin` and has zero
gradient with respect to position.

**`--repulsion-r0` is a distance, so it can be read off the data.** Ground-truth
spacing has a hard floor at 3.0 mm because the ground truth is farthest-point
sampled, so 3.0 matches it; 2.0 is the conservative default, matching the
`clump<2mm` metric. `r0` is normalised while skull radii span 88.3-133.1 mm, so
one value enforces 1.70-2.57 mm depending on the skull — noise, not bias.

**Why there is no regularisation.** Dropout and AdamW were both added and
removed: dropout made val CD_t twice as bad, and the model was never overfitting
— val/train CD_t is 1.03-1.08x on every run. It under-fits. Read this before
adding any regulariser.

## Training guards, and what each one is for

**`--epochs` / `--minutes` are ceilings, not budgets.** EarlyStopping normally
stops a run; one that reaches a ceiling was still descending and cannot be quoted.

**`--lr-patience` must stay below `--early-stop-patience`, checked at startup.**
Both watch val_loss failing to improve, so the LR only ever drops if its patience
is the shorter. It was 40 against 20, the decay never fired, and four runs are
unusable because of it.

**`--early-stop-patience` is 20 because 30 was unaffordable.** Five of the nine
single-split runs came within 1-5 epochs of being ended during a mid-run plateau,
worth up to 0.15 mm. 30 was adopted then reverted for the k-fold sweep: the first
run at 30 hit the epoch ceiling, ~28 h against ~12 h across twenty runs. Every
archived run used 20, so the folds stay comparable. No run was ever trained at 30.

**`--overwrite` exists because re-using a run name used to corrupt the record
silently.** The checkpoint is overwritten from epoch 1 while run.json is written
only at the end, so an abandoned re-run left the record describing one training
and the weights another.

**`--from-run` replays another run's hyper-parameters** rather than trusting you
to retype them; two runs once differed in one flag out of nine unnoticed. Six
flags are not replayed — they say where a run writes, not what is trained.

**`--defect-every` logs the metric the thesis reports, and must not drive
selection.** Choosing a checkpoint by it would pick the epoch that looks best on
the very skulls the result is reported on. Missing labels are a hard error, never
a silent fall back to the old distance rule.

## Two Keras defaults that are wrong here

**Checkpoints are `best.h5`, never `best.weights.h5`.** The latter selects the
newer format, which stores Adam's slots even under `save_weights_only=True` —
187.5M parameters become 562M values, 2.25 GB, rewritten on every improvement,
against 750 MB for weights alone.

**`ReduceLROnPlateau` uses `min_delta=0`.** Keras's 1e-4 default is an *absolute*
threshold: `cd_dcd` sits near 1.0 and clears it, plain `cd` sits near 0.07 and
would be declared stalled every epoch, collapsing the LR and quietly wrecking any
run that changes `--loss`.

## Why the sweep has a driver

A shell loop tells you at the end; twenty hours is long enough that the failure
modes matter more. `run_kfold.py` aborts on a ceiling hit or low disk, skips
finished runs so re-running resumes, self-checks and archives each run — including
runs it skips, since an interruption before the copy would leave a run marked
finished and its record would never reach git. It decides nothing: the four cells
are the ones in `KFOLD.md`, and `--list` prints them for comparison.

**Config-major by default, fold-major with `--all`.** One configuration's five
folds together means its mean and spread can be read as soon as it finishes; the
cost is that nothing is comparable until the second model does. `--all` inverts
that. Config-major wins because a reviewable result every five hours is worth
more over a twenty-hour job.

**`last.h5` is deleted when it matches `best.h5`.** `restore_best_weights=True`
makes the two byte-identical, and keeping both would need 28.6 GB instead of
14.3 GB. One that *differs* is kept — that means the run was truncated.

## Implementation notes

**Neighbour selection in `repulsion_loss` is chunked and gradient-free.** Done in
one go it costs an extra 4.8 GiB of peak memory and OOMs a 24 GB card.

**`_min_dists` recomputes distances on the matched pairs**, using the big matrix
only for the argmin under `stop_gradient`. `|a|^2 - 2a.b + |b|^2` cancels
catastrophically exactly where it matters most, since nearest neighbours are the
smallest distances in it. The gradient is unchanged.

**`calc_f1` thresholds are normalised** to match the source paper's tables; at
this dataset's mean radius of 103.8 mm, 0.05 and 0.03 are 5.19 mm and 3.11 mm.

**`--tie-qk-init`** restores a one-line weight tie the demo has in its encoder
self-attention. It only sets the starting point. Tested under controlled repeats
and it produced no reproducible change in accuracy, so it stays off.

**`--per-point-attn`** changes what decoder stage 1 attends to. Off, every key row
is identical and the block is a global conditioning offset rather than attention.
It changes no weight shape, so a checkpoint from one setting loads into the other
without raising — which is why it goes into run.json and is read back by
`report.Run.arch_key`. The global vector stays first either way; dropping it
measured worse in the defect region.

## Where the numbers live

Per-epoch curves and hyper-parameters are archived under `experiments_log/<run>/`
and tracked in git; the checkpoints are not. Per-skull metrics are in
`experiments_log/eval_all_runs.csv`. Training is not bit-reproducible on a GPU
while evaluation is, so deleting a checkpoint is the irreversible step.
