"""
MSN (PCT + text) point-cloud completion, reworked so it can actually be trained
on one 24 GB GPU.

This is a rewrite of notebooks/demo/MSN_model_training_Demo.ipynb. The layer
topology is the same (LBR / offset self-attention / cross-attention decoder /
copy-and-mapping upsampling); what changed is listed below, grouped by whether
the change alters the maths.

MATHEMATICALLY IDENTICAL, JUST FASTER
  1. `pairwise_distance` / `distance_matrix`. The demo materialises
     (B, N, M, 3) by tiling both clouds, then subtracts. At the demo's own
     settings (B=8, 6144 output points) that single loss tensor is ~12 GB
     forward, and autodiff keeps it -- which is why the inference notebook had
     to fall back to the CPU. Replaced with the |a|^2 - 2a.b + |b|^2 identity,
     which only ever materialises (B, N, M) and runs on cuBLAS. ~10x less
     memory. The returned values keep the demo's convention exactly: Euclidean
     distances, NOT squared (the demo's `distance_matrix` returns tf.norm), so
     CD_p / CD_t stay comparable with skullfix_eval_results.csv.
  2. `calc_dcd` looped over the batch with `tf.map_fn`. Replaced with batched
     `tf.math.bincount(..., axis=-1)`.
  3. The frozen BERT branch is precomputed. BERT is `trainable=False` and this
     dataset has exactly one class ("skull"), so its output is a constant for
     every sample of every epoch. Running 110M parameters each step to
     recompute a constant is pure waste. `encode_class_name()` produces the
     pooled vector once; the model takes it as an input.

DELIBERATE BEHAVIOUR CHANGES (each is a fix, and each is flagged)
  4. `UniformSampler` drew indices with `tf.random.uniform`, i.e. WITH
     replacement, so the "farthest point" centroids contained duplicates and
     the groups around them were degenerate. `sampler="unique"` (default) draws
     distinct indices instead. `sampler="original"` restores the demo.
  5. The demo's encoder called `UniformSampler(4096)` on a 4096-point input --
     sampling as many centroids as there are points, i.e. no downsampling at
     all, while still paying the O(N^2) kNN. The configs here always downsample.
  6. eye_seed. The demo trains with `np.random.rand(1,1)` but infers with
     `tf.zeros` -- a train/inference mismatch. Fixed to zeros everywhere.
  7. Final layer forced to float32 so mixed precision cannot corrupt the
     coordinates or the distance computation.

SIZING
  `MSNConfig.paper()` is the published architecture, 4096 in / 6144 out, 187M
  parameters, and it is what train_skullfix.py uses by default. Once the
  distance kernels above are fixed it costs 372 ms/step at batch 4 and 15.5 GiB
  on one RTX 4090 -- so there is no need to shrink the model to train it. Batch
  8 still OOMs at 24 GB.

  `MSNConfig.small()` is a ~20x cheaper 9.4M-parameter variant (2048 in /
  1536 out, 25 ms/step, 0.9 GiB) for debugging the pipeline.

  NEITHER config can load msn_downloads/MSN_weights3.h5. An earlier version of
  this docstring claimed `paper()` could -- that was wrong, and wrong in the
  worst way, because the failure is silent: `load_weights(by_name=True,
  skip_mismatch=True)` returns without raising while matching only 3 of the
  checkpoint's 32 weight groups (`D-OUT_lin`, `D1-IN`, `D2-IN`), leaving ~96% of
  the network randomly initialised. Shapes are compatible; the layer NESTING is
  not. The demo wraps each Dense+ReLU in a nested Model, so the checkpoint holds
  `E-IN_LBR1/E-IN_LBR1_lin/kernel`, while `LBR` here emits a top-level Dense
  named `E-IN_LBR1_lin` and looks for `E-IN_LBR1_lin/kernel`. Attention blocks
  differ likewise (`E-SA1` as one group vs `E-SA1_Q`/`_K`/`_V`).

  To run the published weights use `msn_demo_arch.py`, which is that topology
  lifted verbatim and does load them strictly. Keep the two modules separate:
  this one for models trained by this project, that one for the vendor weights.

LOSS
  Do NOT train from scratch with DCD alone -- see the warning on `dcd_loss`. Use
  `cd_dcd_loss` (Chamfer + the density-aware term) or `chamfer_loss`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

EPS = 1e-12

# Query points per block when selecting repulsion neighbours. Trades a few extra
# kernel launches for peak memory; see `repulsion_loss`.
_REPULSION_CHUNK = 1024


# --------------------------------------------------------------------------- #
# distances and losses
# --------------------------------------------------------------------------- #
def squared_distance_matrix(a, b):
    """(B,N,3),(B,M,3) -> (B,N,M) squared euclidean, without an (B,N,M,3) blowup."""
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    a2 = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)        # (B,N,1)
    b2 = tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)        # (B,M,1)
    ab = tf.matmul(a, b, transpose_b=True)                          # (B,N,M)
    d2 = a2 - 2.0 * ab + tf.transpose(b2, (0, 2, 1))
    return tf.maximum(d2, 0.0)


def distance_matrix(a, b):
    """Euclidean (not squared) distances -- matches the demo's tf.norm convention."""
    return tf.sqrt(tf.maximum(squared_distance_matrix(a, b), EPS))


def _min_dists(array1, array2):
    """Nearest-neighbour distances both ways, exactly and cheaply.

    The (B,N,M) matrix is used ONLY to find the argmin, under stop_gradient. The
    distances themselves are then recomputed directly on the matched pairs. Two
    reasons this beats reducing over the big matrix:

      * Exactness. |a|^2 - 2a.b + |b|^2 catastrophically cancels precisely where
        it matters most -- nearest neighbours are the smallest distances in the
        matrix. Recomputing on (B,N,3) removes that error entirely.
      * Cost. The backward pass no longer flows through the (B,N,M) matmul; it
        only touches the O(N) matched pairs.

    The gradient is unchanged: d/dx of min_j ||a_i - b_j|| is (a.e.) the gradient
    of the distance to the arg-min point, which is what reduce_min computes too.
    """
    d2 = tf.stop_gradient(squared_distance_matrix(array1, array2))
    idx1 = tf.argmin(d2, axis=-1, output_type=tf.int32)   # (B,N) -> indexes array2
    idx2 = tf.argmin(d2, axis=-2, output_type=tf.int32)   # (B,M) -> indexes array1

    match1 = tf.gather(array2, idx1, batch_dims=1)        # (B,N,3)
    dist1 = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(array1 - match1), -1), EPS))
    match2 = tf.gather(array1, idx2, batch_dims=1)        # (B,M,3)
    dist2 = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(array2 - match2), -1), EPS))
    return dist1, dist2, idx1, idx2


def calc_cd(pred, gt, return_raw=False):
    """Chamfer, with the demo's exact definitions so numbers stay comparable.

    dist1: gt -> pred (length n_gt), idx1 indexes pred
    dist2: pred -> gt (length n_pred), idx2 indexes gt
    """
    dist1, dist2, idx1, idx2 = _min_dists(gt, pred)
    cd_p = (tf.sqrt(tf.reduce_mean(dist1, axis=1)) + tf.sqrt(tf.reduce_mean(dist2, axis=1))) / 2
    cd_t = tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1)
    if return_raw:
        return cd_p, cd_t, dist1, dist2, idx1, idx2
    return cd_p, cd_t


def calc_f1(pred, gt, threshold=0.05):
    """F-score at a distance threshold, the metric the source paper reports.

    Returns (f1, precision, recall), each (B,).
      precision = fraction of PREDICTED points with a ground-truth point within
                  `threshold` -- "how much of what I drew is real"
      recall    = fraction of GROUND-TRUTH points with a predicted point within
                  `threshold` -- "how much of the real thing did I cover"

    `threshold` is in NORMALISED units, matching the source paper's Table 1/2
    (F-score@0.05 and @0.03). At this dataset's mean radius of 103.8 mm those
    are 5.19 mm and 3.11 mm. Reporting both lets our numbers sit next to their
    Table 2 (the 4,800-shape run), which is the closest analogue to this setting.

    Unlike Chamfer this says WHERE the error lives: a low precision with high
    recall means spurious points off the surface, the reverse means an
    incomplete reconstruction. Chamfer averages the two failure modes together.
    """
    dist1, dist2, _, _ = _min_dists(gt, pred)          # gt->pred, pred->gt
    recall = tf.reduce_mean(tf.cast(dist1 < threshold, tf.float32), axis=1)
    precision = tf.reduce_mean(tf.cast(dist2 < threshold, tf.float32), axis=1)
    f1 = 2 * precision * recall / tf.maximum(precision + recall, EPS)
    return f1, precision, recall


def calc_hausdorff(pred, gt, percentile=95.0):
    """Symmetric Hausdorff distance, in NORMALISED units. (B,)

    Returns the `percentile`-th percentile of each direction's nearest-neighbour
    distances, then the max of the two. Defaults to 95 rather than 100 because
    the true Hausdorff distance is a single worst point and is dominated by one
    outlier -- medical shape papers almost always report HD95 for that reason.

    Worth reporting alongside Chamfer because it measures the WORST-CASE gap,
    which Chamfer's mean hides. For implant design a 6 mm average with one 15 mm
    hole is not clinically equivalent to a uniform 6 mm error.
    """
    dist1, dist2, _, _ = _min_dists(gt, pred)
    q = percentile / 100.0
    h1 = tfp_percentile(dist1, q)
    h2 = tfp_percentile(dist2, q)
    return tf.maximum(h1, h2)


def tfp_percentile(x, q):
    """Per-row quantile of a (B,N) tensor, without a tensorflow_probability dep."""
    n = tf.shape(x)[-1]
    k = tf.maximum(tf.cast(tf.math.ceil(tf.cast(n, tf.float32) * q), tf.int32), 1)
    # top_k of the smallest k values -> the k-th smallest is the q-quantile
    return -tf.math.top_k(-x, k=k).values[:, -1]


def _dcd_from_raw(pred, gt, dist1, dist2, idx1, idx2, alpha=1.0, n_lambda=1.0):
    """DCD's density-weighted part, given distances that were already computed."""
    n_pred = tf.shape(pred)[1]
    n_gt = tf.shape(gt)[1]
    frac_12 = tf.cast(n_pred, tf.float32) / tf.cast(n_gt, tf.float32)
    frac_21 = tf.cast(n_gt, tf.float32) / tf.cast(n_pred, tf.float32)

    # idx1 indexes pred -> count over n_pred; idx2 indexes gt -> count over n_gt.
    count1 = tf.math.bincount(idx1, minlength=n_pred, axis=-1)
    weight1 = tf.cast(tf.gather(count1, idx1, batch_dims=1), tf.float32)
    weight1 = tf.pow(tf.pow(weight1, n_lambda) + 1e-6, -1.0) * frac_21
    loss1 = tf.reduce_mean(-tf.exp(-dist1 * alpha) * weight1 + 1.0, axis=1)

    count2 = tf.math.bincount(idx2, minlength=n_gt, axis=-1)
    weight2 = tf.cast(tf.gather(count2, idx2, batch_dims=1), tf.float32)
    weight2 = tf.pow(tf.pow(weight2, n_lambda) + 1e-6, -1.0) * frac_12
    loss2 = tf.reduce_mean(-tf.exp(-dist2 * alpha) * weight2 + 1.0, axis=1)

    return tf.reduce_mean(loss1 + loss2)


def calc_dcd(pred, gt, alpha=1.0, n_lambda=1.0):
    """Density-aware Chamfer Distance. Vectorised over the batch."""
    pred = tf.cast(pred, tf.float32)
    gt = tf.cast(gt, tf.float32)
    _, _, dist1, dist2, idx1, idx2 = calc_cd(pred, gt, return_raw=True)
    return _dcd_from_raw(pred, gt, dist1, dist2, idx1, idx2, alpha, n_lambda)


def dcd_loss(y_true, y_pred):
    """The demo's loss. Keras passes (ground truth, prediction).

    WARNING: DCD cannot bootstrap a randomly initialised model. It is bounded in
    [0, 2] and both of its factors vanish when the prediction is far from the
    target. Measured at init on this model (paper config, seed 42):

        mean nearest-neighbour distance 2.79  ->  exp(-2.79) = 0.067
        all 6144 GT points collapse onto 6 distinct predicted points,
        so the density weight 1/count^lambda is ~1/1970

    Their product leaves a loss of 1.9995 against a ceiling of 2.0, with
    effectively no gradient -- an LR sweep over 1e-7 / 1e-4 / 3e-4 / 1e-3 moved
    it by less than 0.03 in 40 steps. Use `chamfer_loss` (or `cd_dcd_loss`) to
    train from scratch and keep this as a refinement objective / metric.
    """
    return calc_dcd(y_pred, y_true)


def chamfer_loss(y_true, y_pred):
    """Plain (bidirectional) Chamfer -- unbounded, so it bootstraps from noise."""
    return tf.reduce_mean(calc_cd(y_pred, y_true)[1])


def cd_dcd_loss(y_true, y_pred, dcd_weight=1.0, n_lambda=1.0):
    """Chamfer + DCD. No scheduling needed: CD starts around 5 and dominates
    while the shape is still wrong, then decays below the bounded DCD term,
    which takes over as the density-aware refinement signal."""
    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(y_pred, y_true, return_raw=True)
    return tf.reduce_mean(cd_t) + dcd_weight * _dcd_from_raw(
        y_pred, y_true, dist1, dist2, idx1, idx2, n_lambda=n_lambda)


def repulsion_loss(pred, r0, k=4):
    """Hinge repulsion between predicted points. Nothing to do with the target.

    WHY THIS EXISTS, given DCD is already the "density-aware" term: DCD's density
    factor is `1/count^lambda` where `count` comes from `argmin`, a piecewise
    constant function. Its gradient w.r.t. point positions is exactly zero --
    measured, `tf.gradients` returns None for that path. So DCD can only reweight
    Chamfer's gradients; it can never apply a force that pushes two predicted
    points apart. Worse, two coincident predictions each matched by a different
    ground-truth point have count=1 and draw no DCD penalty at all, which is the
    dominant form of the clumping seen here. This loss supplies exactly the term
    DCD structurally cannot.

    Hinge rather than PU-Net's `-d*exp(-d^2/h^2)`: once points are `r0` apart the
    gradient is exactly zero, so the term stops fighting Chamfer instead of
    pushing forever with a decaying weight. And `r0` is a distance, so it can be
    read off the ground truth rather than tuned blind -- measured GT nearest
    neighbour spacing on this dataset has a hard floor at 3.0 mm (0.028% below
    3 mm, 0% below 2 mm), because the GT is farthest-point sampled.

    `r0` is in NORMALISED units. Each skull was divided by its own radius, and
    those range 88.3-133.1 mm, so one fixed normalised r0 enforces 1.70-2.57 mm
    depending on the skull. That spread is noise, not bias; if this direction
    proves out, derive r0 per sample from the target's own spacing instead.

    DIMENSIONLESS ON PURPOSE. The shortfall is divided by r0 before squaring, so
    the per-pair penalty is a *fraction* in [0,1] -- 1 when two points coincide,
    0 once they are r0 apart -- instead of a squared length. The first version
    returned the raw squared length and was unusable: at r0=0.01927 normalised,
    the largest possible per-pair penalty is r0^2 = 3.7e-4, and after averaging
    over mostly-compliant pairs the term measured 1e-6 against CD 0.066 and DCD
    1.147. At weight 0.5 it contributed 0.0095% of the gradient -- the run that
    used it was, in effect, repulsion-free. Reaching a 30% gradient share would
    have needed weight ~2249, a number that is an artefact of r0 and would have
    to be retuned every time r0 moved. Normalised, the gradient norm is ~0.51
    against CD's 0.72, so weights of 0.1-1 mean what they look like, and r0 and
    the weight are independent knobs.
    """
    n = pred.shape[1]
    # Neighbour SELECTION is chunked and gradient-free. A full (B,N,N) matrix is
    # 604 MB at B=4/N=6144, and squared_distance_matrix needs a few copies of it;
    # measured, doing it in one go costs +4.8 GiB of peak memory and OOMs a 24 GB
    # card once the clump metric is also live. Chunking caps it at (B,chunk,N)
    # while changing nothing about the result -- the differentiable part below
    # only ever sees the (B,N,k,3) gathered neighbours.
    idx_parts = []
    for start in range(0, n, _REPULSION_CHUNK):
        stop = min(start + _REPULSION_CHUNK, n)
        blk = pred[:, start:stop]
        d2 = tf.stop_gradient(squared_distance_matrix(blk, pred))   # (B,chunk,N)
        # Push each query point's own column out of contention so it is never
        # returned as its own neighbour.
        self_mask = tf.one_hot(tf.range(start, stop), n, on_value=1e10, off_value=0.0)
        _, idx_blk = tf.math.top_k(-(d2 + self_mask), k=k)
        idx_parts.append(idx_blk)
    idx = tf.concat(idx_parts, axis=1)                     # (B,N,k) nearest others
    nb = tf.gather(pred, idx, batch_dims=1)                # (B,N,k,3)
    # Distances recomputed on the selected pairs only, so the backward pass never
    # touches the (B,N,N) matrix -- same two-stage trick as `_min_dists`.
    d = tf.sqrt(tf.maximum(
        tf.reduce_sum(tf.square(tf.expand_dims(pred, 2) - nb), axis=-1), EPS))
    shortfall = tf.maximum(0.0, r0 - d) / tf.maximum(r0, EPS)   # in [0,1]
    return tf.reduce_mean(tf.square(shortfall))


def make_clump_metric(thresh, n_sample=512):
    """Fraction of predicted points whose nearest neighbour is closer than `thresh`.

    This is the quantity repulsion targets, so it needs to be visible per epoch;
    otherwise tuning is blind until the surface-quality notebook is run. Ground
    truth scores exactly 0.0% here, which makes it a clean reference.

    Subsampled to `n_sample` query points against all N: the exact version needs
    an (N,N) matrix every step for a number that is only ever read by a human.
    Sampling error is ~1/sqrt(n_sample) (~3% relative at 1024), far below the
    epoch-to-epoch variation. `thresh` is in normalised units.
    """
    def clump_metric(y_true, y_pred):
        pred = tf.cast(y_pred, tf.float32)
        n = tf.shape(pred)[1]
        sel = tf.random.shuffle(tf.range(n))[:n_sample]
        sub = tf.gather(pred, sel, axis=1)                          # (B,S,3)
        d2 = tf.stop_gradient(squared_distance_matrix(sub, pred))   # (B,S,N)
        # top_k of -d2: [0] is the query point matching itself at 0, [1] is its
        # nearest genuine neighbour.
        vals, _ = tf.math.top_k(-d2, k=2)
        nn = tf.sqrt(tf.maximum(-vals[:, :, 1], 0.0))
        return tf.reduce_mean(tf.cast(nn < thresh, tf.float32))
    return clump_metric


def make_loss(name, dcd_weight=1.0, n_lambda=1.0,
              repulsion_weight=0.0, repulsion_r0=0.0, repulsion_k=4):
    """Build a Keras-compatible loss, with DCD's two knobs exposed.

    dcd_weight scales the whole DCD term. Read it as a gradient share, not as a
    share of the loss value -- the two are wildly different here, and the loss
    value is the misleading one. Measured on the trained model: DCD is 92.6% of
    the loss VALUE (0.874 of 0.944) but only ~36% of the gradient, because DCD
    is bounded in [0,2] and saturates while Chamfer does not. So the weight is
    doing much less than the printed loss suggests:

        dcd_weight   1     2     3     5    10    20
        DCD's share 36%   53%   63%   74%   85%   92%   of |grad|

    Past ~10 Chamfer barely gets a vote and shape accuracy is at risk; 1-5 is
    the useful range.

    n_lambda is the exponent in DCD's density weight 1/count^n_lambda, i.e. how
    hard a predicted point is punished for being the nearest neighbour of many
    ground-truth points. Raising it targets clumping SPECIFICALLY, whereas
    dcd_weight scales DCD's distance and density factors together -- DCD is
    `1 - exp(-dist*alpha) * 1/count^n_lambda`, a product of both. If the goal is
    the 11.2%-of-points-within-2mm problem rather than accuracy in general,
    n_lambda is the more targeted knob of the two.

    repulsion_weight adds `repulsion_loss` on top of whichever base loss `name`
    selects, so the ablation that matters -- does DCD still earn its place once a
    real repulsion term exists? -- is reachable as `--loss cd --repulsion-weight w`
    versus `--loss cd_dcd --repulsion-weight w`. 0.0 (default) reproduces the
    previous behaviour exactly.
    """
    if name == "cd":
        base = chamfer_loss
        tag = "cd"
    elif name == "dcd":
        base = dcd_loss
        tag = "dcd"
    elif name == "cd_dcd":
        def base(y_true, y_pred):
            return cd_dcd_loss(y_true, y_pred, dcd_weight=dcd_weight, n_lambda=n_lambda)
        tag = f"cd_dcd_w{dcd_weight:g}_l{n_lambda:g}"
    else:
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(LOSSES)}")

    if repulsion_weight <= 0.0:
        # Return the bare base loss rather than a wrapper that adds 0.0 -- keeps
        # the graph (and any saved run) identical to before this feature existed.
        # Wrapped in functools.partial rather than renamed in place: `base` is the
        # module-level `chamfer_loss` / `dcd_loss` for those two names, so
        # assigning to its __name__ renamed the global function for the rest of
        # the process and the next make_loss call in the same session inherited
        # the wrong label.
        named = functools.wraps(base)(lambda y_true, y_pred: base(y_true, y_pred))
        named.__name__ = tag
        return named

    def loss(y_true, y_pred):
        return base(y_true, y_pred) + repulsion_weight * repulsion_loss(
            tf.cast(y_pred, tf.float32), repulsion_r0, repulsion_k)

    # Keras logs this name; keep the settings visible in history.csv headers.
    loss.__name__ = f"{tag}_rep{repulsion_weight:g}"
    return loss


LOSSES = {"cd": chamfer_loss, "dcd": dcd_loss, "cd_dcd": cd_dcd_loss}


def cd_t_metric(y_true, y_pred):
    return tf.reduce_mean(calc_cd(y_pred, y_true)[1])


def cd_p_metric(y_true, y_pred):
    return tf.reduce_mean(calc_cd(y_pred, y_true)[0])


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def knn_point(k, xyz, new_xyz):
    """k nearest neighbours of each `new_xyz` centroid among `xyz`."""
    d = squared_distance_matrix(new_xyz, xyz)          # (B, M, N)
    val, idx = tf.math.top_k(-d, k)
    return -val, idx


class PointSampler(L.Layer):
    """Picks `num_points` centroid indices per cloud.

    mode="unique"   : distinct indices (argsort of uniform noise)
    mode="original" : the demo's tf.random.uniform, i.e. WITH replacement
    """

    def __init__(self, num_points, mode="unique", seed=42, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.mode = mode
        self.seed = seed

    def call(self, inputs, training=None):
        """Stateful (fresh) draw while training, stateless (fixed) draw otherwise.

        The demo samples centroids with a stateful `tf.random.uniform` in both
        modes, which makes INFERENCE NON-DETERMINISTIC: calling the trained model
        twice on the same skull moved the output by 1.03 in normalised units in a
        direct check. Any metric computed that way carries sampling variance on
        top of model error and cannot be reproduced run to run.

        Training keeps the stateful draw -- re-drawing centroids every step is
        free augmentation, which matters at 40 training skulls. Evaluation uses a
        stateless draw with a fixed seed, so repeated calls agree exactly.
        """
        batch = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]

        if training:
            noise = tf.random.uniform((batch, n), seed=self.seed)
        else:
            noise = tf.random.stateless_uniform((batch, n), seed=[self.seed, 0])

        if self.mode == "original":                       # with replacement (demo)
            return tf.cast(noise[:, : self.num_points] * tf.cast(n, tf.float32), tf.int32)
        return tf.argsort(noise, axis=1)[:, : self.num_points]

    def get_config(self):
        return {**super().get_config(), "num_points": self.num_points,
                "mode": self.mode, "seed": self.seed}


def sample_and_group(args, nsample):
    xyz, pts, idx_c = args
    new_xyz = tf.gather(xyz, idx_c, batch_dims=1)
    new_pts = tf.gather(pts, idx_c, batch_dims=1)
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped = tf.gather(pts, idx, batch_dims=1)
    centred = grouped - tf.expand_dims(new_pts, 2)
    out = tf.concat([centred, tf.tile(tf.expand_dims(new_pts, 2), (1, 1, nsample, 1))], axis=-1)
    return new_xyz, out


def LBR(tensor, C, name, use_bias=True, leaky=0.0):
    x = L.Dense(C, use_bias=use_bias, name=name + "_lin")(tensor)
    if leaky == 0.0:
        return L.ReLU(name=name + "_ReLU")(x)
    return L.LeakyReLU(alpha=leaky, name=name + "_ReLU")(x)


def _offset_attention(query_src, key_src, name, tie_qk=False):
    """PCT offset attention. query_src is the residual stream.

    Dropout was tried here (on the sublayer output, the standard transformer
    placement) and removed -- it made val CD_t 2x worse and the model was never
    overfitting to begin with. See the 2026-08-06 devlog entry before adding any
    regulariser: measured val/train CD_t is 1.03-1.08x across every run.
    """
    C = key_src.shape[-1]
    out_dim = query_src.shape[-1]
    q_layer = L.Dense(C // 4, use_bias=False, name=name + "_Q")
    k_layer = L.Dense(C // 4, use_bias=False, name=name + "_K")
    q = q_layer(query_src)
    k = k_layer(key_src)
    if tie_qk:
        # The published demo does this in Self_Attention (and only there):
        # `W_k.set_weights(W_q.get_weights())`. It is a one-off copy at build
        # time -- the two stay separate trainable variables -- so it only sets
        # the starting point. With W_k == W_q the initial energy is a Gram
        # matrix whose diagonal dominates, i.e. "attend to yourself" is the
        # prior. Without it the scores start isotropic. This rewrite omitted the
        # line from the start (2026-07-28); see devlog 2026-08-21.
        k_layer.set_weights(q_layer.get_weights())
    v = L.Dense(out_dim, use_bias=False, name=name + "_V")(key_src)

    energy = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True), name=name + "_matmul1")([q, k])
    att = L.Softmax(axis=1, name=name + "_softmax")(energy)
    att = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=name + "_l1norm")(att)
    r = L.Lambda(lambda t: tf.matmul(t[0], t[1]), name=name + "_matmul2")([att, v])
    r = L.Subtract(name=name + "_subtract")([query_src, r])
    r = LBR(r, out_dim, name + "_LBR")
    return L.Add(name=name + "_add")([query_src, r])


def self_attention(x, name, tie_qk=False):
    return _offset_attention(x, x, name, tie_qk=tie_qk)


def cross_attention(enc, dec, name):
    return _offset_attention(dec, enc, name)


def copy_and_mapping(x, nmul, name):
    x = L.Lambda(lambda t: tf.expand_dims(t, 2), name=name + "_expand")(x)
    C = x.shape[-1] // nmul
    x1 = L.Conv2DTranspose(C, (1, nmul), (1, nmul), name=name + "_convT")(x)
    x2 = L.Dense(C, name=name + "_lin")(x)
    x2 = L.Lambda(lambda t: tf.tile(t, [1, 1, nmul, 1]), name=name + "_tile")(x2)
    x = L.Add(name=name + "_add")([x1, x2])
    npoint = x.shape[1] * x.shape[2]
    return L.Lambda(lambda t: tf.reshape(t, [-1, npoint, t.shape[3]]), name=name + "_reshape")(x)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class MSNConfig:
    n_in: int = 2048
    sg1_sample: int = 1024
    sg2_sample: int = 256
    sg_knn: int = 32
    lbr_in: tuple = (64, 128)
    sg1_dim: int = 128
    sg2_dim: int = 256
    n_sa_stage1: int = 2
    n_sa_stage2: int = 2
    enc_mid_dim: int = 512
    enc_out_dim: int = 1024
    dec_seed: int = 256          # output points = dec_seed * 3 * 2
    dec_head: tuple = (128, 128, 64)
    sampler: str = "unique"
    use_text: bool = True
    # Decoder stage 1 keys. False (default) reproduces the published behaviour:
    # m1 is the global vector tiled `dec_seed` times, so every key row is
    # identical, the attention weights collapse to a uniform 1/dec_seed, and the
    # block reduces to `x <- x + LBR(x - V(g))` -- a shared global conditioning
    # offset, not attention. True concatenates the encoder's per-point features
    # after the global vector, so the four D1 blocks can actually attend.
    #
    # This changes NO weight shape (only the key sequence length), so a
    # checkpoint from one setting loads into the other without raising. It is
    # recorded in run.json and read back by report.Run.arch_key for exactly that
    # reason -- see the 2026-08-21 devlog entry.
    per_point_attn: bool = False
    # Restores the demo's one-line Q/K weight tie in the encoder's self-attention
    # (cross-attention cannot tie -- its Q and K have different input widths, and
    # the demo does not tie there either). Off by default = this rewrite's
    # behaviour since 2026-07-28, which is what every run so far used.
    tie_qk_init: bool = False
    text_in_dim: int = 768       # BERT pooler width

    @property
    def n_out(self) -> int:
        return self.dec_seed * 6

    @property
    def dec_d1_dim(self) -> int:
        return self.enc_out_dim // 4

    @property
    def dec_d2_dim(self) -> int:
        return self.dec_d1_dim // 4

    @staticmethod
    def small() -> "MSNConfig":
        return MSNConfig()

    @staticmethod
    def paper() -> "MSNConfig":
        """The published architecture: 4096 in / 6144 out, 187M parameters.

        Measured at 372 ms/step (batch 4, 15.5 GiB) on one RTX 4090 once the
        distance kernels above are fixed -- so this, not a shrunken proxy, is
        what you should train. Batch 8 still OOMs at 24 GB; the paper's 6xA6000
        (40 GB each) is what that setting needs.

        `sampler` stays "unique" here: with-replacement centroid sampling is a
        defect, not part of the architecture, and the choice is non-parametric
        so weight compatibility with MSN_weights3.h5 is unaffected either way.
        """
        return MSNConfig(
            n_in=4096, sg1_sample=4096, sg2_sample=2048,
            sg1_dim=512, sg2_dim=1024,
            n_sa_stage1=4, n_sa_stage2=4,
            enc_mid_dim=2048, enc_out_dim=4096,
            dec_seed=1024,
        )


# --------------------------------------------------------------------------- #
# encoder / decoder / model
# --------------------------------------------------------------------------- #
def build_encoder(xyz, cfg: MSNConfig):
    x = LBR(xyz, cfg.lbr_in[0], "E-IN_LBR1", use_bias=False)
    x = LBR(x, cfg.lbr_in[1], "E-IN_LBR2", use_bias=False)

    idx = PointSampler(cfg.sg1_sample, cfg.sampler, name="E-SG1_sample")(xyz)
    new_xyz, feat = L.Lambda(sample_and_group, arguments={"nsample": cfg.sg_knn}, name="E-SG1")([xyz, x, idx])
    x = LBR(feat, cfg.sg1_dim, "E-SG1_LBR1", use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name="E-SG1_MaxPool")(x)

    idx = PointSampler(cfg.sg2_sample, cfg.sampler, name="E-SG2_sample")(new_xyz)
    new_xyz, feat = L.Lambda(sample_and_group, arguments={"nsample": cfg.sg_knn}, name="E-SG2")([new_xyz, x, idx])
    x = LBR(feat, cfg.sg2_dim, "E-SG2_LBR1", use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name="E-SG2_MaxPool")(x)

    outs, h = [], x
    for i in range(cfg.n_sa_stage1):
        h = self_attention(h, f"E-SA{i + 1}", tie_qk=cfg.tie_qk_init)
        outs.append(h)
    x0 = L.Concatenate(axis=2, name="E-SA_Concat")(outs) if len(outs) > 1 else outs[0]
    x = L.Concatenate(axis=2, name="E-OUT_Concat")([x0, x])
    x = LBR(x, cfg.enc_mid_dim, "E-OUT_LBR", use_bias=False, leaky=0.2)

    outs, h = [], x
    for i in range(cfg.n_sa_stage2):
        h = self_attention(h, f"E-SA{cfg.n_sa_stage1 + i + 1}", tie_qk=cfg.tie_qk_init)
        outs.append(h)
    x0 = L.Concatenate(axis=2, name="E-SA_Concat2")(outs) if len(outs) > 1 else outs[0]
    x = LBR(x0, cfg.enc_out_dim, "E-OUT_LBR1", use_bias=False, leaky=0.2)
    # `x` is (B, sg2_sample, enc_out_dim) and the max-pool throws all but one row
    # of it away -- that pool is the information bottleneck. Return both: the
    # pooled vector is what the model has always used, the per-point tensor costs
    # nothing extra because it is already computed.
    pooled = L.Lambda(lambda t: tf.reduce_max(t, axis=1, keepdims=True), name="E-OUT_MaxPool")(x)
    return pooled, x


def build_decoder(feats, cfg: MSNConfig, per_point=None):
    seed1 = cfg.dec_seed
    seed2 = seed1 * 3

    if per_point is None:
        m1 = L.Lambda(lambda t: tf.tile(t, [1, seed1, 1]), name="D-IN_replicate")(feats)
    else:
        # Global vector FIRST, then the per-point rows. Keeping it means the
        # previous behaviour stays reachable (attend to row 0 only) and the text
        # branch keeps its one and only path into the decoder -- `feats` reaches
        # nothing else here, D1-eye/D2-eye use it for batch size alone. Dropping
        # it measured worse in the defect region; see devlog 2026-08-21.
        m1 = L.Concatenate(axis=1, name="D-IN_keys")([feats, per_point])
    # eye_seed is fixed to zeros, so this is a learned (seed, dim) embedding table
    # expressed as Dense(identity) -- kept in this form to mirror the demo.
    e1 = L.Lambda(lambda t: tf.tile(tf.expand_dims(tf.eye(seed1), 0), [tf.shape(t)[0], 1, 1]),
                  name="D1-eye")(feats)
    x = L.Dense(cfg.dec_d1_dim, use_bias=False, name="D1-IN")(e1)
    outs = []
    for i in range(4):
        x = cross_attention(m1, x, f"D1-STA{i + 1}")
        outs.append(x)
    x = L.Concatenate(axis=2, name="D1-STA_Concat")(outs)
    x = L.Concatenate(axis=2, name="D1-OUT_Concat")([x, outs[0]])
    m2 = copy_and_mapping(x, 3, "D1-OUT_CopyAndMapping")

    e2 = L.Lambda(lambda t: tf.tile(tf.expand_dims(tf.eye(seed2), 0), [tf.shape(t)[0], 1, 1]),
                  name="D2-eye")(feats)
    x = L.Dense(cfg.dec_d2_dim, use_bias=False, name="D2-IN")(e2)
    outs = []
    for i in range(4):
        x = cross_attention(m2, x, f"D2-STA{i + 1}")
        outs.append(x)
    x = L.Concatenate(axis=2, name="D2-STA_Concat")(outs)
    x = L.Concatenate(axis=2, name="D2-OUT_Concat")([x, outs[0]])
    x = copy_and_mapping(x, 2, "D2-OUT_CopyAndMapping")

    for i, c in enumerate(cfg.dec_head):
        leaky = 0.2 if i == len(cfg.dec_head) - 1 else 0.0
        x = LBR(x, c, f"D-OUT_LBR{i + 1}", use_bias=False, leaky=leaky)
    # float32 so mixed precision never touches the coordinates
    return L.Dense(3, name="D-OUT_lin", dtype="float32")(x)


def build_model(cfg: MSNConfig = None) -> tf.keras.Model:
    cfg = cfg or MSNConfig.small()
    xyz = Input(shape=(cfg.n_in, 3), name="input_points")
    inputs = [xyz]

    encoded, per_point = build_encoder(xyz, cfg)

    if cfg.use_text:
        text = Input(shape=(cfg.text_in_dim,), name="text_feat")
        inputs.append(text)
        t = L.Dense(cfg.enc_out_dim, activation="relu", name="text_proj")(text)
        t = L.Lambda(lambda z: tf.expand_dims(z, 1), name="text_expand")(t)
        encoded = L.Add(name="multimodal_add")([encoded, t])

    out = build_decoder(encoded, cfg,
                        per_point=per_point if cfg.per_point_attn else None)
    return M.Model(inputs=inputs, outputs=out, name="MSN_PCT_skullfix")


# --------------------------------------------------------------------------- #
# text branch: run BERT once, not every step
# --------------------------------------------------------------------------- #
def encode_class_name(class_name="skull", model_name="bert-base-uncased", max_length=128):
    """Frozen BERT + a single class == a constant. Compute it once."""
    from transformers import BertTokenizer, TFBertModel

    tok = BertTokenizer.from_pretrained(model_name)
    bert = TFBertModel.from_pretrained(model_name, use_safetensors=False)
    enc = tok(class_name, add_special_tokens=True, max_length=max_length,
              padding="max_length", truncation=True, return_tensors="tf")
    pooled = bert(enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output
    return np.asarray(pooled, dtype=np.float32)[0]


def dcd_metric(y_true, y_pred):
    return calc_dcd(y_pred, y_true)
