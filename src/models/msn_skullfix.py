"""MSN (PCT + text) point-cloud completion, reworked to train on one 24 GB GPU.

A rewrite of notebooks/upstream_msn/MSN_model_training_Demo.ipynb. The layer
topology is the published one; what changed inside the model, and the sizing
numbers, are in src/models/README.md.

`MSNConfig.paper()` is the published architecture -- 4096 in, 6144 out, 187M
parameters -- and the default for training. `MSNConfig.small()` is a 9.4M
variant for debugging the pipeline.

Warning: NEITHER config can load msn_downloads/MSN_weights3.h5, and the failure
is SILENT -- `load_weights(by_name=True, skip_mismatch=True)` returns without
raising while matching 3 of the checkpoint's 32 weight groups, leaving most of
the network randomly initialised. Shapes are compatible; the layer NESTING is
not. Use msn_demo_arch.py for the released weights, and keep the two modules
apart: this one for models this project trains, that one for the vendor's.

Warning: do not train from scratch with DCD alone -- see `dcd_loss`. Use
`cd_dcd_loss` or `chamfer_loss`.
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

    The (B,N,M) matrix is used ONLY for the argmin, under stop_gradient; the
    distances are then recomputed on the matched pairs. That is both more exact
    and cheaper, and leaves the gradient unchanged. Why: src/models/README.md.
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

    Returns (f1, precision, recall), each (B,). Precision is the fraction of
    PREDICTED points with a ground-truth point within `threshold`, recall the
    fraction of GROUND-TRUTH points with a predicted one -- so unlike Chamfer
    this says where the error lives. `threshold` is in NORMALISED units, to match
    the source paper's Table 1 and 2. See src/models/README.md.
    """
    dist1, dist2, _, _ = _min_dists(gt, pred)          # gt->pred, pred->gt
    recall = tf.reduce_mean(tf.cast(dist1 < threshold, tf.float32), axis=1)
    precision = tf.reduce_mean(tf.cast(dist2 < threshold, tf.float32), axis=1)
    f1 = 2 * precision * recall / tf.maximum(precision + recall, EPS)
    return f1, precision, recall


def calc_hausdorff(pred, gt, percentile=95.0):
    """Symmetric Hausdorff distance, in NORMALISED units. (B,)

    The `percentile`-th percentile of each direction's nearest-neighbour
    distances, then the max of the two. 95 rather than 100 because the true
    Hausdorff distance is one worst point; reported alongside Chamfer because it
    catches the worst-case gap a mean hides.
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

    Warning: DCD CANNOT bootstrap a randomly initialised model. It is bounded in
    [0, 2] and both factors vanish when the prediction is far from the target,
    which measured at init leaves the loss pinned at 1.9995 against a ceiling of
    2.0 with effectively no gradient. Train from scratch with `chamfer_loss` or
    `cd_dcd_loss` and keep this as a refinement objective or a metric.
    Measurements in notebooks/README.md.
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

    This is the only term that can push two predicted points apart. DCD's density
    factor is `1/count^lambda` with `count` from an `argmin`, so its gradient
    with respect to position is exactly zero -- DCD can reweight Chamfer's
    gradients but cannot apply that force, and two coincident predictions matched
    by different ground-truth points draw no DCD penalty at all.

    `r0` is a distance in NORMALISED units, so it can be read off the ground
    truth rather than tuned blind. The penalty is DIMENSIONLESS on purpose -- the
    shortfall is divided by r0 before squaring, giving a fraction in [0, 1]
    rather than a squared length, which is what makes a weight of 0.5 mean what
    it looks like. Both choices, and what happened before the second one, are in
    src/models/README.md.
    """
    n = pred.shape[1]
    # Neighbour SELECTION is chunked and gradient-free: done in one go it costs
    # +4.8 GiB of peak memory and OOMs a 24 GB card. Chunking changes nothing
    # about the result. See src/models/README.md.
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

    The quantity repulsion targets, so it is worth seeing per epoch; ground truth
    scores exactly 0.0%, which makes it a clean reference. Subsampled to
    `n_sample` queries against all N -- the exact version needs an (N,N) matrix
    every step for a number only a human reads, and the sampling error is far
    below the epoch-to-epoch variation. `thresh` is in normalised units.
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

    `dcd_weight` scales the DCD term. Read it as a share of the GRADIENT, not of
    the loss value -- DCD saturates, so the loss value badly overstates what it
    is doing. Measured shares are in src/models/README.md; 1-5 is the useful
    range and past ~10 Chamfer barely gets a vote.

    `n_lambda` is the exponent in DCD's density weight. By construction it should
    target clumping specifically, but nothing valid has ever been measured about
    it here -- the only run that tested it sits in the voided tier, and the
    explanation once built on that run was withdrawn untested.

    `repulsion_weight` adds `repulsion_loss` on top of whichever base loss `name`
    selects, which is what makes the ablation that matters reachable: does DCD
    still earn its place once a real repulsion term exists? 0.0 (default)
    reproduces the previous behaviour exactly.
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
        # Return the bare base loss rather than a wrapper adding 0.0, so the
        # graph stays identical to before this feature existed. functools.partial
        # rather than renaming in place: `base` IS the module-level function, and
        # assigning to its __name__ renamed the global for the rest of the
        # process -- the next make_loss call inherited the wrong label.
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

        The demo draws centroids statefully in both modes, which makes INFERENCE
        NON-DETERMINISTIC -- measured, two calls on the same skull moved the
        output by 1.03 normalised units, so any metric from it carries sampling
        variance on top of model error. Training keeps the fresh draw as free
        augmentation; evaluation is stateless with a fixed seed, so repeated
        calls agree exactly.
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

    Dropout was tried here, at the standard placement, and removed: it made val
    CD_t twice as bad and the model was never overfitting -- val/train CD_t is
    1.03-1.08x on every run. Read src/models/README.md before adding any
    regulariser.
    """
    C = key_src.shape[-1]
    out_dim = query_src.shape[-1]
    q_layer = L.Dense(C // 4, use_bias=False, name=name + "_Q")
    k_layer = L.Dense(C // 4, use_bias=False, name=name + "_K")
    q = q_layer(query_src)
    k = k_layer(key_src)
    if tie_qk:
        # The published demo does this in Self_Attention, and only there. A
        # one-off copy at build time, so it sets the starting point and nothing
        # else; this rewrite omitted it from the start. Tested and found to make
        # no reproducible difference -- see src/models/README.md.
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
    # Decoder stage 1 keys. False (default) reproduces the published behaviour,
    # where every key row is identical and the block is a global conditioning
    # offset rather than attention. Changes NO weight shape, so a checkpoint from
    # one setting loads into the other without raising -- which is why it goes
    # into run.json and is read back by report.Run.arch_key.
    # See src/models/README.md.
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

        Trainable as it stands -- 372 ms/step at batch 4, 15.5 GiB on one RTX
        4090 -- so train this rather than a shrunken proxy. `sampler` stays
        "unique": with-replacement centroid sampling is a defect, not part of the
        architecture, and the choice is non-parametric so it cannot affect weight
        compatibility. See src/models/README.md.
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
        # Global vector FIRST, then the per-point rows: it keeps the previous
        # behaviour reachable and is the text branch's only path into the
        # decoder. Dropping it measured worse in the defect region.
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
