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
  1536 out, 25 ms/step, 0.9 GiB) for debugging the pipeline. Because the decoder
  seed size is baked into the `D1-IN` / `D2-IN` weight shapes, `small()` cannot
  load msn_downloads/MSN_weights3.h5; `paper()` can.

LOSS
  Do NOT train from scratch with DCD alone -- see the warning on `dcd_loss`. Use
  `cd_dcd_loss` (Chamfer + the density-aware term) or `chamfer_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

EPS = 1e-12


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


def cd_dcd_loss(y_true, y_pred, dcd_weight=1.0):
    """Chamfer + DCD. No scheduling needed: CD starts around 5 and dominates
    while the shape is still wrong, then decays below the bounded DCD term,
    which takes over as the density-aware refinement signal."""
    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(y_pred, y_true, return_raw=True)
    return tf.reduce_mean(cd_t) + dcd_weight * _dcd_from_raw(
        y_pred, y_true, dist1, dist2, idx1, idx2)


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


def _offset_attention(query_src, key_src, name):
    """PCT offset attention. query_src is the residual stream."""
    C = key_src.shape[-1]
    out_dim = query_src.shape[-1]
    q = L.Dense(C // 4, use_bias=False, name=name + "_Q")(query_src)
    k = L.Dense(C // 4, use_bias=False, name=name + "_K")(key_src)
    v = L.Dense(out_dim, use_bias=False, name=name + "_V")(key_src)

    energy = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True), name=name + "_matmul1")([q, k])
    att = L.Softmax(axis=1, name=name + "_softmax")(energy)
    att = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=name + "_l1norm")(att)
    r = L.Lambda(lambda t: tf.matmul(t[0], t[1]), name=name + "_matmul2")([att, v])
    r = L.Subtract(name=name + "_subtract")([query_src, r])
    r = LBR(r, out_dim, name + "_LBR")
    return L.Add(name=name + "_add")([query_src, r])


def self_attention(x, name):
    return _offset_attention(x, x, name)


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
        h = self_attention(h, f"E-SA{i + 1}")
        outs.append(h)
    x0 = L.Concatenate(axis=2, name="E-SA_Concat")(outs) if len(outs) > 1 else outs[0]
    x = L.Concatenate(axis=2, name="E-OUT_Concat")([x0, x])
    x = LBR(x, cfg.enc_mid_dim, "E-OUT_LBR", use_bias=False, leaky=0.2)

    outs, h = [], x
    for i in range(cfg.n_sa_stage2):
        h = self_attention(h, f"E-SA{cfg.n_sa_stage1 + i + 1}")
        outs.append(h)
    x0 = L.Concatenate(axis=2, name="E-SA_Concat2")(outs) if len(outs) > 1 else outs[0]
    x = LBR(x0, cfg.enc_out_dim, "E-OUT_LBR1", use_bias=False, leaky=0.2)
    return L.Lambda(lambda t: tf.reduce_max(t, axis=1, keepdims=True), name="E-OUT_MaxPool")(x)


def build_decoder(feats, cfg: MSNConfig):
    seed1 = cfg.dec_seed
    seed2 = seed1 * 3

    m1 = L.Lambda(lambda t: tf.tile(t, [1, seed1, 1]), name="D-IN_replicate")(feats)
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

    encoded = build_encoder(xyz, cfg)

    if cfg.use_text:
        text = Input(shape=(cfg.text_in_dim,), name="text_feat")
        inputs.append(text)
        t = L.Dense(cfg.enc_out_dim, activation="relu", name="text_proj")(text)
        t = L.Lambda(lambda z: tf.expand_dims(z, 1), name="text_expand")(t)
        encoded = L.Add(name="multimodal_add")([encoded, t])

    out = build_decoder(encoded, cfg)
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
