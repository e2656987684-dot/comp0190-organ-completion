"""Evaluation + presentation figures, shared by the notebooks.

Everything that used to be copy-pasted into notebook cells lives here, because
the copies drifted: the training notebook was loading one run's history next to
another run's weights, and its curve title still claimed the train/validation
gap was memorisation after that had been measured and disproved.

Two rules carried over from mesh_viz:
  * metrics come from the raw point clouds, never from a reconstructed mesh
  * numbers before pictures -- a figure is for showing a conclusion, not
    reaching one

Typical use:

    import report as rp
    runs = rp.load_runs(REPO, ["baseline_es20", "msn_skullfix/rep_w05"])
    rp.fig_curves(runs).show()
    df = rp.eval_runs(REPO, runs)          # per-skull metrics, needs a GPU
    print(rp.format_summary(df))
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Same palette across every figure so slides look like one deck.
C_TRAIN = "#1565C0"
C_VAL = "#E64A19"
C_GT = "#2E7D32"
C_SERIES = ["#1565C0", "#E64A19", "#6A1B9A", "#00838F", "#EF6C00", "#37474F"]

# F-score thresholds in normalised units. These two are what the source paper
# reports (Table 1/2), so keeping them makes our numbers directly comparable.
F1_THRESHOLDS = (0.05, 0.03)

# A ground-truth point counts as "in the defect" when the nearest point of the
# DEFECTIVE INPUT is further away than this, in millimetres. 5 mm sits well clear
# of the sampling spacing -- measured GT-to-input nearest distance is 2.75 mm at
# the median (shared surface) against 20.1 mm at p99 (the hole) -- and puts 6.7%
# of ground-truth points inside the defect, range 4.5-9.3% across the validation
# skulls. Raising it shrinks the region and makes the metric noisier; lowering it
# starts admitting shared surface.
DEFECT_MM = 5.0


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
class Run:
    """One training run: its config, its curve, and where its weights are."""

    def __init__(self, repo, rel, label=None):
        self.rel = rel
        self.dir = os.path.join(repo, "experiments", rel)
        self.label = label or os.path.basename(rel)
        with open(os.path.join(self.dir, "run.json")) as f:
            self.meta = json.load(f)
        self.hist = pd.read_csv(os.path.join(self.dir, "history.csv"))
        self.scale_mm = float(self.meta["scale_mm"])

    @property
    def weights(self):
        return os.path.join(self.dir, "best.h5")

    @property
    def arch_key(self):
        """Which network topology these weights belong to.

        Every field that changes the topology WITHOUT changing any weight shape
        has to be listed here, because those are exactly the ones `load_weights`
        cannot catch. Runs recorded before a field existed default to the value
        that was in force at the time, so old run.json files stay readable.
        """
        m = self.meta
        return (m.get("config", "paper"),
                bool(m.get("per_point_attn", False)),
                bool(m.get("use_text", True)))

    @property
    def arch_label(self):
        name, per_point, use_text = self.arch_key
        bits = [name]
        if per_point:
            bits.append("pp_attn")
        if not use_text:
            bits.append("no_text")
        return "+".join(bits)

    @property
    def best_epoch(self):
        """1-indexed epoch of lowest val CD_t -- what best.h5 holds."""
        return int(self.hist["val_cd_t_metric"].idxmin()) + 1

    @property
    def lr_drops(self):
        """Epochs (1-indexed) where the learning rate was reduced.

        Warm-up ramps up over the first ~5 epochs, so only decreases count.
        Marking these on the curves is the point: the single largest gain in
        this project came from making these fire at all.
        """
        lr = self.hist["lr"].to_numpy()
        return [i + 1 for i in range(1, len(lr)) if lr[i] < lr[i - 1] - 1e-12]

    def config_str(self):
        m = self.meta
        bits = [f"loss={m.get('loss', 'cd_dcd')}"]
        if m.get("dcd_lambda", 1) != 1:
            bits.append(f"λ={m['dcd_lambda']:g}")
        if m.get("dcd_weight", 1) != 1:
            bits.append(f"w_dcd={m['dcd_weight']:g}")
        if m.get("repulsion_weight"):
            bits.append(f"rep={m['repulsion_weight']:g}@{m.get('repulsion_r0_mm', 2):g}mm")
        return "  ".join(bits)


def load_runs(repo, specs):
    """specs: list of "dir" or ("label", "dir")."""
    out = []
    for s in specs:
        label, rel = (s, None) if isinstance(s, str) else (s[0], s[1])
        out.append(Run(repo, rel if rel else label, label if rel else None))
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _dcd_numpy(dist1, dist2, idx1, idx2, n_pred, n_gt, alpha=1.0, n_lambda=1.0):
    """msn_skullfix._dcd_from_raw, transcribed to numpy. Reported at lambda=1
    regardless of what a run trained with, so the column stays comparable."""
    w1 = np.bincount(idx1, minlength=n_pred)[idx1].astype(np.float64)
    w1 = (w1 ** n_lambda + 1e-6) ** -1 * (n_gt / n_pred)
    w2 = np.bincount(idx2, minlength=n_gt)[idx2].astype(np.float64)
    w2 = (w2 ** n_lambda + 1e-6) ** -1 * (n_pred / n_gt)
    return float(np.mean(1.0 - np.exp(-dist1 * alpha) * w1)
                 + np.mean(1.0 - np.exp(-dist2 * alpha) * w2))


def metrics_from_points(pred, gt, scale_mm, inp=None):
    """Every point-cloud metric, from ONE pair of nearest-neighbour lookups.

    Deliberately numpy + cKDTree rather than the TensorFlow versions in
    msn_skullfix: those each materialise a (6144, 6144) matrix, so calling
    calc_cd / calc_hausdorff / calc_f1 in sequence allocated three of them and
    ran a 24 GB card out of memory. A KD-tree needs no such matrix, evaluation
    then needs no GPU at all, and the definitions are identical (verified
    against the TF implementations to within 1e-5).

    dist1: gt -> pred      dist2: pred -> gt      both in normalised units.

    Pass `inp` (the defective input cloud) to also get the defect-region columns;
    without it those are simply absent, so existing callers keep working.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt, dtype=np.float64)
    dist1, idx1 = cKDTree(P).query(G, k=1, workers=-1)   # idx1 indexes P
    dist2, idx2 = cKDTree(G).query(P, k=1, workers=-1)   # idx2 indexes G

    out = {
        "DCD": _dcd_numpy(dist1, dist2, idx1, idx2, len(P), len(G)),
        # matches msn_skullfix.calc_cd's cd_t: mean of each direction, summed
        "CD_t_mm": (dist1.mean() + dist2.mean()) * scale_mm,
        "HD95_mm": max(np.percentile(dist1, 95), np.percentile(dist2, 95)) * scale_mm,
    }
    for t in F1_THRESHOLDS:
        rec = float((dist1 < t).mean())          # of the real surface, how much covered
        prec = float((dist2 < t).mean())         # of what was drawn, how much is real
        out[f"F1@{t:g}"] = 2 * prec * rec / max(prec + rec, 1e-12)
        if t == F1_THRESHOLDS[0]:
            out["precision"], out["recall"] = prec, rec

    if inp is not None:
        out.update(_defect_metrics(P, G, np.asarray(inp, dtype=np.float64),
                                   dist1, scale_mm))
    return out


def _defect_metrics(P, G, I, dist1, scale_mm):
    """The same metrics, restricted to the region the input does not already show.

    WHY: only 6.7% of ground-truth points lie in the defect -- the other 93.3% are
    surface the model was handed in the input and merely has to reproduce. The
    whole-cloud numbers are therefore dominated by copying, and a model that
    reproduced the visible surface perfectly while filling the hole with garbage
    would still score well. These columns measure the part of the task that is
    actually hard.

    THE TWO MASKS ARE DIFFERENT, ON PURPOSE.

      ground truth : a GT point is in the defect if the nearest INPUT point is
                     more than DEFECT_MM away. The GT->input distance is cleanly
                     bimodal (peak at 2-3 mm for shared surface, valley at 5-6 mm,
                     second peak past 15 mm), and DEFECT_MM = 5 sits in the
                     valley: it selects 6.7% of GT points, against 42% at a 3 mm
                     cut and 4.2% at 8 mm, i.e. the count is flat once past the
                     valley. The scale that sets this is the sampling spacing --
                     4.03 mm between GT points, 4.96 mm between input points --
                     since two independent samplings of one surface land about
                     half a spacing apart.

      prediction   : a predicted point is in the defect if it is within DEFECT_MM
                     of a defect GT point. The same "far from input" rule does
                     NOT work here: a prediction floating 6 mm above intact bone
                     also satisfies it, so that version silently measured how far
                     predictions drift off the surface -- which CD_t and HD95
                     already cover -- rather than how well the hole is filled.
                     Measured, it inflated this column by 43% (4.37 -> 2.91 mm on
                     cd_rep05_full) and inflated it MOST for the worst model
                     (baseline 5.42 -> 3.01), because a poorer prediction drifts
                     past the threshold more often.

    Defining the prediction's region from the ground truth is not circular: the
    region is a fact about (input, GT), and the prediction never enters its
    definition. Restricting a metric to a GT-derived band and then scoring
    predictions inside it is what boundary DSC does as well.

    Reported per direction rather than summed, because the two carry different
    weight:

      coverage (gt -> pred)  : of the missing surface, how close is the nearest
                               predicted point. Cannot be gamed -- ignoring the
                               hole makes this worse. THIS is the column that
                               discriminates: 3.24-3.91 mm across runs, against a
                               0.007 mm repeat difference.
      precision (pred -> gt) : of the points placed in the defect, how close they
                               are to the real surface. Every configuration lands
                               at 2.89-3.01 mm, i.e. it does NOT discriminate --
                               the models differ in how completely they cover the
                               hole, not in how accurately they fill it. Can also
                               be gamed by placing no points there, hence n_pred.
    """
    from scipy.spatial import cKDTree

    thr = DEFECT_MM / scale_mm                       # mm -> normalised
    gt_mask = cKDTree(I).query(G, k=1, workers=-1)[0] > thr

    out = {"defect_gt_%": 100.0 * gt_mask.mean()}
    if not gt_mask.any():                            # no hole found: leave blank
        return {**out, "defect_n_pred": 0, "defect_cov_mm": np.nan,
                "defect_HD95_mm": np.nan, "defect_prec_mm": np.nan,
                "defect_F1@0.05": np.nan}

    cov = dist1[gt_mask]                             # reuse: gt -> nearest pred
    out["defect_cov_mm"] = float(cov.mean()) * scale_mm
    out["defect_HD95_mm"] = float(np.percentile(cov, 95)) * scale_mm

    pr_mask = cKDTree(G[gt_mask]).query(P, k=1, workers=-1)[0] < thr
    out["defect_n_pred"] = int(pr_mask.sum())

    if pr_mask.any():
        # Target is the FULL ground truth, not just its defect points. Restricting
        # it inflates the number badly (measured 12-16 mm): a predicted point at
        # the edge of the hole has its nearest real surface just outside the
        # defect, and excluding those forces it to match something much further
        # in. The question here is "is this point on the skull at all", and the
        # skull is all of G.
        d_pr = cKDTree(G).query(P[pr_mask], k=1, workers=-1)[0]
        out["defect_prec_mm"] = float(d_pr.mean()) * scale_mm
        t = F1_THRESHOLDS[0]
        r, pr = float((cov < t).mean()), float((d_pr < t).mean())
        out["defect_F1@0.05"] = 2 * pr * r / max(pr + r, 1e-12)
    else:                                            # nothing placed in the hole
        out["defect_prec_mm"] = np.nan
        out["defect_F1@0.05"] = 0.0
    return out


def arch_config(msn, arch):
    """Rebuild the MSNConfig a run was trained with, from its arch_key."""
    name, per_point, use_text = arch
    try:
        cfg = getattr(msn.MSNConfig, name)()
    except AttributeError:
        raise ValueError(f"run.json names an unknown config {name!r}") from None
    if per_point:
        # Guards the one-way failure: a run trained with per-point decoder keys
        # being evaluated by a checkout that predates the option. Silently
        # falling back to the replicated global vector would load the weights
        # fine and report numbers for the wrong network.
        if not hasattr(cfg, "per_point_attn"):
            raise ValueError(
                "this run was trained with per_point_attn=True but msn_skullfix "
                "has no such option -- check out the code version that matches it")
        cfg.per_point_attn = True
    cfg.use_text = use_text
    return cfg


def eval_runs(repo, runs, n_skulls=None, device="/GPU:0"):
    """Per-skull metrics for every run, on that run's own validation split.

    Returns a long-form DataFrame: one row per (run, skull). Millimetre columns
    use each skull's own scale, not the dataset mean.

    CD_p is deliberately absent. Its definition -- sqrt of a MEAN DISTANCE --
    takes the square root of a length, so it has no meaningful unit. The bug is
    inherited verbatim from the source project; the value is not reported here.
    """
    import sys
    sys.path.insert(0, os.path.join(repo, "src", "models"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import mesh_viz as mv

    data = np.load(os.path.join(repo, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    # Only needed by text-branch runs; a no-text checkout may not have cached it.
    text_path = os.path.join(repo, "data", "cache", "bert_skull.npy")
    text = np.load(text_path) if os.path.exists(text_path) else None

    # One model per ARCHITECTURE, reused across the runs that share it. Building a
    # fresh 187M-parameter model per run exhausted a 24 GB card at the fourth one:
    # `del model` + clear_session() does not make TF's allocator hand memory back,
    # so each rebuild stacked.
    #
    # Grouping, rather than asserting a single architecture, is what keeps a
    # cross-architecture comparison honest. `load_weights` will NOT protect you
    # here: changing where the decoder's keys come from leaves every weight SHAPE
    # untouched (only the key sequence length moves), so an old checkpoint loads
    # into the new topology without raising, and quietly reports numbers for a
    # network that was never trained. Same failure mode as the MSN_weights3.h5
    # mismatch documented in msn_skullfix -- shapes agree, meaning does not.
    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)
    if len(groups) > 1:
        print(f"[eval_runs] {len(groups)} architectures present, one model each: "
              + ", ".join(f"{k[0]}{'+pp_attn' if k[1] else ''} x{len(v)}"
                          for k, v in groups.items()))

    rows = []
    with tf.device(device):
        for arch, group in groups.items():
            cfg = arch_config(msn, arch)
            if cfg.use_text and text is None:
                raise FileNotFoundError(
                    f"{group[0].label} was trained with the text branch, but "
                    f"{text_path} is missing")
            model = msn.build_model(cfg)
            for run in group:
                model.load_weights(run.weights)
                val = run.meta["val_ids"][:n_skulls] if n_skulls else run.meta["val_ids"]
                pos = [int(np.where(ids == sid)[0][0]) for sid in val]
                # model.predict, NOT model(x) in a loop. Measured: calling the model
                # directly on one sample at a time leaks 0.29 GiB per call and never
                # returns it, so 80 skulls exhausts a 24 GB card partway through the
                # fourth run. predict() holds flat at 0.78 GiB across any number of
                # batches.
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                preds = model.predict(x, batch_size=1, verbose=0)
                for sid, i, p in zip(val, pos, preds):
                    s = float(scales[i])
                    row = {"run": run.label, "id": sid}
                    row.update(metrics_from_points(p, gt[i], s, inp=inputs[i]))
                    st = mv.surface_stats(p, gt[i], s)
                    row["clump_%"] = st["clump_pct"]
                    row["spacing_CV"] = st["spacing_cv"]
                    rows.append(row)
                print(f"  {run.label}: {len(val)} skulls")
    return pd.DataFrame(rows)


SUMMARY_COLS = ["CD_t_mm", "HD95_mm", "F1@0.05", "F1@0.03", "DCD", "clump_%", "spacing_CV"]
# Reported as a separate table: mixing them into the one above invites reading a
# whole-cloud number and a defect-only number off the same row as if comparable.
DEFECT_COLS = ["defect_cov_mm", "defect_HD95_mm", "defect_prec_mm",
               "defect_F1@0.05", "defect_gt_%", "defect_n_pred"]
LOWER_IS_BETTER = {"CD_t_mm", "HD95_mm", "DCD", "clump_%", "spacing_CV",
                   "defect_cov_mm", "defect_HD95_mm", "defect_prec_mm"}


def summarise(df):
    """Mean per run, in the order the runs were given."""
    order = list(dict.fromkeys(df["run"]))
    out = df.groupby("run")[SUMMARY_COLS].mean().loc[order]
    return out.round(4)


def format_defect_summary(df):
    """The defect-region table. Kept separate from format_summary on purpose --
    these millimetres are one-directional and cover 6.7% of the surface, so
    reading them on the same row as the whole-cloud numbers would invite treating
    a 12 mm coverage error as if it were four times worse than a 6 mm CD_t."""
    cols = [c for c in DEFECT_COLS if c in df.columns]
    if not cols:
        return "(no defect-region columns -- pass inp= to metrics_from_points)"
    order = list(dict.fromkeys(df["run"]))
    s = df.groupby("run")[cols].mean().loc[order]
    lines = [f"{'run':<20}" + "".join(f"{c:>16}" for c in cols),
             "-" * (20 + 16 * len(cols))]
    for name, r in s.iterrows():
        lines.append(f"{name:<20}" + "".join(f"{r[c]:>16.3f}" for c in cols))
    return "\n".join(lines)


def format_summary(df, gt_row=None):
    """Plain-text table for a slide or the terminal."""
    s = summarise(df)
    lines = [f"{'run':<20}" + "".join(f"{c:>12}" for c in SUMMARY_COLS)]
    lines.append("-" * len(lines[0]))
    for name, r in s.iterrows():
        lines.append(f"{name:<20}" + "".join(f"{r[c]:>12.4f}" for c in SUMMARY_COLS))
    if gt_row is not None:
        lines.append("-" * len(lines[0]))
        lines.append(f"{'ground truth':<20}" +
                     "".join(f"{gt_row.get(c, float('nan')):>12.4f}" for c in SUMMARY_COLS))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_curves(runs, scale_mm=None, height=400, settle_epoch=20):
    """Training curves: loss, CD_t in mm, and clump %, with LR drops marked.

    The LR markers are the point of this figure. Every run before 2026-08-07 has
    none, because ReduceLROnPlateau's patience was larger than EarlyStopping's
    and it never fired -- and fixing that was worth more than any loss change.

    Y ranges are taken from epoch `settle_epoch` onwards. Autoscaling includes
    the start-up transient -- CD_t begins near 165 mm and settles around 6 -- so
    an autoscaled axis renders every run as the same flat line at the bottom and
    hides the entire result. The transient is still drawn, just clipped.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_clump = any("val_clump_metric" in r.hist for r in runs)
    titles = ["Loss", "val CD_t (mm)"] + (["val clumping <2mm (%)"] if has_clump else [])
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles)

    series = [("val_loss", 1)] + [("val_cd_t_metric", None)] + \
             ([("val_clump_metric", 100)] if has_clump else [])
    settled = {k: [] for k, _ in series}

    for n, run in enumerate(runs):
        c = C_SERIES[n % len(C_SERIES)]
        s = scale_mm or run.scale_mm
        show = True
        for col, (key, mul) in enumerate(series, start=1):
            if key not in run.hist:
                continue
            y = run.hist[key] * (s if mul is None else mul)
            settled[key].append(y[settle_epoch:])
            fig.add_trace(go.Scatter(y=y, name=run.label,
                                     line=dict(color=c, width=1.6), legendgroup=run.label,
                                     showlegend=show), row=1, col=col)
            show = False
        for ep in run.lr_drops:
            fig.add_vline(x=ep, line=dict(color=c, width=0.8, dash="dot"),
                          opacity=0.45, row=1, col=2)

    for col, (key, _) in enumerate(series, start=1):
        vals = np.concatenate([v for v in settled[key] if len(v)]) if settled[key] else None
        if vals is None or not len(vals):
            continue
        lo, hi = float(np.min(vals)), float(np.percentile(vals, 99))
        pad = max((hi - lo) * 0.08, 1e-9)
        fig.update_yaxes(range=[max(0.0, lo - pad), hi + pad], row=1, col=col)

    fig.update_xaxes(title_text="epoch")
    note = "dotted = learning-rate drop" if any(r.lr_drops for r in runs) else \
           "no learning-rate drops: ReduceLROnPlateau never fired in these runs"
    fig.update_layout(height=height,
                      title=f"Validation curves ({note}; y-axis from epoch {settle_epoch})",
                      legend=dict(orientation="h", y=-0.18), margin=dict(t=80, b=70))
    return fig


def fig_progress(df, baseline=None, height=380):
    """One bar per run per metric, normalised so 'better' always points down.

    For a progress report: shows at a glance which metrics moved and which did
    not. Bars are % change against `baseline` (defaults to the first run).
    """
    import plotly.graph_objects as go

    s = summarise(df)
    base = baseline or s.index[0]
    fig = go.Figure()
    for n, m in enumerate(SUMMARY_COLS):
        b = s.loc[base, m]
        if not np.isfinite(b) or b == 0:
            continue
        vals = (s[m] - b) / abs(b) * 100
        if m not in LOWER_IS_BETTER:      # flip so down = better everywhere
            vals = -vals
        fig.add_trace(go.Bar(x=s.index, y=vals, name=m,
                             marker_color=C_SERIES[n % len(C_SERIES)]))
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.update_layout(barmode="group", height=height,
                      title=f"Change vs {base} — negative is better (all metrics flipped)",
                      yaxis_title="% change", legend=dict(orientation="h", y=-0.22),
                      margin=dict(t=60, b=80))
    return fig


def fig_per_skull(df, metric="CD_t_mm", height=380):
    """Per-skull spread, so a mean is never read as if it were the whole story.

    Run-to-run variance was measured at 0.49 mm on this setup, which is larger
    than most of the differences being compared -- this figure is what stops a
    within-noise gap being presented as an improvement.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for n, run in enumerate(dict.fromkeys(df["run"])):
        d = df[df["run"] == run]
        fig.add_trace(go.Box(y=d[metric], name=run, boxpoints="all", jitter=0.4,
                             pointpos=0, marker=dict(color=C_SERIES[n % len(C_SERIES)], size=5),
                             line=dict(color=C_SERIES[n % len(C_SERIES)])))
    fig.update_layout(height=height, yaxis_title=metric, showlegend=False,
                      title=f"{metric} per validation skull (n={df.groupby('run').size().iloc[0]})",
                      margin=dict(t=60))
    return fig
