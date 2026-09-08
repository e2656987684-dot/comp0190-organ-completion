"""Train the MSN PCT completion model from scratch on SkullFix.

Replaces the `AE.fit(...)` cell of
notebooks/upstream_msn/MSN_model_training_Demo.ipynb. What this rewrite changed
against that cell, and why, is in src/models/README.md; the differences that
concern data preparation are in notebooks/README.md.

    python src/models/train_skullfix.py --run-name cd_rep05_full_f0 \
        --n-folds 5 --fold 0 --loss cd --repulsion-weight 0.5

For a k-fold sweep use src/models/run_kfold.py, which adds the guards a bare
loop does not have.

Three things that are easy to get wrong:

  * --epochs and --minutes are CEILINGS, not budgets; a run that reaches one has
    to be discarded.
  * --defect-every is DIAGNOSTIC ONLY and must never drive selection.
  * --from-run replays another run's flags; retyping them by hand is how a
    repeat stops being one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Kept equal to eval.mesh_viz.CLUMP_MM so the per-epoch clump_metric and the
# after-the-fact surface-quality numbers mean the same thing.
CLUMP_MM = 2.0

# ⚠️ NO LONGER DEFINES THE DEFECT REGION (2026-08-28). That is now the implant
# ground truth the dataset ships, read from experiments_log/defect_mask_labels.npz
# by `make_defect_callback` below, so the per-epoch diagnostic stays directly
# comparable with the defect_cov_mm column of eval_all_runs.csv -- which is the
# only reason that column is worth logging. Kept here because it is still
# `eval.report.DEFECT_MM`'s value and the two must not drift apart.
DEFECT_MM = 5.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "experiments", "msn_skullfix"))
    ap.add_argument("--run-name", default="",
                    help="optional subfolder under --out-dir (e.g. 'minutes10'). Empty (default) "
                         "writes straight into --out-dir, unchanged from before -- set this when "
                         "sweeping so successive runs don't overwrite each other's history.csv/best.h5.")
    ap.add_argument("--config", choices=["paper", "small"], default="paper")
    ap.add_argument("--epochs", type=int, default=600,
                    help="hard ceiling, NOT a budget: EarlyStopping is what normally stops a run, so "
                         "raising this costs nothing unless it is reached -- and a run that "
                         "does reach it was still descending and has to be discarded.")
    ap.add_argument("--minutes", type=float, default=180.0,
                    help="wall-clock safety net, sized to sit above what --epochs would take, so it "
                         "only binds if a run never plateaus.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="linear LR ramp; without it the first few steps spike (CD ~5 -> ~40)")
    ap.add_argument("--early-stop-patience", type=int, default=20,
                    help="stop once val_loss hasn't improved for this many epochs (0 disables). 20 is "
                         "the affordable value rather than the safe one; see src/models/README.md.")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="ReduceLROnPlateau patience. Must stay below --early-stop-patience or the "
                         "decay never fires -- that exact conflict voided this project's first "
                         "four runs, so it is checked at startup.")
    ap.add_argument("--loss", choices=["cd_dcd", "cd", "dcd"], default="cd_dcd",
                    help="cd_dcd = Chamfer + the paper's density-aware term. "
                         "'dcd' alone is the demo's loss and will NOT train from scratch "
                         "(see the warning on dcd_loss).")
    ap.add_argument("--dcd-weight", type=float, default=1.0,
                    help="scales the DCD term of --loss cd_dcd. Judge it by gradient share, not by the "
                         "loss value: DCD saturates, so at weight 1 it is 92%% of the loss but "
                         "only ~36%% of the gradient. Measured shares in src/models/README.md.")
    ap.add_argument("--dcd-lambda", type=float, default=1.0,
                    help="exponent in DCD's density weight 1/count^lambda. Targets clumping BY "
                         "CONSTRUCTION, but nothing valid has ever been measured here, and the "
                         "adopted configuration drops DCD entirely. See src/models/README.md.")
    ap.add_argument("--repulsion-weight", type=float, default=0.0,
                    help="0 = off (default). Hinge penalty on predicted points closer than "
                         "--repulsion-r0 to each other -- the only term that can actually push "
                         "points apart. Dimensionless, so 0.5 is about a quarter of the "
                         "gradient. Watch that cd_t does not rise. See src/models/README.md.")
    ap.add_argument("--repulsion-r0", type=float, default=2.0,
                    help="target minimum spacing in MILLIMETRES, converted with this dataset's mean "
                         "scale_mm. 2.0 matches the clump<2mm metric; 3.0 matches ground-truth "
                         "spacing. See src/models/README.md.")
    ap.add_argument("--repulsion-k", type=int, default=4,
                    help="how many nearest neighbours each point is repelled from. k=1 "
                         "tracks the clump metric most directly but oscillates as the "
                         "nearest neighbour keeps changing; 4 is steadier.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="used only when --n-folds is 0 (single random split)")
    ap.add_argument("--n-folds", type=int, default=0,
                    help="0 = single random --val-frac split (default). >0 = k-fold CV; pick the "
                         "held-out fold with --fold. At 100 skulls a single 80/20 split is noisy; "
                         "k-fold gives a less lucky/unlucky read on val performance.")
    ap.add_argument("--fold", type=int, default=0,
                    help="which fold (0-indexed) is validation, when --n-folds > 0")
    ap.add_argument("--no-text", action="store_true", help="drop the (constant) text branch entirely")
    ap.add_argument("--tie-qk-init", action="store_true",
                    help="restore the published demo's one-line Q/K weight tie in the encoder "
                         "self-attention (W_k.set_weights(W_q.get_weights())). This rewrite has "
                         "omitted it since 2026-07-28; it raises the initial attention-score "
                         "scale ~10x. Topology-affecting, so it goes into run.json.")
    ap.add_argument("--per-point-attn", action="store_true",
                    help="feed the decoder's first four cross-attention blocks the encoder's per-point "
                         "features instead of only the tiled global vector. Off by default = "
                         "published behaviour. Changes no weight shape, so it is recorded in "
                         "run.json and read back by report.Run.arch_key.")
    ap.add_argument("--from-run", default="",
                    help="replay another run's hyper-parameters, so a repeat is a repeat. Takes a run "
                         "name (experiments_log/ first, then experiments/) or a path to a "
                         "run.json; anything typed explicitly still wins.")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow this run to write into a --run-name directory that already "
                         "holds artifacts. Off by default: the checkpoint and the CSV log are "
                         "rewritten from epoch 1 while run.json survives until the new run "
                         "ends, so a re-used name leaves a record and a set of weights that "
                         "describe different trainings (and nothing raises).")
    ap.add_argument("--defect-every", type=int, default=10,
                    help="log validation defect-region coverage (the metric the thesis reports) every "
                         "N epochs; 0 disables. DIAGNOSTIC ONLY -- it must never drive stopping "
                         "or checkpoint selection. See make_defect_callback.")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# run.json field -> (argparse dest, default in force before that field existed).
# The second element matters: a run trained before a flag existed recorded nothing
# for it, and replaying it must reproduce the behaviour it actually had, not
# today's default. Same convention as report.Run.arch_key.
_REPLAY = {
    "config": ("config", "paper"),
    "epochs": ("epochs", 300),          # the ceiling in force before it was recorded
    "minutes": ("minutes", 90.0),
    "lr": ("lr", 3e-4),
    "batch_size": ("batch_size", 4),
    "seed": ("seed", 42),
    "n_folds": ("n_folds", 0),
    "fold": ("fold", 0),
    "early_stop_patience": ("early_stop_patience", 20),
    # Both added 2026-08-25. The fallbacks are what the earlier runs effectively
    # used: lr_patience was derived as max(3, 20//2) = 10, and there was no
    # per-epoch defect logging at all.
    "lr_patience": ("lr_patience", 10),
    "defect_every": ("defect_every", 0),
    "loss": ("loss", "cd_dcd"),
    "dcd_weight": ("dcd_weight", 1.0),
    "dcd_lambda": ("dcd_lambda", 1.0),
    "repulsion_weight": ("repulsion_weight", 0.0),
    "repulsion_r0_mm": ("repulsion_r0", 2.0),
    "repulsion_k": ("repulsion_k", 4),
    "per_point_attn": ("per_point_attn", False),
    "tie_qk_init": ("tie_qk_init", False),
}
# Recorded fields that are outputs, not settings -- never replayed.
_REPLAY_IGNORE = {"params", "epochs_run", "scale_mm", "final", "best_val_loss",
                  "best_val_cd_t_mm", "train_ids", "val_ids", "clump_thresh_mm",
                  "use_text", "dropout", "weight_decay"}


def apply_from_run(args, argv):
    """Fill `args` from another run's run.json, leaving anything typed explicitly alone.

    `--data`, `--out-dir`, `--run-name`, `--overwrite`, `--mixed-precision` and
    `--warmup-steps` are deliberately NOT replayed: they say where a run writes
    and how it is driven, not what is being trained.
    """
    if not args.from_run:
        return args

    for cand in (os.path.join(REPO_ROOT, "experiments_log", args.from_run, "run.json"),
                 os.path.join(REPO_ROOT, "experiments", args.from_run, "run.json"),
                 os.path.join(REPO_ROOT, "experiments", "msn_skullfix", args.from_run, "run.json"),
                 args.from_run):
        if os.path.isfile(cand):
            src = cand
            break
    else:
        raise SystemExit(f"--from-run {args.from_run!r}: no run.json found "
                         f"(looked in experiments_log/, experiments/, experiments/msn_skullfix/)")

    with open(src) as fh:
        meta = json.load(fh)

    typed = {a.split("=")[0] for a in argv if a.startswith("--")}
    replayed, missing, overridden = [], [], []

    for key, (dest, fallback) in _REPLAY.items():
        flag = "--" + dest.replace("_", "-")
        value = meta.get(key, fallback)
        if key not in meta:
            missing.append(key)
        if value is None:                      # e.g. "fold": null when n_folds == 0
            value = fallback
        if flag in typed:
            overridden.append(f"{dest}={getattr(args, dest)!r} (the record says {value!r})")
            continue
        setattr(args, dest, value)
        replayed.append(f"{dest}={value!r}")

    # use_text is stored positively but driven by --no-text, so it inverts.
    if "--no-text" not in typed:
        args.no_text = not bool(meta.get("use_text", True))
        replayed.append(f"no_text={args.no_text!r}")
    else:
        overridden.append(f"no_text={args.no_text!r}")

    unknown = set(meta) - set(_REPLAY) - _REPLAY_IGNORE
    print(f"[--from-run] copied from {os.path.relpath(src, REPO_ROOT)}")
    print(f"[--from-run] replayed {len(replayed)} fields: {', '.join(sorted(replayed))}")
    if overridden:
        print(f"[--from-run] Warning: set by hand, not replayed: {'; '.join(overridden)}")
        print("[--from-run] Warning: this is no longer a strict repeat -- it differs from "
              "the original run in the fields above.")
    if missing:
        print(f"[--from-run] not in the record, filled with the default that was in "
              f"force at the time: {', '.join(sorted(missing))}")
    if unknown:
        print(f"[--from-run] Warning: run.json holds fields this script does not know, "
              f"so they were not copied: {', '.join(sorted(unknown))}")
        print("[--from-run] Warning: that run came from newer code -- check before continuing.")
    return args


def guard_out_dir(out_dir, overwrite):
    """Refuse to start a run on top of another run's artifacts.

    Re-using a --run-name used to leave the record describing one training and
    the weights another, without raising. An empty or missing directory is fine;
    this only blocks one that already holds output. See src/models/README.md.
    """
    names = ("run.json", "best.h5", "last.h5", "history.csv")
    existing = [n for n in names if os.path.exists(os.path.join(out_dir, n))]
    if not existing or overwrite:
        return

    meta_path = os.path.join(out_dir, "run.json")
    detail = ""
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            m = json.load(fh)
        detail = (f"\n  it holds: {m.get('epochs_run', '?')} epochs, "
                  f"loss={m.get('loss', '?')}, best val_loss {m.get('best_val_loss', float('nan')):.5f}")
    else:
        detail = "\n  it holds no run.json, so it is most likely an ABORTED run"

    raise SystemExit(
        f"\n{out_dir}\nalready contains {', '.join(existing)} -- refusing to overwrite it."
        f"{detail}\n\n"
        "  Pick a different --run-name, or move that directory aside, or pass --overwrite\n"
        "  if you really mean to discard it. Training is not bit-reproducible on GPU, so a\n"
        "  checkpoint replaced here cannot be recreated.\n")


def _nn_dist(query, ref, chunk=1024):
    """Nearest-neighbour distance from each `query` point to `ref`, chunked numpy.

    Brute force on purpose: the arrays here are small (at most 6144 x 4096, and
    only ~400 query points once the defect mask is applied), and this keeps the
    training script free of a KD-tree dependency it otherwise does not need.
    """
    ref2 = (ref ** 2).sum(1)
    out = np.empty(len(query), dtype=np.float64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] - 2.0 * (q @ ref.T) + ref2[None, :]
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def make_defect_callback(keras, x_val, gt_val, inputs_val, scales_val, every,
                         val_ids=None, repo=None):
    """Log validation defect-region coverage every `every` epochs. DIAGNOSTIC ONLY.

    It must never drive stopping or checkpoint selection -- that would bias the
    reported result. Uses the same implant labels as `report._defect_metrics`, so
    the column is comparable with `defect_cov_mm`; missing labels are a hard
    error, never a silent fall back. Reasons in src/models/README.md.
    """
    if val_ids is None or repo is None:
        raise ValueError("make_defect_callback needs val_ids and repo to read the "
                         "defect ground-truth labels")
    labels_path = os.path.join(repo, "experiments_log", "defect_mask_labels.npz")
    if not os.path.exists(labels_path):
        raise SystemExit(
            f"no defect ground-truth labels at {labels_path}\n"
            f"run first: python src/eval/make_defect_labels.py\n"
            f"(or pass --defect-every 0 to switch this diagnostic column off)")
    store = np.load(labels_path)
    missing = [s for s in val_ids if s not in store]
    if missing:
        raise SystemExit(f"no defect ground-truth labels for these skulls: {missing}\n"
                         f"run first: python src/eval/make_defect_labels.py")
    masks = [store[sid] for sid in val_ids]

    class _Defect(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            # The key has to exist on the very first epoch or CSVLogger, which
            # fixes its field names then, drops the column for the whole run.
            due = epoch == 0 or (epoch + 1) % every == 0
            if not due:
                logs["val_defect_cov_mm"] = float("nan")
                return
            preds = self.model.predict(x_val, batch_size=1, verbose=0)
            vals = [float(_nn_dist(g[m].astype(np.float64), p.astype(np.float64)).mean() * s)
                    for p, g, m, s in zip(preds, gt_val, masks, scales_val)]
            logs["val_defect_cov_mm"] = float(np.mean(vals))

    return _Defect()


class TimeBudget:
    """Stop cleanly when the wall-clock budget is spent."""

    def __init__(self, minutes):
        self.limit = minutes * 60.0
        self.start = None

    def make_callback(self, keras):
        outer = self

        class _CB(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                outer.start = time.time()

            def on_epoch_end(self, epoch, logs=None):
                elapsed = time.time() - outer.start
                if elapsed >= outer.limit:
                    print(f"\n[budget] {elapsed / 60:.1f} min reached at epoch {epoch + 1}; stopping.")
                    self.model.stop_training = True

        return _CB()


def make_warmup(keras, target_lr, steps):
    """Linear LR ramp over the first `steps` batches.

    Without it the loss spikes hard on the first few updates (CD ~5 at init,
    peaking near 40 at lr 3e-4) before recovering. Kept separate from
    ReduceLROnPlateau, which needs a plain mutable LR variable to write to.
    """

    class _Warmup(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.step = 0

        def on_train_batch_begin(self, batch, logs=None):
            if self.step < steps:
                self.step += 1
                self.model.optimizer.learning_rate.assign(target_lr * self.step / steps)

    return _Warmup()


def main():
    args = apply_from_run(parse_args(), sys.argv[1:])
    # Before TensorFlow, so a re-used --run-name fails in a second rather than
    # after ten seconds of CUDA start-up.
    out_dir = os.path.join(args.out_dir, args.run_name) if args.run_name else args.out_dir
    guard_out_dir(out_dir, args.overwrite)

    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    import msn_skullfix as msn

    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- data ----------------
    data = np.load(args.data)
    ids, inputs, gt = data["ids"], data["inputs"], data["gt"]
    scale_mm = float(data["scale_mm"].mean())

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(ids))       # split on the skull id, never mid-skull
    if args.n_folds > 0:
        if not (0 <= args.fold < args.n_folds):
            raise SystemExit(f"--fold must be in [0, {args.n_folds}) for --n-folds {args.n_folds}")
        folds = np.array_split(order, args.n_folds)
        val_idx = folds[args.fold]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != args.fold])
    else:
        n_val = max(1, int(round(len(ids) * args.val_frac)))
        val_idx, train_idx = order[:n_val], order[n_val:]
    n_val = len(val_idx)

    cfg = msn.MSNConfig.paper() if args.config == "paper" else msn.MSNConfig.small()
    cfg.use_text = not args.no_text
    cfg.per_point_attn = args.per_point_attn
    cfg.tie_qk_init = args.tie_qk_init
    if inputs.shape[1] != cfg.n_in or gt.shape[1] != cfg.n_out:
        raise SystemExit(
            f"cache is {inputs.shape[1]} in / {gt.shape[1]} gt but config '{args.config}' "
            f"wants {cfg.n_in} / {cfg.n_out}. Regenerate with:\n"
            f"  python src/data/prepare_skullfix.py --n-in {cfg.n_in} --n-out {cfg.n_out} "
            f"--n-dense {max(16384, cfg.n_out * 2)} --out {args.data}"
        )

    def pack(idx):
        x = [inputs[idx]]
        if cfg.use_text:
            x.append(np.tile(text_feat[None], (len(idx), 1)))
        return x, gt[idx]

    text_feat = None
    if cfg.use_text:
        cache = os.path.join(REPO_ROOT, "data", "cache", "bert_skull.npy")
        if os.path.exists(cache):
            text_feat = np.load(cache)
        else:
            print("encoding class name with BERT (once)...")
            text_feat = msn.encode_class_name("skull")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, text_feat)
        print(f"text feature {text_feat.shape} (constant: frozen BERT, single class)")

    x_train, y_train = pack(train_idx)
    x_val, y_val = pack(val_idx)

    # ---------------- model ----------------
    model = msn.build_model(cfg)
    # Plain Adam on purpose. AdamW and dropout were both added and removed: this
    # model does not overfit (val/train CD_t 1.03-1.08x on every run), so there is
    # nothing for a regulariser to fix. See src/models/README.md.
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    # The losses run in normalised coordinates, so the mm thresholds are divided
    # by this dataset's mean radius. Per-skull radii span 88.3-133.1 mm, so the
    # effective threshold varies ~+-20% across samples; that is noise, not bias.
    clump_thresh_norm = CLUMP_MM / scale_mm
    repulsion_r0_norm = args.repulsion_r0 / scale_mm
    clump_metric = msn.make_clump_metric(clump_thresh_norm)

    model.compile(optimizer=optimizer,
                  loss=msn.make_loss(args.loss, dcd_weight=args.dcd_weight,
                                     n_lambda=args.dcd_lambda,
                                     repulsion_weight=args.repulsion_weight,
                                     repulsion_r0=repulsion_r0_norm,
                                     repulsion_k=args.repulsion_k),
                  metrics=[msn.cd_t_metric, msn.cd_p_metric, msn.dcd_metric, clump_metric])

    print(f"\nconfig={args.config}  params={model.count_params() / 1e6:.1f}M  "
          f"in={cfg.n_in} out={cfg.n_out}")
    print(f"train={len(train_idx)} skulls  val={len(val_idx)} skulls "
          f"(ids {', '.join(ids[val_idx][:5])}{'...' if n_val > 5 else ''})")
    if args.loss == "cd_dcd":
        print(f"dcd_weight={args.dcd_weight:g}  dcd_lambda={args.dcd_lambda:g}")
    if args.repulsion_weight > 0:
        print(f"repulsion w={args.repulsion_weight:g}  r0={args.repulsion_r0:g}mm "
              f"(={repulsion_r0_norm:.5f} norm)  k={args.repulsion_k}")
    else:
        print("repulsion off")
    print(f"loss={args.loss}  lr={args.lr:g}  batch={args.batch_size}  "
          f"budget={args.minutes:g} min  scale={scale_mm:.1f} mm")
    if args.n_folds > 0:
        print(f"cv: fold {args.fold}/{args.n_folds}")
    print(f"artifacts -> {out_dir}\n")

    budget = TimeBudget(args.minutes)

    # ReduceLROnPlateau and EarlyStopping watch the same signal, so the LR drop
    # only ever fires if its patience is the shorter of the two. Enforced here
    # rather than derived from --early-stop-patience, which used to couple the two
    # knobs. Why that matters: src/models/README.md.
    lr_patience = args.lr_patience
    if args.early_stop_patience > 0 and lr_patience >= args.early_stop_patience:
        raise SystemExit(
            f"\n--lr-patience ({lr_patience}) must be smaller than --early-stop-patience "
            f"({args.early_stop_patience}).\nOtherwise early stopping fires first and the "
            f"learning-rate decay never runs -- that exact conflict (40 vs 20) is what voided\n"
            f"this project's first four runs. See experiments_log/README.md.\n")

    callbacks = [
        budget.make_callback(tf.keras),
        make_warmup(tf.keras, args.lr, args.warmup_steps),
        # The filename must end in plain ".h5", NOT ".weights.h5" -- the latter
        # selects the newer Keras format, which also stores Adam's slots and turns
        # 750 MB into 2.25 GB on every improvement. See src/models/README.md.
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        # min_delta=0, not Keras's default 1e-4, which is an ABSOLUTE threshold
        # and so depends on how large the loss happens to be -- it silently
        # wrecks any run that changes --loss. See src/models/README.md.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6,
            min_delta=0.0, verbose=1),
    ]
    if args.defect_every > 0:
        # BEFORE CSVLogger: callbacks run in list order, and this one writes its
        # value into `logs` for CSVLogger to pick up.
        callbacks.append(make_defect_callback(
            tf.keras, x_val, gt[val_idx], inputs[val_idx],
            [float(v) for v in data["scale_mm"][val_idx]], args.defect_every,
            val_ids=[str(v) for v in data["ids"][val_idx]], repo=REPO_ROOT))
    callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")))
    if args.early_stop_patience > 0:
        # A run previously stopped purely because --minutes ran out, one epoch
        # past its best val_loss, with nothing noticing (ModelCheckpoint still
        # saved the right weights, but the run itself kept going past its peak).
        # This makes the run stop itself instead of relying on the wall clock.
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stop_patience,
            restore_best_weights=True, verbose=1))

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, verbose=2, callbacks=callbacks,
    )

    model.save_weights(os.path.join(out_dir, "last.h5"))
    meta = {
        "config": args.config, "params": int(model.count_params()),
        # Topology switches. These have to be recorded, not inferred: they change
        # what the network IS, and report.Run.arch_key reads them to decide which
        # checkpoints may share a model. A run.json without them is read as the
        # defaults that were in force before the field existed.
        "use_text": bool(cfg.use_text),
        "per_point_attn": bool(cfg.per_point_attn),
        "tie_qk_init": bool(cfg.tie_qk_init),
        # Both stopping ceilings, so "did this run stop early or hit a wall?" is
        # answerable from the record instead of by arithmetic on history.csv.
        "epochs": args.epochs, "minutes": args.minutes,
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "n_folds": args.n_folds, "fold": args.fold if args.n_folds > 0 else None,
        "early_stop_patience": args.early_stop_patience,
        "lr_patience": args.lr_patience,
        "defect_every": args.defect_every,
        "loss": args.loss, "dcd_weight": args.dcd_weight, "dcd_lambda": args.dcd_lambda,
        "repulsion_weight": args.repulsion_weight,
        "repulsion_r0_mm": args.repulsion_r0, "repulsion_k": args.repulsion_k,
        "clump_thresh_mm": CLUMP_MM,
        "epochs_run": len(history.history["loss"]),
        "scale_mm": scale_mm,
        "train_ids": ids[train_idx].tolist(), "val_ids": ids[val_idx].tolist(),
        "final": {k: float(v[-1]) for k, v in history.history.items()},
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_cd_t_mm": float(np.min(history.history["val_cd_t_metric"]) * scale_mm),
    }
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nepochs run       {meta['epochs_run']}")
    print(f"best val_loss    {meta['best_val_loss']:.4f}")
    print(f"best val CD_t    {meta['best_val_cd_t_mm']:.2f} mm")
    print(f"train CD_t       {history.history['cd_t_metric'][-1] * scale_mm:.2f} mm  "
          f"(gap vs val = memorisation, expected at this data size)")
    print(f"artifacts        {out_dir}")


if __name__ == "__main__":
    main()
