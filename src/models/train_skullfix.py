"""
Train the MSN PCT completion model from scratch on SkullFix.

Replaces the `AE.fit(...)` cell of notebooks/demo/MSN_model_training_Demo.ipynb.
Differences from that cell, and why:

  lr 1e-7 -> 3e-4 (configurable).  The demo's 1e-7 with Adam is three orders of
      magnitude below normal. It is a large part of why the published run needed
      322 epochs / six weeks: at that step size the weights barely move. Nothing
      about the architecture requires it.

  batch 8 -> 4.  8 OOMs at 24 GB even with the fixed distance kernels. The model
      contains no BatchNorm (`LBR` is Dense+ReLU despite the name), so a smaller
      batch only adds gradient noise; it does not break any normalisation
      statistics.

  validation_split=0.1 -> explicit split by skull id.  Keras slices off the LAST
      10% *before* shuffling. That is survivable here, but it silently becomes
      leakage the moment you generate more than one partial per skull (the demo's
      `preprocess_data` generates two), because sibling crops of the same skull
      would straddle the split. Splitting on the id makes that impossible.

  A wall-clock budget (--minutes) stops training on time regardless of epoch
      count, so a run is guaranteed to produce a usable checkpoint.

Expect the training loss to fall much faster than the validation loss: 187M
parameters against a few dozen skulls will memorise the training set. That is
the intended signal for a pilot run -- a pipeline that CANNOT overfit 40 samples
is broken. Do not quote validation numbers from this run as a result.

Added once data grew from 50 to 100 skulls (see devlog 2026-08-05):

  --early-stop-patience.  A run previously stopped purely because --minutes
      ran out, at epoch 18 -- one epoch past its best val_loss (epoch 17) --
      with nothing noticing. EarlyStopping(restore_best_weights=True) makes
      the run stop itself instead of relying on the wall clock happening to
      land in the right place.

  --n-folds / --fold.  A single 80/20 split of 100 skulls means a 20-skull
      validation set, which is noisy. k-fold CV lets you average over several
      splits instead of trusting one lucky/unlucky one. Default (--n-folds 0)
      keeps the old single random --val-frac split, unchanged.

  --run-name.  Optional subfolder under --out-dir so sweeping --minutes (to
      re-measure the convergence curve now that the data volume changed)
      doesn't overwrite the previous sweep point's history.csv/best.h5. Empty
      (default) keeps writing straight to --out-dir, so notebooks that
      hardcode experiments/msn_skullfix/best.h5 are unaffected.

  --overwrite.  Re-using a --run-name used to silently replace the previous
      run's best.h5 and history.csv while leaving its run.json (written only at
      the END of a run) untouched, so the record and the weights stopped
      describing the same training. It happened: see the cd_only incident in
      devlog 2026-08-24. Starting a run now refuses to write into a directory
      that already holds artifacts unless this flag is passed.

  --epochs 10000 -> 300 -> 600, --minutes 55 -> 90 -> 180.  Both are ceilings,
      not budgets: EarlyStopping is what stops a run, so raising them costs
      nothing unless they are reached. They are set high because a run that hits
      the ceiling was still descending and has to be discarded, which is far
      more expensive than a ceiling nobody reaches.

Added 2026-08-25, all three from the same finding (see devlog):

  --early-stop-patience 20 -> 30 -> BACK TO 20 (2026-09-06).  The 08-25 finding
      stands and is not withdrawn: across the nine valid runs, five survived a
      mid-run gap of 15-20 epochs without a new val_loss record and then improved
      again -- patience=20 was 1-5 epochs from killing them early, and which side
      of that line a run lands on is a lottery worth up to 0.15 mm (tie_qk stopped
      at 411 epochs, its repeat at 246).

      What changed is the cost. Measured on the first k-fold run: at patience=30
      the run was still setting records at epoch 599 and hit the 600 ceiling,
      which makes it unusable; the same curve under patience=20 stops at 502.
      Across the sweep that is ~28 h against ~12 h, and the schedule does not have
      28 h. Reverted to 20 for the k-fold, deliberately, with two things making it
      acceptable:
        * The 0.09 mm standard error the sweep was sized around was derived from a
          0.15 mm training variance that ALREADY INCLUDED this lottery, so 20 buys
          exactly the precision that was planned -- it does not degrade it.
        * Every run in experiments_log/ used 20, so the folds stay directly
          comparable with the single-split results rather than forming a second
          tier.
      ⚠️ NO RUN WAS EVER TRAINED AT 30. Verified three ways: the change landed in
      09aad3a on 2026-08-25 00:40, the newest run.json predates it (notext_r2,
      08-24 19:30), and not one of the 26 run.json files carries `lr_patience` or
      `defect_every` -- fields added by that same commit. So this revert orphans
      nothing and splits no tier.

  --lr-patience.  Was derived from --early-stop-patience, so raising the stop
      patience would have slowed the LR schedule too -- two changes at once, and
      a different annealing pace from every earlier run. Now explicit, pinned at
      the 10 they all used. The ordering constraint that voided the first four
      runs (LR patience must stay below early-stop patience) is now checked at
      startup instead of guaranteed by the formula.

  --defect-every.  Every decision during training watches val_loss, while the
      thesis reports defect-region coverage, and nothing recorded that metric
      per epoch -- so "did this run converge on the metric that matters?" could
      not be answered. This logs it every N epochs. DIAGNOSTIC ONLY: it must
      never drive stopping or checkpoint selection, because choosing the epoch
      by the same number the thesis reports, on the same 20 skulls, biases that
      number optimistically.
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
                    help="hard ceiling, NOT a budget: EarlyStopping is what normally stops a "
                         "run, so raising this costs nothing unless it is actually reached. It "
                         "is set high on purpose -- a run that hits the ceiling stopped while "
                         "still descending and has to be discarded (see cd_rep05_truncated), "
                         "which is far more expensive than a ceiling nobody reaches. At "
                         "~10.6s/epoch, 600 would be ~106 min if nothing ever stopped it.")
    ap.add_argument("--minutes", type=float, default=180.0,
                    help="wall-clock safety net -- sized to sit above what --epochs would take, "
                         "so in normal operation EarlyStopping (or the --epochs ceiling) binds "
                         "first and this is just a backstop against a run that never plateaus.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="linear LR ramp; without it the first few steps spike (CD ~5 -> ~40)")
    ap.add_argument("--early-stop-patience", type=int, default=20,
                    help="stop once val_loss hasn't improved for this many epochs (0 disables). "
                         "Raised from 20 on 2026-08-25. Measured across the nine valid runs, "
                         "five of them survived a mid-run gap of 15-20 epochs without a new "
                         "val_loss record and then improved again -- i.e. patience=20 was 1-5 "
                         "epochs away from killing them early, and which side of that line a "
                         "run lands on is a lottery worth up to 0.15mm (tie_qk 411 epochs vs "
                         "tie_qk_r2 246). Runs whose longest mid-run gap is under 15 simply "
                         "wait 10 more epochs before stopping, i.e. ~2 minutes.")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="ReduceLROnPlateau patience. Pinned rather than derived from "
                         "--early-stop-patience: it used to be `max(3, early_stop//2)`, so "
                         "raising the stop patience silently slowed the LR schedule too and "
                         "changed two things at once. Must stay below --early-stop-patience or "
                         "the decay never fires -- that exact conflict (40 vs 20) voided this "
                         "project's first four runs.")
    ap.add_argument("--loss", choices=["cd_dcd", "cd", "dcd"], default="cd_dcd",
                    help="cd_dcd = Chamfer + the paper's density-aware term. "
                         "'dcd' alone is the demo's loss and will NOT train from scratch "
                         "(see the warning on dcd_loss).")
    ap.add_argument("--dcd-weight", type=float, default=1.0,
                    help="scales the DCD term of --loss cd_dcd. Judge it by gradient share, "
                         "not by the loss value: DCD is already 92%% of the loss value but "
                         "only ~36%% of the gradient at 1.0, because it saturates. Measured "
                         "shares: 1->36%%, 2->53%%, 3->63%%, 5->74%%, 10->85%%. Past ~10 "
                         "Chamfer barely votes and shape accuracy is at risk.")
    ap.add_argument("--dcd-lambda", type=float, default=1.0,
                    help="exponent in DCD's density weight 1/count^lambda. BY CONSTRUCTION it "
                         "targets clumping specifically, where --dcd-weight scales DCD's distance "
                         "and density factors together -- but that is design intent: the only run "
                         "that tested it is in the voided pre-LR-fix tier, so nothing valid has "
                         "been measured here. Moot in practice, since the best configuration "
                         "drops DCD entirely in favour of --repulsion-weight.")
    ap.add_argument("--repulsion-weight", type=float, default=0.0,
                    help="0 = off (default, identical to previous behaviour). Adds a hinge "
                         "penalty on predicted points closer than --repulsion-r0 to each "
                         "other. This is the one term that can actually push points apart: "
                         "DCD's density factor comes from argmin and has zero gradient "
                         "w.r.t. position, so it cannot. The term is dimensionless (penalty "
                         "1.0 for a coincident pair, 0 once r0 apart), so its gradient norm "
                         "is ~0.51 against CD's 0.72 and these weights read normally: 0.5 is "
                         "roughly a quarter of the gradient. Watch that cd_t does not rise -- "
                         "repulsion knows nothing about the surface, so too much inflates "
                         "the cloud.")
    ap.add_argument("--repulsion-r0", type=float, default=2.0,
                    help="target minimum spacing in MILLIMETRES, converted to normalised "
                         "units with this dataset's mean scale_mm. Measured ground truth "
                         "spacing has a hard floor at 3.0mm (it is farthest-point sampled), "
                         "so 3.0 is the value that actually matches GT; 2.0 is the "
                         "conservative default because it matches the existing 'clump<2mm' "
                         "metric and disturbs far fewer points.")
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
                    help="feed the decoder's first four cross-attention blocks the encoder's "
                         "per-point features instead of only the tiled global vector. Off by "
                         "default = published behaviour, where every key row is identical and "
                         "the attention weights collapse to a uniform 1/dec_seed. The global "
                         "vector is kept as the first key row, so this only ADDS keys. Changes "
                         "no weight shape, so old checkpoints load into it silently -- the flag "
                         "goes into run.json and report.Run.arch_key reads it back.")
    ap.add_argument("--from-run", default="",
                    help="replay another run's hyper-parameters, so a repeat is a repeat. Takes a "
                         "run name (looked up in experiments_log/ first, then experiments/) or a "
                         "path to a run.json. Every field that run recorded becomes the default "
                         "here; anything you type explicitly still wins. Fields that run predates "
                         "fall back to the default that was in force when it was trained, the same "
                         "convention report.Run.arch_key uses. Use it for repeats: "
                         "--from-run tie_qk --run-name tie_qk_r2.")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow this run to write into a --run-name directory that already "
                         "holds artifacts. Off by default: the checkpoint and the CSV log are "
                         "rewritten from epoch 1 while run.json survives until the new run "
                         "ends, so a re-used name leaves a record and a set of weights that "
                         "describe different trainings (and nothing raises).")
    ap.add_argument("--defect-every", type=int, default=10,
                    help="log validation defect-region coverage (the metric the thesis reports) "
                         "every N epochs; 0 disables. DIAGNOSTIC ONLY -- it never drives "
                         "stopping or checkpoint selection, and it must not: selecting the "
                         "epoch on the same 20 skulls the number is reported on would bias that "
                         "number optimistically. It exists because the stopping rule watches "
                         "val_loss while the conclusions read defect coverage, so until now "
                         "there was no way to tell whether a run had converged on the metric "
                         "that matters. Costs one extra validation forward pass every N epochs "
                         "(~2-4%% at N=10).")
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

    Repeats are only informative if they are actually the same configuration, and
    retyping eight flags by hand is exactly where a repeat silently stops being a
    repeat -- `notext` vs `cd_rep05_full` differed in one flag out of nine, and
    that only held up because the flags were checked field by field afterwards.
    This does that check up front instead.

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
            overridden.append(f"{dest}={getattr(args, dest)!r} (记录里是 {value!r})")
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
    print(f"[--from-run] 复制自 {os.path.relpath(src, REPO_ROOT)}")
    print(f"[--from-run] 复制了 {len(replayed)} 个字段: {', '.join(sorted(replayed))}")
    if overridden:
        print(f"[--from-run] ⚠ 你手动指定的，未被覆盖: {'; '.join(overridden)}")
        print("[--from-run] ⚠ 这不再是一次严格的重复实验 —— 它和原轮相差上面这些字段。")
    if missing:
        print(f"[--from-run] 记录里没有这些字段，按「它当时生效的默认值」补: {', '.join(sorted(missing))}")
    if unknown:
        print(f"[--from-run] ⚠ run.json 里有本脚本不认识的字段，没有复制: {', '.join(sorted(unknown))}")
        print("[--from-run] ⚠ 说明这个 run 是更新的代码跑的，先确认再继续。")
    return args


def guard_out_dir(out_dir, overwrite):
    """Refuse to start a run on top of another run's artifacts.

    ModelCheckpoint and CSVLogger both start writing at epoch 1 regardless of
    what is already in the directory, while run.json is written only when the
    run finishes. Re-using a --run-name therefore replaces best.h5 and
    history.csv but leaves the previous run.json in place: the record then
    describes one training and the weights another, with nothing raising.

    Measured, on cd_only (devlog 2026-08-24): an aborted re-run left a 57-epoch
    checkpoint behind a run.json still reporting the original 305 epochs. On the
    same validation skull the archived number was 7.62 mm and the file on disk
    gave 8.29 mm. The run survived only because save_weights writes last.h5 at
    the end of a run and the aborted one never got there.

    An empty (or missing) directory is fine -- this only blocks a directory that
    already holds a run's output.
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

    WHY THIS EXISTS. Every decision during training watches `val_loss`: when to
    stop, which epoch to checkpoint, when to drop the learning rate. The thesis
    reports defect-region coverage. Until this callback there was no per-epoch
    record of that metric at all (history.csv held cd_t / cd_p / dcd / clump), so
    "had the run converged on the metric that matters?" was unanswerable -- and
    measured, every run was still descending on val_cd_t when it stopped.

    WHY IT MUST NOT DRIVE SELECTION. Choosing the checkpoint by this number would
    pick the epoch that looks best on the same 20 skulls the number is reported
    on, which biases the headline result optimistically. Watching `val_loss` -- a
    different, merely correlated quantity -- and reporting defect coverage is the
    clean arrangement. This callback only writes a column.

    THE MASK IS THE SAME ONE report._defect_metrics USES, and staying that way is
    the whole point of the column -- it is only worth logging if it is directly
    comparable with `defect_cov_mm` in eval_all_runs.csv. Since 2026-08-28 that
    means the implant ground truth the dataset ships, read from the labels file,
    NOT the old "nearest input point further than DEFECT_MM" rule (audited at
    precision 0.79 / recall 0.81 against the implant; see devlog 2026-08-27).

    ⚠️ Missing labels are a hard error, not a silent fall back to the old rule:
    a run whose logged column quietly used a different region from the one the
    thesis reports would be worse than no column at all. No run has ever recorded
    this column yet (`--defect-every` landed 2026-08-25 and nothing has trained
    since), so there is no back-compatibility to preserve.
    """
    if val_ids is None or repo is None:
        raise ValueError("make_defect_callback 需要 val_ids 与 repo 才能读缺损区真值标签")
    labels_path = os.path.join(repo, "experiments_log", "defect_mask_labels.npz")
    if not os.path.exists(labels_path):
        raise SystemExit(
            f"缺损区真值标签不存在：{labels_path}\n"
            f"先跑：python src/eval/make_defect_labels.py\n"
            f"（或用 --defect-every 0 关掉这个诊断列）")
    store = np.load(labels_path)
    missing = [s for s in val_ids if s not in store]
    if missing:
        raise SystemExit(f"缺少这些颅骨的缺损区真值标签 {missing}\n"
                         f"先跑：python src/eval/make_defect_labels.py")
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
    # Plain Adam on purpose. AdamW/dropout were added and removed: this model does
    # not overfit (val/train CD_t 1.03-1.08x on every run), so there is nothing for
    # a regulariser to fix. See the 2026-08-06 devlog entry.
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

    # ReduceLROnPlateau and EarlyStopping watch the same signal (val_loss not
    # improving), so the LR drop only ever happens if its patience is the shorter
    # of the two. This was 40 against an early-stop patience of 20, which meant it
    # never fired once in any experiment: every run trained at a flat 3e-4 from the
    # end of warmup onwards, with nothing to damp late oscillation -- rep05 went
    # from 1.25 to 2.07 train loss after epoch 104.
    #
    # It used to be DERIVED from --early-stop-patience (half of it, floored at 3),
    # which prevented that conflict but coupled two knobs: raising the stop patience
    # also slowed the LR schedule, so a run got longer for two reasons at once and
    # its annealing no longer matched the earlier runs'. Now explicit, defaulting to
    # the 10 that every run so far actually used, with the ordering enforced here
    # rather than by construction.
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
        # NOTE: the filename must end in plain ".h5", NOT ".weights.h5".
        # ".weights.h5" selects the new Keras format, which stores the optimizer
        # slots alongside the weights even under save_weights_only=True -- for
        # Adam that is 3x the parameters (187.5M -> 562M values, 2.25 GB), and it
        # gets rewritten on every val_loss improvement. Legacy ".h5" writes the
        # weights only (750 MB). The cost is that optimizer state is not kept, so
        # a run cannot be resumed bit-exactly; for inference/eval it makes no
        # difference.
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        # min_delta=0, not Keras's default 1e-4. That default is an ABSOLUTE
        # threshold, so what counts as "no improvement" depends on how big the
        # loss happens to be: cd_dcd sits near 1.0 and improves 2-4e-4 per epoch
        # late on, which clears 1e-4 -- but plain `cd` sits near 0.07, improves
        # ~1.5e-5, and would be declared stalled every single epoch, collapsing
        # the LR to min_lr and quietly wrecking any run that changes --loss.
        # Zero is the only scale-free choice, and it matches EarlyStopping, which
        # has always used 0.
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
