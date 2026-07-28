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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "experiments", "msn_skullfix"))
    ap.add_argument("--config", choices=["paper", "small"], default="paper")
    ap.add_argument("--epochs", type=int, default=10000, help="upper bound; --minutes usually binds first")
    ap.add_argument("--minutes", type=float, default=55.0, help="wall-clock budget")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="linear LR ramp; without it the first few steps spike (CD ~5 -> ~40)")
    ap.add_argument("--loss", choices=["cd_dcd", "cd", "dcd"], default="cd_dcd",
                    help="cd_dcd = Chamfer + the paper's density-aware term. "
                         "'dcd' alone is the demo's loss and will NOT train from scratch "
                         "(see the warning on dcd_loss).")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-text", action="store_true", help="drop the (constant) text branch entirely")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


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
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    import msn_skullfix as msn

    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- data ----------------
    data = np.load(args.data)
    ids, inputs, gt = data["ids"], data["inputs"], data["gt"]
    scale_mm = float(data["scale_mm"].mean())

    n_val = max(1, int(round(len(ids) * args.val_frac)))
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(ids))       # split on the skull id, never mid-skull
    val_idx, train_idx = order[:n_val], order[n_val:]

    cfg = msn.MSNConfig.paper() if args.config == "paper" else msn.MSNConfig.small()
    cfg.use_text = not args.no_text
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=msn.LOSSES[args.loss],
                  metrics=[msn.cd_t_metric, msn.cd_p_metric, msn.dcd_metric])

    print(f"\nconfig={args.config}  params={model.count_params() / 1e6:.1f}M  "
          f"in={cfg.n_in} out={cfg.n_out}")
    print(f"train={len(train_idx)} skulls  val={len(val_idx)} skulls "
          f"(ids {', '.join(ids[val_idx][:5])}{'...' if n_val > 5 else ''})")
    print(f"loss={args.loss}  lr={args.lr:g}  batch={args.batch_size}  "
          f"budget={args.minutes:g} min  scale={scale_mm:.1f} mm\n")

    budget = TimeBudget(args.minutes)
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
            os.path.join(args.out_dir, "best.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=40, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, "history.csv")),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, verbose=2, callbacks=callbacks,
    )

    model.save_weights(os.path.join(args.out_dir, "last.h5"))
    meta = {
        "config": args.config, "params": int(model.count_params()),
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "epochs_run": len(history.history["loss"]),
        "scale_mm": scale_mm,
        "train_ids": ids[train_idx].tolist(), "val_ids": ids[val_idx].tolist(),
        "final": {k: float(v[-1]) for k, v in history.history.items()},
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_cd_t_mm": float(np.min(history.history["val_cd_t_metric"]) * scale_mm),
    }
    with open(os.path.join(args.out_dir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nepochs run       {meta['epochs_run']}")
    print(f"best val_loss    {meta['best_val_loss']:.4f}")
    print(f"best val CD_t    {meta['best_val_cd_t_mm']:.2f} mm")
    print(f"train CD_t       {history.history['cd_t_metric'][-1] * scale_mm:.2f} mm  "
          f"(gap vs val = memorisation, expected at this data size)")
    print(f"artifacts        {args.out_dir}")


if __name__ == "__main__":
    main()
