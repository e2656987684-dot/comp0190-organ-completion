r"""Verify that the text branch is exactly four bias vectors, by folding it away.

The published architecture conditions the decoder on a text embedding. With a
single class that embedding is a CONSTANT -- the same 768 numbers for every
sample of every epoch -- so the branch cannot carry per-sample information. This
script proves it carries nothing at all, rather than arguing it.

Write `c` for the constant after `text_proj`, `g` for the pooled global vector,
`x` for the decoder's residual stream. Each of the four D1 cross-attention blocks
computes

    r    = x - att @ V(m1)                m1 = tile(g + c)
         = x - V(g) - V(c)                att is uniform, V is linear and unbiased
    LBR  = relu(W.r + b)
         = relu( W.(x - V(g))  -  W.V(c)  +  b )
                                \________/
                                a constant the bias can absorb

so a model WITHOUT the branch computes the same function once its bias is set to
`b' = b - (c @ W_V) @ W_LBR`. The step from `att @ V(m1)` to `V(g + c)` needs the
attention weights to be uniform, and they are structurally: every row of `m1` is
the same vector, so every key is identical and softmax returns 1/dec_seed
whatever Q and K learned.

This shows redundancy in EXPRESSIVE POWER only. Removing the branch and
retraining measurably hurts the defect region, so the difference is in
optimisation -- gradient descent does not find those bias values on its own once
the branch is gone. Why it does not remains a hypothesis.

The algebraic conclusion is a property of the architecture and cannot change, but
the norms printed below belong to the checkpoint passed in, so point `--run` at
whichever one is being reported.

    python src/eval/fold_text_branch.py [--run msn_skullfix/cd_rep05_full_f0] [--n 5]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

BLOCKS = [f"D1-STA{i}" for i in range(1, 5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="msn_skullfix/cd_rep05_full_f0",
                    help="run directory under experiments/, must be a use_text=True run")
    ap.add_argument("--n", type=int, default=5, help="validation skulls to compare on")
    args = ap.parse_args()

    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import report as rp

    run = rp.Run(REPO, args.run)
    cfg = rp.arch_config(msn, run.arch_key)
    if not cfg.use_text:
        raise SystemExit(f"{args.run} was trained without the text branch; nothing to fold")

    data = np.load(os.path.join(REPO, rp.DATA_CACHE))
    ids, inputs = data["ids"], data["inputs"]
    pos = [int(np.where(ids == s)[0][0]) for s in run.meta["val_ids"][:args.n]]
    text = np.load(os.path.join(REPO, rp.BERT_CACHE))

    # ---------------- the original, with the text branch ----------------
    model = msn.build_model(cfg)
    model.load_weights(run.weights)
    x = [inputs[pos], np.tile(text[None], (len(pos), 1))]
    ref = model.predict(x, batch_size=1, verbose=0)

    # c = relu(t @ W + b), the constant the branch injects
    w_text, b_text = model.get_layer("text_proj").get_weights()
    c = np.maximum(text @ w_text + b_text, 0.0)
    print(f"run                {run.label}  ({run.arch_label})")
    print(f"text constant c    dim {c.shape[0]}, {int((c > 0).sum())} non-zero, "
          f"||c|| = {np.linalg.norm(c):.4f}")

    folded = {}
    for name in BLOCKS:
        w_v = model.get_layer(f"{name}_V").get_weights()[0]          # (enc_out, out_dim)
        w_lbr, b_lbr = model.get_layer(f"{name}_LBR_lin").get_weights()
        delta = (c @ w_v) @ w_lbr                                    # (out_dim,)
        folded[f"{name}_LBR_lin"] = [w_lbr, b_lbr - delta]
        print(f"  {name}: ||b|| = {np.linalg.norm(b_lbr):.4f} -> folded "
              f"||b'|| = {np.linalg.norm(b_lbr - delta):.4f}   (||Δ|| = {np.linalg.norm(delta):.4f})")

    weights = {l.name: l.get_weights() for l in model.layers if l.get_weights()}
    del model
    tf.keras.backend.clear_session()

    # ---------------- the same network with the branch removed ----------------
    cfg_nt = rp.arch_config(msn, run.arch_key)
    cfg_nt.use_text = False
    plain = msn.build_model(cfg_nt)
    missing = []
    for layer in plain.layers:
        if not layer.get_weights():
            continue
        if layer.name in folded:
            layer.set_weights(folded[layer.name])
        elif layer.name in weights:
            layer.set_weights(weights[layer.name])
        else:
            missing.append(layer.name)
    if missing:
        raise SystemExit(f"no source weights for: {', '.join(missing)}")
    got = plain.predict([inputs[pos]], batch_size=1, verbose=0)

    # ---------------- what the no-text model would have had to learn ----------------
    # The folded biases are the target: a network trained WITHOUT the branch has to
    # reach them on its own, through a bias that gets no amplification. Comparing
    # them with what such a run actually converged to is the whole argument.
    nt = os.path.join(REPO, "experiments", "msn_skullfix", "notext", "best.h5")
    if os.path.exists(nt):
        import h5py
        print("\nwhat a no-text run actually learned, against the folded ||b'|| above:")
        with h5py.File(nt, "r") as f:
            for name in BLOCKS:
                key = f"{name}_LBR_lin/{name}_LBR_lin/bias:0"
                b_nt = np.array(f[key]) if key in f else None
                if b_nt is None:                       # older layouts nest differently
                    grp = f[f"{name}_LBR_lin"]
                    b_nt = np.array(grp[list(grp)[0]]["bias:0"])
                target = np.linalg.norm(folded[f"{name}_LBR_lin"][1])
                got_norm = np.linalg.norm(b_nt)
                print(f"  {name}: target {target:.4f}  reached {got_norm:.4f}  "
                      f"= {100 * got_norm / target:.0f}% of it")

    # ---------------- did it change anything? ----------------
    d = np.abs(ref - got)
    scale = float(data["scale_mm"][pos].mean())
    print(f"\ncompared on        {len(pos)} validation skulls, {ref.shape[1]} points each")
    print(f"max |Δ|            {d.max():.3e} normalised  =  {d.max() * scale:.6f} mm")
    print(f"mean |Δ|           {d.mean():.3e} normalised  =  {d.mean() * scale:.6f} mm")
    print(f"prediction scale   coordinates span {np.ptp(ref):.3f} normalised units")
    # float32 matmuls over 4096-wide layers do not reproduce bit for bit; anything
    # at 1e-5 or below is arithmetic noise, not a difference in the function.
    import report as _rp
    gt = data["gt"]
    print("\nand at the level of the reported metric:")
    print(f"  {'skull':8}{'CD_t original':>15}{'CD_t folded':>13}{'diff':>12}")
    for k, i in enumerate(pos[:3]):
        s_mm = float(data["scale_mm"][i])
        a = _rp.metrics_from_points(ref[k], gt[i], s_mm)["CD_t_mm"]
        b = _rp.metrics_from_points(got[k], gt[i], s_mm)["CD_t_mm"]
        print(f"  {ids[i]:8}{a:>12.6f}{b:>13.6f}{abs(a - b):>12.2e}")

    print("\n" + ("EQUIVALENT: the text branch's 3,149,824 parameters are four bias vectors"
                  if d.max() < 1e-4 else
                  "NOT EQUIVALENT: outputs differ -- check that the attention really is uniform"))


if __name__ == "__main__":
    main()
