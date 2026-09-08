r"""Surface roughness, prediction against ground truth, at several neighbourhood sizes.

This is the measurement behind the decision NOT to add a smoothness term to the
loss: the prediction is not measurably rougher than the target it is fitted to,
so a smoothness prior has no headroom, and pushing on it would only make the
surface flatter than ground truth and cost Chamfer accuracy. The effort went to
point density instead. An earlier version of this measurement was never
committed, and it reported GT 0.736 against prediction 0.760 -- numbers this
script exists to replace with ones that can be recomputed.

For every point: fit a plane to its k nearest neighbours, itself excluded, and
take the absolute residual along the normal. Divide the median residual by the
median nearest-neighbour spacing, so the two clouds compare despite differing in
density. Ground truth and prediction go through identical code.

Warning: the absolute value is NOT "how rough this surface is". A skull is a
shell 5-7 mm thick and any usable neighbourhood reaches the far surface, so the
fitted plane straddles both -- `spread_mm` in the output is that thickness,
measured. Enlarging k does not escape it, it trades shell thickness for the
skull's own curvature, which is what the k sweep shows: `spread_mm` climbs past
the shell thickness instead of levelling off at it.

Warning: what survives the contamination is the ground-truth-vs-prediction
DIFFERENCE, since the bias is largely common-mode -- though not entirely, as the
two clouds differ in density. Report the comparison, never the absolute number.

Ground-truth roughness depends on the data alone; the prediction side depends on
which checkpoint is quoted.

    python src/eval/roughness.py [--runs A B ...] [--n 8]
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
import pandas as pd

OUT_CSV = os.path.join("experiments_log", "roughness.csv")

# k=16 is where the original claim was made. The rest are the sweep that shows
# no neighbourhood size is clean -- small k contaminated by shell thickness,
# large k by curvature.
KS = (8, 16, 24, 48, 128)

DEFAULT_RUNS = ["msn_skullfix/cd_rep05_full_f0"]


def analyse(repo, specs, n_skulls=8, device="/GPU:0"):
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import mesh_viz as mv
    import report as rp

    runs = rp.load_runs(repo, specs)
    data = np.load(os.path.join(repo, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    text_path = os.path.join(repo, rp.BERT_CACHE)
    text = np.load(text_path) if os.path.exists(text_path) else None

    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)

    rows = []
    with tf.device(device):
        for arch, group in groups.items():
            cfg = rp.arch_config(msn, arch)
            if cfg.use_text and text is None:
                raise FileNotFoundError(f"{group[0].label} needs {text_path}")
            model = msn.build_model(cfg)
            for run in group:
                model.load_weights(run.weights)
                # NOT the same 8 skulls as surface_quality.csv, which takes the
                # first 8 in `ids` order while this takes the first 8 in `val_ids`
                # order, as do normal_quality.py and point_to_surface.py. The two
                # sets share exactly ONE skull, so numbers from here must never be
                # placed beside surface_quality's.
                val = run.meta["val_ids"][:n_skulls]
                pos = [int(np.where(ids == sid)[0][0]) for sid in val]
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                preds = model.predict(x, batch_size=1, verbose=0)

                for sid, i, pred in zip(val, pos, preds):
                    s = float(scales[i])
                    sp_p = np.median(mv.local_spacing(pred, s))
                    sp_g = np.median(mv.local_spacing(gt[i], s))
                    for k in KS:
                        rp_mm, sp_spread = mv.local_roughness(pred, s, k=k)
                        rg_mm, sg_spread = mv.local_roughness(gt[i], s, k=k)
                        rows.append({
                            "run": run.label, "id": sid, "k": k,
                            "rough_mm": float(np.median(rp_mm)),
                            "rough_mm_gt": float(np.median(rg_mm)),
                            "rough_norm": float(np.median(rp_mm) / sp_p),
                            "rough_norm_gt": float(np.median(rg_mm) / sp_g),
                            "spread_mm": float(np.median(sp_spread)),
                            "spread_mm_gt": float(np.median(sg_spread)),
                            "spacing_mm": float(sp_p), "spacing_mm_gt": float(sp_g),
                        })
                report(run, [r for r in rows if r["run"] == run.label])
            del model
            tf.keras.backend.clear_session()
    return pd.DataFrame(rows)


def report(run, rows):
    df = pd.DataFrame(rows)
    n = df["id"].nunique()
    print(f"\n{'=' * 78}\n{run.label}   ({run.arch_label})   {n} validation skulls\n{'=' * 78}")
    head = (f"{'k':>5}{'rough_norm GT':>15}{'pred':>9}{'Δ':>9}{'  |':>4}"
            f"{'rough_mm GT':>13}{'pred':>8}{'  |':>4}{'spread_mm GT':>14}{'pred':>8}")
    print(head)
    print("-" * len(head))
    for k, g in df.groupby("k"):
        rn_g, rn_p = g["rough_norm_gt"].mean(), g["rough_norm"].mean()
        print(f"{k:>5}{rn_g:>15.3f}{rn_p:>9.3f}{rn_p - rn_g:>+9.3f}{'  |':>4}"
              f"{g['rough_mm_gt'].mean():>13.2f}{g['rough_mm'].mean():>8.2f}{'  |':>4}"
              f"{g['spread_mm_gt'].mean():>14.2f}{g['spread_mm'].mean():>8.2f}")

    k0 = df[df["k"] == 16]
    d = (k0["rough_norm"] - k0["rough_norm_gt"])
    worse = int((d > 0).sum())
    print(f"\n  k=16, the size the original claim was made at: GT "
          f"{k0['rough_norm_gt'].mean():.3f} vs pred {k0['rough_norm'].mean():.3f}, "
          f"delta {d.mean():+.3f} ({worse}/{len(d)} skulls rougher in the prediction)")
    print(f"  for reference, the superseded measurement was GT 0.736 / pred 0.760, delta +0.024")

    # The whole reason the metric was abandoned. Spread that keeps climbing means
    # the neighbourhood never settles onto one surface -- it just trades one
    # contaminant for another.
    sw = df.groupby("k")["spread_mm_gt"].mean()
    print(f"\n  ground-truth neighbourhood spread along the normal: " +
          " -> ".join(f"k={k} {v:.2f}mm" for k, v in sw.items()))
    print(f"  The shell is 5-7 mm thick. If thickness were the only contaminant this "
          f"row would level off near 6 mm; it reaches {sw.iloc[-1]:.2f}mm at k=128, " +
          ("still climbing -- at large k the contaminant becomes curvature, so NO "
           "neighbourhood size is clean"
           if sw.iloc[-1] > sw.loc[24] * 1.1 else "and has levelled off, which "
           "contradicts the archived conclusion and needs looking at"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--n", type=int, default=8, help="validation skulls (surface_quality uses 8)")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    df = analyse(REPO, args.runs, n_skulls=args.n)

    out = os.path.join(REPO, args.out)
    if os.path.exists(out):
        # dtype={"id": str} is required, and astype(str) does not substitute for it:
        # ids carry a leading zero ('083'), pandas reads them back as the integer 83,
        # and str(83) is '83'. The key then fails to match and the old rows survive
        # as duplicates -- silently, because their other columns are identical.
        old = pd.read_csv(out, dtype={"id": str})
        KEY = ['run', 'id', 'k']
        k_old = old[KEY].astype(str).apply(tuple, axis=1)
        k_new = set(df[KEY].astype(str).apply(tuple, axis=1))
        df = pd.concat([old[~k_old.isin(k_new)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
