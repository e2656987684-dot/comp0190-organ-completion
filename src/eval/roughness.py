r"""Archive the surface-roughness comparison that decided NOT to add a smoothness term.

WHY THIS EXISTS
  The supervisor proposed penalising surface roughness during training. A quick
  measurement on 2026-08-05 said there was nothing to gain -- normalised local
  roughness came out at 0.736 for ground truth against 0.760 for the prediction,
  i.e. the prediction is not measurably rougher than the target it is being
  fitted to, so a smoothness prior has no headroom and pushing on it would only
  make the surface flatter than ground truth and hurt Chamfer. That measurement
  redirected the whole surface-quality effort onto point DENSITY instead, which
  is where repulsion came from and is the hardest result this project has.

  But the script that produced 0.736/0.760 was never committed, so the number
  backing that decision could not be recomputed -- flagged in devlog 2026-08-07
  and left standing ever since. This is that script, written so the claim can be
  cited. Two of the project's other un-archived numbers (the sampling floor, the
  D2 attention readings) turned out to be WRONG when finally measured properly,
  so "it was probably fine" is not a safe assumption about the third.

WHAT IT MEASURES, AND WHAT THE NUMBERS ARE NOT
  For every point: fit a plane to its k nearest neighbours (itself excluded) and
  take the absolute residual along the normal. Divide the median residual by the
  median nearest-neighbour spacing so the two clouds are comparable despite
  having different densities. Ground truth and prediction go through exactly the
  same code.

  ⚠️ The absolute value is not "how rough this surface is". A skull is a shell
  5-7 mm thick and any usable neighbourhood reaches the far surface, so the
  fitted plane straddles both -- `spread_mm` in the output is that thickness,
  measured. Enlarging k does not escape it, it swaps shell thickness for the
  skull's own curvature. The k sweep below archives that argument (previously
  also only in devlog, also from a lost script): watch `spread_mm` climb past
  the shell thickness instead of levelling off at it.

  ⚠️ What survives is the ground-truth-vs-prediction DIFFERENCE, because the bias
  is largely common-mode. Not entirely: the two clouds differ in density, so
  they are not contaminated to quite the same degree. Report the comparison,
  never the absolute number.

⚠️ k 折之后：只在论文引用这个比较时才需要重跑（`--runs <最终模型>`）。
  Ground-truth roughness depends on the data alone; the prediction side depends
  on which checkpoint is quoted.

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

DEFAULT_RUNS = ["msn_skullfix/cd_rep05_full"]


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
    text_path = os.path.join(repo, "data", "cache", "bert_skull.npy")
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
                # ⚠️ NOT the same 8 skulls as surface_quality.csv. That table uses
                # "the first 8 validation skulls in `ids` order"; this uses the
                # first 8 in `val_ids` order, as do normal_quality.py and
                # point_to_surface.py. The two sets share exactly ONE skull, so
                # numbers here must never be put beside surface_quality's without
                # recomputing one of them on the other's cohort (2026-08-27).
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
    print(f"\n{'=' * 78}\n{run.label}   ({run.arch_label})   {n} 颗验证颅骨\n{'=' * 78}")
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
    print(f"\n  k=16（原始论断的口径）：GT {k0['rough_norm_gt'].mean():.3f} vs "
          f"pred {k0['rough_norm'].mean():.3f}，Δ {d.mean():+.3f} "
          f"（{worse}/{len(d)} 颗颅骨预测更粗糙）")
    print(f"  参照：devlog 2026-08-05 那个丢失的脚本记的是 GT 0.736 / pred 0.760、Δ +0.024")

    # The whole reason the metric was abandoned. Spread that keeps climbing means
    # the neighbourhood never settles onto one surface -- it just trades one
    # contaminant for another.
    sw = df.groupby("k")["spread_mm_gt"].mean()
    print(f"\n  GT 邻域沿法向的展开：" +
          " → ".join(f"k={k} {v:.2f}mm" for k, v in sw.items()))
    print(f"  骨壳厚度约 5~7mm。若污染只来自厚度，这一列应当在 6mm 附近**走平**；"
          f"实测 k=128 到 {sw.iloc[-1]:.2f}mm，" +
          ("**仍在爬** → 大 k 处污染换成了曲率，**没有哪个 k 是干净的**"
           if sw.iloc[-1] > sw.loc[24] * 1.1 else "⚠️ 走平了 —— 与 devlog 的结论不符，要重看"))


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
        old = pd.read_csv(out)
        # ⚠️ 键要先统一成字符串再比：`id` 是 '083' 这种带前导零的编号，
        # 写进 CSV 再读回来会被 pandas 解析成整数 83，于是 ('run', 83) 对不上
        # ('run', '083')，旧行被当成不同的行留下来 —— 实测重跑一次就变成 16 行。
        KEY = ['run', 'id', 'k']
        k_old = old[KEY].astype(str).apply(tuple, axis=1)
        k_new = set(df[KEY].astype(str).apply(tuple, axis=1))
        df = pd.concat([old[~k_old.isin(k_new)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
