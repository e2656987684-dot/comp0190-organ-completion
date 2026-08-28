r"""Dry run: what every defect-region number becomes under a different mask.

WHAT THIS IS FOR
  `defect_mask_audit.py` measured that the 5 mm distance rule recovers the true
  implant region with precision 0.79 and recall 0.81 -- the count is nearly right
  (6.44% against 6.18% of ground-truth points) while the SET is about a third
  wrong (Jaccard 0.664, and per skull 87 false positives against 71 false
  negatives that very nearly cancel).

  That says the mask is imperfect. It does not say by how much the numbers built
  on it would move. This script answers that, and answers it WITHOUT CHANGING
  ANYTHING: the same predictions and the same `report._defect_metrics` are run
  through three region definitions side by side, and the result goes to its own
  CSV. `eval_all_runs.csv` is not touched, `report.DEFECT_MM` is not changed, and
  no default behaviour anywhere moves.

      5mm       the rule in force -- distance to nearest input point > 5 mm
      6mm       the same rule at the threshold `defect_mask_audit` found best
                (F1 0.848 against 0.796; precision 0.975 against 0.790)
      implant   the dataset's own ground truth: is this point on the implant

⭐ THE CHECK THAT MAKES THE REST READABLE
  The `5mm` column must reproduce the frozen `eval_all_runs.csv` exactly. It is
  computed here through the same function with the same arguments, so any
  difference means this script is wrong and none of the other columns can be
  trusted. It is verified per run per skull and the script aborts on mismatch.

WHAT TO LOOK AT IN THE OUTPUT
  Not "did the numbers move" -- they will, the region changed. The questions are:
    1. do the RANKINGS between configurations survive? The mask never sees a
       prediction, so every configuration is scored through the identical set of
       points and the comparisons ought to be untouched. Ought to. This checks.
    2. do the three claims that rest on defect coverage still hold -- repulsion's
       effect, `notext` being real, `tie_qk` being rejected?
  A number moving is expected and harmless. A ranking moving would be serious.

⚠️ 关于预测侧
  Only the GROUND-TRUTH side of the mask has a truth to switch to. A predicted
  point is an arbitrary point in space; asking whether it "is on the implant"
  is not well posed, so that side keeps a tolerance (points within `pred_mm` of
  a defect ground-truth point). The `implant` variant therefore holds the
  prediction tolerance at the current 5 mm and changes only the ground-truth
  side, which is where the main metric `defect_cov_mm` lives.

⚠️ k 折之后：这是试算不是结论，除非真的换口径，否则不用重跑。

    python src/eval/defect_mask_switch.py [--runs A B ...] [--n 20]
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

OUT_CSV = os.path.join("experiments_log", "defect_mask_switch.csv")
# Per-point implant labels, cached because they cost ~20 s a skull and do not
# depend on any run. Also the thing that would let report.py adopt the implant
# definition WITHOUT needing the raw nrrd at evaluation time.
LABELS = os.path.join("experiments_log", "defect_mask_labels.npz")

DEFECT_COLS = ["defect_gt_%", "defect_cov_mm", "defect_HD95_mm",
               "defect_n_pred", "defect_prec_mm", "defect_F1@0.05"]

# Every run whose weights still exist. `baseline` / `dcd_l2` cannot be included:
# their checkpoints were deleted, so their defect columns can never be recomputed
# under any definition -- which is itself one of the costs of switching.
DEFAULT_RUNS = ["msn_skullfix/lr_fix_only", "msn_skullfix/rep_w05",
                "msn_skullfix/cd_only", "msn_skullfix/cd_rep05_full",
                "msn_skullfix/cd_rep05_r2", "msn_skullfix/tie_qk",
                "msn_skullfix/tie_qk_r2", "msn_skullfix/notext",
                "msn_skullfix/notext_r2", "msn_skullfix/pp_attn"]


def implant_labels(repo, sids, raw_root):
    """Per-point 'is this ground-truth point on the implant' — cached."""
    import defect_mask_audit as dma
    import normal_quality as nq

    path = os.path.join(repo, LABELS)
    have = dict(np.load(path)) if os.path.exists(path) else {}
    todo = [s for s in sids if s not in have]
    if todo:
        print(f"给 {len(todo)} 颗颅骨算 implant 真值标签（约 20 秒/颗，只需算这一次）…")
        data = np.load(os.path.join(repo, "data", "cache",
                                    "skullfix_pairs_4096_6144.npz"))
        ids, gt, scales = data["ids"], data["gt"], data["scale_mm"]
        for sid in todo:
            j = int(np.where(ids == sid)[0][0])
            s_mm = float(scales[j])
            pts_n, _, scale, _ = nq.truth_for(sid, raw_root)
            if abs(scale - s_mm) > 1e-3 or np.abs(pts_n - gt[j]).max() > 1e-5:
                raise SystemExit(f"{sid}: 管线复现失败，坐标系没对上，已中止")
            seed = nq._task_seed(sid, raw_root)
            dense_def, _, _ = nq._dense_with_faces(
                os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"),
                16384, 0.5, seed * 2 + 1)
            centroid = dense_def.mean(axis=0)
            mesh_i, vol_i = dma._mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
            mesh_d, vol_d = dma._mesh_mm(os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"))
            _, vol_c = dma._mesh_mm(os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"))
            if not np.array_equal((vol_d > 0) | (vol_i > 0), vol_c > 0) or \
                    ((vol_i > 0) & (vol_d > 0)).any():
                raise SystemExit(f"{sid}: defective + implant != complete，已中止")
            for m in (mesh_i, mesh_d):
                m.vertices = (np.asarray(m.vertices) - centroid) / scale * s_mm
            is_imp, _, _, d_min = dma.label_by_implant(gt[j] * s_mm, mesh_i, mesh_d)
            if (d_min > dma.ON_SURFACE_MM).mean() > 0.02:
                raise SystemExit(f"{sid}: GT 点离两张面都太远，并集不变量不成立，已中止")
            have[sid] = is_imp
            print(f"  {sid} ✓ 真实缺损 {100 * is_imp.mean():.2f}%")
        np.savez_compressed(path, **have)
        print(f"标签已缓存 -> {LABELS}（{len(have)} 颗）")
    return have


def analyse(repo, specs, n_skulls=20, device="/GPU:0"):
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import report as rp

    runs = rp.load_runs(repo, specs)
    raw_root = os.path.join(repo, "data", "14161307", "SkullFix", "training_set")
    data = np.load(os.path.join(repo, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    text_path = os.path.join(repo, "data", "cache", "bert_skull.npy")
    text = np.load(text_path) if os.path.exists(text_path) else None

    # Every run must share the split, because the frozen-CSV check and every
    # comparison below assume the same skulls. They do (one split across all 16
    # runs, checked 2026-08-27), but assuming it silently is how the cohort trap
    # that bit `point_to_surface.py` happened.
    splits = {tuple(r.meta["val_ids"]) for r in runs}
    if len(splits) != 1:
        raise SystemExit(f"这些 run 的验证集划分不止一种（{len(splits)} 种），无法并排比较")
    sids = runs[0].meta["val_ids"][:n_skulls]
    labels = implant_labels(repo, sids, raw_root)

    frozen = pd.read_csv(os.path.join(repo, "experiments_log", "eval_all_runs.csv"))
    frozen["id"] = frozen["id"].astype(str).str.zfill(3)

    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)

    rows, n_checked, unmatched = [], 0, []
    with tf.device(device):
        for arch, group in groups.items():
            cfg = rp.arch_config(msn, arch)
            model = msn.build_model(cfg)
            for run in group:
                model.load_weights(run.weights)
                pos = [int(np.where(ids == s)[0][0]) for s in sids]
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                preds = model.predict(x, batch_size=1, verbose=0)

                for sid, j, pred in zip(sids, pos, preds):
                    s_mm = float(scales[j])
                    # dist1 = gt -> nearest pred, exactly as report.eval_runs feeds it
                    from scipy.spatial import cKDTree
                    dist1 = cKDTree(pred).query(gt[j], k=1, workers=-1)[0]
                    variants = {
                        "5mm": dict(defect_mm=5.0),
                        "6mm": dict(defect_mm=6.0),
                        "implant": dict(gt_mask=labels[sid], defect_mm=5.0),
                    }
                    # Decompose the change instead of only showing it. The two
                    # sets share their TP; what differs is 87 false positives
                    # (intact bone, dropped) against 71 false negatives (implant
                    # near the rim, added). BOTH are relatively easy points, so
                    # which way the mean moves is not something to reason out --
                    # it depends on their actual distances, which is what these
                    # three numbers report.
                    m5 = cKDTree(inputs[j]).query(gt[j], k=1, workers=-1)[0] * s_mm > 5.0
                    mi = labels[sid]
                    d_mm = dist1 * s_mm
                    for tag, sel in (("TP", m5 & mi), ("FP_dropped", m5 & ~mi),
                                     ("FN_added", ~m5 & mi)):
                        rows.append({"run": run.label, "id": sid,
                                     "definition": f"_part_{tag}",
                                     "defect_cov_mm": float(d_mm[sel].mean()) if sel.any() else np.nan,
                                     "defect_gt_%": 100.0 * sel.mean()})

                    got5 = None
                    for name, kw in variants.items():
                        m = rp._defect_metrics(pred, gt[j], inputs[j], dist1, s_mm, **kw)
                        rows.append({"run": run.label, "id": sid, "definition": name, **m})
                        if name == "5mm":
                            got5 = m

                    # ⭐ the check everything else depends on.
                    # ⚠️ A run whose label is not in the frozen CSV used to be
                    # skipped in silence -- `lr_fix_only` is stored there under
                    # the label `lr_fix`, so 20 of 200 combinations went
                    # unchecked while the script still reported a pass. Missing
                    # rows are now counted and reported at the end.
                    ref = frozen[(frozen.run == run.label) & (frozen.id == sid)]
                    if len(ref) != 1:
                        unmatched.append((run.label, sid))
                    else:
                        got = got5
                        for c in DEFECT_COLS:
                            a, b = float(ref.iloc[0][c]), float(got[c])
                            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-6:
                                raise SystemExit(
                                    f"{run.label}/{sid}: 5mm 口径复现失败，{c} "
                                    f"冻结值 {a} vs 本次 {b} —— 本脚本不可信，已中止")
                        n_checked += 1
                print(f"  {run.label} ✓")
            del model
            tf.keras.backend.clear_session()
    total = len(runs) * len(sids)
    print(f"\n⭐ 5mm 口径与 eval_all_runs.csv 逐颅骨逐列核对：{n_checked}/{total} 组一致 ✅")
    if unmatched:
        miss = sorted({r for r, _ in unmatched})
        print(f"⚠️ 另有 {len(unmatched)} 组在冻结 CSV 里找不到对应行，**未被核对**：{miss}\n"
              f"   （通常是 run 目录名与 CSV 里的标签不同，例如 lr_fix_only vs lr_fix）")
    return pd.DataFrame(rows)


def _paired(df, base, other, col="defect_cov_mm"):
    from scipy.stats import wilcoxon
    a = df[df.run == base].set_index("id")[col]
    b = df[df.run == other].set_index("id")[col]
    d = (b - a).dropna()
    p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
    return d.mean(), int((d < 0).sum()), len(d), p


def report(df):
    df = df[~df.definition.str.startswith("_part_")] if False else df
    piv = df[~df.definition.str.startswith("_part_")].pivot_table(index="run", columns="definition",
                         values="defect_cov_mm", sort=False)[["5mm", "6mm", "implant"]]
    print(f"\n{'=' * 74}\n主指标 defect_cov_mm，三种定义并排\n{'=' * 74}")
    print(f"{'run':<18}{'5mm(现行)':>12}{'6mm':>10}{'implant':>11}"
          f"{'Δ implant':>12}{'排名变化':>10}")
    print("-" * 74)
    r5 = piv["5mm"].rank(); ri = piv["implant"].rank()
    for run in piv.index:
        move = int(ri[run] - r5[run])
        print(f"{run:<18}{piv.loc[run,'5mm']:>12.3f}{piv.loc[run,'6mm']:>10.3f}"
              f"{piv.loc[run,'implant']:>11.3f}{piv.loc[run,'implant']-piv.loc[run,'5mm']:>+12.3f}"
              f"{('—' if move == 0 else f'{move:+d}'):>10}")
    moved = int((ri != r5).sum())
    print(f"\n  ⭐ 排名变化：{moved}/{len(piv)} 个 run 换了位次 "
          f"{'✅ 全部不变，比较未受影响' if moved == 0 else '⚠️ 有变化，要细看'}")

    parts = df[df.definition.str.startswith("_part_")]
    if len(parts):
        print(f"\n{'=' * 74}\n⭐ 变化从哪来：两个集合共享 TP，差别在这两批点\n{'=' * 74}")
        print(f"  {'':16}{'点数/颅骨':>11}{'到最近预测点':>14}   说明")
        for tag, note in (("TP", "两种定义都算，共有"),
                          ("FP_dropped", "5mm 划进来但其实在完好骨上 —— implant 定义**剔除**"),
                          ("FN_added", "真在 implant 上但离输入点近 —— implant 定义**加入**")):
            g = parts[parts.definition == f"_part_{tag}"]
            print(f"  {tag:16}{g['defect_gt_%'].mean()*6144/100:>11.0f}"
                  f"{g.defect_cov_mm.mean():>13.3f}mm   {note}")
        fp = parts[parts.definition == "_part_FP_dropped"].defect_cov_mm.mean()
        fn = parts[parts.definition == "_part_FN_added"].defect_cov_mm.mean()
        print(f"\n  剔除的那批平均 {fp:.3f}mm，加入的那批平均 {fn:.3f}mm —— "
              f"{'剔除的更容易 → 主指标会升高' if fp < fn else '加入的更容易 → 主指标会降低'}")

    print(f"\n{'=' * 74}\n其余各列的平均变化（implant vs 5mm）\n{'=' * 74}")
    main = df[~df.definition.str.startswith("_part_")]
    for c in DEFECT_COLS:
        p = main.pivot_table(index="run", columns="definition", values=c, sort=False)
        a, b = p["5mm"].mean(), p["implant"].mean()
        pct = f"{100 * (b - a) / a:+6.1f}%" if a else "    n/a"
        print(f"  {c:<18}{a:>10.3f} → {b:>8.3f}   {b - a:>+8.3f}   {pct}")

    print(f"\n{'=' * 74}\n三条靠 defect_cov 的结论，在每种定义下重判\n{'=' * 74}")
    claims = [("repulsion 有效", "cd_only", "cd_rep05_full"),
              ("notext 更差", "cd_rep05_full", "notext"),
              ("tie_qk 无效", "cd_rep05_full", "tie_qk")]
    for label, base, other in claims:
        print(f"\n  {label}（{other} vs {base}）")
        for d in ("5mm", "6mm", "implant"):
            sub = main[main.definition == d]
            if not {base, other} <= set(sub.run):
                continue
            mean, better, n, p = _paired(sub, base, other)
            print(f"    {d:<9} Δ {mean:+.3f}mm   {other} 更好 {better}/{n}   "
                  f"p={p:.4f}   {'显著' if p < 0.002 else '不显著'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    df = analyse(REPO, args.runs, n_skulls=args.n)
    report(df)
    df.to_csv(os.path.join(REPO, args.out), index=False)
    print(f"\n{len(df)} rows -> {args.out}"
          f"\n⚠️ 这是试算。eval_all_runs.csv 与 report.DEFECT_MM 都没有被改动。")


if __name__ == "__main__":
    main()
