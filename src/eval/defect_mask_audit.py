r"""Check the defect-region mask against the implant the dataset actually ships.

WHAT IS BEING CHECKED
  The defect region is the main metric's domain, and it is currently INFERRED,
  not known: a ground-truth point counts as "in the defect" when the nearest
  point of the defective input is more than 5 mm away. That threshold was argued
  from a histogram -- ground-truth-to-input distance is clearly bimodal (a peak
  at 2-3 mm where the two clouds sample the same surface, a trough at 5-6 mm,
  the real hole beyond 15 mm) and a sensitivity sweep flattens out past 5 mm
  (3 mm -> 42.0%, 4 mm -> 16.1%, 5 mm -> 6.7%, 6 mm -> 5.0%, 8 mm -> 4.2%).

  An argument, not a measurement. And a measurement is available and has never
  been used: `training_set/implant/` holds the missing piece itself, and it is
  exact, not approximate -- verified here per skull, `defective + implant ==
  complete` voxel for voxel with zero overlap.

  So every ground-truth point can be labelled by ground truth rather than by
  proxy, and the proxy can be scored against it: precision, recall, and a sweep
  to see whether 5 mm is actually the best threshold.

WHY THE LABELLING IS UNAMBIGUOUS
  The complete skull's surface is the union of two disjoint pieces of surface:
  the part bounding the remaining bone, and the part bounding the implant. Their
  interface -- where the implant is cut from the skull -- is INTERIOR to the
  complete skull and therefore not on its surface at all. So a ground-truth
  point, which lies on the complete surface, is over one or the other, and the
  nearer of the two meshes says which. Points within a voxel of both are counted
  and reported as `seam_pct`; they sit on the rim where the two surfaces meet.

WHAT THIS CANNOT TELL YOU
  Nothing about the model. No prediction is loaded, no GPU is used; this is a
  property of the DATA. If the proxy mask turns out to be inaccurate, every
  defect-region number moves -- but only in absolute terms. Every configuration
  was scored through the same mask, so the comparisons between them are
  unaffected, in the same way the sampling floor is common-mode.

⚠️ k 折之后不用重跑。数据的性质，与划分和权重无关（同 `sampling_floor.csv`）。

    python src/eval/defect_mask_audit.py [--n 20] [--self-test]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT_CSV = os.path.join("experiments_log", "defect_mask.csv")

# The threshold in force (report.DEFECT_MM), plus the sweep around it.
CURRENT_MM = 5.0
SWEEP = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0)

# A ground-truth point is "on" a surface if it is within this of it. Voxels here
# are 0.45-0.63 mm on a side, so this is about one voxel: tight enough to be a
# real test, loose enough not to trip over the marching-cubes surface sitting
# fractionally off the voxel boundary.
ON_SURFACE_MM = 0.6


def _mesh_mm(nrrd_path, level=0.5):
    """The volume's surface, in millimetres, exactly as prepare_skullfix builds it."""
    import nrrd
    import trimesh
    from skimage import measure

    volume, header = nrrd.read(nrrd_path)
    verts, faces, _, _ = measure.marching_cubes(volume, level=level)
    spacing = np.asarray(header["space directions"], dtype=np.float64)
    return trimesh.Trimesh(vertices=verts @ spacing, faces=faces), volume


def label_by_implant(gt_mm, mesh_implant, mesh_defective):
    """Which side of the complete surface each ground-truth point is on.

    Returns (is_implant, d_implant, d_defective, d_min). Nearest surface wins;
    `d_min` is how far the point is from EITHER, and must be ~0 because the
    complete surface is covered by the two of them -- checked by the caller.
    """
    import point_to_surface as p2s

    d_i, _, _ = p2s.point_to_mesh(gt_mm, mesh_implant)
    d_d, _, _ = p2s.point_to_mesh(gt_mm, mesh_defective)
    return d_i < d_d, d_i, d_d, np.minimum(d_i, d_d)


def label_one(repo, sid, raw_root, gt_pts, scale_mm):
    """The per-skull labelling, in one place because three scripts need it.

    Returns a length-6144 bool array: True where that ground-truth point sits on
    the implant rather than on the bone that was left. Every guard the audit
    relies on runs here too -- the pipeline must reproduce the cache, the dataset
    must actually satisfy `defective + implant == complete`, and every point must
    land on one of the two surfaces -- because a wrong label here would silently
    become a wrong defect region everywhere downstream.
    """
    import normal_quality as nq

    pts_n, _, scale, _ = nq.truth_for(sid, raw_root)
    if abs(scale - scale_mm) > 1e-3 or np.abs(pts_n - gt_pts).max() > 1e-5:
        raise SystemExit(f"{sid}: 管线复现失败，坐标系没对上，已中止")
    seed = nq._task_seed(sid, raw_root)
    dense_def, _, _ = nq._dense_with_faces(
        os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2 + 1)
    centroid = dense_def.mean(axis=0)

    mesh_i, vol_i = _mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
    mesh_d, vol_d = _mesh_mm(os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"))
    _, vol_c = _mesh_mm(os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"))
    if not np.array_equal((vol_d > 0) | (vol_i > 0), vol_c > 0) or ((vol_i > 0) & (vol_d > 0)).any():
        raise SystemExit(f"{sid}: defective + implant != complete，真值前提不成立，已中止")
    for m in (mesh_i, mesh_d):
        m.vertices = (np.asarray(m.vertices) - centroid) / scale * scale_mm

    is_imp, d_i, d_d, d_min = label_by_implant(gt_pts * scale_mm, mesh_i, mesh_d)
    off = float((d_min > ON_SURFACE_MM).mean())
    if off > 0.02:
        raise SystemExit(f"{sid}: {100*off:.1f}% 的 GT 点离两张表面都太远，并集不变量不成立，已中止")
    return is_imp, {"seam_pct": 100.0 * float(((d_i < ON_SURFACE_MM) & (d_d < ON_SURFACE_MM)).mean()),
                    "off_surface_pct": 100.0 * off,
                    "watertight": bool(mesh_i.is_watertight),
                    "winding_consistent": bool(mesh_i.is_winding_consistent)}


def analyse(repo, sids, raw_root):
    import normal_quality as nq

    data = np.load(os.path.join(repo, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]

    rows = []
    for sid in sids:
        j = int(np.where(ids == sid)[0][0])
        s_mm = float(scales[j])

        # ---- rebuild the frame; `truth_for` already verifies it reproduces the cache
        pts_n, _, scale, _ = nq.truth_for(sid, raw_root)
        if abs(scale - s_mm) > 1e-3 or np.abs(pts_n - gt[j]).max() > 1e-5:
            raise SystemExit(f"{sid}: 管线复现失败，坐标系没对上，已中止")
        # The centroid must come from the SAME draw prepare_skullfix used, or the
        # meshes land in a different frame from the cached points.
        seed = nq._task_seed(sid, raw_root)
        dense_def, _, _ = nq._dense_with_faces(
            os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2 + 1)
        centroid = dense_def.mean(axis=0)

        mesh_i, vol_i = _mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
        mesh_d, vol_d = _mesh_mm(os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"))
        _, vol_c = _mesh_mm(os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"))

        # ---- invariant: the dataset's own claim about what `implant` is ----
        ok_union = bool(np.array_equal((vol_d > 0) | (vol_i > 0), vol_c > 0))
        n_overlap = int(((vol_i > 0) & (vol_d > 0)).sum())
        if not ok_union or n_overlap:
            raise SystemExit(
                f"{sid}: defective + implant != complete（并集相同 {ok_union}，"
                f"重叠 {n_overlap} 体素）—— 真值标签的前提不成立，已中止")

        for m in (mesh_i, mesh_d):                       # into the cache's frame, in mm
            m.vertices = (np.asarray(m.vertices) - centroid) / scale * s_mm

        gt_mm = gt[j] * s_mm
        is_imp, d_i, d_d, d_min = label_by_implant(gt_mm, mesh_i, mesh_d)

        # ---- invariant: every GT point lies on ONE of the two surfaces ----
        off = float((d_min > ON_SURFACE_MM).mean())
        if off > 0.02:
            raise SystemExit(
                f"{sid}: {100*off:.1f}% 的 GT 点离两张表面都超过 {ON_SURFACE_MM}mm，"
                f"「complete 表面 = implant 表面 ∪ defective 表面」不成立，已中止")
        seam = float(((d_i < ON_SURFACE_MM) & (d_d < ON_SURFACE_MM)).mean())

        # ---- the proxy under test ----
        d_to_input = cKDTree(inputs[j]).query(gt[j], k=1, workers=-1)[0] * s_mm
        for t in SWEEP:
            geom = d_to_input > t
            tp = int((geom & is_imp).sum())
            fp = int((geom & ~is_imp).sum())
            fn = int((~geom & is_imp).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            rows.append({
                "id": sid, "thresh_mm": t, "n_gt": len(gt[j]),
                "n_true_defect": int(is_imp.sum()), "n_geom_defect": int(geom.sum()),
                "true_pct": 100.0 * is_imp.mean(), "geom_pct": 100.0 * geom.mean(),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
                "jaccard": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                "seam_pct": 100.0 * seam, "off_surface_pct": 100.0 * off,
            })
        print(f"  {sid} ✓  真实缺损占 {100*is_imp.mean():5.2f}%  "
              f"(5mm 规则给 {100*(d_to_input > CURRENT_MM).mean():5.2f}%)  "
              f"缝隙点 {100*seam:.2f}%  离面 {100*off:.3f}%")
    return pd.DataFrame(rows)


def report(df):
    n = df["id"].nunique()
    print(f"\n{'=' * 76}\n缺损区掩码 vs 数据集自带的 implant 真值（{n} 颗颅骨）\n{'=' * 76}")
    print(f"{'阈值mm':>8}{'真实占比%':>11}{'规则占比%':>11}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'Jaccard':>9}")
    print("-" * 76)
    best_f1 = df.groupby("thresh_mm")["f1"].mean().idxmax()
    for t, g in df.groupby("thresh_mm"):
        mark = "  ← 当前" if t == CURRENT_MM else ("  ← F1 最优" if t == best_f1 else "")
        print(f"{t:>8.1f}{g.true_pct.mean():>11.2f}{g.geom_pct.mean():>11.2f}"
              f"{g.precision.mean():>11.3f}{g.recall.mean():>9.3f}"
              f"{g.f1.mean():>8.3f}{g.jaccard.mean():>9.3f}{mark}")

    cur = df[df.thresh_mm == CURRENT_MM]
    print(f"\n  当前 5mm：precision {cur.precision.mean():.3f}（划进来的点里有这么多真在 implant 上）")
    print(f"            recall    {cur.recall.mean():.3f}（真实 implant 表面被划进来这么多）")
    print(f"            真实缺损占 GT 的 {cur.true_pct.mean():.2f}%，规则给出 {cur.geom_pct.mean():.2f}%")
    print(f"  跨颅骨 std: precision {cur.precision.std():.3f}  recall {cur.recall.std():.3f}")
    print(f"  缝隙点（离两张表面都 <{ON_SURFACE_MM}mm）{cur.seam_pct.mean():.2f}% —— "
          f"标签在这里天然模糊，是精度的下限")

    f1c, f1b = cur.f1.mean(), df[df.thresh_mm == best_f1].f1.mean()
    if best_f1 == CURRENT_MM:
        print(f"\n  ✅ **5mm 就是扫描里 F1 最优的那一档**，那个直方图论证被真值确认了。")
    else:
        print(f"\n  ⚠️ F1 最优在 {best_f1:.1f}mm（{f1b:.3f}）而非当前的 5.0mm（{f1c:.3f}），"
              f"差 {f1b - f1c:+.3f}")
    if cur.precision.mean() > 0.9 and cur.recall.mean() > 0.9:
        print("  ✅ precision 与 recall 都 >0.9 —— 代理规则可靠，缺损区指标不必改口径。")
    else:
        print("  ⚠️ precision 或 recall 低于 0.9 —— 代理规则与真值有实质差距。\n"
              "     ⚠️ 但注意：所有配置用的是同一个掩码，误差是**共模**的，\n"
              "        受影响的是缺损区指标的**绝对值**，不是配置之间的**比较**。")


def self_test():
    """Two adjacent boxes: the labelling must follow which box a point sits on."""
    import trimesh
    import point_to_surface as p2s
    print("=== 自检：把'哪张面更近'的判定放在已知答案的几何上 ===")
    a = trimesh.creation.box(extents=[10, 10, 10])                  # 「剩余骨」
    b = trimesh.creation.box(extents=[10, 10, 10]); b.apply_translation([10, 0, 0])  # 「implant」
    rng = np.random.default_rng(0)
    # 各自外表面上的点（避开交界面 x=5）
    pa, _ = trimesh.sample.sample_surface(a, 2000, seed=1)
    pb, _ = trimesh.sample.sample_surface(b, 2000, seed=2)
    pa = np.asarray(pa)[np.abs(np.asarray(pa)[:, 0] - 5) > 0.5]
    pb = np.asarray(pb)[np.abs(np.asarray(pb)[:, 0] - 5) > 0.5]
    is_imp_a, *_ = label_by_implant(pa, b, a)
    is_imp_b, *_ = label_by_implant(pb, b, a)
    print(f"① 在'剩余骨'上的点被判成 implant 的比例 {100*is_imp_a.mean():.1f}%  "
          f"{'✅' if is_imp_a.mean() < 0.01 else '❌'}（应 ~0）")
    print(f"② 在'implant'上的点被判成 implant 的比例 {100*is_imp_b.mean():.1f}%  "
          f"{'✅' if is_imp_b.mean() > 0.99 else '❌'}（应 ~100）")
    # 两张面覆盖了并集的表面：任一点到最近那张面的距离必须 ~0
    _, _, _, dmin = label_by_implant(np.r_[pa, pb], b, a)
    print(f"③ 到最近那张面的最大距离 {dmin.max():.2e}  "
          f"{'✅' if dmin.max() < 1e-9 else '❌'}（点就在面上，应为 0）")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--from-run", default="msn_skullfix/cd_rep05_full",
                    help="取哪一轮的验证颅骨（只读 run.json，不建模型、不要 GPU）")
    ap.add_argument("--n", type=int, default=20, help="颅骨数（默认整个验证集 20 颗）")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import report as rp
    sids = rp.Run(REPO, args.from_run).meta["val_ids"][:args.n]
    raw_root = os.path.join(REPO, "data", "14161307", "SkullFix", "training_set")
    print(f"用数据集自带的 implant 给 GT 点打真值标签（{len(sids)} 颗，每颗 4 个体数据）…")
    df = analyse(REPO, sids, raw_root)
    report(df)

    out = os.path.join(REPO, args.out)
    if os.path.exists(out):
        old = pd.read_csv(out)
        KEY = ["id", "thresh_mm"]
        k_old = old[KEY].astype(str).apply(tuple, axis=1)
        k_new = set(df[KEY].astype(str).apply(tuple, axis=1))
        df = pd.concat([old[~k_old.isin(k_new)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
