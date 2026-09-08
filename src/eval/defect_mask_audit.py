"""Check the defect-region mask against the implant the dataset actually ships.

The defect region is the main metric's domain, and it is INFERRED rather than
known: a ground-truth point counts as "in the defect" when the nearest point of
the defective input is more than 5 mm away. That threshold was argued from a
histogram -- ground-truth-to-input distance is clearly bimodal, with a peak where
the two clouds sample the same surface, a trough around 5-6 mm, and the real hole
well beyond -- and from a sensitivity sweep that flattens out past 5 mm.

An argument, not a measurement. And a measurement is available: `training_set/
implant/` holds the missing piece itself, exactly rather than approximately --
verified here per skull, `defective + implant == complete` voxel for voxel with
zero overlap. So every ground-truth point can be labelled by ground truth instead
of by proxy, and the proxy scored against it.

The labelling is unambiguous because the complete skull's surface is the union of
two disjoint pieces: the part bounding the remaining bone and the part bounding
the implant. Their interface, where the implant is cut away, is INTERIOR to the
complete skull and so not on its surface at all. A ground-truth point therefore
lies over one piece or the other, and the nearer mesh says which. Points within a
voxel of both are counted as `seam_pct`; they sit on the rim.

This says nothing about any model -- no prediction is loaded and no GPU is used.
If the proxy mask is inaccurate every defect-region number moves, but only in
absolute terms: every configuration was scored through the same mask, so
comparisons between them are unaffected, in the same way the sampling floor is
common-mode.

    python src/eval/defect_mask_audit.py [--n 20] [--self-test]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import paths                    # every data path is written down once

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
        raise SystemExit(f"{sid}: could not reproduce the pipeline -- the coordinate "
                         f"frames do not line up. Aborted.")
    seed = nq._task_seed(sid, raw_root)
    dense_def, _, _ = nq._dense_with_faces(
        os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2 + 1)
    centroid = dense_def.mean(axis=0)

    mesh_i, vol_i = _mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
    mesh_d, vol_d = _mesh_mm(os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"))
    _, vol_c = _mesh_mm(os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"))
    if not np.array_equal((vol_d > 0) | (vol_i > 0), vol_c > 0) or ((vol_i > 0) & (vol_d > 0)).any():
        raise SystemExit(f"{sid}: defective + implant != complete, so the premise "
                         f"behind the ground-truth labels does not hold. Aborted.")
    for m in (mesh_i, mesh_d):
        m.vertices = (np.asarray(m.vertices) - centroid) / scale * scale_mm

    is_imp, d_i, d_d, d_min = label_by_implant(gt_pts * scale_mm, mesh_i, mesh_d)
    off = float((d_min > ON_SURFACE_MM).mean())
    if off > 0.02:
        raise SystemExit(f"{sid}: {100*off:.1f}% of ground-truth points are far from both "
                         f"surfaces, so the union invariant fails. Aborted.")
    return is_imp, {"seam_pct": 100.0 * float(((d_i < ON_SURFACE_MM) & (d_d < ON_SURFACE_MM)).mean()),
                    "off_surface_pct": 100.0 * off,
                    "watertight": bool(mesh_i.is_watertight),
                    "winding_consistent": bool(mesh_i.is_winding_consistent)}


def analyse(repo, sids, raw_root):
    import normal_quality as nq

    data = np.load(os.path.join(repo, paths.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]

    rows = []
    for sid in sids:
        j = int(np.where(ids == sid)[0][0])
        s_mm = float(scales[j])

        # ---- rebuild the frame; `truth_for` already verifies it reproduces the cache
        pts_n, _, scale, _ = nq.truth_for(sid, raw_root)
        if abs(scale - s_mm) > 1e-3 or np.abs(pts_n - gt[j]).max() > 1e-5:
            raise SystemExit(f"{sid}: could not reproduce the pipeline -- the coordinate "
                             f"frames do not line up. Aborted.")
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
                f"{sid}: defective + implant != complete (union matches: {ok_union}, "
                f"{n_overlap} overlapping voxels). The premise behind the ground-truth "
                f"labels does not hold. Aborted.")

        for m in (mesh_i, mesh_d):                       # into the cache's frame, in mm
            m.vertices = (np.asarray(m.vertices) - centroid) / scale * s_mm

        gt_mm = gt[j] * s_mm
        is_imp, d_i, d_d, d_min = label_by_implant(gt_mm, mesh_i, mesh_d)

        # ---- invariant: every GT point lies on ONE of the two surfaces ----
        off = float((d_min > ON_SURFACE_MM).mean())
        if off > 0.02:
            raise SystemExit(
                f"{sid}: {100*off:.1f}% of ground-truth points are further than "
                f"{ON_SURFACE_MM}mm from both surfaces, so 'complete surface = implant "
                f"surface union defective surface' does not hold. Aborted.")
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
        print(f"  {sid} ok  true defect {100*is_imp.mean():5.2f}%  "
              f"(the 5mm rule says {100*(d_to_input > CURRENT_MM).mean():5.2f}%)  "
              f"seam {100*seam:.2f}%  off-surface {100*off:.3f}%")
    return pd.DataFrame(rows)


def report(df):
    n = df["id"].nunique()
    print(f"\n{'=' * 76}\nthe distance rule against the dataset's implant ground truth "
          f"({n} skulls)\n{'=' * 76}")
    print(f"{'thresh':>8}{'true %':>11}{'rule %':>11}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'Jaccard':>9}")
    print("-" * 76)
    best_f1 = df.groupby("thresh_mm")["f1"].mean().idxmax()
    for t, g in df.groupby("thresh_mm"):
        mark = "  <- current" if t == CURRENT_MM else ("  <- best F1" if t == best_f1 else "")
        print(f"{t:>8.1f}{g.true_pct.mean():>11.2f}{g.geom_pct.mean():>11.2f}"
              f"{g.precision.mean():>11.3f}{g.recall.mean():>9.3f}"
              f"{g.f1.mean():>8.3f}{g.jaccard.mean():>9.3f}{mark}")

    cur = df[df.thresh_mm == CURRENT_MM]
    print(f"\n  at the current 5mm: precision {cur.precision.mean():.3f} -- this much of "
          f"what the rule selects really is on the implant")
    print(f"                      recall    {cur.recall.mean():.3f} -- this much of the "
          f"real implant surface gets selected")
    print(f"                      the true defect is {cur.true_pct.mean():.2f}% of the "
          f"ground truth, the rule says {cur.geom_pct.mean():.2f}%")
    print(f"  across skulls, std: precision {cur.precision.std():.3f}  "
          f"recall {cur.recall.std():.3f}")
    print(f"  seam points, within {ON_SURFACE_MM}mm of both surfaces: "
          f"{cur.seam_pct.mean():.2f}% -- the label is genuinely ambiguous there, so this "
          f"is a floor on precision")

    f1c, f1b = cur.f1.mean(), df[df.thresh_mm == best_f1].f1.mean()
    if best_f1 == CURRENT_MM:
        print(f"\n  5mm is the best F1 in the sweep -- the histogram argument is "
              f"confirmed by the ground truth.")
    else:
        print(f"\n  Warning: the best F1 is at {best_f1:.1f}mm ({f1b:.3f}) rather than the "
              f"current 5.0mm ({f1c:.3f}), a difference of {f1b - f1c:+.3f}")
    if cur.precision.mean() > 0.9 and cur.recall.mean() > 0.9:
        print("  precision and recall are both above 0.9 -- the proxy is reliable and the "
              "defect-region metrics need no change of definition.")
    else:
        print("  Warning: precision or recall is below 0.9 -- the proxy differs materially "
              "from the ground truth.\n"
              "     Note that every configuration used the SAME mask, so the error is "
              "common-mode: it moves\n"
              "     the absolute defect-region numbers, not the comparisons between "
              "configurations.")


def self_test():
    """Two adjacent boxes: the labelling must follow which box a point sits on."""
    import trimesh
    import point_to_surface as p2s
    print("=== self-test: run the 'which surface is nearer' rule on known geometry ===")
    a = trimesh.creation.box(extents=[10, 10, 10])                  # the remaining bone
    b = trimesh.creation.box(extents=[10, 10, 10]); b.apply_translation([10, 0, 0])  # 「implant」
    rng = np.random.default_rng(0)
    # points on each outer surface, keeping clear of the interface at x=5
    pa, _ = trimesh.sample.sample_surface(a, 2000, seed=1)
    pb, _ = trimesh.sample.sample_surface(b, 2000, seed=2)
    pa = np.asarray(pa)[np.abs(np.asarray(pa)[:, 0] - 5) > 0.5]
    pb = np.asarray(pb)[np.abs(np.asarray(pb)[:, 0] - 5) > 0.5]
    is_imp_a, *_ = label_by_implant(pa, b, a)
    is_imp_b, *_ = label_by_implant(pb, b, a)
    print(f"1. points on the bone labelled implant: {100*is_imp_a.mean():.1f}%  "
          f"{'ok' if is_imp_a.mean() < 0.01 else 'FAIL'} (should be ~0)")
    print(f"2. points on the implant labelled implant: {100*is_imp_b.mean():.1f}%  "
          f"{'ok' if is_imp_b.mean() > 0.99 else 'FAIL'} (should be ~100)")
    # the two surfaces cover the union: every point's distance to the nearer one is ~0
    _, _, _, dmin = label_by_implant(np.r_[pa, pb], b, a)
    print(f"3. largest distance to the nearer surface: {dmin.max():.2e}  "
          f"{'ok' if dmin.max() < 1e-9 else 'FAIL'} (points lie on it, so it should be 0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--from-run", default="msn_skullfix/cd_rep05_full_f0",
                    help="whose validation skulls to use; only run.json is read, no model "
                         "is built and no GPU is needed")
    ap.add_argument("--n", type=int, default=20, help="how many skulls (default: the whole "
                                                     "validation set)")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import report as rp
    sids = rp.Run(REPO, args.from_run).meta["val_ids"][:args.n]
    raw_root = os.path.join(REPO, paths.RAW_ROOT)
    print(f"labelling ground-truth points from the dataset's implant "
          f"({len(sids)} skulls, 4 volumes each)...")
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
