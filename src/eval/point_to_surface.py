"""Metrics against the TRUE surface instead of against 6144 sampled points.

Every other metric here is point-to-point: for each ground-truth point, the
distance to the nearest predicted point. Ground truth is itself only 6144 points
sampled off a continuous surface, so a PERFECT model still scores non-zero,
because its points are not the same 6144 points ground truth happened to draw.
That is the sampling floor, and it is most of the reported error.

The mesh those points were sampled from can be recovered exactly. Scoring against
it changes what the ruler measures:

    predicted point -> surface     "is it in the right place"   floor = 0
    surface -> predicted point     "is the surface covered"     floor > 0

Warning: the asymmetry is the point and must not be oversold. Only the first
direction loses its floor. The second keeps one, because a prediction is 6144
points about 4 mm apart and a continuous surface cannot be covered more finely
than that -- and the direction that keeps a floor is the MAIN metric, while the
one that loses it is the column already measured to have almost no discriminative
power. So this is not "the floor is gone". It is a direct measurement of how much
of the reported error is the model, and a re-attribution of the coverage floor
from an artefact of scoring to a property of the output representation.
`s2p_floor_*` measures exactly that: the same coverage number computed for
ground truth's own 6144 points, i.e. what a perfect 6144-point model scores.

Relationship to `mesh_viz.signed_deviation` -- read before dropping either.
`signed_deviation` answers the same question as the accuracy direction, but
against a plane fitted to the 24 nearest ground-truth points, and that plane is
contaminated: the k=24 neighbourhood spreads 8.72 mm along its own normal while
the shell is 5-7 mm thick, so the "local plane" is fitted across BOTH sheets. So
this script computes both, on the same predicted points in the same run, and
prints them side by side. That is not a convenience: doing it across CSVs was
tried and was wrong, because `surface_quality.csv` uses the first 8 validation
skulls in `ids` order while this script uses the first 8 in `val_ids` order, and
those sets share exactly ONE skull. Either way do NOT delete `signed_deviation`:
`surface_quality.csv` holds rows whose weights are gone and can never be
recomputed, so removing the code behind those columns would orphan them.

Distances: a KD-tree over mesh vertices gives the k nearest, their incident faces
are the candidate triangles, and the exact point-to-triangle distance is taken
over those -- about 20x faster than the library's own accelerated query here, and
agreeing with it to 4.4e-16 mm. Correctness is not assumed: `--self-test` checks
the routine against an analytic plane, an analytic sphere and a brute-force
solver, and every real run additionally checks on live data that ground truth's
own points come back at distance ~0, that doubling the candidate set does not
move the answer, and that a sample agrees with an independent exact solver.

The mesh side depends only on the data; the prediction side depends on which
checkpoint is passed in.

    python src/eval/point_to_surface.py [--runs A ...] [--n 8] [--self-test]
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
from scipy.spatial import cKDTree

OUT_CSV = os.path.join("experiments_log", "p2s.csv")

# Candidate triangles per query = the faces touching its VERT_K nearest vertices.
# Not guaranteed in general, but marching-cubes triangles here are ~0.485 mm on a
# ~0.45 mm voxel grid, so the closest triangle is always incident to a very near
# vertex. Verified rather than assumed, in three links:
#   4 == 8          `analyse` recomputes every skull at 2*VERT_K and requires the
#                   answer not to move. (3 and 2 do NOT suffice -- measured, they
#                   differ by 8.7e-04 mm; 4 matches 8 exactly.)
#   8 == exact      cross-checked against trimesh's accelerated closest_point on
#                   the real 2.47M-face mesh: max difference 4.4e-16 mm over 400
#                   points. `analyse` repeats this on its first skull every run.
#   routine correct `--self-test`: analytic plane, analytic sphere, and trimesh's
#                   brute-force closest_point_naive.
# k=4 is also 14x faster than k=8 (5 s vs 70 s per skull) -- fewer candidate
# slots keeps the intermediate arrays in cache.
VERT_K = 4

# Monte-Carlo samples of the surface for the coverage direction. This integrates
# over the surface, so density affects the estimator's variance, not its bias;
# 200k gives ~1.1 mm between samples on a ~250,000 mm^2 skull, well under the
# 2-4 mm being measured.
N_DENSE = 200_000

DEFECT_MM = 5.0        # same threshold as report.DEFECT_MM
DEFAULT_RUNS = ["msn_skullfix/cd_rep05_full_f0"]


# --------------------------------------------------------------------------- #
# exact point-to-triangle distance
# --------------------------------------------------------------------------- #
def _closest_on_triangles(P, A, B, C):
    """Closest point on each triangle to each query. P (N,1,3), A/B/C (N,M,3)."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.einsum("nmi,nmi->nm", ab, ap)
    d2 = np.einsum("nmi,nmi->nm", ac, ap)
    bp = P - B
    d3 = np.einsum("nmi,nmi->nm", ab, bp)
    d4 = np.einsum("nmi,nmi->nm", ac, bp)
    cp = P - C
    d5 = np.einsum("nmi,nmi->nm", ab, cp)
    d6 = np.einsum("nmi,nmi->nm", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = 1.0 / np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = np.clip(vb * denom, 0.0, 1.0)
    w = np.clip(vc * denom, 0.0, 1.0)
    out = A + v[..., None] * ab + w[..., None] * ac          # face interior

    def seg(P0, D, t):
        return P0 + np.clip(t, 0.0, 1.0)[..., None] * D

    # Voronoi regions, applied in the order of Ericson's routine so that later
    # (more specific) cases overwrite earlier ones.
    reg = [
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), seg(A, ab, d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3))),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), seg(A, ac, d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6))),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
         seg(B, C - B, (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)))),
        ((d1 <= 0) & (d2 <= 0), A),
        ((d3 >= 0) & (d4 <= d3), B),
        ((d6 >= 0) & (d5 <= d6), C),
    ]
    for mask, val in reg:
        out = np.where(mask[..., None], val, out)
    return out


def point_to_mesh(points, mesh, vert_tree=None, k=VERT_K):
    """Exact distance from each point to the mesh surface, plus the signed value.

    Sign comes from the winning face's normal: positive means the point sits on
    the outward side of the bone. That is better defined than the sign
    `signed_deviation` derives from a plane fitted across both sheets of the
    shell -- these normals are consistent (checked: `is_winding_consistent`).
    """
    P = np.asarray(points, dtype=np.float64)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    tree = vert_tree if vert_tree is not None else cKDTree(V)
    vidx = tree.query(P, k=k, workers=-1)[1]                  # (N, k)

    vf = np.asarray(mesh.vertex_faces)                        # (n_verts, pad), -1 padded
    cand = vf[vidx].reshape(len(P), -1)                       # (N, k*pad)
    valid = cand >= 0
    safe = np.where(valid, cand, 0)

    tri = V[F[safe]]                                          # (N, M, 3, 3)
    cp = _closest_on_triangles(P[:, None, :], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])
    d = np.linalg.norm(cp - P[:, None, :], axis=-1)
    d = np.where(valid, d, np.inf)
    best = np.argmin(d, axis=1)
    rows = np.arange(len(P))
    dist = d[rows, best]
    face = safe[rows, best]
    signed = dist * np.sign(np.einsum("ni,ni->n",
                                      P - cp[rows, best],
                                      np.asarray(mesh.face_normals)[face]))
    return dist, signed, face


# --------------------------------------------------------------------------- #
def analyse(repo, specs, n_skulls=8, n_dense=N_DENSE, device="/GPU:0"):
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import trimesh
    import mesh_viz as mv
    import msn_skullfix as msn
    import normal_quality as nq
    import report as rp

    runs = rp.load_runs(repo, specs)
    labels = rp.defect_labels(repo)
    data = np.load(os.path.join(repo, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    text_path = os.path.join(repo, rp.BERT_CACHE)
    text = np.load(text_path) if os.path.exists(text_path) else None
    raw_root = os.path.join(repo, rp.RAW_ROOT)

    # ---- predictions first, one model, then release the GPU ----
    preds = {}
    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)
    with tf.device(device):
        for arch, group in groups.items():
            cfg = rp.arch_config(msn, arch)
            model = msn.build_model(cfg)
            for run in group:
                model.load_weights(run.weights)
                val = run.meta["val_ids"][:n_skulls]
                pos = [int(np.where(ids == s)[0][0]) for s in val]
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                preds[run.label] = (val, pos, model.predict(x, batch_size=1, verbose=0))
            del model
            tf.keras.backend.clear_session()

    rows = []
    for label, (val, pos, pr) in preds.items():
        print(f"\n{label}:")
        for sid, j, pred in zip(val, pos, pr):
            s_mm = float(scales[j])
            mesh, mesh_i = _mesh_for(sid, raw_root, nq, trimesh, gt[j], s_mm)
            vtree = cKDTree(np.asarray(mesh.vertices))

            # ---- invariant: GT points were sampled off this mesh -> distance 0 ----
            gt_d, _, _ = point_to_mesh(gt[j] * s_mm, mesh, vtree)
            if gt_d.max() > 1e-3:
                raise SystemExit(
                    f"{sid}: ground-truth points sit up to {gt_d.max():.3e}mm from the "
                    f"mesh, and should be at 0 -- either the frames do not line up or the "
                    f"point-to-triangle routine is wrong. Aborted.")
            # ---- invariant: candidate set is big enough ----
            chk, _, _ = point_to_mesh(pred[:256] * s_mm, mesh, vtree, k=2 * VERT_K)
            ref, _, _ = point_to_mesh(pred[:256] * s_mm, mesh, vtree, k=VERT_K)
            if np.abs(chk - ref).max() > 1e-9:
                raise SystemExit(f"{sid}: k={VERT_K} gives too few candidate triangles -- "
                                 f"doubling it changed the answer")
            # ---- invariant: agrees with an INDEPENDENT exact solver, on live data.
            # Once per run (it costs ~3 s); the synthetic tests cannot cover
            # marching-cubes topology, which is the one thing that could break the
            # candidate-set assumption.
            if not rows:
                try:
                    _, d_tm, _ = trimesh.proximity.closest_point(mesh, pred[:200] * s_mm)
                    gap = float(np.abs(ref[:200] - d_tm).max())
                    if gap > 1e-9:
                        raise SystemExit(f"{sid}: disagrees with the independent exact "
                                         f"solver by {gap:.3e}mm")
                    print(f"  [check] agrees with an independent exact solver over 200 "
                          f"points, largest difference {gap:.2e}mm")
                except ImportError:
                    print("  [check] rtree missing, skipping the independent cross-check")

            # ---- accuracy: predicted points -> surface (floor 0) ----
            dist, signed, _ = point_to_mesh(pred * s_mm, mesh, vtree)
            # The same quantity as mesh_viz measures it, on the SAME points, so the
            # two references (true surface vs plane fitted across both sheets of
            # the shell) can be compared without a cohort confound.
            dev = mv.signed_deviation(pred, gt[j], s_mm)

            # ---- coverage: surface -> nearest predicted point (floor > 0) ----
            dense = np.asarray(trimesh.sample.sample_surface(mesh, n_dense, seed=7)[0])
            dense_n = dense / s_mm                      # into the cache's frame
            s2p = cKDTree(pred).query(dense_n, k=1, workers=-1)[0] * s_mm
            floor = cKDTree(gt[j]).query(dense_n, k=1, workers=-1)[0] * s_mm

            # ---- defect region: the implant ground truth, same as report.py ----
            # (2026-08-28; the distance rule this replaced scored precision 0.79 /
            # recall 0.81 against it.)
            defect_gt = gt[j][labels[sid]]
            # The dense surface points need labelling too, and running the exact
            # point-to-mesh query on 200k of them would cost ~3 min a skull. They
            # lie ON the complete surface, and the complete surface is the union
            # of the implant's and the remaining bone's, so "is this point on the
            # implant" is answered by its distance to a dense sampling OF the
            # implant: at 200k samples over ~13,000 mm^2 those sit ~0.26 mm apart,
            # so a 0.5 mm cut separates the two sides to within a sample spacing.
            imp_pts = np.asarray(trimesh.sample.sample_surface(mesh_i, n_dense, seed=11)[0])
            m_dense = cKDTree(imp_pts / s_mm).query(dense_n, k=1, workers=-1)[0] * s_mm < 0.5
            m_pred = cKDTree(defect_gt).query(pred, k=1, workers=-1)[0] * s_mm < DEFECT_MM

            rows.append({
                "run": label, "id": sid, "n_dense": n_dense,
                "dense_spacing_mm": float(np.sqrt(mesh.area / n_dense)),
                "gt_p2s_max_mm": float(gt_d.max()),
                # accuracy -- named to line up with mesh_viz.signed_deviation
                "p2s_abs_median_mm": float(np.median(dist)),
                "p2s_abs_p95_mm": float(np.percentile(dist, 95)),
                "p2s_bias_mm": float(signed.mean()),
                "p2s_std_mm": float(signed.std()),
                "p2s_outside_pct": float(100.0 * (signed > 0).mean()),
                "p2s_defect_abs_median_mm": float(np.median(dist[m_pred])) if m_pred.any() else np.nan,
                "n_pred_defect": int(m_pred.sum()),
                # mesh_viz.signed_deviation on the identical points, for the audit
                "dev_abs_median_mm": float(np.median(np.abs(dev))),
                "dev_abs_p95_mm": float(np.percentile(np.abs(dev), 95)),
                "dev_bias_mm": float(dev.mean()),
                "dev_std_mm": float(dev.std()),
                "dev_outside_pct": float(100.0 * (dev > 0).mean()),
                # coverage + its floor
                "s2p_mean_mm": float(s2p.mean()),
                "s2p_hd95_mm": float(np.percentile(s2p, 95)),
                "s2p_floor_mean_mm": float(floor.mean()),
                "s2p_floor_hd95_mm": float(np.percentile(floor, 95)),
                "s2p_defect_mean_mm": float(s2p[m_dense].mean()),
                "s2p_defect_hd95_mm": float(np.percentile(s2p[m_dense], 95)),
                "s2p_floor_defect_mean_mm": float(floor[m_dense].mean()),
                "defect_pct_of_surface": float(100.0 * m_dense.mean()),
            })
            print(f"  {sid} ok (ground truth to mesh {gt_d.max():.2e}mm, dense sampling "
                  f"spacing {rows[-1]['dense_spacing_mm']:.2f}mm)")
    return pd.DataFrame(rows)


def _mesh_for(sid, raw_root, nq, trimesh, gt_pts, s_mm):
    """The true surface, in the cache's normalised frame scaled to mm.

    Rebuilt through `normal_quality.truth_for`, which already re-runs
    prepare_skullfix's pipeline AND verifies it reproduces the cache, so the
    frame is not guessed here.
    """
    import nrrd
    from skimage import measure

    seed = nq._task_seed(sid, raw_root)
    _, _, mesh_c = nq._dense_with_faces(
        os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2)
    dense_d, _, _ = nq._dense_with_faces(
        os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2 + 1)
    centroid = dense_d.mean(axis=0)
    scale = float(np.max(np.linalg.norm(dense_d - centroid, axis=1)))
    # cache frame is (x - centroid)/scale; work in mm = that * s_mm, and s_mm == scale
    mesh_c.vertices = (np.asarray(mesh_c.vertices) - centroid) / scale * s_mm
    # The implant's own surface, same frame -- needed to label the dense samples.
    import defect_mask_audit as dma
    mesh_i, _ = dma._mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
    mesh_i.vertices = (np.asarray(mesh_i.vertices) - centroid) / scale * s_mm
    return mesh_c, mesh_i


def report(df):
    for label, g in df.groupby("run", sort=False):
        print(f"\n{'=' * 84}\n{label}   {g['id'].nunique()} skulls"
              f"   dense sampling {int(g.n_dense.iloc[0]):,} points/{g.dense_spacing_mm.mean():.2f}mm"
              f"\n{'=' * 84}")
        print("1. accuracy   predicted points -> true surface   floor = 0")
        print(f"   |distance| median {g.p2s_abs_median_mm.mean():6.3f}mm   p95 {g.p2s_abs_p95_mm.mean():6.3f}mm"
              f"   signed bias {g.p2s_bias_mm.mean():+.3f}mm   std {g.p2s_std_mm.mean():.3f}mm"
              f"   outside {g.p2s_outside_pct.mean():.1f}%")
        print(f"   predicted points inside the defect: median "
              f"{g.p2s_defect_abs_median_mm.mean():.3f}mm"
              f" ({g.n_pred_defect.mean():.0f} points)")
        print("\n2. coverage   true surface -> nearest predicted point   floor > 0 "
              "(a prediction is only 6144 points)")
        print(f"   {'':16}{'whole surface':>15}{'defect':>12}")
        print(f"   {'prediction':16}{g.s2p_mean_mm.mean():>15.3f}{g.s2p_defect_mean_mm.mean():>12.3f}")
        print(f"   {'floor (GT itself)':16}{g.s2p_floor_mean_mm.mean():>15.3f}{g.s2p_floor_defect_mean_mm.mean():>12.3f}")
        print(f"   {'above the floor':16}{g.s2p_mean_mm.mean()-g.s2p_floor_mean_mm.mean():>15.3f}"
              f"{g.s2p_defect_mean_mm.mean()-g.s2p_floor_defect_mean_mm.mean():>12.3f}")
        print(f"   the defect is {g.defect_pct_of_surface.mean():.1f}% of the surface")
        print("\n3. audit: the true surface against signed_deviation's fitted plane, "
              "ON THE SAME POINTS")
        print(f"   {'':18}{'signed_deviation':>18}{'p2s (true surface)':>20}{'ratio':>10}")
        for name, a, b in [("|deviation| median", g.dev_abs_median_mm.mean(), g.p2s_abs_median_mm.mean()),
                           ("|deviation| p95", g.dev_abs_p95_mm.mean(), g.p2s_abs_p95_mm.mean()),
                           ("signed bias", g.dev_bias_mm.mean(), g.p2s_bias_mm.mean()),
                           ("std", g.dev_std_mm.mean(), g.p2s_std_mm.mean()),
                           ("outside %", g.dev_outside_pct.mean(), g.p2s_outside_pct.mean())]:
            r = f"{a / b:.2f}x" if abs(b) > 0.1 else "different in kind"
            print(f"   {name:18}{a:>18.3f}{b:>20.3f}{r:>10}")


def self_test():
    """Analytic geometry + trimesh's own brute force. No project data touched."""
    import trimesh
    rng = np.random.default_rng(0)
    print("=== self-test: exact point-to-triangle distance ===")

    # 1. a plane: the analytic answer is |z|. Two triangles tile a square.
    plane = trimesh.Trimesh(
        vertices=np.array([[-10., -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]))
    q = np.c_[rng.uniform(-8, 8, 500), rng.uniform(-8, 8, 500), rng.uniform(-5, 5, 500)]
    d, s, _ = point_to_mesh(q, plane, k=4)
    err = np.abs(d - np.abs(q[:, 2])).max()
    print(f"1. plane    : largest error {err:.2e}mm  {'ok' if err < 1e-9 else 'FAIL'}"
          f"   sign correct {100*np.mean(np.sign(s) == np.sign(q[:,2])):.0f}%")

    # 2. a sphere, against a brute-force implementation -- independent, and it
    #    needs no extra dependency
    sph = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    u = rng.normal(size=(200, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    q2 = u * rng.uniform(30, 70, 200)[:, None]
    d2, s2, _ = point_to_mesh(q2, sph, k=8)
    _, d_ref, _ = trimesh.proximity.closest_point_naive(sph, q2)
    e2 = np.abs(d2 - d_ref).max()
    print(f"2. sphere vs brute force: largest difference {e2:.2e}mm  "
          f"{'ok' if e2 < 1e-8 else 'FAIL'}")
    inside = np.linalg.norm(q2, axis=1) < 50
    print(f"   sign: points inside should be negative -> "
          f"{100*np.mean((s2 < 0) == inside):.0f}% correct")

    # 3. points lying on the surface must be at distance 0
    on, _ = trimesh.sample.sample_surface(sph, 500, seed=3)
    d3, _, _ = point_to_mesh(np.asarray(on), sph, k=8)
    print(f"3. on-surface: largest distance {d3.max():.2e}mm  "
          f"{'ok' if d3.max() < 1e-9 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--n", type=int, default=8, help="how many skulls (the same cohort as "
                                                    "roughness.py)")
    ap.add_argument("--n-dense", type=int, default=N_DENSE)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    df = analyse(REPO, args.runs, n_skulls=args.n, n_dense=args.n_dense)
    report(df)

    out = os.path.join(REPO, args.out)
    if os.path.exists(out):
        # dtype={"id": str} is required, and astype(str) does not substitute for it:
        # ids carry a leading zero ('083'), pandas reads them back as the integer 83,
        # and str(83) is '83'. The key then fails to match and the old rows survive
        # as duplicates -- silently, because their other columns are identical. This
        # file is where that was first noticed, at twice the expected row count.
        old = pd.read_csv(out, dtype={"id": str})
        KEY = ['run', 'id']
        k_old = old[KEY].astype(str).apply(tuple, axis=1)
        k_new = set(df[KEY].astype(str).apply(tuple, axis=1))
        df = pd.concat([old[~k_old.isin(k_new)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
