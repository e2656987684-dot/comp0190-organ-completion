r"""Can point normals be estimated well enough for surface reconstruction? (Poisson gate)

WHY THIS RUNS BEFORE ANY POISSON WORK
  Screened Poisson reconstruction does not take points, it takes points WITH
  ORIENTED NORMALS, and the normals have to be estimated from the cloud itself
  because a prediction comes with nothing else. This script measures whether
  that estimation is even possible on this data, because there is a specific
  reason to expect it is not:

      a skull is a shell 5-7 mm thick, and the clouds are sampled at ~4 mm

  So a neighbourhood large enough to fit a plane to also reaches the OPPOSITE
  sheet of the shell, whose surface faces the other way. Already measured, on
  ground truth, by `roughness.py`: the neighbourhood's spread along its own
  normal is 5.44 mm at k=8 and 7.87 mm at k=16 -- i.e. the same order as the
  shell thickness, at every k that is usable.

  If normals cannot be recovered, Poisson cannot work, no reconstruction floor
  needs measuring, and TODO 9 ends with a measured negative instead of a table
  of numbers dominated by a reconstructor. That is a cheap answer to buy: this
  script needs no GPU and no new dependencies (open3d is NOT installed, and
  putting it in the environment that holds TF 2.15 + numpy 1.26 is a risk worth
  avoiding unless something is going to be built on it).

TWO WAYS IT CAN FAIL, AND THEY ARE DIFFERENT
  unoriented   the fitted plane itself is wrong, because the neighbourhood
               straddles both sheets. Measured as the angle to the true normal
               ignoring sign: `ang_unoriented`.
  orientation  the plane is right but the SIGN propagates across the gap
               between the sheets and comes out inside-out over a patch.
               Measured by running the standard consistent-orientation pass
               (minimum spanning tree over the kNN graph, weight 1 - |ni.nj|)
               and counting points left pointing the wrong way: `frac_flipped`.

  Poisson needs both. `frac_nb_opposite` is the mechanism behind either one: the
  fraction of a point's neighbours that genuinely sit on the far sheet, known
  exactly here because the truth comes from the mesh, not from an estimate.

WHERE THE TRUTH COMES FROM
  The ground-truth points in the cache were sampled off a mesh, so their true
  normals are that mesh's face normals -- no closest-point query needed, and no
  approximation. `prepare_skullfix.py`'s pipeline is re-run here (marching cubes
  on the raw nrrd, the same anisotropic spacing matmul, the same seeds) to
  recover the mesh, the face index of every sampled point, and the normalisation
  that the cache is expressed in. That re-run is checked, not assumed: `scale`
  and the ground-truth points themselves must come back identical to the cache,
  and the script refuses to report anything if they do not.

⚠️ NOT a statement about the model. Every number here is measured on GROUND
  TRUTH points. A prediction's normals can only be worse -- it has clumping the
  ground truth does not (1.34% against 0.0%) -- so ground truth is the
  optimistic case and the right place to test feasibility.

    python src/eval/normal_quality.py [--n 8] [--self-test]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree

OUT_CSV = os.path.join("experiments_log", "normal_quality.csv")

# k for the normal fit. 6 is about the smallest that still defines a plane
# robustly; 16 is where roughness.py measured a 7.87 mm neighbourhood spread.
KS = (6, 8, 12, 16)

# A fitted plane more than this far from the truth is not a usable normal.
BAD_DEG = 20.0
ORIENT_K = 10          # kNN graph for the orientation pass


def estimate_normals(P, k):
    """Unoriented normals: smallest principal axis of each point's k neighbours.

    The point itself is INCLUDED in its own neighbourhood, which is what
    open3d/PCL do -- excluding it (as `mesh_viz.local_roughness` does, for a
    different purpose) would make the fit ignore the very point it describes.
    """
    idx = cKDTree(P).query(P, k=k, workers=-1)[1]
    X = P[idx] - P[idx].mean(1)[:, None, :]
    _, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X) / k)
    return evec[:, :, 0]


def orient_normals(P, N, k=ORIENT_K):
    """The standard consistent-orientation pass, so the test is realistic.

    Minimum spanning tree over the kNN graph with weight 1 - |ni.nj| (cheap
    edges join near-parallel normals), then flip signs while walking out from a
    seed. This is what open3d's orient_normals_consistent_tangent_plane does,
    and it is the step that a thin shell breaks: the two sheets are ~6 mm apart
    while neighbours are ~4 mm apart, so the tree can hop between sheets and
    carry the wrong sign across with it.

    Seeded at the point furthest from the centroid, pointed away from it -- on
    the outer sheet that is unambiguously outward.
    """
    n = len(P)
    idx = cKDTree(P).query(P, k=k + 1, workers=-1)[1][:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    # +1e-9: a weight of exactly 0 (parallel normals) would read as "no edge".
    w = 1.0 - np.abs(np.einsum("ni,ni->n", N[rows], N[cols])) + 1e-9
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)                       # symmetrise before the MST
    mst = minimum_spanning_tree(G)
    adj = mst + mst.T

    centre = P.mean(0)
    seed = int(np.argmax(np.linalg.norm(P - centre, axis=1)))
    out = N.copy()
    if np.dot(out[seed], P[seed] - centre) < 0:
        out[seed] = -out[seed]

    order, preds = breadth_first_order(adj, seed, directed=False,
                                       return_predecessors=True)
    for node in order[1:]:
        p = preds[node]
        if np.dot(out[p], out[node]) < 0:
            out[node] = -out[node]
    return out, len(order)


def angles(est, true):
    """Degrees. Unoriented ignores sign; oriented does not."""
    d = np.clip(np.einsum("ni,ni->n", est, true), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(d))), np.degrees(np.arccos(d))


# --------------------------------------------------------------------------- #
# ground truth: re-run prepare_skullfix's pipeline to recover mesh + normals
# --------------------------------------------------------------------------- #
def _dense_with_faces(nrrd_path, n_dense, level, seed):
    """prepare_skullfix._volume_to_dense_points, but keeping the face index."""
    import nrrd
    import trimesh
    from skimage import measure

    volume, header = nrrd.read(nrrd_path)
    verts, faces, _, _ = measure.marching_cubes(volume, level=level)
    spacing = np.asarray(header["space directions"], dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=verts @ spacing, faces=faces)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_dense, seed=seed)
    return np.asarray(pts, dtype=np.float64), np.asarray(face_idx), mesh


def _task_seed(sid, raw_root, base_seed=42):
    """The seed `prepare_skullfix.main()` actually handed this skull.

    ⚠️ It is NOT `--seed`. main() does `enumerate(sorted(complete_skull/*.nrrd))`
    and passes `args.seed + i`, so every skull got a different seed. Getting this
    wrong reproduces a valid but DIFFERENT sampling of the same surface, which is
    exactly what the guard in `analyse` caught the first time this ran (scale off
    by 3e-02 mm out of ~99 mm -- the mesh was right, the drawn points were not).

    `i` enumerates the FULL listing: a skull with no defective pair is skipped
    without renumbering the ones after it. So no filtering before enumerate.
    """
    files = sorted(glob.glob(os.path.join(raw_root, "complete_skull", "*.nrrd")))
    for i, f in enumerate(files):
        if os.path.splitext(os.path.basename(f))[0] == sid:
            return base_seed + i
    raise SystemExit(f"{sid}: 在 {raw_root}/complete_skull 下找不到对应的 .nrrd")


def truth_for(sid, raw_root, n_dense=16384, n_out=6144, level=0.5, base_seed=42):
    """The cache's ground-truth points, plus their exact face normals.

    Returns (points_normalised, normals, scale, checks). Everything is
    recomputed rather than looked up, so the caller can check whether the
    recomputation actually reproduced the cache -- see the module docstring.
    """
    import fpsample

    seed = _task_seed(sid, raw_root, base_seed)
    comp = os.path.join(raw_root, "complete_skull", f"{sid}.nrrd")
    defe = os.path.join(raw_root, "defective_skull", f"{sid}.nrrd")
    dense_c, face_c, mesh_c = _dense_with_faces(comp, n_dense, level, seed * 2)
    dense_d, _, _ = _dense_with_faces(defe, n_dense, level, seed * 2 + 1)

    # Fix A: the transform comes from the DEFECTIVE cloud and is applied to both.
    centroid = dense_d.mean(axis=0)
    scale = float(np.max(np.linalg.norm(dense_d - centroid, axis=1)))
    dense_c_n = ((dense_c - centroid) / scale).astype(np.float32)

    idx = fpsample.fps_sampling(dense_c_n, n_out, start_idx=0)
    pts = dense_c_n[idx]
    # A uniform scale + translation does not rotate anything, so face normals
    # are already valid in the normalised frame.
    normals = np.asarray(mesh_c.face_normals[face_c[idx]], dtype=np.float64)
    # `winding_consistent` is the one that matters: it is what makes the face
    # normals agree on which side is out, and therefore what makes the ORIENTED
    # comparison meaningful. `watertight` comes out False here and that is
    # expected, not a defect -- marching cubes leaves open edges where the skull
    # meets the edge of the volume. Recorded so the reader can see it was checked.
    checks = {"watertight": bool(mesh_c.is_watertight),
              "winding_consistent": bool(mesh_c.is_winding_consistent),
              "n_faces": int(len(mesh_c.faces))}
    return pts.astype(np.float64), normals, scale, checks


def analyse(repo, sids, raw_root):
    data = np.load(os.path.join(repo, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ids, gt, scales = data["ids"], data["gt"], data["scale_mm"]

    rows = []
    for sid in sids:
        j = int(np.where(ids == sid)[0][0])
        pts, true_n, scale, checks = truth_for(sid, raw_root)

        # ---- the pipeline must reproduce, or nothing below means anything ----
        d_scale = abs(scale - float(scales[j]))
        d_pts = float(np.abs(pts - gt[j]).max())
        if d_scale > 1e-3 or d_pts > 1e-5:
            raise SystemExit(
                f"{sid}: 复现 prepare_skullfix 失败 —— scale 差 {d_scale:.3e}、"
                f"GT 点最大差 {d_pts:.3e}。坐标系没对上，后面的法向真值全都无意义，"
                f"已中止。（检查 --raw-root、trimesh 版本、以及 prepare_skullfix 的 seed）")
        if not checks["winding_consistent"]:
            print(f"⚠️ {sid}: 网格绕行方向不一致 —— 面法向的**朝向**不可信，"
                  f"本行的 frac_flipped 要打折扣看")

        s_mm = float(scales[j])
        for k in KS:
            est = estimate_normals(pts, k)
            ang_u, _ = angles(est, true_n)
            oriented, visited = orient_normals(pts, est)
            _, ang_o = angles(oriented, true_n)

            # mechanism: how many of the k neighbours really are on the far sheet
            nb = cKDTree(pts).query(pts, k=k, workers=-1)[1]
            opp = (np.einsum("nki,ni->nk", true_n[nb], true_n) < 0).mean()
            kth = np.median(cKDTree(pts).query(pts, k=k, workers=-1)[0][:, -1]) * s_mm

            rows.append({
                "id": sid, "k": k, "n_points": len(pts),
                "ang_unoriented_median": float(np.median(ang_u)),
                "ang_unoriented_p90": float(np.percentile(ang_u, 90)),
                "frac_plane_bad": float((ang_u > BAD_DEG).mean()),
                "frac_flipped": float((ang_o > 90.0).mean()),
                "frac_nb_opposite": float(opp),
                "nb_radius_mm": float(kth),
                "mst_visited": int(visited),
                "watertight": checks["watertight"],
                "winding_consistent": checks["winding_consistent"],
            })
        print(f"  {sid} ✓  (scale 复现差 {d_scale:.2e}，GT 点复现差 {d_pts:.2e})")
    return pd.DataFrame(rows)


def report(df):
    print(f"\n{'=' * 92}\n法向估计质量（{df['id'].nunique()} 颗颅骨的 **GT 点**，"
          f"真值取自网格面法向）\n{'=' * 92}")
    head = (f"{'k':>4}{'邻域半径mm':>12}{'邻居在对面片的比例':>20}"
            f"{'未定向夹角中位':>16}{'平面就错了 >20°':>17}{'定向后翻转':>13}")
    print(head)
    print("-" * 96)
    for k, g in df.groupby("k"):
        print(f"{k:>4}{g.nb_radius_mm.mean():>12.2f}{100*g.frac_nb_opposite.mean():>19.1f}%"
              f"{g.ang_unoriented_median.mean():>15.1f}°"
              f"{100*g.frac_plane_bad.mean():>16.1f}%{100*g.frac_flipped.mean():>12.1f}%")

    best = df.loc[df.groupby("k")["frac_flipped"].mean().idxmin()] if len(df) else None
    fl = df.groupby("k")["frac_flipped"].mean()
    pb = df.groupby("k")["frac_plane_bad"].mean()
    k_best = int(fl.idxmin())
    print(f"\n  最好的一档是 k={k_best}：平面错误 {100*pb[k_best]:.1f}%、"
          f"定向翻转 {100*fl[k_best]:.1f}%")
    print(f"  参照：骨壳厚约 5~7mm，而上表「邻域半径」这一列就是邻域够到多远。")

    if fl.min() > 0.10 or pb.min() > 0.30:
        print("\n  ⛔ **闸门：不通过。** 法向估计在任何一档 k 上都不可用 —— "
              "Poisson 从这些点做不出正确的面。\n"
              "     → TODO 9 的 Step C（Poisson）不必做，也不必装 open3d；\n"
              "       这本身是可写进论文的测量结论：**输出分辨率不足以支撑表面重建**。\n"
              "     → Step B（点到面指标，用真 GT 网格）不受影响，照做。")
    elif fl.min() > 0.02:
        print("\n  ⚠️ **闸门：勉强。** 法向大体可用但翻转比例不低，"
              "Poisson 可能在局部出现内外翻面。\n"
              "     → 若继续做 Step C，重建结果必须逐颗目视检查，不能只看数字。")
    else:
        print("\n  ✅ **闸门：通过。** 法向可用，Step C（先量重建地板）可以做。")


def self_test():
    """Controls: the same code on geometry where the answer is known.

    Point of this: separate "the estimator is broken" from "this geometry is
    hard". A single-sheet sphere must come out clean; a two-sheet shell at the
    skull's own proportions must not.
    """
    rng = np.random.default_rng(0)
    print("=== 自检 ===")

    def sphere(n, r):
        v = rng.normal(size=(n, 3))
        return r * v / np.linalg.norm(v, axis=1, keepdims=True)

    # ① 单层球面：真法向 = 径向。必须几乎全对，且没有翻转。
    P = sphere(6144, 100.0)
    T = P / np.linalg.norm(P, axis=1, keepdims=True)
    est = estimate_normals(P, 8)
    au, _ = angles(est, T)
    ori, _ = orient_normals(P, est)
    _, ao = angles(ori, T)
    print(f"① 单层球面 r=100mm : 未定向夹角中位 {np.median(au):5.1f}°  "
          f"平面错误 {100*(au>BAD_DEG).mean():4.1f}%  翻转 {100*(ao>90).mean():4.1f}%"
          f"   {'✅' if (au>BAD_DEG).mean()<.05 and (ao>90).mean()<.02 else '❌ 估计器本身有问题'}")

    # ② 双层壳，间距 6mm，点间距约 4mm —— 颅骨的真实比例。必须明显退化。
    inner, outer = sphere(3072, 97.0), sphere(3072, 103.0)
    P2 = np.r_[inner, outer]
    T2 = np.r_[-inner / 97.0, outer / 103.0]          # 内层法向朝腔内
    est2 = estimate_normals(P2, 8)
    au2, _ = angles(est2, T2)
    ori2, _ = orient_normals(P2, est2)
    _, ao2 = angles(ori2, T2)
    sp = np.median(cKDTree(P2).query(P2, k=2)[0][:, 1])
    print(f"② 双层壳 6mm/间距{sp:.1f}mm: 未定向夹角中位 {np.median(au2):5.1f}°  "
          f"平面错误 {100*(au2>BAD_DEG).mean():4.1f}%  翻转 {100*(ao2>90).mean():4.1f}%"
          f"   {'✅ 如预期退化' if (au2>BAD_DEG).mean() > (au>BAD_DEG).mean() else '⚠️ 没退化，与预期不符'}")

    # ③ 法向必须是单位向量
    print(f"③ 单位长度        : 最大偏差 {np.abs(np.linalg.norm(est,axis=1)-1).max():.2e}  "
          f"{'✅' if np.abs(np.linalg.norm(est,axis=1)-1).max() < 1e-9 else '❌'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="只跑合成几何的对照，不碰数据")
    ap.add_argument("--from-run", default="msn_skullfix/cd_rep05_full",
                    help="取哪一轮的验证颅骨（只读 run.json，不建模型、不要 GPU）")
    ap.add_argument("--n", type=int, default=8, help="颅骨数（与 surface_quality / roughness 同批）")
    ap.add_argument("--raw-root", default=os.path.join(REPO, "data", "14161307",
                                                       "SkullFix", "training_set"))
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import report as rp
    sids = rp.Run(REPO, args.from_run).meta["val_ids"][:args.n]
    print(f"复现 prepare_skullfix 的管线以取得真值法向（{len(sids)} 颗，每颗 2 个体数据）…")
    df = analyse(REPO, sids, args.raw_root)
    report(df)

    out = os.path.join(REPO, args.out)
    if os.path.exists(out):
        old = pd.read_csv(out)
        # ⚠️ 键要先统一成字符串再比：`id` 是 '083' 这种带前导零的编号，
        # 写进 CSV 再读回来会被 pandas 解析成整数 83，于是 ('run', 83) 对不上
        # ('run', '083')，旧行被当成不同的行留下来 —— 实测重跑一次就变成 16 行。
        KEY = ['id', 'k']
        k_old = old[KEY].astype(str).apply(tuple, axis=1)
        k_new = set(df[KEY].astype(str).apply(tuple, axis=1))
        df = pd.concat([old[~k_old.isin(k_new)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
