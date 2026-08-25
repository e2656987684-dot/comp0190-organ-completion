"""Measure the error floor imposed by representing a skull with N points.

WHAT IT MEASURES
  Every metric in this project compares two point clouds, but the ground truth is
  only a SAMPLE of a continuous surface -- 6144 points drawn from a marching-cubes
  mesh -- and so is any prediction. Two independent samplings of the SAME surface
  therefore differ from each other even though neither has any model error at all.
  That difference is the floor: no model, however good, can score below it.

  Method: read the complete skull's volume, run marching cubes ONCE, then sample
  its surface twice with different seeds, farthest-point-sample each draw down to
  N points, and score one against the other exactly as a prediction is scored.

WHY IT HAS TO BE WRITTEN DOWN
  The thesis needs this number for three separate arguments:
    * a reader who puts CD_t 6.36 mm next to AutoImplant's HD95 1.52 mm (a 0.45 mm
      voxel grid) concludes the method is poor. The floor is what makes that an
      apples-to-oranges comparison rather than a defeat;
    * it bounds what is left to win. Defect coverage sits at 3.24 mm against a
      one-directional floor of 2.31 mm -- 0.93 mm of headroom -- which is why
      chasing that column further was dropped;
    * "our numbers cannot be placed beside voxel-domain results" is otherwise an
      excuse. With this it is a measurement.
  Until 2026-08-25 the figures (CD_t 4.43 mm, HD95 3.79 mm) existed only as prose
  in the devlog, from a script that was never committed -- the same trap that made
  the roughness and attention-collapse numbers unciteable. Measured properly over
  all 100 skulls they come out HIGHER: CD_t 4.619 +- 0.252, HD95 3.974 +- 0.214,
  one-directional 2.310 +- 0.126 mm. That gap is 7.5 standard errors, so it is a
  methodological difference from the lost script rather than sampling variation --
  and it moves in the direction that credits the model LESS, not more.

⚠️ k 折之后**不需要**重跑。
  This depends on the DATA and the point count only: no model, no weights, no
  train/validation split. Changing the split cannot move it. Run it over all 100
  skulls (the default) and the number is fold-independent by construction --
  which is exactly why it is worth measuring that way rather than over one
  validation set of 20.

    python src/eval/sampling_floor.py                      # all 100 skulls, 6144 points
    python src/eval/sampling_floor.py --n-out 3072,6144,12288   # floor vs point count
    python src/eval/sampling_floor.py --n-skulls 2         # smoke test
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))

import fpsample
import nrrd
import numpy as np
import pandas as pd
import trimesh
from skimage import measure

RAW = os.path.join(REPO, "data", "14161307", "SkullFix", "training_set", "complete_skull")


def load_mesh_mm(path, level=0.5):
    """nrrd -> marching cubes -> trimesh, in physical millimetres.

    The `space directions` matmul is not optional: this dataset's voxels are
    anisotropic and sheared (~0.451/0.446/0.625 mm), so working in index space
    would stretch the skull along one axis and every distance below would be
    wrong. Same step as prepare_skullfix.py.
    """
    volume, header = nrrd.read(path)
    verts, faces, _, _ = measure.marching_cubes(volume, level=level)
    verts_mm = verts @ np.asarray(header["space directions"], dtype=np.float64)
    return trimesh.Trimesh(vertices=verts_mm, faces=faces)


def sample_once(mesh, n_dense, n_out, seed):
    """One independent draw: dense surface sample, then farthest-point down to n_out.

    Mirrors the data pipeline exactly (prepare_skullfix.py does the same two steps
    with the same libraries), so the floor is the floor of the data this project
    actually trains and evaluates on -- not of some idealised sampler.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, n_dense, seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    return pts[fpsample.fps_sampling(pts, n_out, start_idx=0)]


def score(a, b):
    """CD_t / HD95 / one-directional mean, in whatever unit the inputs are (mm here).

    Same definitions as report.metrics_from_points, so the floor can be read
    straight against the CD_t_mm and HD95_mm columns of eval_all_runs.csv.
    """
    from scipy.spatial import cKDTree

    d_ab = cKDTree(b).query(a, k=1, workers=-1)[0]
    d_ba = cKDTree(a).query(b, k=1, workers=-1)[0]
    return {"CD_t_mm": d_ab.mean() + d_ba.mean(),
            "one_way_mm": d_ab.mean(),
            "HD95_mm": max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-out", default="6144",
                    help="output points per cloud; comma-separate to sweep. 6144 is what this "
                         "project uses, and the sweep answers 'how much would more points buy' "
                         "without training anything.")
    ap.add_argument("--n-dense", type=int, default=16384,
                    help="dense surface points before farthest-point sampling, as in prepare_skullfix")
    ap.add_argument("--n-skulls", type=int, default=0, help="0 = every complete skull found")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent sampling PAIRS per skull; >1 averages out the draw itself")
    ap.add_argument("--raw-root", default=RAW)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments_log", "sampling_floor.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_root, "*.nrrd")))
    if not files:
        raise SystemExit(f"no .nrrd under {args.raw_root}")
    if args.n_skulls:
        files = files[:args.n_skulls]
    n_outs = [int(v) for v in args.n_out.split(",")]

    rows = []
    for k, path in enumerate(files, 1):
        sid = os.path.splitext(os.path.basename(path))[0]
        mesh = load_mesh_mm(path)                       # marching cubes ONCE per skull
        for n_out in n_outs:
            for rep in range(args.repeats):
                a = sample_once(mesh, args.n_dense, n_out, seed=1000 * rep + 1)
                b = sample_once(mesh, args.n_dense, n_out, seed=1000 * rep + 2)
                rows.append({"id": sid, "n_out": n_out, "rep": rep, **score(a, b)})
        del mesh
        print(f"  [{k}/{len(files)}] skull_{sid}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(files)} 颗完整颅骨 × {args.repeats} 对独立采样  ->  {os.path.relpath(args.out, REPO)}")
    print(f"\n{'点数':>8}{'CD_t 地板':>12}{'(std)':>9}{'单向':>10}{'HD95 地板':>12}{'(std)':>9}")
    print("-" * 60)
    for n_out, g in df.groupby("n_out"):
        print(f"{n_out:>8}{g['CD_t_mm'].mean():>12.3f}{g['CD_t_mm'].std(ddof=1):>9.3f}"
              f"{g['one_way_mm'].mean():>10.3f}{g['HD95_mm'].mean():>12.3f}"
              f"{g['HD95_mm'].std(ddof=1):>9.3f}")
    print("\n单位 mm。**没有模型参与** —— 这是同一张网格采两次的差异，任何模型都低不过它。")
    print("⚠️ k 折之后不需要重跑：它只依赖数据和点数，与划分、权重都无关。")


if __name__ == "__main__":
    main()
