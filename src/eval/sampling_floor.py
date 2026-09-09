"""Measure the error floor that representing a skull with N points imposes.

Every metric here compares two point clouds, but the ground truth is only a
SAMPLE of a continuous surface, and so is any prediction. Two independent
samplings of the SAME surface differ even though neither carries model error.
That difference is the floor: no model can score below it, and it is what makes a
comparison against voxel-domain results apples-to-oranges rather than a defeat.

Method: read the complete skull's volume, run marching cubes once, sample its
surface twice with different seeds, farthest-point-sample each draw down to N,
and score one against the other exactly as a prediction is scored.

It depends on the data and the point count only -- no model, no weights, no
split -- so it never needs recomputing when the models change, and running it
over all 100 skulls (the default) makes it fold-independent by construction.

    python src/eval/sampling_floor.py                          # 100 skulls, 6144 points
    python src/eval/sampling_floor.py --n-out 3072,6144,12288  # floor against point count
    python src/eval/sampling_floor.py --n-skulls 2             # smoke test
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import paths                    # every data path is written down once

import fpsample
import nrrd
import numpy as np
import pandas as pd
import trimesh
from skimage import measure

RAW = os.path.join(REPO, paths.RAW_ROOT, "complete_skull")


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

    print(f"\n{len(files)} complete skulls x {args.repeats} independent pairs"
          f"  ->  {os.path.relpath(args.out, REPO)}")
    print(f"\n{'points':>8}{'CD_t floor':>12}{'(std)':>9}{'one-way':>10}"
          f"{'HD95 floor':>12}{'(std)':>9}")
    print("-" * 60)
    for n_out, g in df.groupby("n_out"):
        print(f"{n_out:>8}{g['CD_t_mm'].mean():>12.3f}{g['CD_t_mm'].std(ddof=1):>9.3f}"
              f"{g['one_way_mm'].mean():>10.3f}{g['HD95_mm'].mean():>12.3f}"
              f"{g['HD95_mm'].std(ddof=1):>9.3f}")
    print("\nMillimetres. NO MODEL IS INVOLVED -- this is one mesh sampled twice, and "
          "no model can score below it.")
    print("It depends only on the data and the point count, so it does not need "
          "recomputing when the models change.")


if __name__ == "__main__":
    main()
