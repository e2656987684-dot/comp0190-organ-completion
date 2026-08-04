"""
Build data/cache/skullfix_pairs_{n_in}_{n_out}.npz from the raw SkullFix nrrd volumes.

Reconstructed after the original script was lost to a machine reset (never
committed to git -- data/ and experiments/ are gitignored on purpose, but this
.py file should not have been). Rebuilt from what survived: the raw-data layout
and baseline nrrd->point-cloud logic in notebooks/explore_skull.ipynb, and the
fixes documented in notebooks/demo/MSN_train_skullfix.ipynb's "相对 demo 改了
什么" section (A and E below). If you find the original script, diff it against
this one before trusting either.

Raw layout (see notebooks/explore_skull.ipynb):
    data/14161307/SkullFix/training_set/complete_skull/*.nrrd
    data/14161307/SkullFix/training_set/defective_skull/*.nrrd
  paired by filename, e.g. 000.nrrd <-> 000.nrrd, 100 pairs total.

Run in the `comp0190-msn` conda env (has scikit-image + everything else this
needs; the old `comp0190` env this used to require has been removed):
    /root/miniconda3/envs/comp0190-msn/bin/python src/data/prepare_skullfix.py \
        --n-samples 0 --n-dense 16384 --n-in 4096 --n-out 6144 --workers 12 \
        --out data/cache/skullfix_pairs_4096_6144.npz

Two fixes over the naive per-file approach in explore_skull.ipynb's
`nrrd_to_point_cloud`:

  A. Pair-alignment. Normalising complete and defective INDEPENDENTLY (each
     with its own centroid/max-radius, as explore_skull.ipynb does) breaks
     correspondence: cropping a defect moves the centroid and shrinks the max
     radius, so the two clouds land in different frames even though they
     describe the same skull. Fixed here by deriving ONE similarity transform
     (centroid + scale) from the DEFECTIVE cloud only -- the only shape
     available at inference time -- and applying that same transform to both.

  E. Voxel spacing. explore_skull.ipynb runs marching_cubes directly on array
     indices and never looks at the nrrd header, so an anisotropic (and
     sheared) voxel spacing silently stretches the skull along one axis.
     Fixed here by multiplying marching-cubes vertices through the header's
     `space directions` matrix first, so every downstream distance is in
     physical space. The defective cloud's normalisation radius is saved per
     sample as `scale_mm`, so a normalised metric can be converted back to
     millimetres via `metric_mm = metric * scale_mm`.
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import fpsample
import nrrd
import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_ROOT = os.path.join(REPO_ROOT, "data", "14161307", "SkullFix", "training_set")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=RAW_ROOT, help="dir containing complete_skull/ and defective_skull/")
    ap.add_argument("--n-samples", type=int, default=0, help="0 = use every pair found under --raw-root")
    ap.add_argument("--n-dense", type=int, default=16384, help="surface points sampled before FPS downsampling")
    ap.add_argument("--n-in", type=int, default=4096, help="points kept for the defective input cloud")
    ap.add_argument("--n-out", type=int, default=6144, help="points kept for the complete ground-truth cloud")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--level", type=float, default=0.5, help="marching-cubes iso-level for the binary mask")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    return ap.parse_args()


def _volume_to_dense_points(nrrd_path, n_dense, level, seed):
    volume, header = nrrd.read(nrrd_path)
    verts, faces, _, _ = measure.marching_cubes(volume, level=level)
    # header["space directions"]: (3,3), row i = the physical-space vector for
    # a unit step along index axis i. Anisotropic and sheared here (measured
    # ~0.451/0.446/0.625 mm on the diagonal), so this must be a full matmul,
    # not a per-axis scalar multiply.
    spacing = np.asarray(header["space directions"], dtype=np.float64)
    verts_mm = verts @ spacing
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces)
    points, _ = trimesh.sample.sample_surface(mesh, n_dense, seed=seed)
    return np.asarray(points, dtype=np.float64)


def _fps(points, n_out, seed):
    if len(points) <= n_out:
        idx = np.random.RandomState(seed).choice(len(points), n_out, replace=True)
    else:
        idx = fpsample.fps_sampling(points, n_out, start_idx=0)
    return points[idx]


def process_one(task):
    complete_path, defective_path, n_dense, n_in, n_out, level, seed = task
    sid = os.path.splitext(os.path.basename(complete_path))[0]
    try:
        dense_complete = _volume_to_dense_points(complete_path, n_dense, level, seed * 2)
        dense_defective = _volume_to_dense_points(defective_path, n_dense, level, seed * 2 + 1)

        # Fix A: one transform derived from the DEFECTIVE cloud, applied to both.
        centroid = dense_defective.mean(axis=0)
        scale = float(np.max(np.linalg.norm(dense_defective - centroid, axis=1)))
        dense_defective_n = ((dense_defective - centroid) / scale).astype(np.float32)
        dense_complete_n = ((dense_complete - centroid) / scale).astype(np.float32)

        inp = _fps(dense_defective_n, n_in, seed)
        gt = _fps(dense_complete_n, n_out, seed + 1)
        return sid, inp, gt, scale, None
    except Exception as e:  # keep going on a single bad file, report it at the end
        return sid, None, None, None, str(e)


def main():
    args = parse_args()
    assert args.n_in <= args.n_dense and args.n_out <= args.n_dense, (
        f"--n-in/--n-out must be <= --n-dense ({args.n_dense})")

    complete_files = sorted(glob.glob(os.path.join(args.raw_root, "complete_skull", "*.nrrd")))
    if not complete_files:
        raise SystemExit(
            f"No .nrrd files under {args.raw_root}/complete_skull. "
            "Download/extract the SkullFix training_set there first "
            "(see notebooks/explore_skull.ipynb for the expected layout)."
        )
    if args.n_samples > 0:
        complete_files = complete_files[: args.n_samples]

    tasks = []
    for i, complete_path in enumerate(complete_files):
        defective_path = complete_path.replace("complete_skull", "defective_skull")
        if not os.path.exists(defective_path):
            print(f"skip {complete_path}: no matching defective file")
            continue
        tasks.append((complete_path, defective_path, args.n_dense, args.n_in, args.n_out,
                      args.level, args.seed + i))

    ids, inputs, gts, scales, failed = [], [], [], [], []
    with Pool(args.workers) as pool:
        for sid, inp, gt, scale, err in tqdm(pool.imap(process_one, tasks), total=len(tasks),
                                              desc="prepare_skullfix"):
            if err is not None:
                failed.append((sid, err))
                continue
            ids.append(sid)
            inputs.append(inp)
            gts.append(gt)
            scales.append(scale)

    if not ids:
        raise SystemExit(f"all {len(tasks)} pairs failed; see errors above")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        ids=np.array(ids),
        inputs=np.stack(inputs).astype(np.float32),
        gt=np.stack(gts).astype(np.float32),
        scale_mm=np.array(scales, dtype=np.float32),
    )

    print(f"\nsaved {len(ids)} pairs ({len(failed)} failed) -> {args.out}")
    print(f"input {inputs[0].shape}  gt {gts[0].shape}  scale_mm mean {np.mean(scales):.1f}")
    for sid, err in failed:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
