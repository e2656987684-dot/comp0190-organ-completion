"""Render one skull as shaded meshes -- input, completion, ground truth.

For looking, not measuring: "is the completed skull a plausible skull, and is the
defect actually filled?" A scatter plot of 6144 loose points cannot answer that, a
shaded surface can. Three panels side by side sharing one camera.

`--truth` adds the two raw volumes, marching-cubed at their native 0.475 mm
voxels, giving four panels:

    truth: defective | input (4096 pts) | completed (6144 pts) | truth: complete
    \_______________/  \_____________________________________/  \_______________/
     0.475 mm voxels            ~4 mm point spacing              0.475 mm voxels

Warning: the outer panels look dramatically crisper, and that gap is the
REPRESENTATION, not the model. Compare 1 with 2 to see what the point cloud
costs, 2 with 3 to see what the model adds, 3 with 4 for the total. Reading 4
against 3 as "the model is bad" is the one wrong way to look at this figure. It is
the sampling floor stated as a picture rather than a number.

`--truth-step` decimates the truth surface FOR DISPLAY, not the data. Even the
lightest setting resolves more finely than the ground-truth points are spaced, so
the point of the figure survives any of them; the default is the one that is
comfortable to open.

The two rules from mesh_viz still apply, and neither is a formality:

  1. These meshes are for looking ONLY. Reconstruction inflates the shape by the
     same order as the model's own error, so every metric keeps coming from the
     raw point clouds.
  2. Do not use `--preset` to compare two runs. Each smoothing knob changes how
     smooth the surface LOOKS, so tuning per figure lets a parameter change
     masquerade as a model improvement. Comparisons use the locked RECON.

`--res` is exposed and the smoothing knobs are not, because they are different in
kind: `radius_mm` / `sigma` / `taubin` change appearance, while `res` changes
fidelity -- raising it resolves the same isosurface more finely and cannot make a
rough surface look smooth. So `--res 160` is an honest view of the same
reconstruction; `--preset heavy` is a different reconstruction.

Output is PNG by default -- about 1 MB, opens natively in an editor, and what a
figure has to be anyway. `--html` additionally writes the interactive version,
worth it to rotate the skull, since all panels share one camera.

    python src/eval/mesh_preview.py                        # best run, its first val skull
    python src/eval/mesh_preview.py --skull 070 --truth    # + the two raw volumes
    python src/eval/mesh_preview.py --skull 070 --html     # also the rotatable version
    python src/eval/mesh_preview.py --preset heavy         # exploration only

Needs a GPU (one 187M model), so restart the notebook kernel first. Produces no
number that goes into any table.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import paths                    # every data path is written down once
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

# Anchor the picture to the numbers: a surface that looks fine can still be the
# worse model, and the whole density result is invisible in a mesh.
FROZEN = os.path.join("experiments_log", "eval_all_runs.csv")
SHOW_COLS = ["defect_cov_mm", "CD_t_mm", "HD95_mm", "clump_%", "defect_n_pred"]


def truth_meshes(sid, raw_root, step):
    """The defective and complete volumes as meshes, in the point clouds' frame.

    The transform is prepare_skullfix's "fix A": it comes from the DEFECTIVE
    dense cloud and is applied to everything, which is the whole reason the pair
    lines up at all (normalising each cloud on its own put them in different
    coordinate systems -- the bug that started this project). Recomputed here
    through normal_quality's helpers rather than re-derived, so there is one
    implementation of it, and checked against the cached input cloud by the
    caller: a wrong transform would not look wrong, because every panel
    auto-scales to its own data.

    `step` decimates the surface FOR DISPLAY ONLY (see the module docstring).
    """
    import nrrd
    import trimesh
    from skimage import measure

    import normal_quality as nq

    seed = nq._task_seed(sid, raw_root)
    # step_size stays 1 here: this cloud defines the transform and has to
    # reproduce prepare_skullfix exactly. Only the rendered surface is decimated.
    dense_d, _, _ = nq._dense_with_faces(
        os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"), 16384, 0.5, seed * 2 + 1)
    centroid = dense_d.mean(axis=0)
    scale = float(np.max(np.linalg.norm(dense_d - centroid, axis=1)))

    out = []
    for sub_dir in ("defective_skull", "complete_skull"):
        vol, hdr = nrrd.read(os.path.join(raw_root, sub_dir, f"{sid}.nrrd"))
        v, f, _, _ = measure.marching_cubes(vol, level=0.5, step_size=step)
        spacing = np.asarray(hdr["space directions"], dtype=np.float64)
        out.append(trimesh.Trimesh(vertices=((v @ spacing) - centroid) / scale, faces=f))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cd_rep05_full_f0",
                    help="run directory under experiments/ ('msn_skullfix/' is added if "
                         "you give a bare name). Default is the best configuration.")
    ap.add_argument("--skull", default=None,
                    help="skull id, e.g. 083. Default: that run's first validation skull.")
    ap.add_argument("--res", type=int, default=None,
                    help="distance-field grid; default = the locked mesh_viz.RECON value "
                         "(128). Fidelity, not appearance -- see the module docstring.")
    ap.add_argument("--preset", choices=["raw", "light", "default", "heavy"], default=None,
                    help="⚠️ EXPLORATION ONLY. Changes how smooth the surface looks, so a "
                         "figure made with it must never be put beside another run.")
    ap.add_argument("--truth", action="store_true",
                    help="also render the defective and complete VOLUMES (raw nrrd, "
                         "0.475 mm voxels) either side of the two point-cloud panels. "
                         "⚠️ the outer panels look far crisper because of the "
                         "representation, not the model -- read the module docstring.")
    ap.add_argument("--truth-step", type=int, default=3,
                    help="marching-cubes step for the truth surfaces, DISPLAY ONLY. "
                         "Measured four-panel file size: step 2 = 64 MB (0.95 mm), "
                         "step 3 = 39 MB (1.42 mm, default), step 4 = 31 MB (1.90 mm). "
                         "All of them out-resolve the 4.03 mm point spacing.")
    ap.add_argument("--raw-root", default=os.path.join(REPO, paths.RAW_ROOT))
    ap.add_argument("--camera", default="defect", choices=["defect", "default"],
                    help="defect (the default) faces the hole; 'default' is the view the "
                         "existing figures use, which is nearly orthogonal to the defect "
                         "direction and does not show it.")
    ap.add_argument("--html", action="store_true",
                    help="also write the interactive HTML (20-40 MB). Worth it to rotate "
                         "the skull -- all panels share one camera. ⚠️ VS Code cannot "
                         "preview a file that size; you need a real browser.")
    ap.add_argument("--out", default=None,
                    help="output path WITHOUT extension "
                         "(default reports/preview/<run>_<skull>)")
    ap.add_argument("--device", default="/GPU:0")
    args = ap.parse_args()

    rel = args.run if "/" in args.run else os.path.join("msn_skullfix", args.run)
    weights = os.path.join(REPO, "experiments", rel, "best.h5")
    if not os.path.exists(weights):
        raise SystemExit(f"no checkpoint at {os.path.relpath(weights, REPO)}\n"
                         f"   runs that do have one: " +
                         ", ".join(sorted(os.listdir(os.path.join(REPO, "experiments",
                                                                  "msn_skullfix")))))

    import report as rp
    import mesh_viz as mv

    run = rp.Run(REPO, rel)
    data = np.load(os.path.join(REPO, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]

    sid = args.skull or run.meta["val_ids"][0]
    if sid not in set(ids):
        raise SystemExit(f"skull {sid} is not in the data")
    if sid not in set(run.meta["val_ids"]):
        print(f"Warning: skull {sid} is in {run.label}'s TRAINING set, so it will of "
              f"course look better. Do not take it as representative. That run validates "
              f"on: {' '.join(run.meta['val_ids'][:8])} ...")
    k = int(np.where(ids == sid)[0][0])
    s = float(scales[k])

    # ---- the numbers first ----
    frozen = os.path.join(REPO, FROZEN)
    if os.path.exists(frozen):
        df = pd.read_csv(frozen, dtype={"id": str})        # '083' must not become 83
        row = df[(df.run == run.label) & (df.id == sid)]
        if len(row):
            r = row.iloc[0]
            print(f"skull_{sid} / {run.label}  (from {FROZEN}, "
                  f"defect definition: {r.get('defect_def', '?')})")
            print("  " + "  ".join(f"{c}={r[c]:.3f}" for c in SHOW_COLS if c in r))
            print("  Warning: the figure below CANNOT show the strongest of these numbers. "
                  "A mesh smooths\n  point density away, which is why the notebook has a "
                  "separate density diagnostic.\n")

    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn

    cfg = rp.arch_config(msn, run.arch_key)
    x = [inputs[k][None]]
    if cfg.use_text:
        x.append(np.load(os.path.join(REPO, paths.BERT_CACHE))[None])
    with tf.device(args.device):
        model = msn.build_model(cfg)
        model.load_weights(weights)
        pred = model.predict(x, batch_size=1, verbose=0)[0]
    print(f"inference done ({run.arch_label})")

    kw = dict(mv.PRESETS[args.preset]) if args.preset else {}
    if args.res:
        kw["res"] = args.res
    res = kw.get("res", mv.RECON["res"])

    print(f"reconstructing three panels (res={res}"
          + (f", preset={args.preset} -- EXPLORATION ONLY" if args.preset
             else ", smoothing locked")
          + ")...")
    items = [(mv.pc_to_mesh(inputs[k], s, **kw), "input (defective, 4096 pts)"),
             (mv.pc_to_mesh(pred, s, **kw), f"completed — {run.label}"),
             (mv.pc_to_mesh(gt[k], s, **kw), "ground truth (6144 pts)")]

    if args.truth:
        if not os.path.isdir(args.raw_root):
            raise SystemExit(f"no raw volumes under {args.raw_root}\n"
                             f"   --truth needs the nrrd files; without it only the point "
                             f"cloud cache is used.")
        print(f"reading the raw volumes and running marching cubes "
              f"(step={args.truth_step})...")
        m_def, m_comp = truth_meshes(sid, args.raw_root, args.truth_step)
        # A wrong transform would not LOOK wrong, since every panel autoscales to its
        # own data. So check it here: the input cloud was sampled off the defective
        # surface, so their bounding boxes have to coincide.
        gap = float(np.abs(np.array(m_def.bounds) -
                           np.array([inputs[k].min(0), inputs[k].max(0)])).max())
        print(f"  frame check: bounding boxes differ by at most {gap:.4f} normalised units"
              + ("  ok" if gap < 0.05
                 else "  -- too large, the transform may be wrong; do not draw conclusions"))
        items = [(m_def, "truth: defective (nrrd, 0.475mm)")] + items[:2] + \
                [(m_comp, "truth: complete (nrrd, 0.475mm)")]
    for m, lbl in items:
        print(f"  {lbl:34} {len(m.faces):8,d} faces")

    note = f"preset={args.preset} — EXPLORATION ONLY, do NOT compare runs" if args.preset \
        else "smoothing locked (mesh_viz.RECON)"
    if args.truth:
        note += " · outer panels = raw volumes; the crispness gap is the REPRESENTATION, not the model"
    # Face the defect by default: the whole point here is whether the hole got
    # filled, and the "default" camera is nearly orthogonal to the defect direction,
    # which renders a defective and a complete skull almost identically.
    fig = mv.fig_meshes(items, f"skull_{sid} — {run.label} — res={res}, {note}",
                        height=600, camera=args.camera)

    stem = args.out or os.path.join(REPO, "reports", "preview", f"{run.label}_{sid}")
    stem = os.path.splitext(stem)[0]
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    # PNG by default: it opens natively in an editor, and a figure has to be one
    # anyway. Width scales with the panel count, or four panels come out squashed.
    png = stem + ".png"
    fig.write_image(png, width=max(1200, 600 * len(items)), height=700, scale=2)
    print(f"\n-> {os.path.relpath(png, REPO)}   ({os.path.getsize(png) / 1e6:.2f} MB)"
          f"   opens directly in the editor")
    if args.html:
        html = stem + ".html"
        fig.write_html(html, include_plotlyjs=True)   # plotly.js inlined, so it works offline
        print(f"-> {os.path.relpath(html, REPO)}   ({os.path.getsize(html) / 1e6:.1f} MB)"
              f"   for rotating it; needs a real browser")
    print("   For looking only, never for a metric: reconstruction moves the original "
          "points by a median 5.3mm, the same order as the model's own error.")
    print("   The input panel should show the hole and the completed one should not. "
          "The reconstruction deliberately does NOT use Poisson, which would close it.")
    if args.truth:
        print("   Warning: the outer panels are raw volumes at 0.475mm voxels and the "
              "inner two are point-cloud\n   reconstructions at about 4mm spacing. The "
              "crispness gap between them is the REPRESENTATION,\n   not model quality -- "
              "the sampling floor stated as a picture. 1 against 2 is what the point\n"
              "   cloud costs, 2 against 3 is what the model adds.")


if __name__ == "__main__":
    main()
