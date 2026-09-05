r"""Render one skull as a shaded mesh -- input, completion, ground truth -- to an HTML file.

WHAT THIS IS FOR
  Looking. "Is the completed skull a plausible skull, and is the defect actually
  filled?" A scatter plot of 6144 loose points cannot answer that; a shaded
  surface can. Three panels side by side, sharing one camera:

      input (defective, 4096 pts)  ->  completed (6144)  ->  ground truth (6144)

--truth: THE DEFECTIVE AND COMPLETE SKULL AS THE DATASET ACTUALLY HOLDS THEM
  Panels 1 and 3 above are built from POINT CLOUDS -- the same lossy
  representation the model works in (4096 / 6144 points, ~4-5 mm spacing).
  `--truth` adds the two raw volumes, marching-cubed at their native 0.475 mm
  voxels, giving four panels:

      truth: defective  |  input (4096 pts)  |  completed (6144 pts)  |  truth: complete
      \________________/   \_______________________________________/   \_______________/
        0.475 mm voxels              ~4 mm point spacing                 0.475 mm voxels

  ⚠️⚠️ THE OUTER PANELS WILL LOOK DRAMATICALLY CRISPER, AND THAT GAP IS THE
  REPRESENTATION, NOT THE MODEL. Compare 1 vs 2 to see what the point cloud
  costs, 2 vs 3 to see what the model adds, 3 vs 4 for the total. Reading 4
  against 3 as "the model is bad" is the one wrong way to look at this figure.
  It is the same fact the sampling floor states numerically: 73% of the reported
  CD_t is the representation (4.619 of 6.355 mm), and 6144 points at ~4 mm
  cannot resolve what 0.45 mm voxels do -- matching AutoImplant's density would
  need ~485k points, which `tf.eye(dec_seed)` makes structurally impossible.

  ⚠️ `--truth-step` is a DISPLAY decimation of the truth surface (marching cubes
  step_size), not of the data. Measured on skull_070, whole four-panel file:
      step 1  2,327,456 faces/panel   unusable (~63 MB for that panel alone)
      step 2    574,620   0.95 mm     64 MB   sluggish over a remote link
      step 3    251,738   1.42 mm     39 MB   default
      step 4    138,646   1.90 mm     31 MB   lightest
  Even step 4 resolves 2.1x finer than the 4.03 mm the ground-truth points are
  spaced at, so the point of the figure survives any of these; the default is
  the one that is comfortable to open.

⚠️ THE TWO RULES FROM mesh_viz STILL APPLY, AND THEY ARE NOT FORMALITIES
  1. These meshes are for looking ONLY -- never compute a metric on them.
     Reconstruction inflates the shape: the original points sit a median 5.3 mm
     (p95 12.0) from the reconstructed surface, the same order as the model's own
     CD_t. Every Chamfer/DCD number keeps coming from the raw point clouds.
  2. Do not use `--preset` to compare two runs. Each smoothing knob changes how
     smooth the surface LOOKS, so tuning per figure lets a parameter change
     masquerade as a model improvement. Comparisons use the locked mesh_viz.RECON.

WHY `--res` IS EXPOSED AND THE OTHER KNOBS ARE NOT
  They are different kinds of knob and the distinction is the whole reason this
  script can offer a "nicer" picture without cheating:
    radius_mm / sigma / taubin  change how smooth the surface looks  -> APPEARANCE
    res                         distance-field grid density          -> FIDELITY
  Raising res resolves the same isosurface more finely; it does not make a rough
  surface look smooth. So `--res 160` is still an honest view of the same
  reconstruction, while `--preset heavy` is a different reconstruction.

WHY AN HTML FILE AND NOT A NOTEBOOK CELL
  Measured on skull_083: three panels serialise to ~26 MB at res=128 and ~40 MB
  at res=160. Inline, that goes into the .ipynb and into git. The training
  notebook already carries a comment about a 60 MB output incident from exactly
  this. A file is opened once and thrown away; reports/preview/ is gitignored.

  Sizes and times measured on this machine (skull_083, three panels):
      --res  96    ~17 MB    ~2 s      lighter, good enough to see the shape
      --res 128    ~26 MB    ~3 s      the locked RECON value (default)
      --res 160    ~40 MB    ~5 s      finest; slowest to open in a browser

    python src/eval/mesh_preview.py                        # best run, its first val skull
    python src/eval/mesh_preview.py --skull 070 --truth    # + the two raw volumes
    python src/eval/mesh_preview.py --skull 053 --res 160  # the outlier, finest grid
    python src/eval/mesh_preview.py --preset heavy         # ⚠️ exploration only

  Needs a GPU (one 187M model, ~15.5 GiB) -- restart the notebook kernel first.
  ⚠️ k 折之后：不用重跑。这是看一眼的工具，不产出任何入库的数字。
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

# Anchor the picture to the numbers. "数字先行" is a project rule precisely
# because a surface that looks fine can still be the worse model -- the whole
# repulsion result is invisible in a mesh.
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
    ap.add_argument("--run", default="cd_rep05_full",
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
    ap.add_argument("--raw-root", default=os.path.join(REPO, "data", "14161307",
                                                       "SkullFix", "training_set"))
    ap.add_argument("--out", default=None,
                    help="output .html (default reports/preview/<run>_<skull>.html)")
    ap.add_argument("--device", default="/GPU:0")
    args = ap.parse_args()

    rel = args.run if "/" in args.run else os.path.join("msn_skullfix", args.run)
    weights = os.path.join(REPO, "experiments", rel, "best.h5")
    if not os.path.exists(weights):
        raise SystemExit(f"⛔ 没有权重: {os.path.relpath(weights, REPO)}\n"
                         f"   现存的 run: " +
                         ", ".join(sorted(os.listdir(os.path.join(REPO, "experiments",
                                                                  "msn_skullfix")))))

    import report as rp
    import mesh_viz as mv

    run = rp.Run(REPO, rel)
    data = np.load(os.path.join(REPO, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]

    sid = args.skull or run.meta["val_ids"][0]
    if sid not in set(ids):
        raise SystemExit(f"⛔ 数据里没有 skull {sid}")
    if sid not in set(run.meta["val_ids"]):
        print(f"⚠️  skull {sid} 在 {run.label} 的**训练集**里 —— 看着当然会更好，"
              f"别拿它当效果代表。该 run 的验证集: {' '.join(run.meta['val_ids'][:8])} …")
    k = int(np.where(ids == sid)[0][0])
    s = float(scales[k])

    # ---- 数字先行 ----
    frozen = os.path.join(REPO, FROZEN)
    if os.path.exists(frozen):
        df = pd.read_csv(frozen, dtype={"id": str})        # ⚠️ '083' 不能变成 83
        row = df[(df.run == run.label) & (df.id == sid)]
        if len(row):
            r = row.iloc[0]
            print(f"skull_{sid} / {run.label}（取自 {FROZEN}，口径 {r.get('defect_def', '?')}）")
            print("  " + "  ".join(f"{c}={r[c]:.3f}" for c in SHOW_COLS if c in r))
            print("  ⚠️ 下面那张图**看不见**这些数字里最硬的那条（密度/扎堆）—— mesh 会把点的"
                  "疏密抹平，这正是 MSN_surface_quality.ipynb 第 5 节存在的理由。\n")

    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn

    cfg = rp.arch_config(msn, run.arch_key)
    x = [inputs[k][None]]
    if cfg.use_text:
        x.append(np.load(os.path.join(REPO, "data", "cache", "bert_skull.npy"))[None])
    with tf.device(args.device):
        model = msn.build_model(cfg)
        model.load_weights(weights)
        pred = model.predict(x, batch_size=1, verbose=0)[0]
    print(f"推理完成（{run.arch_label}）")

    kw = dict(mv.PRESETS[args.preset]) if args.preset else {}
    if args.res:
        kw["res"] = args.res
    res = kw.get("res", mv.RECON["res"])

    print(f"重建三格（res={res}"
          + (f"、preset={args.preset} ⚠️ 探索用" if args.preset else "、平滑参数=锁定值")
          + ")…")
    items = [(mv.pc_to_mesh(inputs[k], s, **kw), "input (defective, 4096 pts)"),
             (mv.pc_to_mesh(pred, s, **kw), f"completed — {run.label}"),
             (mv.pc_to_mesh(gt[k], s, **kw), "ground truth (6144 pts)")]

    if args.truth:
        if not os.path.isdir(args.raw_root):
            raise SystemExit(f"⛔ 找不到原始体数据: {args.raw_root}\n"
                             f"   --truth 需要 nrrd；不带这个开关就只用点云缓存。")
        print(f"读原始体数据并 marching cubes（step={args.truth_step}）…")
        m_def, m_comp = truth_meshes(sid, args.raw_root, args.truth_step)
        # 变换错了不会「看起来不对」——每格都按自己的数据自动缩放。所以在这里核一次：
        # 输入点云是从 defective 那张表面上采下来的，两者的包围盒必须基本重合。
        gap = float(np.abs(np.array(m_def.bounds) -
                           np.array([inputs[k].min(0), inputs[k].max(0)])).max())
        print(f"  坐标系核对：真值网格与输入点云的包围盒最大差 {gap:.4f}（归一化单位）"
              + ("  ✅" if gap < 0.05 else "  ⚠️ 偏大，变换可能没对上，别据此下结论"))
        items = [(m_def, "truth: defective (nrrd, 0.475mm)")] + items[:2] + \
                [(m_comp, "truth: complete (nrrd, 0.475mm)")]
    for m, lbl in items:
        print(f"  {lbl:34} {len(m.faces):8,d} faces")

    note = f"preset={args.preset} — EXPLORATION ONLY, do NOT compare runs" if args.preset \
        else "smoothing locked (mesh_viz.RECON)"
    if args.truth:
        note += " · outer panels = raw volumes; the crispness gap is the REPRESENTATION, not the model"
    fig = mv.fig_meshes(items, f"skull_{sid} — {run.label} — res={res}, {note}", height=600)

    out = args.out or os.path.join(REPO, "reports", "preview", f"{run.label}_{sid}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs=True)      # 内嵌 plotly.js，离线也能开
    print(f"\n-> {os.path.relpath(out, REPO)}   ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("   浏览器打开，可以拖动旋转。三格共用一个相机。")
    print("   ⚠️ 只能看，不能算指标（重建把原始点外扩中位 5.3mm，和模型自己的 CD_t 同量级）。")
    print("   ⚠️ 输入那格应该看得见洞，补全那格应该没有 —— 重建**不**用 Poisson，"
          "正因为 Poisson 会把洞补掉。")
    if args.truth:
        print("   ⚠️⚠️ 最外两格是原始体数据（0.475mm 体素），中间两格是点云重建（~4mm 间距）。"
              "\n        它们之间的清晰度差距是**表示**，不是模型好坏 —— 这正是采样地板"
              "\n        「CD_t 里 73% 是地板」那句话的图像版。1↔2 看点云的代价，"
              "2↔3 看模型的贡献。")


if __name__ == "__main__":
    main()
