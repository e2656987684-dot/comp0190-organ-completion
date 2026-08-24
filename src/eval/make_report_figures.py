"""Export the progress-report figures as PNGs, straight from the real runs.

Every image the slides use is produced here, from `experiments/` and the data
cache -- nothing is redrawn by hand, so a slide can never drift from what the
code actually outputs. Re-run after new experiments and the deck's figures
update with it.

    python src/eval/make_report_figures.py [--out reports/figures]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

# ⚠️ THIS LIST NO LONGER RUNS AS-IS (2026-08-24).
# The first four entries are the pre-LR-fix runs, which were reclassified as
# erroneous experiments on 2026-08-21 and whose weights were then pruned -- see
# experiments_log/README.md. Training is not bit-reproducible on GPU, so they
# cannot be recreated; the PNGs those runs produced are now tracked in git
# instead (reports/figures/, force-added past .gitignore).
#
# Before reusing this script for thesis figures, point RUNS at runs that still
# have weights: lr_fix_only, rep_w05, cd_only, cd_rep05_full, cd_rep05_r2,
# tie_qk, notext, pp_attn. That is a narrative choice (which runs the figure
# tells its story with), so it is deliberately left to whoever writes that
# chapter rather than silently rewritten here.
RUNS = [
    ("baseline", "baseline_es20"),
    ("dcd_w3", "msn_skullfix/dcd_w3"),
    ("dcd_l2", "msn_skullfix/dcd_l2"),
    ("lr_fix", "msn_skullfix/lr_fix_only"),
    ("rep_w05", "msn_skullfix/rep_w05"),
    ("cd_rep05", "msn_skullfix/cd_rep05"),
]
# The density figure drops dcd_l2: it is the same story as baseline and a fourth
# panel only makes each one narrower.
DENSITY_RUNS = ["baseline", "lr_fix", "rep_w05"]
SHOW_SKULL = None          # None -> first validation skull
SCALE = 2                  # PNG upscale, so text stays sharp when projected
C_PRED = "#1565C0"
WARN_HEX = "#B73B1F"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import mesh_viz as mv
    import report as rp

    runs = rp.load_runs(REPO, RUNS)
    data = np.load(os.path.join(REPO, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    text = np.load(os.path.join(REPO, "data", "cache", "bert_skull.npy"))

    def save(fig, name, w=1400, h=None):
        p = os.path.join(args.out, name)
        fig.write_image(p, width=w, height=h or fig.layout.height or 500, scale=SCALE)
        print(f"  {name}")
        return p

    # ---- 1. training curves, with the learning-rate drops marked -------------
    save(rp.fig_curves(runs, height=420), "curves.png", w=1500)

    # ---- 2. metrics table (reuse the cached CSV if the runs have not moved) --
    csv = os.path.join(REPO, "experiments_log", "eval_all_runs.csv")
    if os.path.exists(csv):
        eval_df = pd.read_csv(csv)
    else:
        eval_df = rp.eval_runs(REPO, runs)
        eval_df.to_csv(csv, index=False)
    summary = rp.summarise(eval_df)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print("  summary.csv")

    # ---- 3. per-skull spread -----------------------------------------------
    save(rp.fig_per_skull(eval_df, "CD_t_mm", height=400), "per_skull.png", w=1000)

    # ---- 4. predictions for the density + mesh figures ----------------------
    val = runs[0].meta["val_ids"]
    k = int(np.where(ids == (SHOW_SKULL or val[0]))[0][0])
    preds = {}
    # One model for all of DENSITY_RUNS, so they must share a topology -- swapping
    # weights across architectures loads without error and draws the wrong picture.
    shown = [r for r in runs if r.label in DENSITY_RUNS]
    archs = {r.arch_key for r in shown}
    if len(archs) > 1:
        raise ValueError("DENSITY_RUNS spans several architectures "
                         f"({sorted(r.arch_label for r in shown)}); this figure "
                         "reuses one model and cannot mix them")
    cfg = rp.arch_config(msn, archs.pop())
    model = msn.build_model(cfg)
    for r in shown:
        model.load_weights(r.weights)
        x = [inputs[k][None]] + ([text[None]] if cfg.use_text else [])
        preds[r.label] = model.predict(x, batch_size=1, verbose=0)[0]
    s = float(scales[k])

    items = [(gt[k], "ground truth")] + [(preds[n], n) for n in DENSITY_RUNS]
    save(mv.fig_spacing_grid(items, s, height=430,
                             title=f"skull_{ids[k]} — nearest-neighbour spacing "
                                   f"(dark = clumped, colour scale locked)"),
         "density.png", w=1600)

    # Problem-statement version: ground truth against the starting point only.
    # The four-panel figure already contains the fix, so using it to introduce the
    # problem gives the answer away two slides early.
    save(mv.fig_spacing_grid([(gt[k], "ground truth"), (preds["baseline"], "baseline")],
                             s, height=430, sp_max=6.0,
                             title=f"skull_{ids[k]} — nearest-neighbour spacing "
                                   f"(dark = points sitting on top of each other)"),
         "density_problem.png", w=1000)

    best = DENSITY_RUNS[-1]
    save(mv.fig_meshes([(mv.pc_to_mesh(gt[k], s), "ground truth"),
                        (mv.pc_to_mesh(preds[best], s), best)],
                       title=f"skull_{ids[k]} — reconstructed surface", height=460),
         "mesh.png", w=1100)

    # ---- 5. defective input -> prediction -> ground truth --------------------
    # Three panels, not an overlay. Overlaying the prediction on the input hides
    # the input completely (the prediction covers the same surface plus the
    # defect), so the one thing the figure exists to show -- the hole being
    # filled -- becomes invisible.
    import plotly.graph_objects as go  # noqa: E402
    from plotly.subplots import make_subplots

    # Point the camera AT the defect. A fixed viewpoint shows whichever side of
    # the skull it happens to face, and on most of these that is intact bone --
    # all three panels then look identical and the figure says nothing.
    from scipy.spatial import cKDTree
    d_to_input = cKDTree(inputs[k]).query(gt[k], k=1)[0]
    defect = gt[k][d_to_input > np.percentile(d_to_input, 97)]
    v = defect.mean(0) - gt[k].mean(0)
    v = v / (np.linalg.norm(v) + 1e-9) * 1.9
    print(f"  (defect centroid direction {np.round(v / 1.9, 2)}, "
          f"{(d_to_input > 0.05).mean() * 100:.1f}% of GT points are defect-side)")

    # Colour the defect region rather than relying on the hole being visible.
    # A point cloud viewed face-on at a defect still reads as solid -- the
    # surrounding points fill the silhouette -- so three uncoloured panels look
    # identical no matter where the camera is.
    # The defect is a REGION OF SPACE, defined once from the ground truth. Masking
    # predicted points by their own distance to the input instead would colour the
    # model's error everywhere -- predictions sit ~6 mm off the surface, which is
    # already past any threshold that isolates a defect -- and the panel would
    # show scattered blue over the whole skull rather than the filled hole.
    THR = 0.05                                   # normalised; ~5 mm
    defect_gt = d_to_input > THR
    in_defect = cKDTree(gt[k][defect_gt]).query(preds[best], k=1)[0] < THR
    base = dict(size=1.4, color="#C9CDD2", opacity=0.5)
    panels = [
        (gt[k], defect_gt, "Ground truth — missing region in red", "#E64A19"),
        (preds[best], in_defect, f"Prediction ({best}) — points inside that region", C_PRED),
    ]
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.02,
                        specs=[[{"type": "scatter3d"}] * len(panels)],
                        subplot_titles=[t for _, _, t, _ in panels])
    for i, (pts, mask, _, col) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter3d(x=pts[~mask, 0], y=pts[~mask, 1], z=pts[~mask, 2],
                                   mode="markers", marker=base, hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter3d(x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                                   mode="markers", hoverinfo="skip",
                                   marker=dict(size=2.6, color=col)), row=1, col=i)
    scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                 zaxis_visible=False,
                 camera=dict(eye=dict(x=float(v[0]), y=float(v[1]), z=float(v[2]))))
    fig.update_layout(height=460, showlegend=False,
                      title=dict(text=f"skull_{ids[k]} — the defect region, "
                                      f"grey = surface already present in the input",
                                 y=0.97, yanchor="top"),
                      margin=dict(l=0, r=0, t=95, b=0),
                      **{f"scene{i if i > 1 else ''}": scene
                         for i in range(1, len(panels) + 1)})
    for a in fig.layout.annotations:
        a.y = min(a.y, 0.9)
    save(fig, "completion.png", w=1100)

    # ---- 6. schematic: the case DCD is blind to ------------------------------
    # A drawing, not data. The blind spot is a statement about the matching, and
    # no rendering of a real skull shows it -- the two coincident predictions are
    # one pixel apart and the argmin that lets them off is invisible.
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("DCD: no penalty (count = 1 each)",
                                        "Repulsion: a force that separates them"))
    gtx, gty = [0.25, 0.75], [0.62, 0.62]
    px, py = [0.48, 0.52], [0.34, 0.34]
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=gtx, y=gty, mode="markers+text", text=["GT", "GT"],
                                 textposition="top center", showlegend=False,
                                 marker=dict(size=17, color="#2E7D32")), row=1, col=col)
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", showlegend=False,
                                 marker=dict(size=17, color=C_PRED)), row=1, col=col)
        for i in (0, 1):                      # each GT matches a different prediction
            fig.add_annotation(x=px[i], y=py[i], ax=gtx[i], ay=gty[i], xref=f"x{col}",
                               yref=f"y{col}", axref=f"x{col}", ayref=f"y{col}",
                               showarrow=True, arrowhead=2, arrowwidth=1.6,
                               arrowcolor="#7A8794")
    for dx in (-1, 1):                        # repulsion arrows, right panel only
        fig.add_annotation(x=0.5 + dx * 0.22, y=0.34, ax=0.5 + dx * 0.05, ay=0.34,
                           xref="x2", yref="y2", axref="x2", ayref="y2",
                           showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=WARN_HEX)
    fig.add_annotation(x=0.5, y=0.16, text="two predictions on top of each other",
                       xref="x", yref="y", showarrow=False,
                       font=dict(size=13, color="#5B6673"))
    fig.add_annotation(x=0.5, y=0.16, text="pushed apart to the target spacing",
                       xref="x2", yref="y2", showarrow=False,
                       font=dict(size=13, color="#5B6673"))
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0.05, 0.8], visible=False, scaleanchor=None)
    fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=45, b=10))
    save(fig, "dcd_blindspot.png", w=1200)

    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

# NOTE: reports/figures/pred_vs_gt.png is NOT generated here. It is exported by
# hand from MSN_train_skullfix.ipynb's show_pred_vs_gt cell (plotly's camera icon)
# and copied in, because the interactive figure is rotated to a viewpoint worth
# keeping and this script has no way to know which one that was.
