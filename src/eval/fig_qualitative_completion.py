"""Figure for the write-up: one case, defective input -> completion -> ground truth.

All three panels are rebuilt from point clouds by one reconstruction -- the
locked RECON settings on one shared voxel grid, so the blur is identical in
millimetres and not just in voxels. They share one camera, one view centre and
one fixed axis box, so size and orientation compare directly. Nothing is read
from the raw volumes, and no GPU or checkpoint is needed.

Arrays are matched by skull id, never by position: a skull sits at a different
index in the point-cloud cache (100 skulls) than in the fold-0 predictions (20).

    python src/eval/fig_qualitative_completion.py                  # case 039, oblique defect view
    python src/eval/fig_qualitative_completion.py --camera defect  # face the defect squarely
    python src/eval/fig_qualitative_completion.py --smoothing light # less smoothing, all else equal
"""

from __future__ import annotations

import argparse
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import numpy as np
from PIL import Image                  # Pillow, installed with matplotlib

import mesh_viz as mv
import paths

PREDS = os.path.join("experiments_log", "preds_fold0.npz")
LABELS = os.path.join("experiments_log", "defect_mask_labels.npz")
CONFIG = "cd_rep05_full"                       # CD + repulsion

# Written out so the figure states its own settings; must equal the locked RECON.
RECON = dict(res=128, radius_mm=6.0, sigma=2.5, taubin=60, pad=0.14)

# `light` swaps in mesh_viz's light preset (radius, blur, Taubin) for all three
# panels at once. Grid, padding, camera and zoom stay put, so the two versions
# compare at the same scale.
SMOOTHING = {"default": RECON, "light": {**RECON, **mv.PRESETS["light"]}}

SURFACE, BACKGROUND, INK = "#DDD8CC", "#FFFFFF", "#222222"
FONT = "Arial, Helvetica, 'DejaVu Sans', sans-serif"
PANEL_LABELS = ("(a) Defective input",
                "(b) Completed prediction<br>CD + repulsion",
                "(c) Complete ground truth")

# 2400 x 880 px, tagged 369 dpi so it lands at 16.5 cm; 26 CSS px of text at
# scale 2 prints at about 10 pt. Near-square panels, so the skulls fill them.
WIDTH, HEIGHT, SCALE, FONT_SIZE, LABEL_BAND, DPI = 1200, 440, 2, 26, 78, 369

# Brings the one shared camera closer along its fixed direction, for every panel
# at once. Calibrated per camera on rendered pixels so the skulls fill about 90% of
# their panels; `implant` is 7 degrees from `defect` and borrows its value.
ZOOM = {"defect_oblique": 1.724, "defect": 1.732, "implant": 1.732}


def _row(ids, case, source):
    hit = np.flatnonzero(np.asarray(ids).astype(str) == case)
    if hit.size != 1:
        raise KeyError(f"skull {case!r} appears {hit.size} times in {source}")
    return int(hit[0])


def load_case(case):
    """(inputs, prediction, ground truth, scale_mm, implant mask), matched by id."""
    pairs = np.load(os.path.join(REPO, paths.DATA_CACHE))
    preds = np.load(os.path.join(REPO, PREDS))
    labels = np.load(os.path.join(REPO, LABELS))
    i = _row(pairs["ids"], case, paths.DATA_CACHE)
    j = _row(preds["ids"], case, PREDS)
    if case not in labels.files:
        raise KeyError(f"no implant labels for skull {case!r} in {LABELS}")

    inputs, gt, pred = pairs["inputs"][i], pairs["gt"][i], preds[CONFIG][j]
    implant = labels[case].astype(bool)
    scale = float(pairs["scale_mm"][i])
    if not np.isclose(scale, float(preds["scale_mm"][j])):
        raise ValueError(f"scale_mm disagrees for {case}: {scale} in the cache, "
                         f"{float(preds['scale_mm'][j])} in the predictions")
    for name, arr, n in (("inputs", inputs, 4096), ("gt", gt, 6144), (CONFIG, pred, 6144)):
        if arr.shape != (n, 3):
            raise ValueError(f"{name} for {case} has shape {arr.shape}, expected ({n}, 3)")
    if implant.shape != (len(gt),):
        raise ValueError(f"implant mask has shape {implant.shape}, expected ({len(gt)},)")
    print(f"skull {case}: row {i} of the cache, row {j} of the predictions, "
          f"scale {scale:.3f} mm, {int(implant.sum())} implant points")
    return inputs, pred, gt, scale, implant


def camera_eye(kind, gt, implant, zoom):
    """A locked mesh_viz camera, or the implant direction at the defect camera's
    distance; divided by `zoom`."""
    locked = np.array(list(mv.CAMERAS["defect"].values()), dtype=float)
    toward = gt[implant].mean(0) - gt.mean(0)
    toward /= np.linalg.norm(toward)
    if kind == "implant":
        eye = toward * np.linalg.norm(locked)
    else:
        eye = np.array(list(mv.CAMERAS[kind].values()), dtype=float)
    eye = eye / zoom
    angle = np.degrees(np.arccos(np.clip(eye @ toward / np.linalg.norm(eye), -1, 1)))
    print(f"camera {kind}: eye {np.round(eye, 3)}, {angle:.1f} deg from the implant direction")
    return dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2]))


def build(case, camera, zoom=None, smoothing="default"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if RECON != mv.RECON:
        raise ValueError(f"RECON here {RECON} no longer matches mesh_viz.RECON {mv.RECON}")
    inputs, pred, gt, scale, implant = load_case(case)
    clouds = (inputs, pred, gt)

    # One grid for all three, so no panel is reconstructed at a finer voxel size.
    lo = np.min([c.min(0) for c in clouds], axis=0) - RECON["pad"]
    hi = np.max([c.max(0) for c in clouds], axis=0) + RECON["pad"]
    recon = SMOOTHING[smoothing]
    meshes = [mv.pc_to_mesh(c, scale, bounds=(lo, hi), **recon) for c in clouds]
    voxel = (hi - lo) / (RECON["res"] - 1) * scale
    print(f"{smoothing} smoothing, shared grid: voxel {' / '.join(f'{v:.3f}' for v in voxel)} "
          f"mm; faces " + ", ".join(str(len(m.faces)) for m in meshes))

    zoom = ZOOM[camera] if zoom is None else zoom
    eye = camera_eye(camera, gt, implant, zoom)

    # One cubic box, so the eye vector is a true direction in data space and every
    # panel has the same scale. It is centred on the complete skull's outline as
    # the camera sees it, perspective included: from any oblique view the
    # bounding-box centre sits off that outline, and zooming in then crops one side.
    e = np.array([eye["x"], eye["y"], eye["z"]])
    view = e / np.linalg.norm(e)
    right = np.cross([0.0, 0.0, 1.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    v_gt = meshes[2].vertices
    centre = (v_gt.max(0) + v_gt.min(0)) / 2
    for _ in range(4):
        half = 1.02 * max(np.abs(m.vertices - centre).max() for m in meshes)
        rel = (v_gt - centre) / (2 * half) - e          # scene units: the box is 1 wide
        depth = -rel @ view
        sx, sy = (rel @ right) / depth, (rel @ up) / depth
        mid = np.array([sx.max() + sx.min(), sy.max() + sy.min()]) / 2 * depth.mean()
        centre = centre + (right * mid[0] + up * mid[1]) * 2 * half
    half = 1.02 * max(np.abs(m.vertices - centre).max() for m in meshes)

    def axis(c):
        return dict(range=[c - half, c + half], autorange=False, visible=False,
                    showbackground=False, showgrid=False, zeroline=False)

    scene = dict(xaxis=axis(centre[0]), yaxis=axis(centre[1]), zaxis=axis(centre[2]),
                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=1), bgcolor=BACKGROUND,
                 camera=dict(eye=eye, center=dict(x=0, y=0, z=0),
                             up=dict(x=0, y=0, z=1), projection=dict(type="perspective")))

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        horizontal_spacing=0.0)
    for col, m in enumerate(meshes, start=1):
        v, f = m.vertices, m.faces
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=SURFACE, opacity=1.0, flatshading=False,
                                lighting=mv.MESH_LIGHTING, lightposition=mv.MESH_LIGHT_POSITION,
                                showscale=False, hoverinfo="skip"), row=1, col=col)
    for col, text in enumerate(PANEL_LABELS):
        fig.add_annotation(text=text, x=(col + 0.5) / 3, y=0, xref="paper", yref="paper",
                           xanchor="center", yanchor="top", yshift=-6, showarrow=False, align="center",
                           font=dict(family=FONT, size=FONT_SIZE, color=INK))
    fig.update_layout(scene=scene, scene2=scene, scene3=scene, showlegend=False,
                      paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
                      margin=dict(l=0, r=0, t=0, b=LABEL_BAND))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="039")
    ap.add_argument("--camera", default="defect_oblique", choices=sorted(ZOOM),
                    help="defect_oblique: the defect view turned aside so the vault reads "
                         "as a skull. defect: faces the defect squarely. implant: the "
                         "direction of the implant centroid, at the same distance")
    ap.add_argument("--out", default=None,
                    help="default reports/figures/qualitative_completion_case<case>.png, "
                         "with _<camera> and _<smoothing> appended for non-default choices")
    ap.add_argument("--smoothing", default="default", choices=sorted(SMOOTHING),
                    help="default: the locked RECON. light: mesh_viz's light preset, "
                         "applied to all three panels")
    args = ap.parse_args()
    suffix = "" if args.camera == ap.get_default("camera") else f"_{args.camera}"
    suffix += "" if args.smoothing == ap.get_default("smoothing") else f"_{args.smoothing}"
    out = args.out or os.path.join(REPO, "reports", "figures",
                                   f"qualitative_completion_case{args.case}{suffix}.png")
    png = build(args.case, args.camera, smoothing=args.smoothing).to_image(format="png", width=WIDTH, height=HEIGHT,
                                                  scale=SCALE)
    # Kaleido writes RGBA. Composite onto white and keep RGB, so no page or viewer
    # can show anything else behind the skulls, and tag the print resolution.
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    Image.alpha_composite(Image.new("RGBA", rgba.size, BACKGROUND), rgba).convert("RGB") \
        .save(out, dpi=(DPI, DPI))
    print(f"-> {os.path.relpath(out, REPO)}  {WIDTH * SCALE} x {HEIGHT * SCALE} px RGB, "
          f"{DPI} dpi, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
