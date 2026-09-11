"""Thesis figure: nearest-neighbour spacing on case 039, ground truth vs four losses.

Reads saved point clouds only -- no training, no inference -- and writes one PNG.
All five panels share one camera, one axis box and one colour range, so colour
and size compare directly across panels. Arrays are matched by skull id, never
by position: case 039 sits at a different row in the cache than in the fold-0
predictions.

Colour and clump rate come from the same spacing array. The colour range only
clamps what is drawn; the rate is computed on the unclamped values.

    python thesis/figure_scripts/make_spacing_case039.py
    python thesis/figure_scripts/make_spacing_case039.py --overwrite
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join("data", "cache", "skullfix_pairs_4096_6144.npz")
PREDS = os.path.join("experiments_log", "preds_fold0.npz")
OUT = os.path.join("reports", "preview", "thesis_spacing_case039_v1.png")
CASE = "039"
N_POINTS = 6144

# (panel label, key in PREDS); None means the ground truth from CACHE.
PANELS = (("(a) Complete ground truth", None),
          ("(b) CD", "cd_only"),
          ("(c) CD + DCD", "lr_fix_only"),
          ("(d) CD + repulsion", "cd_rep05_full"),
          ("(e) CD + DCD + repulsion", "rep_w05"))

# Clump rates the figure was checked against. A drift above TOLERANCE_PP stops
# the script before anything is drawn.
EXPECTED_PCT = (0.0, 19.9, 10.8, 3.7, 1.4)
TOLERANCE_PP = 0.1
CLUMP_MM = 2.0                     # strict: spacing < 2.0 mm counts as clumped

CMIN, CMAX, TICKS = 0.0, 6.0, (0, 2, 4, 6)
EYE = dict(x=0.0, y=-1.5, z=1.15)
MARKER_SIZE, OPACITY = 1.9, 0.9
BOX_PAD = 0.02                     # fraction of the largest joint extent, each side
# One magnification for all five scenes, applied to the aspect ratio so the eye
# stays exactly at EYE; same as moving the camera closer along that direction.
# Calibrated on rendered pixels: the tallest cloud fills 86% of its panel, unclipped.
ZOOM = 1.30

BACKGROUND, INK, INK_2 = "#FFFFFF", "#222222", "#555555"
FONT = "Arial, Helvetica, 'DejaVu Sans', sans-serif"

# 1500 x 850 CSS px at scale 2 -> 3000 x 1700 px, tagged 462 dpi so it lands at
# 16.5 cm; 30 CSS px titles then print at about 9 pt.
WIDTH, HEIGHT, SCALE, DPI = 1500, 850, 2, 462
TITLE_SIZE, SUB_SIZE, BAR_TITLE_SIZE, TICK_SIZE = 30, 26, 26, 24
MARGIN_PX, TITLE_BAND_PX = 10, 80
NOTE_OFFSET_PX = 78                # threshold note, below the colour bar's centre


def _row(ids, source):
    hit = np.flatnonzero(np.asarray(ids).astype(str) == CASE)
    if hit.size != 1:
        raise KeyError(f"skull {CASE!r} appears {hit.size} times in {source}")
    return int(hit[0])


def load_clouds():
    """Five (label, points, spacing_mm, clump_pct) tuples, in panel order."""
    pairs = np.load(os.path.join(REPO, CACHE))
    preds = np.load(os.path.join(REPO, PREDS))
    i, j = _row(pairs["ids"], CACHE), _row(preds["ids"], PREDS)
    scale = float(pairs["scale_mm"][i])
    if not np.isclose(scale, float(preds["scale_mm"][j])):
        raise ValueError(f"scale_mm disagrees for {CASE}: {scale} in the cache, "
                         f"{float(preds['scale_mm'][j])} in the predictions")
    print(f"skull {CASE}: row {i} of the cache, row {j} of the predictions, scale {scale:.4f} mm")

    out = []
    for label, key in PANELS:
        points = np.asarray(pairs["gt"][i] if key is None else preds[key][j], dtype=np.float64)
        if points.shape != (N_POINTS, 3) or not np.isfinite(points).all():
            raise ValueError(f"{label}: bad points, shape {points.shape}")
        spacing_mm = cKDTree(points).query(points, k=2)[0][:, 1] * scale
        clump_pct = 100.0 * np.mean(spacing_mm < CLUMP_MM)
        out.append((label, points, spacing_mm, clump_pct))
    return out


def check_rates(clouds):
    bad = []
    for (label, _, spacing, pct), want in zip(clouds, EXPECTED_PCT):
        diff = pct - want
        print(f"  {label:28s} clumped {pct:.1f}%  (raw {pct:.4f}, expected {want:.1f}, "
              f"diff {diff:+.3f} pp; spacing {spacing.min():.2f}-{spacing.max():.2f} mm)")
        if abs(diff) > TOLERANCE_PP:
            bad.append(label)
    if bad:
        sys.exit(f"stopped: clump rate off by more than {TOLERANCE_PP} pp for {', '.join(bad)}")


def shared_scene(clouds):
    """One scene dict for every panel: joint axis box, equal scaling, fixed camera."""
    pts = np.concatenate([p for _, p, _, _ in clouds])
    lo, hi = pts.min(0), pts.max(0)
    pad = BOX_PAD * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad

    # From this oblique view the joint box centre projects below the middle of
    # the outline, so zooming in cuts off the skull base. Move the one shared
    # centre onto the joint outline as the camera sees it.
    view = np.array([EYE["x"], EYE["y"], EYE["z"]])
    view /= np.linalg.norm(view)
    right = np.cross([0.0, 0.0, 1.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    centre = (lo + hi) / 2
    for d in (right, up):
        t = (pts - centre) @ d
        centre = centre + d * (t.max() + t.min()) / 2

    # Symmetric about that centre and still holding every point, with the same
    # scene units per data unit on all three axes.
    half = np.maximum(hi - centre, centre - lo)
    ratio = ZOOM * 2 * half / float((hi - lo).max())
    lo, hi = centre - half, centre + half

    def axis(a, b):
        return dict(range=[float(a), float(b)], autorange=False, visible=False,
                    showgrid=False, zeroline=False, showbackground=False)

    return dict(xaxis=axis(lo[0], hi[0]), yaxis=axis(lo[1], hi[1]), zaxis=axis(lo[2], hi[2]),
                aspectmode="manual", aspectratio=dict(x=float(ratio[0]), y=float(ratio[1]),
                                                      z=float(ratio[2])),
                bgcolor=BACKGROUND,
                camera=dict(eye=EYE, center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1),
                            projection=dict(type="perspective")))


def build(clouds):
    import plotly.graph_objects as go

    scene = shared_scene(clouds)
    fig = go.Figure()
    row_h = (HEIGHT - 2 * MARGIN_PX) / 2

    def cell(n):
        """Paper-coordinate box of grid cell n (0..5, row-major) and its scene domain."""
        r, c = divmod(n, 3)
        top = 1 - (MARGIN_PX + r * row_h) / HEIGHT
        bottom = 1 - (MARGIN_PX + (r + 1) * row_h) / HEIGHT
        return (c / 3, (c + 1) / 3), (bottom, top), (bottom, top - TITLE_BAND_PX / HEIGHT)

    for n, (label, p, spacing, pct) in enumerate(clouds):
        (x0, x1), (_, top), dom_y = cell(n)
        name = "scene" if n == 0 else f"scene{n + 1}"
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", scene=name, hoverinfo="skip",
            marker=dict(size=MARKER_SIZE, opacity=OPACITY, color=spacing, colorscale="Viridis",
                        cmin=CMIN, cmax=CMAX, showscale=(n == 0),
                        colorbar=_colorbar(cell(5)) if n == 0 else None)))
        fig.update_layout({name: dict(scene, domain=dict(x=[x0, x1], y=list(dom_y)))})
        fig.add_annotation(text=label, x=(x0 + x1) / 2, y=top, xref="paper", yref="paper",
                           xanchor="center", yanchor="top", showarrow=False,
                           font=dict(family=FONT, size=TITLE_SIZE, color=INK))
        fig.add_annotation(text=f"Clumped points: {pct:.1f}%", x=(x0 + x1) / 2,
                           y=top - (TITLE_SIZE + 10) / HEIGHT, xref="paper", yref="paper",
                           xanchor="center", yanchor="top", showarrow=False,
                           font=dict(family=FONT, size=SUB_SIZE, color=INK_2))

    (x0, x1), _, (bottom, top) = cell(5)
    fig.add_annotation(text=f"Clumped points: spacing &lt; {CLUMP_MM:g} mm",
                       x=(x0 + x1) / 2, y=(bottom + top) / 2 - NOTE_OFFSET_PX / HEIGHT,
                       xref="paper", yref="paper", xanchor="center", yanchor="top",
                       showarrow=False, font=dict(family=FONT, size=TICK_SIZE, color=INK_2))

    fig.update_layout(width=WIDTH, height=HEIGHT, showlegend=False,
                      paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
                      margin=dict(l=0, r=0, t=0, b=0))
    _assert_shared(fig, len(clouds))
    return fig


def _colorbar(box):
    """Horizontal shared colour bar, level with the clouds in the sixth cell."""
    (x0, x1), _, (bottom, top) = box
    return dict(orientation="h", x=(x0 + x1) / 2, xanchor="center",
                y=(bottom + top) / 2, yanchor="middle", len=0.62 * (x1 - x0), lenmode="fraction",
                thickness=30, outlinewidth=0, tickvals=list(TICKS),
                ticktext=[str(t) for t in TICKS], ticks="outside", ticklen=8, tickwidth=2,
                tickcolor=INK, tickfont=dict(family=FONT, size=TICK_SIZE, color=INK),
                title=dict(text="Nearest-neighbour spacing (mm)", side="top",
                           font=dict(family=FONT, size=BAR_TITLE_SIZE, color=INK)))


def _assert_shared(fig, n):
    """Every scene must carry the identical camera, axis ranges and aspect ratio."""
    ref = fig.layout.scene
    keys = lambda s: (s.camera.to_plotly_json(), s.aspectratio.to_plotly_json(), s.aspectmode,
                      tuple(s.xaxis.range), tuple(s.yaxis.range), tuple(s.zaxis.range))
    for k in range(2, n + 1):
        if keys(fig.layout[f"scene{k}"]) != keys(ref):
            raise AssertionError(f"scene{k} differs from scene")
    print(f"shared scene x{n}: eye {ref.camera.eye.to_plotly_json()}, "
          f"x {np.round(ref.xaxis.range, 4).tolist()}, y {np.round(ref.yaxis.range, 4).tolist()}, "
          f"z {np.round(ref.zaxis.range, 4).tolist()}, aspect "
          f"{ {a: round(v, 4) for a, v in ref.aspectratio.to_plotly_json().items()} }")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, OUT))
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite to replace it")

    clouds = load_clouds()
    check_rates(clouds)
    png = build(clouds).to_image(format="png", width=WIDTH, height=HEIGHT, scale=SCALE)

    # Kaleido writes RGBA: composite onto white and keep RGB.
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    Image.alpha_composite(Image.new("RGBA", rgba.size, BACKGROUND), rgba).convert("RGB") \
        .save(args.out, dpi=(DPI, DPI))
    print(f"-> {os.path.relpath(args.out, REPO)}  {WIDTH * SCALE} x {HEIGHT * SCALE} px RGB, "
          f"{DPI} dpi, colour range {CMIN:g}-{CMAX:g} mm")


if __name__ == "__main__":
    main()
