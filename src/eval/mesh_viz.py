"""
Surface-quality visualisation and diagnostics for predicted point clouds.

WHAT THIS IS FOR
  Scatter plots of 6144 loose points make it very hard to judge whether a
  predicted skull surface is any good. This module turns a point cloud into a
  shaded mesh you can actually read, plus the two diagnostics that a mesh alone
  hides, plus the numbers that settle "did it really improve?" objectively.

  Intended workflow: render a model's output now, change the training code,
  render again, and compare -- both by eye and by `surface_stats`.

⚠ VISUALISATION ONLY -- NEVER COMPUTE METRICS ON THESE MESHES
  Reconstruction inflates the shape: measured on skull_083, the original points
  sit a median of 5.3 mm (p95 12.0 mm) from the reconstructed surface. That is
  the same order as the model's own CD_t (~7 mm), so a metric computed on the
  mesh would be dominated by reconstruction artefacts. All Chamfer/DCD numbers
  must keep coming from the raw point clouds via msn_skullfix.calc_cd/calc_dcd.

⚠ NEVER TUNE RECON PER-FIGURE WHEN COMPARING RUNS
  RECON below is deliberately a module-level constant, not a default argument
  you are invited to override. Every knob in it changes how smooth the result
  LOOKS, so tuning it per figure would let a parameter change masquerade as a
  model improvement. Fix it once, keep it fixed across every run you compare.

WHY THESE RECON VALUES (measured on 4096/6144-point skulls, ~99 mm radius)
  radius_mm=6.0  Each point becomes a ball; the surface is the r-isosurface of
      the distance field. Point spacing is ~3.5-4.3 mm median (p95 ~5-6 mm), and
      r below ~4 mm leaves the balls disconnected -- the skull renders as a
      sponge. r must also stay well under the defect's ~20 mm+ extent or the
      hole gets bridged and you lose the thing you are looking at. 5-6 mm is the
      usable window; 6 mm sits at the smooth end of it.
  sigma=2.5   Gaussian blur of the distance field, in voxels. Removes the
      "cobblestone" bumps left by individual balls. Without it you cannot see
      real surface waviness through the reconstruction's own texture.
  taubin=60   Taubin mesh smoothing. Taubin, NOT Laplacian: Laplacian shrinks
      the model steadily with each iteration, Taubin does not.
  res=128     Distance-field grid. 96 is noticeably faster and fine for a quick
      look; 128 is the default because ~1 s per cloud is already cheap.

  Counter-intuitive but measured: MORE smoothing makes model differences MORE
  visible, not less. At low smoothing both ground truth and prediction render as
  the same pile of balls -- the reconstruction artefact swamps the signal. The
  blur removes that texture and leaves the genuine low-frequency waviness, which
  is where predictions actually differ from ground truth.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure

# Locked reconstruction settings -- see the module docstring before touching.
RECON = {"res": 128, "radius_mm": 6.0, "sigma": 2.5, "taubin": 60, "pad": 0.14}

# Named smoothing levels for LOOKING AT ONE MODEL, e.g.
#   pc_to_mesh(pred, scale_mm, **PRESETS["raw"])
# or all of them at once via fig_smoothing_ladder().
#
# These exist for exploring what a surface actually looks like. They are NOT for
# comparing two models -- for that, leave the settings alone and let RECON apply,
# or every figure is reconstructed differently and the comparison means nothing.
#
# "raw" is the honest view of where the points are and nothing else; because
# each point becomes a ball, it mostly shows the reconstruction's own texture,
# and ground truth looks just as lumpy as any prediction. Smoothing is what
# removes that shared texture and leaves the differences between models.
PRESETS = {
    # no smoothing at all: ball-per-point, tight radius. Shows individual points
    # and any clumping directly, at the cost of looking like gravel.
    "raw":     {"radius_mm": 4.0, "sigma": 0.0, "taubin": 0},
    # balls are bridged, but the cobblestone texture is still there
    "light":   {"radius_mm": 5.0, "sigma": 1.0, "taubin": 15},
    # the locked default, repeated here so ladders include it
    "default": {"radius_mm": 6.0, "sigma": 2.5, "taubin": 60},
    # aggressive: only gross shape survives. Useful to check overall form, but
    # it will hide real surface defects too -- do not judge quality from this.
    "heavy":   {"radius_mm": 7.0, "sigma": 4.0, "taubin": 120},
}

# Ground truth is farthest-point sampled, so its spacing is near-uniform and it
# contains literally no pair closer than ~3.5 mm. Predictions are not, and the
# fraction of points sitting on top of a neighbour is the clearest single number
# for that gap (measured: 0.0% for GT vs 11.2% for the current model).
CLUMP_MM = 2.0


def pc_to_mesh(points, scale_mm, **overrides):
    """Point cloud -> shaded-renderable mesh, via a KD-tree distance field.

    Poisson reconstruction would be the usual choice and is deliberately not
    used here: it needs per-point normals, which a predicted cloud does not
    have and which are unreliable to estimate on a skull (a thin bone shell,
    where inner and outer surface normals flip against each other), and it
    tends to close the defect -- exactly the feature being inspected.

    `overrides` take precedence over RECON, for exploring one model:

        pc_to_mesh(pred, scale_mm)                      # locked default
        pc_to_mesh(pred, scale_mm, sigma=0, taubin=0)   # no smoothing at all
        pc_to_mesh(pred, scale_mm, **PRESETS["heavy"])  # named level

    Knobs, and the ranges that behave sensibly at 4096-6144 points:
      radius_mm  ball radius per point, 4-7. Below ~4 the balls stop touching
          and the skull renders as a sponge; above ~8 the defect starts to be
          bridged over and you lose it.
      sigma      distance-field blur in voxels, 0-4. The main smoothness knob.
      taubin     mesh smoothing passes, 0-120. Cheap; does not shrink the model.
      res        grid size, 96 (fast) to 160 (fine). Cost is roughly res^3.

    Overriding for a single model is fine and useful. Overriding while comparing
    models is not -- see the module docstring.
    """
    cfg = {**RECON, **overrides}
    P = np.asarray(points, dtype=np.float64)
    r = cfg["radius_mm"] / scale_mm

    lo, hi = P.min(0) - cfg["pad"], P.max(0) + cfg["pad"]
    res = cfg["res"]
    axes = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    field = cKDTree(P).query(grid, workers=-1)[0].reshape(res, res, res)
    if cfg["sigma"] > 0:
        field = gaussian_filter(field, cfg["sigma"])

    verts, faces, _, _ = measure.marching_cubes(field, level=r)
    step = np.array([(hi[i] - lo[i]) / (res - 1) for i in range(3)])
    mesh = trimesh.Trimesh(vertices=verts * step + lo, faces=faces)

    # The isosurface picks up small detached blobs around outlying points; keep
    # only the skull itself so they do not clutter the render.
    parts = mesh.split(only_watertight=False)
    if len(parts):
        mesh = max(parts, key=lambda m: len(m.faces))
    if cfg["taubin"]:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=cfg["taubin"])
    return mesh


def local_spacing(points, scale_mm):
    """Distance from each point to its nearest neighbour, in mm."""
    P = np.asarray(points, dtype=np.float64)
    return cKDTree(P).query(P, k=2, workers=-1)[0][:, 1] * scale_mm


def signed_deviation(pred, gt, scale_mm, k=24):
    """Signed distance from each predicted point to the local ground-truth surface, mm.

    Positive is outside the surface, negative inside. A local plane is fitted to
    the k nearest ground-truth points and the residual taken along its normal,
    so this separates "the surface is in the wrong place" from "the points are
    unevenly spread", which the unsigned Chamfer distance mixes together.
    """
    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt, dtype=np.float64)
    idx = cKDTree(G).query(P, k=k, workers=-1)[1]
    nb = G[idx]
    centre = nb.mean(1)
    X = nb - centre[:, None, :]
    _, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X) / k)
    normal = evec[:, :, 0]                      # smallest eigenvector = surface normal
    return np.einsum("ni,ni->n", P - centre, normal) * scale_mm


def surface_stats(pred, gt, scale_mm):
    """The numbers that decide whether a training change actually helped.

    Eyeballing two renders is not evidence; these are. Report them alongside
    every before/after figure.
    """
    sp_p, sp_g = local_spacing(pred, scale_mm), local_spacing(gt, scale_mm)
    dev = signed_deviation(pred, gt, scale_mm)
    return {
        # density uniformity -- the clearest current gap vs ground truth
        "clump_pct": 100.0 * (sp_p < CLUMP_MM).mean(),
        "clump_pct_gt": 100.0 * (sp_g < CLUMP_MM).mean(),
        "spacing_cv": sp_p.std() / sp_p.mean(),
        "spacing_cv_gt": sp_g.std() / sp_g.mean(),
        "spacing_median_mm": float(np.median(sp_p)),
        # surface accuracy -- signed, so bias and scatter stay distinguishable
        "dev_abs_median_mm": float(np.median(np.abs(dev))),
        "dev_abs_p95_mm": float(np.percentile(np.abs(dev), 95)),
        "dev_bias_mm": float(dev.mean()),
        "dev_std_mm": float(dev.std()),
        "outside_pct": 100.0 * (dev > 0).mean(),
    }


def format_stats(rows):
    """rows: list of (name, stats dict) -> a plain-text comparison table."""
    head = (f"{'':22s} {'clump<2mm':>10s} {'spacingCV':>10s} {'|dev|med':>10s} "
            f"{'|dev|p95':>10s} {'dev std':>9s} {'outside':>9s}")
    out = [head, "-" * len(head)]
    for name, s in rows:
        out.append(f"{name:22s} {s['clump_pct']:9.1f}% {s['spacing_cv']:10.3f} "
                   f"{s['dev_abs_median_mm']:8.2f}mm {s['dev_abs_p95_mm']:8.2f}mm "
                   f"{s['dev_std_mm']:7.2f}mm {s['outside_pct']:8.1f}%")
    if rows:
        s = rows[0][1]
        out += ["-" * len(head),
                f"{'ground truth (ref)':22s} {s['clump_pct_gt']:9.1f}% {s['spacing_cv_gt']:10.3f}"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# plotly figures
# --------------------------------------------------------------------------- #
def fig_meshes(items, title="", height=620):
    """items: list of (mesh, label). Side-by-side shaded meshes, shared camera."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=len(items), specs=[[{"type": "mesh3d"}] * len(items)],
                        subplot_titles=[lbl for _, lbl in items], horizontal_spacing=0.01)
    for i, (m, _) in enumerate(items, start=1):
        v, f = m.vertices, m.faces
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="#d8d2c4", flatshading=False,
                                lighting=dict(ambient=0.42, diffuse=0.85, specular=0.12,
                                              roughness=0.85, fresnel=0.1),
                                lightposition=dict(x=120, y=180, z=200),
                                showscale=False, hoverinfo="skip"), row=1, col=i)
    scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                 zaxis_visible=False, camera=dict(eye=dict(x=0.0, y=-1.5, z=1.15)))
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=60, b=0),
                      **{f"scene{i if i > 1 else ''}": scene for i in range(1, len(items) + 1)})
    return fig


def fig_smoothing_ladder(points, scale_mm, levels=("raw", "light", "default", "heavy"),
                         title="", height=560, res=80):
    """One cloud rendered at several smoothing levels, for picking a level.

    Use this to see what the reconstruction is doing to your surface before
    trusting any single render. Run it on ground truth too: whatever texture
    shows up there is the reconstruction's, not your model's.

    `res` defaults BELOW RECON's 128 on purpose. The panels each carry a full
    mesh, and at 128 the "raw" preset alone emits 372k faces; the four-level
    figure serialises to 30 MB, so a notebook holding two of them gains 60 MB of
    output and takes tens of minutes to write. At 80 the figure is a few MB and
    still shows what it needs to -- the point here is the relative texture
    between levels, not absolute surface detail.

    NOT a comparison tool. Two models rendered through this will differ by
    whatever the reconstruction does, so use it on one cloud at a time; model
    comparison goes through `fig_meshes` at the locked RECON settings.

    A single level can be passed as a bare string: levels="raw" behaves the
    same as levels=("raw",). Without this, the missing-comma version is a
    string, iterating it yields characters, and the failure surfaces as a
    baffling KeyError: 'r'.
    """
    if isinstance(levels, str):
        levels = (levels,)
    unknown = [k for k in levels if k not in PRESETS]
    if unknown:
        raise KeyError(f"unknown preset(s) {unknown}; available: {sorted(PRESETS)}")
    return fig_meshes(
        [(pc_to_mesh(points, scale_mm, res=res, **PRESETS[k]),
          f"{k}<br><sub>r={PRESETS[k]['radius_mm']} σ={PRESETS[k]['sigma']} "
          f"taubin={PRESETS[k]['taubin']}</sub>") for k in levels],
        title=title, height=height)


def fig_diagnostic(pred, gt, scale_mm, title="", height=620, dev_lim=None, sp_max=None):
    """The two things a shaded mesh cannot show, side by side.

    Left  -- signed deviation from the ground-truth surface: red outside, blue
             inside, white on it. Symmetric red/blue speckle means the points
             scatter around the surface rather than sitting on it.
    Right -- nearest-neighbour spacing: dark points are clumped onto a
             neighbour, i.e. wasted, since they cover no new surface.

    `dev_lim` / `sp_max` fix the colour limits. Leave them None to auto-scale to
    THIS cloud, which is fine for looking at one model and wrong for comparing
    two: auto-scaling renders the best and the worst model in the same range of
    colours, so a real difference disappears. `fig_spacing_grid` locks them for
    you; pass them explicitly if you build a comparison by hand.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    P = np.asarray(pred, dtype=np.float64)
    dev = signed_deviation(P, gt, scale_mm)
    sp = local_spacing(P, scale_mm)
    lim = float(dev_lim if dev_lim is not None else np.percentile(np.abs(dev), 95))

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}] * 2],
                        subplot_titles=(f"Signed deviation (±{lim:.1f} mm, red = outside)",
                                        "Nearest-neighbour spacing (dark = clumped)"),
                        horizontal_spacing=0.02)
    sp_hi = float(sp_max if sp_max is not None else np.percentile(sp, 95))
    for col, (vals, cs, cmin, cmax, bar) in enumerate(
            [(dev, "RdBu_r", -lim, lim, "mm"),
             (sp, "Viridis", 0.0, sp_hi, "mm")], start=1):
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.9, color=vals, colorscale=cs, cmin=cmin, cmax=cmax,
                        opacity=0.9, colorbar=dict(title=bar, len=0.72,
                                                   x=0.46 if col == 1 else 1.0)),
            hoverinfo="skip"), row=1, col=col)
    scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                 zaxis_visible=False, camera=dict(eye=dict(x=0.0, y=-1.5, z=1.15)))
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=60, b=0),
                      showlegend=False, scene=scene, scene2=scene)
    return fig


def fig_spacing_grid(items, scale_mm, title="", height=520, sp_max=None, clump_mm=CLUMP_MM):
    """Several predictions side by side, coloured by nearest-neighbour spacing.

    This is the figure for the density result. `fig_diagnostic` renders one model
    at a time and auto-scales its colours to that model, so putting two of its
    outputs next to each other makes a 12.8% clump rate and a 1.4% one look
    alike. Here every panel shares one scale, computed across all of them, so
    dark really does mean "closer together than in the panel next door".

    Each panel's subtitle carries its own clump rate, since colour alone is hard
    to read quantitatively. Ground truth scores 0.0% and is worth including as
    the leftmost panel for reference.

    items: list of (points, label).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sps = [local_spacing(np.asarray(p, dtype=np.float64), scale_mm) for p, _ in items]
    hi = float(sp_max if sp_max is not None else
               np.percentile(np.concatenate(sps), 97))
    labels = [f"{lbl}<br><sub>clumped &lt;{clump_mm:g} mm: "
              f"{(s < clump_mm).mean() * 100:.1f}%</sub>"
              for (_, lbl), s in zip(items, sps)]

    fig = make_subplots(rows=1, cols=len(items), horizontal_spacing=0.01,
                        specs=[[{"type": "scatter3d"}] * len(items)], subplot_titles=labels)
    # Panel titles sit at the very top of each cell, so the figure title has to be
    # lifted clear of them or the two overlap.
    for a in fig.layout.annotations:
        a.y = min(a.y, 0.93)
    for i, ((pts, _), sp) in enumerate(zip(items, sps), start=1):
        P = np.asarray(pts, dtype=np.float64)
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.9, color=sp, colorscale="Viridis", cmin=0.0, cmax=hi,
                        opacity=0.9, showscale=(i == len(items)),
                        colorbar=dict(title="spacing<br>(mm)", len=0.72)),
            hoverinfo="skip"), row=1, col=i)
    scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                 zaxis_visible=False, camera=dict(eye=dict(x=0.0, y=-1.5, z=1.15)))
    fig.update_layout(title=dict(text=title or ("Nearest-neighbour spacing "
                                 f"(dark = clumped; colour scale locked 0–{hi:.1f} mm)"),
                                 y=0.98, yanchor="top"),
                      height=height, showlegend=False, margin=dict(l=0, r=0, t=110, b=0),
                      **{f"scene{i if i > 1 else ''}": scene for i in range(1, len(items) + 1)})
    return fig
