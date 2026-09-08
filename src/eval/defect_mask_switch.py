"""Dry run: what every defect-region number becomes under a different mask.

The audit measured that the 5 mm distance rule recovers the true implant region
with precision 0.79 and recall 0.81 -- the count is nearly right while the SET is
about a third wrong, the false positives per skull very nearly cancelling the
false negatives. That says the mask is imperfect. It does not say by how much the
numbers built on it would move.

This answers that WITHOUT CHANGING ANYTHING: the same predictions go through the
same metric function under three region definitions side by side, and the result
goes to its own CSV. The frozen table is not touched, the threshold constant is
not changed, and no default behaviour moves.

    5mm       the rule in force -- distance to nearest input point > 5 mm
    6mm       the same rule at the threshold the audit found best
    implant   the dataset's own ground truth: is this point on the implant

The `5mm` column must reproduce the frozen table exactly. It is computed here
through the same function with the same arguments, so any difference means this
script is wrong and none of the other columns can be trusted. It is verified per
run per skull, and a mismatch aborts.

What to look at is not "did the numbers move" -- they will, the region changed.
The questions are whether the RANKINGS between configurations survive, since the
mask never sees a prediction and every configuration is scored through the
identical set of points, and whether the claims that rest on defect coverage
still hold. A number moving is expected and harmless. A ranking moving would be
serious.

Only the GROUND-TRUTH side of the mask has a truth to switch to. A predicted
point is an arbitrary point in space, so asking whether it "is on the implant" is
not well posed, and that side keeps a tolerance. The `implant` variant therefore
holds the prediction tolerance where it is and changes only the ground-truth
side, which is where the main metric lives.

This is a trial calculation, not a result: it only needs re-running if the
definition actually changes.

    python src/eval/defect_mask_switch.py [--runs A B ...] [--n 20]
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

OUT_CSV = os.path.join("experiments_log", "defect_mask_switch.csv")
# Per-point implant labels, cached because they cost ~20 s a skull and do not
# depend on any run. Also the thing that would let report.py adopt the implant
# definition WITHOUT needing the raw nrrd at evaluation time.
LABELS = os.path.join("experiments_log", "defect_mask_labels.npz")

DEFECT_COLS = ["defect_gt_%", "defect_cov_mm", "defect_HD95_mm",
               "defect_n_pred", "defect_prec_mm", "defect_F1@0.05"]

# Every run whose weights still exist. `baseline` / `dcd_l2` cannot be included:
# their checkpoints were deleted, so their defect columns can never be recomputed
# under any definition -- which is itself one of the costs of switching.
DEFAULT_RUNS = ["msn_skullfix/cd_only_f0",
                "msn_skullfix/cd_only_f1",
                "msn_skullfix/cd_only_f2",
                "msn_skullfix/cd_only_f3",
                "msn_skullfix/cd_only_f4",
                "msn_skullfix/lr_fix_only_f0",
                "msn_skullfix/lr_fix_only_f1",
                "msn_skullfix/lr_fix_only_f2",
                "msn_skullfix/lr_fix_only_f3",
                "msn_skullfix/lr_fix_only_f4",
                "msn_skullfix/rep_w05_f0",
                "msn_skullfix/rep_w05_f1",
                "msn_skullfix/rep_w05_f2",
                "msn_skullfix/rep_w05_f3",
                "msn_skullfix/rep_w05_f4",
                "msn_skullfix/cd_rep05_full_f0",
                "msn_skullfix/cd_rep05_full_f1",
                "msn_skullfix/cd_rep05_full_f2",
                "msn_skullfix/cd_rep05_full_f3",
                "msn_skullfix/cd_rep05_full_f4"]


def implant_labels(repo, sids, raw_root):
    """Per-point 'is this ground-truth point on the implant' — cached."""
    import defect_mask_audit as dma
    import normal_quality as nq

    path = os.path.join(repo, LABELS)
    have = dict(np.load(path)) if os.path.exists(path) else {}
    todo = [s for s in sids if s not in have]
    if todo:
        print(f"computing implant ground-truth labels for {len(todo)} skulls "
              f"(about 20 s each, and only once)...")
        data = np.load(os.path.join(repo, paths.DATA_CACHE))
        ids, gt, scales = data["ids"], data["gt"], data["scale_mm"]
        for sid in todo:
            j = int(np.where(ids == sid)[0][0])
            s_mm = float(scales[j])
            pts_n, _, scale, _ = nq.truth_for(sid, raw_root)
            if abs(scale - s_mm) > 1e-3 or np.abs(pts_n - gt[j]).max() > 1e-5:
                raise SystemExit(f"{sid}: could not reproduce the pipeline -- the "
                                 f"coordinate frames do not line up. Aborted.")
            seed = nq._task_seed(sid, raw_root)
            dense_def, _, _ = nq._dense_with_faces(
                os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"),
                16384, 0.5, seed * 2 + 1)
            centroid = dense_def.mean(axis=0)
            mesh_i, vol_i = dma._mesh_mm(os.path.join(raw_root, "implant", f"{sid}.nrrd"))
            mesh_d, vol_d = dma._mesh_mm(os.path.join(raw_root, "defective_skull", f"{sid}.nrrd"))
            _, vol_c = dma._mesh_mm(os.path.join(raw_root, "complete_skull", f"{sid}.nrrd"))
            if not np.array_equal((vol_d > 0) | (vol_i > 0), vol_c > 0) or \
                    ((vol_i > 0) & (vol_d > 0)).any():
                raise SystemExit(f"{sid}: defective + implant != complete. Aborted.")
            for m in (mesh_i, mesh_d):
                m.vertices = (np.asarray(m.vertices) - centroid) / scale * s_mm
            is_imp, _, _, d_min = dma.label_by_implant(gt[j] * s_mm, mesh_i, mesh_d)
            if (d_min > dma.ON_SURFACE_MM).mean() > 0.02:
                raise SystemExit(f"{sid}: ground-truth points are far from both surfaces, "
                                 f"so the union invariant fails. Aborted.")
            have[sid] = is_imp
            print(f"  {sid} ok  true defect {100 * is_imp.mean():.2f}%")
        np.savez_compressed(path, **have)
        print(f"labels cached -> {LABELS} ({len(have)} skulls)")
    return have


def analyse(repo, specs, n_skulls=20, device="/GPU:0"):
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import report as rp

    runs = rp.load_runs(repo, specs)
    raw_root = os.path.join(repo, paths.RAW_ROOT)
    data = np.load(os.path.join(repo, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    text_path = os.path.join(repo, paths.BERT_CACHE)
    text = np.load(text_path) if os.path.exists(text_path) else None

    # Every run must share the split, because the frozen-CSV check and every
    # comparison below assume the same skulls. They do (one split across all 16
    # runs, checked 2026-08-27), but assuming it silently is how the cohort trap
    # that bit `point_to_surface.py` happened.
    splits = {tuple(r.meta["val_ids"]) for r in runs}
    if len(splits) != 1:
        raise SystemExit(f"these runs use {len(splits)} different validation splits, so "
                         f"they cannot be compared side by side")
    sids = runs[0].meta["val_ids"][:n_skulls]
    labels = implant_labels(repo, sids, raw_root)

    frozen = pd.read_csv(os.path.join(repo, "experiments_log", "eval_all_runs.csv"))
    frozen["id"] = frozen["id"].astype(str).str.zfill(3)

    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)

    rows, n_checked, unmatched = [], 0, []
    with tf.device(device):
        for arch, group in groups.items():
            cfg = rp.arch_config(msn, arch)
            model = msn.build_model(cfg)
            for run in group:
                model.load_weights(run.weights)
                pos = [int(np.where(ids == s)[0][0]) for s in sids]
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                preds = model.predict(x, batch_size=1, verbose=0)

                for sid, j, pred in zip(sids, pos, preds):
                    s_mm = float(scales[j])
                    # dist1 = gt -> nearest pred, exactly as report.eval_runs feeds it
                    from scipy.spatial import cKDTree
                    dist1 = cKDTree(pred).query(gt[j], k=1, workers=-1)[0]
                    variants = {
                        "5mm": dict(defect_mm=5.0),
                        "6mm": dict(defect_mm=6.0),
                        "implant": dict(gt_mask=labels[sid], defect_mm=5.0),
                    }
                    # Decompose the change instead of only showing it. The two
                    # sets share their TP; what differs is 87 false positives
                    # (intact bone, dropped) against 71 false negatives (implant
                    # near the rim, added). BOTH are relatively easy points, so
                    # which way the mean moves is not something to reason out --
                    # it depends on their actual distances, which is what these
                    # three numbers report.
                    m5 = cKDTree(inputs[j]).query(gt[j], k=1, workers=-1)[0] * s_mm > 5.0
                    mi = labels[sid]
                    d_mm = dist1 * s_mm
                    for tag, sel in (("TP", m5 & mi), ("FP_dropped", m5 & ~mi),
                                     ("FN_added", ~m5 & mi)):
                        rows.append({"run": run.label, "id": sid,
                                     "definition": f"_part_{tag}",
                                     "defect_cov_mm": float(d_mm[sel].mean()) if sel.any() else np.nan,
                                     "defect_gt_%": 100.0 * sel.mean()})

                    got5 = None
                    for name, kw in variants.items():
                        m = rp._defect_metrics(pred, gt[j], inputs[j], dist1, s_mm, **kw)
                        rows.append({"run": run.label, "id": sid, "definition": name, **m})
                        if name == "5mm":
                            got5 = m

                    # ⭐ the check everything else depends on.
                    # ⚠️ A run whose label is not in the frozen CSV used to be
                    # skipped in silence -- `lr_fix_only` is stored there under
                    # the label `lr_fix`, so 20 of 200 combinations went
                    # unchecked while the script still reported a pass. Missing
                    # rows are now counted and reported at the end.
                    ref = frozen[(frozen.run == run.label) & (frozen.id == sid)]
                    if len(ref) != 1:
                        unmatched.append((run.label, sid))
                    else:
                        got = got5
                        for c in DEFECT_COLS:
                            a, b = float(ref.iloc[0][c]), float(got[c])
                            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-6:
                                raise SystemExit(
                                    f"{run.label}/{sid}: the 5mm definition did not "
                                    f"reproduce -- {c} is {a} in the frozen table against "
                                    f"{b} here. This script cannot be trusted. Aborted.")
                        n_checked += 1
                print(f"  {run.label} ✓")
            del model
            tf.keras.backend.clear_session()
    total = len(runs) * len(sids)
    print(f"\nthe 5mm definition matches eval_all_runs.csv column by column and skull by "
          f"skull: {n_checked}/{total} pairs agree")
    if unmatched:
        miss = sorted({r for r, _ in unmatched})
        print(f"Warning: {len(unmatched)} pairs had no matching row in the frozen table and "
              f"were NOT checked: {miss}\n"
              f"   (usually the run's directory name differs from its label in the CSV)")
    return pd.DataFrame(rows)


def _paired(df, base, other, col="defect_cov_mm"):
    from scipy.stats import wilcoxon
    a = df[df.run == base].set_index("id")[col]
    b = df[df.run == other].set_index("id")[col]
    d = (b - a).dropna()
    p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
    return d.mean(), int((d < 0).sum()), len(d), p


def report(df):
    df = df[~df.definition.str.startswith("_part_")] if False else df
    piv = df[~df.definition.str.startswith("_part_")].pivot_table(index="run", columns="definition",
                         values="defect_cov_mm", sort=False)[["5mm", "6mm", "implant"]]
    print(f"\n{'=' * 74}\nthe main metric under all three definitions\n{'=' * 74}")
    print(f"{'run':<18}{'5mm (current)':>15}{'6mm':>10}{'implant':>11}"
          f"{'d implant':>12}{'rank move':>11}")
    print("-" * 74)
    r5 = piv["5mm"].rank(); ri = piv["implant"].rank()
    for run in piv.index:
        move = int(ri[run] - r5[run])
        print(f"{run:<18}{piv.loc[run,'5mm']:>12.3f}{piv.loc[run,'6mm']:>10.3f}"
              f"{piv.loc[run,'implant']:>11.3f}{piv.loc[run,'implant']-piv.loc[run,'5mm']:>+12.3f}"
              f"{('—' if move == 0 else f'{move:+d}'):>10}")
    moved = int((ri != r5).sum())
    print(f"\n  ranking: {moved}/{len(piv)} runs changed position "
          f"{'-- none moved, so the comparisons are unaffected' if moved == 0 else '-- something moved, look closely'}")

    parts = df[df.definition.str.startswith("_part_")]
    if len(parts):
        print(f"\n{'=' * 74}\nwhere the change comes from: the two sets share their true "
              f"positives, and differ in these\n{'=' * 74}")
        print(f"  {'':16}{'points/skull':>14}{'to nearest pred':>17}   what it is")
        for tag, note in (("TP", "counted by both definitions"),
                          ("FP_dropped", "selected by 5mm but actually on intact bone -- "
                                         "the implant definition DROPS these"),
                          ("FN_added", "really on the implant but close to an input point -- "
                                       "the implant definition ADDS these")):
            g = parts[parts.definition == f"_part_{tag}"]
            print(f"  {tag:16}{g['defect_gt_%'].mean()*6144/100:>14.0f}"
                  f"{g.defect_cov_mm.mean():>16.3f}mm   {note}")
        fp = parts[parts.definition == "_part_FP_dropped"].defect_cov_mm.mean()
        fn = parts[parts.definition == "_part_FN_added"].defect_cov_mm.mean()
        print(f"\n  the dropped points average {fp:.3f}mm and the added ones {fn:.3f}mm -- "
              f"{'the dropped ones were easier, so the main metric rises' if fp < fn else 'the added ones are easier, so the main metric falls'}")

    print(f"\n{'=' * 74}\nmean change in every other column (implant against 5mm)\n{'=' * 74}")
    main = df[~df.definition.str.startswith("_part_")]
    for c in DEFECT_COLS:
        p = main.pivot_table(index="run", columns="definition", values=c, sort=False)
        a, b = p["5mm"].mean(), p["implant"].mean()
        pct = f"{100 * (b - a) / a:+6.1f}%" if a else "    n/a"
        print(f"  {c:<18}{a:>10.3f} → {b:>8.3f}   {b - a:>+8.3f}   {pct}")

    print(f"\n{'=' * 74}\nthe claims that rest on defect coverage, re-judged under each "
          f"definition\n{'=' * 74}")
    claims = [("repulsion helps", "cd_only", "cd_rep05_full"),
              ("notext is worse", "cd_rep05_full", "notext"),
              ("tie_qk does nothing", "cd_rep05_full", "tie_qk")]
    for label, base, other in claims:
        print(f"\n  {label}  ({other} vs {base})")
        for d in ("5mm", "6mm", "implant"):
            sub = main[main.definition == d]
            if not {base, other} <= set(sub.run):
                continue
            mean, better, n, p = _paired(sub, base, other)
            print(f"    {d:<9} delta {mean:+.3f}mm   {other} better on {better}/{n}   "
                  f"p={p:.4f}   {'significant' if p < 0.002 else 'not significant'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    df = analyse(REPO, args.runs, n_skulls=args.n)
    report(df)
    df.to_csv(os.path.join(REPO, args.out), index=False)
    print(f"\n{len(df)} rows -> {args.out}"
          f"\nThis was a dry run: neither the frozen table nor the threshold constant "
          f"was changed.")


if __name__ == "__main__":
    main()
