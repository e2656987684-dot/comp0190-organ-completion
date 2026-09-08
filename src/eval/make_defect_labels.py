"""Produce the defect-region ground truth that evaluation scores against.

The defect region used to be inferred: a ground-truth point counted as "in the
defect" if the nearest input point was more than 5 mm away. Audited against the
implant the dataset ships, that rule had precision 0.79 and recall 0.81 -- the
count was nearly right while the SET was about a third wrong, the false positives
almost exactly cancelling the false negatives. The region is now the implant
itself, and this writes that ground truth down.

Two reasons it is a file rather than something computed on demand: cost, since it
needs marching cubes over three volumes per skull against a lookup that is
instant; and dependency, since computing it needs the raw volumes while the
labels are small enough to track in version control. Evaluation then keeps
working on a machine that does not have the raw data.

The default is every skull, on purpose: cross-validation uses each one as
validation in some fold, and generating on demand would stall a long run partway
through. Existing labels are kept and only missing skulls computed, so re-running
is cheap.

A wrong label here would become a wrong defect region everywhere downstream, so
per skull the labelling refuses to return unless the data pipeline reproduces the
cached points exactly, the dataset really satisfies `defective + implant ==
complete` voxel for voxel, and every ground-truth point lands on one of the two
surfaces. Any failure aborts rather than writing a label.

Warning: read `defect_mask_labels.csv` with `dtype={"id": str}`. Ids carry a
leading zero, and read as integers any id-keyed comparison fails silently.

    python src/eval/make_defect_labels.py            # all 100 skulls, ~35 minutes
    python src/eval/make_defect_labels.py --ids 083 053
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import paths                    # every data path is written down once

import numpy as np
import pandas as pd

OUT_NPZ = os.path.join("experiments_log", "defect_mask_labels.npz")
OUT_CSV = os.path.join("experiments_log", "defect_mask_labels.csv")   # per-skull checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=None, help="default: every skull in the data")
    ap.add_argument("--raw-root", default=os.path.join(REPO, paths.RAW_ROOT))
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--force", action="store_true", help="recompute labels that already exist")
    args = ap.parse_args()

    import defect_mask_audit as dma

    data = np.load(os.path.join(REPO, paths.DATA_CACHE))
    ids, gt, scales = data["ids"], data["gt"], data["scale_mm"]
    want = [str(s) for s in (args.ids if args.ids else ids)]

    out = os.path.join(REPO, args.out)
    have = dict(np.load(out)) if os.path.exists(out) and not args.force else {}
    todo = [s for s in want if s not in have]
    print(f"{len(have)} already labelled, {len(todo)} to compute "
          f"(about 20 s each, so roughly {len(todo) * 20 / 60:.0f} minutes)")
    if not todo:
        print("nothing to compute.")
        return

    rows = []
    for n, sid in enumerate(todo, 1):
        j = int(np.where(ids == sid)[0][0])
        lab, chk = dma.label_one(REPO, sid, args.raw_root, gt[j], float(scales[j]))
        have[sid] = lab
        rows.append({"id": sid, "defect_pct": 100.0 * float(lab.mean()),
                     "n_defect": int(lab.sum()), **chk})
        print(f"  [{n:>3}/{len(todo)}] {sid} ok  defect points {int(lab.sum()):>4} "
              f"({100 * lab.mean():.2f}%)  seam {chk['seam_pct']:.2f}%  "
              f"off-surface {chk['off_surface_pct']:.3f}%")

    np.savez_compressed(out, **have)
    df = pd.DataFrame(rows)
    csv = os.path.join(REPO, OUT_CSV)
    if os.path.exists(csv):
        old = pd.read_csv(csv, dtype={"id": str})
        df = pd.concat([old[~old.id.isin(set(df.id))], df], ignore_index=True)
    df.sort_values("id").to_csv(csv, index=False)

    print(f"\n{len(have)} skulls -> {args.out}  ({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"per-skull checks -> {OUT_CSV}")
    # ⚠️ The CSV only gains a row for skulls computed in THIS invocation; ones
    # already cached are skipped and leave no record. That is a gap in the
    # bookkeeping, not in the guarantee: `label_one` raises before returning, so
    # a label existing in the npz at all means it passed every check at the time
    # it was written. Reported rather than left to be noticed.
    if len(df) < len(have):
        print(f"Note: the checks CSV covers {len(df)}/{len(have)} skulls. The rest were "
              f"cached earlier and passed the same checks then -- labelling does not return "
              f"unless they pass.\n   Use --force to recompute everything and fill the "
              f"record in (about {len(have) * 20 / 60:.0f} minutes).")
    print(f"defect fraction {df.defect_pct.mean():.2f}% +- {df.defect_pct.std():.2f}%"
          f"  range {df.defect_pct.min():.2f}-{df.defect_pct.max():.2f}%")
    print(f"largest seam {df.seam_pct.max():.2f}%   "
          f"largest off-surface {df.off_surface_pct.max():.3f}%")


if __name__ == "__main__":
    main()
