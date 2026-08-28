r"""Produce the defect-region ground truth that `report.py` now scores against.

WHY THIS FILE EXISTS
  The defect region used to be inferred -- a ground-truth point counted as "in
  the defect" if the nearest input point was more than 5 mm away. Audited against
  the implant the dataset ships (`defect_mask_audit.py`), that rule turned out to
  have precision 0.79 and recall 0.81: the count was nearly right while the SET
  was about a third wrong, with 87 false positives per skull almost exactly
  cancelling 71 false negatives. Since 2026-08-28 the region is the implant
  itself, and this is what writes that ground truth down.

  Two reasons it is a file rather than something `report.py` computes:
    * cost -- it needs marching cubes over three volumes per skull, ~20 s each,
      against a lookup that is instant.
    * ⭐ dependency -- computing it needs the raw nrrd, and `data/` is gitignored
      on a machine whose `/root` is wiped by redeploys. The labels are tracked in
      git, so evaluation keeps working when the raw data does not.

DEFAULT IS ALL 100 SKULLS, ON PURPOSE
  k-fold will use every skull as validation in some fold. Generating on demand
  would stall that run partway through; generating all of them once (~35 min)
  never does. Existing labels are kept and only missing skulls are computed, so
  re-running is cheap and safe.

WHAT IS CHECKED (a wrong label here becomes a wrong defect region everywhere)
  Per skull, `defect_mask_audit.label_one` refuses to return unless the
  prepare_skullfix pipeline reproduces the cached points exactly, the dataset
  really satisfies `defective + implant == complete` voxel for voxel, and every
  ground-truth point lands on one of the two surfaces. Any failure aborts rather
  than writing a label.

⚠️ k 折之后不用重跑 —— 这是数据的性质，与划分和权重无关（同 `sampling_floor.csv`）。

    python src/eval/make_defect_labels.py            # 全部 100 颗，约 35 分钟
    python src/eval/make_defect_labels.py --ids 083 053
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))

import numpy as np
import pandas as pd

OUT_NPZ = os.path.join("experiments_log", "defect_mask_labels.npz")
OUT_CSV = os.path.join("experiments_log", "defect_mask_labels.csv")   # 逐颅骨的自检记录


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=None, help="默认：数据里全部 100 颗")
    ap.add_argument("--raw-root", default=os.path.join(REPO, "data", "14161307",
                                                       "SkullFix", "training_set"))
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--force", action="store_true", help="重算已有的标签")
    args = ap.parse_args()

    import defect_mask_audit as dma

    data = np.load(os.path.join(REPO, "data", "cache", "skullfix_pairs_4096_6144.npz"))
    ids, gt, scales = data["ids"], data["gt"], data["scale_mm"]
    want = [str(s) for s in (args.ids if args.ids else ids)]

    out = os.path.join(REPO, args.out)
    have = dict(np.load(out)) if os.path.exists(out) and not args.force else {}
    todo = [s for s in want if s not in have]
    print(f"已有 {len(have)} 颗，本次要算 {len(todo)} 颗"
          f"（约 20 秒/颗，合计约 {len(todo) * 20 / 60:.0f} 分钟）")
    if not todo:
        print("没有要算的。")
        return

    rows = []
    for n, sid in enumerate(todo, 1):
        j = int(np.where(ids == sid)[0][0])
        lab, chk = dma.label_one(REPO, sid, args.raw_root, gt[j], float(scales[j]))
        have[sid] = lab
        rows.append({"id": sid, "defect_pct": 100.0 * float(lab.mean()),
                     "n_defect": int(lab.sum()), **chk})
        print(f"  [{n:>3}/{len(todo)}] {sid} ✓ 缺损点 {int(lab.sum()):>4} "
              f"({100 * lab.mean():.2f}%)  缝隙 {chk['seam_pct']:.2f}%  "
              f"离面 {chk['off_surface_pct']:.3f}%")

    np.savez_compressed(out, **have)
    df = pd.DataFrame(rows)
    csv = os.path.join(REPO, OUT_CSV)
    if os.path.exists(csv):
        old = pd.read_csv(csv, dtype={"id": str})
        df = pd.concat([old[~old.id.isin(set(df.id))], df], ignore_index=True)
    df.sort_values("id").to_csv(csv, index=False)

    print(f"\n{len(have)} 颗 -> {args.out}  ({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"逐颅骨自检 -> {OUT_CSV}")
    print(f"缺损占比 {df.defect_pct.mean():.2f}% ± {df.defect_pct.std():.2f}%"
          f"  范围 {df.defect_pct.min():.2f}~{df.defect_pct.max():.2f}%")
    print(f"缝隙点最大 {df.seam_pct.max():.2f}%   离面最大 {df.off_surface_pct.max():.3f}%")


if __name__ == "__main__":
    main()
