r"""Recompute eval_all_runs.csv under the implant defect region, keeping what cannot be recomputed.

WHY A SCRIPT AND NOT A ONE-LINER
  This overwrites the project's most important frozen record, and the first
  version of it was going to be an inline `python -c`. That version had a bug:
  the merge key is (run, id), `eval_all_runs.csv` reads its `id` back as the
  integer 83 while `eval_runs` produces the string '083', so nothing would have
  matched and every run would have ended up in the file TWICE -- old rows under
  the old definition sitting beside new ones, with the reader silently averaging
  across both. That is the same leading-zero trap already fixed in three scripts
  this week; it does not get a fourth chance.

WHAT IT DOES
  Recomputes every row it can, under the implant ground truth, and stamps each
  row with which definition produced it:

      defect_def = "implant"      recomputed now, 2026-08-28 onwards
      defect_def = "5mm_legacy"   kept from before, CANNOT be recomputed

  ⚠️ Only `baseline` and `dcd_l2` fall in the second group -- their checkpoints
  were deleted (devlog 2026-08-24), so their defect columns are frozen under the
  old rule forever. Both are ⛔ invalid runs excluded from the thesis anyway, so
  no reported number mixes definitions; the column exists so that stays checkable
  rather than remembered.

  ⚠️ Columns that do NOT depend on the region -- CD_t, HD95, F1, DCD, clump_%,
  spacing_CV -- are unaffected by the switch, and this verifies that rather than
  assuming it: every recomputed row must reproduce its old values for those to
  1e-9, and the script aborts if not. If they moved, something other than the
  mask changed and the whole recompute is suspect.

    python src/eval/recompute_eval_all.py [--runs A B ...]
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

CSV = os.path.join("experiments_log", "eval_all_runs.csv")

# Region-independent: these must not move when only the mask changes.
INVARIANT = ["DCD", "CD_t_mm", "HD95_mm", "F1@0.05", "precision", "recall",
             "F1@0.03", "clump_%", "spacing_CV"]
DEFECT = ["defect_gt_%", "defect_cov_mm", "defect_HD95_mm", "defect_n_pred",
          "defect_prec_mm", "defect_F1@0.05"]

DEFAULT_RUNS = ["msn_skullfix/lr_fix_only", "msn_skullfix/rep_w05",
                "msn_skullfix/cd_only", "msn_skullfix/cd_rep05_full",
                "msn_skullfix/cd_rep05_r2", "msn_skullfix/tie_qk",
                "msn_skullfix/tie_qk_r2", "msn_skullfix/notext",
                "msn_skullfix/notext_r2", "msn_skullfix/pp_attn"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=CSV)
    args = ap.parse_args()

    import report as rp

    out = os.path.join(REPO, args.out)
    old = pd.read_csv(out)
    old["id"] = old["id"].astype(str).str.zfill(3)          # ⚠️ 83 -> '083'
    if "defect_def" not in old:
        old["defect_def"] = "5mm_legacy"

    rp.defect_labels(REPO)                                   # fail now, not after 15 min of GPU
    runs = rp.load_runs(REPO, args.runs)
    print(f"重算 {len(runs)} 个 run（缺损区 = implant 真值）…")
    new = rp.eval_runs(REPO, runs)
    new["id"] = new["id"].astype(str).str.zfill(3)
    new["defect_def"] = "implant"

    # ---- the region-independent columns must not have moved ----
    key = ["run", "id"]
    j = new.merge(old, on=key, suffixes=("", "_old"))
    print(f"\n⭐ 与旧记录重叠 {len(j)} 行，核对与掩码无关的 {len(INVARIANT)} 列：")
    worst = 0.0
    for c in INVARIANT:
        if f"{c}_old" in j:
            d = float((j[c] - j[f"{c}_old"]).abs().max())
            worst = max(worst, d)
    if worst > 1e-9:
        bad = {c: float((j[c] - j[f"{c}_old"]).abs().max()) for c in INVARIANT
               if f"{c}_old" in j and (j[c] - j[f"{c}_old"]).abs().max() > 1e-9}
        raise SystemExit(
            f"⛔ 与掩码无关的列发生了变化：{bad}\n"
            f"   只换掩码不应该动这些列 —— 说明还有别的东西变了，整次重算不可信，已中止。")
    print(f"   最大绝对差 {worst:.3e} ✅（只有缺损区那几列变了，符合预期）")

    print(f"\n=== 缺损区各列的变化（{len(j)} 行重叠）===")
    for c in DEFECT:
        a, b = j[f"{c}_old"].mean(), j[c].mean()
        print(f"  {c:<18}{a:>10.3f} → {b:>9.3f}   {b - a:>+8.3f}"
              f"   ({100 * (b - a) / a:+.1f}%)" if a else "")

    keep = ~old.set_index(key).index.isin(new.set_index(key).index)
    merged = pd.concat([old[keep.tolist()], new], ignore_index=True)
    dup = int(merged.duplicated(key).sum())
    if dup:
        raise SystemExit(f"⛔ 合并后有 {dup} 个重复的 (run, id) —— 键没对上，已中止")
    merged.to_csv(out, index=False)

    print(f"\n{len(merged)} 行 -> {args.out}")
    print(merged.groupby("defect_def").run.nunique().to_string())
    legacy = sorted(merged[merged.defect_def == "5mm_legacy"].run.unique())
    if legacy:
        print(f"⚠️ 仍为旧口径（权重已删、再也算不出来）：{legacy}"
              f"\n   它们是 ⛔ 错误性实验、不进论文，所以不会有数字混引；"
              f"\n   `defect_def` 这一列就是为了让这件事可核查而不是靠记。")


if __name__ == "__main__":
    main()
