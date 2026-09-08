"""Recompute eval_all_runs.csv under the implant defect region, keeping what cannot be.

This overwrites the project's most important frozen record, which is why it is a
script rather than a one-liner. The merge key is (run, id), and `id` reads back
from CSV as an integer while evaluation produces a zero-padded string, so a naive
merge matches nothing and every run lands in the file TWICE -- old rows under the
old definition beside new ones, with any reader silently averaging across both.

Every row that can be recomputed is, under the implant ground truth, and each row
is stamped with the definition that produced it:

    defect_def = "implant"      recomputed now
    defect_def = "5mm_legacy"   kept from before, CANNOT be recomputed

Only runs whose checkpoints were deleted fall in the second group, and those are
invalid runs excluded from the write-up anyway, so no reported number mixes
definitions. The column exists so that stays checkable rather than remembered.

Warning: columns that do NOT depend on the region -- CD_t, HD95, F1, DCD,
clump_%, spacing_CV -- are unaffected by the switch, and this verifies that rather
than assuming it. Every recomputed row must reproduce its old values for those to
1e-9 or the script aborts: if they moved, something other than the mask changed
and the whole recompute is suspect.

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

# ⚠️ The only rows allowed to stay under the old definition. Anything else left
# behind means a run was not matched and its stale rows survived beside the fresh
# ones -- which is exactly what happened on the first run of this script:
# eval_all_runs.csv labels lr_fix_only as `lr_fix`, `Run.label` takes the
# directory name, so the two never matched and the same experiment ended up in
# the file twice, under two names AND two definitions. Silent, and invisible to a
# duplicate-(run, id) check because the run names genuinely differ.
EXPECTED_LEGACY = {"baseline", "dcd_l2"}

# The 20 k-fold runs. The ten single-split runs this used to list are superseded
# -- their rows stay frozen in eval_all_runs.csv, and their weights moved to cold
# storage on /workspace, so recomputing them needs those restored first.
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
    print(f"recomputing {len(runs)} runs (defect region = the implant ground truth)...")
    new = rp.eval_runs(REPO, runs)
    new["id"] = new["id"].astype(str).str.zfill(3)
    new["defect_def"] = "implant"

    # ---- the region-independent columns must not have moved ----
    key = ["run", "id"]
    j = new.merge(old, on=key, suffixes=("", "_old"))
    print(f"\n{len(j)} rows overlap the old record; checking the {len(INVARIANT)} columns "
          f"that do not depend on the mask:")
    worst = 0.0
    for c in INVARIANT:
        if f"{c}_old" in j:
            d = float((j[c] - j[f"{c}_old"]).abs().max())
            worst = max(worst, d)
    if worst > 1e-9:
        bad = {c: float((j[c] - j[f"{c}_old"]).abs().max()) for c in INVARIANT
               if f"{c}_old" in j and (j[c] - j[f"{c}_old"]).abs().max() > 1e-9}
        raise SystemExit(
            f"mask-independent columns moved: {bad}\n"
            f"   Changing only the mask cannot touch these, so something else changed and "
            f"the whole recompute is untrustworthy. Aborted.")
    print(f"   largest absolute difference {worst:.3e} -- only the defect columns moved, "
          f"as expected")

    print(f"\n=== how the defect columns moved ({len(j)} overlapping rows) ===")
    for c in DEFECT:
        a, b = j[f"{c}_old"].mean(), j[c].mean()
        print(f"  {c:<18}{a:>10.3f} → {b:>9.3f}   {b - a:>+8.3f}"
              f"   ({100 * (b - a) / a:+.1f}%)" if a else "")

    keep = ~old.set_index(key).index.isin(new.set_index(key).index)
    merged = pd.concat([old[keep.tolist()], new], ignore_index=True)
    dup = int(merged.duplicated(key).sum())
    if dup:
        raise SystemExit(f"the merge produced {dup} duplicate (run, id) pairs -- the key "
                         f"did not match. Aborted.")
    merged.to_csv(out, index=False)

    print(f"\n{len(merged)} rows -> {args.out}")
    print(merged.groupby("defect_def").run.nunique().to_string())
    legacy = sorted(merged[merged.defect_def == "5mm_legacy"].run.unique())
    unexpected = set(legacy) - EXPECTED_LEGACY
    if unexpected:
        raise SystemExit(
            f"these runs should have been recomputed but stayed on the old definition: "
            f"{sorted(unexpected)}\n"
            f"   Most likely the label in the CSV differs from the run's directory name, so "
            f"the old rows\n"
            f"   were not replaced and now sit beside the new ones. Check the names and run "
            f"again.\n"
            f"   (The file just written is already the merged result, so the stale rows have "
            f"to be fixed first.)")
    if legacy:
        print(f"Still on the old definition, their checkpoints being gone: {legacy}"
              f"\n   They are invalid runs, excluded from the write-up, so no reported "
              f"number mixes definitions;"
              f"\n   the `defect_def` column exists so that stays checkable rather than "
              f"remembered.")


if __name__ == "__main__":
    main()
