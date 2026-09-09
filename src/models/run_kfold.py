r"""Drive the k-fold sweep: 4 configurations x 5 folds, one run after another.

    python src/models/run_kfold.py cd_only          # this model's 5 folds, ~4-5 h
    python src/models/run_kfold.py --all            # all 20, fold-major
    python src/models/run_kfold.py cd_only --folds 0 1
    python src/models/run_kfold.py --dry-run cd_only
    python src/models/run_kfold.py --list           # the four cells, to check against KFOLD.md

One configuration at a time is the default, so its five folds finish together and
its mean and spread can be read before the next five hours are committed. The
cost is that nothing is comparable until the SECOND model finishes -- `--all`
trades that the other way. Reasons in src/models/README.md.

Hard failures abort, soft ones warn and carry on:

  abort   non-zero exit, the --epochs ceiling reached, disk below --min-free-gb
  warn    fewer than 5 LR decays, last-30-epoch std >= 0.02 mm, val/train >= 1.25

Warnings are repeated in the closing summary; --strict turns them into aborts.

This driver decides nothing. The four cells and their flags are the ones in
KFOLD.md and the two must agree -- `--list` prints them for comparison.

Warning: needs the GPU, so restart the notebook kernel first (one 187M model
holds 15.5 of 24 GiB), and run it inside tmux -- twenty hours outlives any ssh
session.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

# The clean 2x2. ⚠️ These must stay identical to KFOLD.md -- that file is the
# human-facing copy (it also gives the notebook control-panel form), this one is
# what actually runs. `--list` prints it for comparison.
CONFIGS = {
    "cd_only":       ["--loss", "cd"],
    "lr_fix_only":   ["--loss", "cd_dcd", "--dcd-lambda", "2"],
    "rep_w05":       ["--loss", "cd_dcd", "--dcd-lambda", "2", "--repulsion-weight", "0.5"],
    "cd_rep05_full": ["--loss", "cd", "--repulsion-weight", "0.5"],
}
N_FOLDS = 5
OUT_SUB = os.path.join("experiments", "msn_skullfix")


def state(name):
    """'done' | 'partial' | 'new' -- what resuming depends on.

    run.json is written only when a run ENDS, while ModelCheckpoint and CSVLogger
    write from epoch 1, so an interrupted run leaves weights without a record.
    guard_out_dir rightly refuses to reuse such a directory; naming that state
    here is what turns its refusal into a message that explains itself.
    """
    d = os.path.join(REPO, OUT_SUB, name)
    has_w = os.path.exists(os.path.join(d, "best.h5"))
    has_m = os.path.exists(os.path.join(d, "run.json"))
    if has_w and has_m:
        return "done"
    if any(os.path.exists(os.path.join(d, f))
           for f in ("best.h5", "last.h5", "history.csv", "run.json")):
        return "partial"
    return "new"


def done(name):
    return state(name) == "done"


def self_check(name):
    """(hard_error, warnings, summary_line) for a finished run."""
    import pandas as pd

    d = os.path.join(REPO, OUT_SUB, name)
    m = json.load(open(os.path.join(d, "run.json")))
    h = pd.read_csv(os.path.join(d, "history.csv"))
    n = len(h)
    best = int(h["val_loss"].idxmin()) + 1
    drops = sum(1 for i in range(1, n) if h["lr"][i] < h["lr"][i - 1] - 1e-12)
    std = float((h["val_cd_t_metric"] * m["scale_mm"]).tail(30).std())
    ratio = m["final"]["val_cd_t_metric"] / m["final"]["cd_t_metric"]

    hard, warn = None, []
    # Early stopping is decisive: it stops exactly `patience` epochs after the
    # best one. Check it FIRST -- an early run.json without `epochs` has a
    # fallback ceiling lower than what it actually used, and testing the ceiling
    # first mislabels it (that is how cd_only was once reported as truncated).
    if n - best == m["early_stop_patience"]:
        stop = "EarlyStopping"
    elif n >= m.get("epochs", 10 ** 9):
        stop = "hit --epochs"
        hard = (f"{name} reached the --epochs={m['epochs']} ceiling -- it was cut off "
                f"while still descending, so it cannot be quoted. Raise the ceiling "
                f"and run it again.")
    else:
        stop = "wall clock"
        warn.append(f"{name} was stopped by --minutes, not by early stopping")
    if drops < 5:
        warn.append(f"{name} decayed the LR only {drops} times (<5) -- check the config")
    if std >= 0.02:
        warn.append(f"{name} last-30-epoch std {std:.4f} mm (>=0.02): not well annealed, "
                    f"read its numbers with care")
    if ratio >= 1.25:
        warn.append(f"{name} val/train = {ratio:.2f} (>=1.25)")
    line = (f"{stop:14} {n:>4} epochs (best {best:>4})  LR drops {drops:>2}  "
            f"tail30 std {std:.4f}  val/train {ratio:.2f}  "
            f"best_val_CD_t {m['best_val_cd_t_mm']:.3f}mm")
    return hard, warn, line


def archive(name, quiet=False):
    """Copy run.json + history.csv into experiments_log/, the copy git tracks.

    Idempotent, and called for already-finished runs too: an interruption between
    the end of training and this copy would leave a run marked 'done' and skipped
    forever, so its record would never reach git.
    """
    dst = os.path.join(REPO, "experiments_log", name)
    if all(os.path.exists(os.path.join(dst, f)) for f in ("run.json", "history.csv")):
        return False
    os.makedirs(dst, exist_ok=True)
    for f in ("run.json", "history.csv"):
        shutil.copy2(os.path.join(REPO, OUT_SUB, name, f), os.path.join(dst, f))
    if not quiet:
        print(f"  archived late -> experiments_log/{name}/ (missed after the last run)")
    return True


def drop_redundant_last(name):
    """Delete last.h5 when it is byte-identical to best.h5. Never delete blindly.

    EarlyStopping(restore_best_weights=True) means fit() returns holding the best
    epoch's weights, so the two files come out identical and the sweep would need
    28.6 GB instead of 14.3 GB -- it would fill the disk around run 16. A
    last.h5 that DIFFERS is kept: that means the run was truncated rather than
    stopped by itself, and such a run is discarded anyway.
    """
    d = os.path.join(REPO, OUT_SUB, name)
    last = os.path.join(d, "last.h5")
    best = os.path.join(d, "best.h5")
    if not os.path.exists(last) or not os.path.exists(best):
        return
    import hashlib

    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    if md5(last) == md5(best):
        n = os.path.getsize(last)
        os.remove(last)
        print(f"  removed redundant last.h5 (same md5 as best.h5, {n/1e9:.2f} GB saved)")
    else:
        print("  Warning: last.h5 differs from best.h5, so it is kept. That usually "
              "means this run did not stop by itself")


def backup():
    """rsync to /workspace. Warning: /root is ephemeral -- a redeploy wipes experiments/."""
    t = time.time()
    r = subprocess.run(["bash", "sync_workspace.sh", "backup"], cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        print(f"  Warning: backup failed (training continues): {r.stderr.strip()[:120]}")
    else:
        print(f"  backed up to /workspace ({time.time() - t:.0f}s)")


def summarise_model(cfg, names):
    """Five-fold summary for one configuration, so a 20-hour job can be checked
    part way through rather than only at the end."""
    import numpy as np
    import pandas as pd

    print(f"\n{'='*78}\n`{cfg}`: all five folds finished\n{'='*78}")
    print(f"{'run':22}{'epochs':>8}{'best':>6}{'LRdrops':>9}{'tail30std':>11}{'best_val_CD_t':>15}")
    vals = []
    for n in names:
        m = json.load(open(os.path.join(REPO, OUT_SUB, n, "run.json")))
        h = pd.read_csv(os.path.join(REPO, OUT_SUB, n, "history.csv"))
        best = int(h["val_loss"].idxmin()) + 1
        drops = sum(1 for i in range(1, len(h)) if h["lr"][i] < h["lr"][i - 1] - 1e-12)
        std = float((h["val_cd_t_metric"] * m["scale_mm"]).tail(30).std())
        v = m["best_val_cd_t_mm"]
        vals.append(v)
        print(f"{n:22}{len(h):>8}{best:>6}{drops:>9}{std:>11.4f}{v:>15.3f}")
    a = np.array(vals)
    print(f"{'five-fold mean +- std':22}{'':>34}{a.mean():>10.3f} +- {a.std(ddof=1):.3f}")

    print(f"""
Warning: that column is run.json's CD_t -- training-time, dataset-average scale,
   best over the whole run. It reads about 0.09 mm below the per-skull figure the
   thesis quotes, and CD_t is not the main metric anyway. Use it to judge whether
   these five folds ran normally; quote eval_all_runs.csv.

Main metric (defect coverage) for this model -- needs the GPU, 5-7 minutes,
restart the notebook kernel first:

  $PY -c "
  import os, sys; sys.path.insert(0, 'src/eval'); import report as rp
  R = os.path.abspath('.')
  runs = rp.load_runs(R, [f'msn_skullfix/{cfg}_f{{f}}' for f in range({N_FOLDS})])
  fdf = rp.fold_frame(rp.eval_runs(R, runs), runs)
  print(rp.fold_summary(fdf).to_string())"

Look at what it completes:
  $PY src/eval/mesh_preview.py --run {cfg}_f0 --skull 070 --truth

The 2x2 comparison has to wait for a second model -- fold_paired needs two.""")


def free_gb():
    return shutil.disk_usage(REPO).free / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", choices=list(CONFIGS),
                    help="which model to run; does its 5 folds. Omit it only with --all")
    ap.add_argument("--all", action="store_true",
                    help="run all 20, FOLD-MAJOR: after each fold all four cells are comparable")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, train nothing")
    ap.add_argument("--list", action="store_true", help="print the four cells, to check against KFOLD.md")
    ap.add_argument("--strict", action="store_true", help="abort on soft warnings too")
    ap.add_argument("--min-free-gb", type=float, default=2.5,
                    help="free disk required before each run (one best.h5 is 0.72 GB)")
    ap.add_argument("--clean-partial", action="store_true",
                    help="delete an interrupted run's directory and redo that fold. Warning: "
                         "this deletes weights and cannot be undone; it only applies to "
                         "directories holding a best.h5 with no run.json")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the rsync to /workspace after each run. Warning: /root is "
                         "ephemeral and a redeploy wipes experiments/")
    ap.add_argument("--log-dir", default=os.path.join(REPO, "experiments", "kfold_logs"))
    args = ap.parse_args()

    if args.list:
        print("The four cells (must match KFOLD.md exactly):")
        for k, v in CONFIGS.items():
            print(f"  {k:16} {' '.join(v)}")
        return

    if not args.model and not args.all:
        sys.exit("name a model (this runs its 5 folds), or pass --all for all 20:\n"
                 "  python src/models/run_kfold.py cd_only\n"
                 "  python src/models/run_kfold.py --list      # what the four models are")
    if args.model and args.all:
        sys.exit("--all and naming a model are mutually exclusive")

    if args.model:
        # config-major: one model's five folds together, so its mean and spread
        # can be read as soon as it finishes
        configs, plan = [args.model], [(f, args.model) for f in args.folds]
    else:
        # --all is fold-major: stop at any point and every completed fold is
        # comparable across all four cells
        configs = list(CONFIGS)
        plan = [(f, c) for f in args.folds for c in configs]
    partial = [f"{c}_f{f}" for f, c in plan if state(f"{c}_f{f}") == "partial"]
    if partial and args.clean_partial:
        for n in partial:
            shutil.rmtree(os.path.join(REPO, OUT_SUB, n))
            print(f"deleted the interrupted directory: {n}")
        partial = []
    elif partial:
        sys.exit(
            "These directories hold an INTERRUPTED run -- weights but no run.json,\n"
            "so training never finished:\n"
            + "".join(f"     {n}\n" for n in partial) +
            "   train_skullfix's guard_out_dir refuses to reuse them, correctly: the\n"
            "   checkpoint is overwritten from epoch 1 while run.json is written only at\n"
            "   the end, so mixing the two makes the record and the weights describe\n"
            "   different trainings.\n\n"
            "   Partial weights are of no use -- incomplete, and with no record to match.\n"
            "   Delete them and redo that fold:\n"
            "     python src/models/run_kfold.py " + " ".join(sys.argv[1:]) + " --clean-partial")

    # Archive anything finished but not yet archived: an interruption between the
    # end of training and the copy would leave the run marked done, skipped
    # forever, and its record would never reach git.
    for f, c in plan:
        if done(f"{c}_f{f}"):
            archive(f"{c}_f{f}")

    todo = [(f, c) for f, c in plan if not done(f"{c}_f{f}")]
    skip = len(plan) - len(todo)

    print(f"planned {len(plan)}; {skip} already finished and skipped; {len(todo)} to run")
    print(f"{free_gb():.1f} GB free, about {0.72*len(todo):.1f} GB needed "
          f"(0.72 GB per run for best.h5; a last.h5 of the same size exists briefly "
          f"at the end of each run and is deleted once its md5 matches)\n")
    for i, (f, c) in enumerate(plan, 1):
        name = f"{c}_f{f}"
        mark = {"done": "done      ", "partial": "INTERRUPTED", "new": "to run    "}[state(name)]
        print(f"  {i:2d}/{len(plan)}  {mark}  {name:22} "
              f"{' '.join(['--n-folds', str(N_FOLDS), '--fold', str(f)] + CONFIGS[c])}")
    if args.dry_run or not todo:
        print("\n(--dry-run: nothing was trained)" if args.dry_run else "\nAll finished.")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    all_warn, t0 = [], time.time()
    for i, (f, c) in enumerate(todo, 1):
        name = f"{c}_f{f}"
        if free_gb() < args.min_free_gb:
            sys.exit(f"\nOnly {free_gb():.1f} GB free (< --min-free-gb {args.min_free_gb}).\n"
                     f"   Everything finished so far is archived; free some space and "
                     f"re-run this command to resume.")

        cmd = [PY, os.path.join("src", "models", "train_skullfix.py"),
               "--run-name", name, "--n-folds", str(N_FOLDS), "--fold", str(f)] + CONFIGS[c]
        el = time.time() - t0
        eta = f", ~{el / (i - 1) * (len(todo) - i + 1) / 3600:.1f}h left at this rate" if i > 1 else ""
        print(f"\n{'='*78}\n[{i}/{len(todo)}] {name}   {el/3600:.1f}h elapsed{eta}\n"
              f"{' '.join(cmd)}\n{'='*78}", flush=True)

        log = os.path.join(args.log_dir, f"{name}.log")
        with open(log, "w") as fh:
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end="", flush=True)
                fh.write(line)
            p.wait()
        if p.returncode != 0:
            sys.exit(f"\n{name} exited {p.returncode} -- aborted. Log: {log}")

        hard, warn, line = self_check(name)
        print(f"\n  self-check {name}: {line}")
        for w in warn:
            print(f"  ⚠️  {w}")
        all_warn += warn
        if hard:
            sys.exit(f"\n{hard}\n   Everything finished so far is archived; fix it and "
                     f"re-run this command to resume.")
        if warn and args.strict:
            sys.exit("\n--strict: aborting on a warning.")
        drop_redundant_last(name)      # before the backup, so rsync sends half as much
        archive(name, quiet=True)
        print(f"  archived -> experiments_log/{name}/")
        if not args.no_backup:
            backup()

    print(f"\n{'='*78}\nAll {len(todo)} finished in {(time.time()-t0)/3600:.1f}h")
    if all_warn:
        print(f"\nWarning: {len(all_warn)} advisory checks tripped. The runs are usable, "
              f"but read their numbers with care:")
        for w in all_warn:
            print("   -", w)

    for c in configs:
        names = [f"{c}_f{f}" for f in args.folds]
        if all(done(n) for n in names):
            summarise_model(c, names)
    print("\nWhat to do next: KFOLD.md.")


if __name__ == "__main__":
    main()
