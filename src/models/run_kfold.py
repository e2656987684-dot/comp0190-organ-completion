r"""Drive the whole k-fold sweep: 4 configurations x 5 folds, one after another.

WHY A DRIVER AND NOT A for-LOOP IN THE SHELL
  A bare loop runs all twenty and tells you at the end. Twenty hours is long
  enough that the failure modes matter more than the convenience:

    * A run that hits the --epochs ceiling is UNUSABLE (it was still descending
      when truncated -- see cd_rep05_truncated). A loop would keep going and you
      would find out sixteen hours later. This aborts.
    * Twenty checkpoints need ~14 GB and /root had ~19 GB free. Filling the disk
      mid-sweep kills the remaining runs in a way that looks like a training bug.
      This checks before each run.
    * Anything can drop a 20-hour job. This SKIPS runs that already finished, so
      re-running the command resumes instead of starting over -- and it does not
      fight guard_out_dir, which (correctly) refuses to overwrite a finished run.
    * The per-run self-check otherwise happens by hand twenty times, which is
      exactly the kind of thing that gets skipped at 3am.

  What it deliberately does NOT do: pick a configuration, or decide anything.
  The four cells and their flags are the same ones in KFOLD.md, and the two must
  agree -- `--list` prints them so you can eyeball it against that file.

HARD FAILURES ABORT, SOFT ONES WARN AND CONTINUE
  Abort:   non-zero exit, --epochs ceiling hit, disk below --min-free-gb.
  Warn:    fewer than 5 LR decays, late-30-epoch std >= 0.02 mm, val/train >= 1.25.
  Aborting the sweep because fold 2 annealed slightly worse would cost more than
  it saves; those are read-with-care signals, not broken runs. Every warning is
  repeated in the summary at the end. --strict turns them into aborts.

    python src/models/run_kfold.py --dry-run     # 先看它打算干什么
    python src/models/run_kfold.py               # 全部 20 个，约 17~20 小时
    python src/models/run_kfold.py --folds 0     # 只跑 fold 0 的四格（推荐先跑这个）
    python src/models/run_kfold.py --list        # 只打印四格配置

  ⚠️ Needs the GPU: restart the notebook kernel first (one 187M model is
  15.5 / 24 GiB). ⚠️ Run it inside tmux -- twenty hours outlives any ssh session.
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


def done(name):
    """A run counts as finished only if BOTH its weights and its record exist."""
    d = os.path.join(REPO, OUT_SUB, name)
    return (os.path.exists(os.path.join(d, "best.h5"))
            and os.path.exists(os.path.join(d, "run.json")))


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
        stop = "撞 --epochs 上限"
        hard = (f"{name} 撞了 --epochs={m['epochs']} 上限 —— 它还在下降时被截断，"
                f"这一轮不可引用。调高上限重跑。")
    else:
        stop = "墙钟掐停"
        warn.append(f"{name} 是被 --minutes 掐停的，不是自行早停")
    if drops < 5:
        warn.append(f"{name} LR 只衰减了 {drops} 次（<5），检查配置")
    if std >= 0.02:
        warn.append(f"{name} 末 30 轮 std {std:.4f} mm（>=0.02），没退火好，读它的数要当心")
    if ratio >= 1.25:
        warn.append(f"{name} val/train = {ratio:.2f}（>=1.25）")
    line = (f"{stop:14} {n:>4} 轮(最优 {best:>4})  LR降 {drops:>2}  "
            f"末30std {std:.4f}  val/train {ratio:.2f}  "
            f"best_val_CD_t {m['best_val_cd_t_mm']:.3f}mm")
    return hard, warn, line


def archive(name):
    dst = os.path.join(REPO, "experiments_log", name)
    os.makedirs(dst, exist_ok=True)
    for f in ("run.json", "history.csv"):
        shutil.copy2(os.path.join(REPO, OUT_SUB, name, f), os.path.join(dst, f))


def free_gb():
    return shutil.disk_usage(REPO).free / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不训练")
    ap.add_argument("--list", action="store_true", help="只打印四格配置，和 KFOLD.md 对照用")
    ap.add_argument("--strict", action="store_true", help="软警告也中止")
    ap.add_argument("--min-free-gb", type=float, default=2.5,
                    help="每一轮开跑前要求的最小剩余磁盘（一个 best.h5 是 0.72 GB）")
    ap.add_argument("--log-dir", default=os.path.join(REPO, "experiments", "kfold_logs"))
    args = ap.parse_args()

    if args.list:
        print("四格配置（应与 KFOLD.md 完全一致）：")
        for k, v in CONFIGS.items():
            print(f"  {k:16} {' '.join(v)}")
        return

    # 按折走，不按配置走：任何时候停下来都有完整可比的整折
    plan = [(f, c) for f in args.folds for c in args.configs]
    todo = [(f, c) for f, c in plan if not done(f"{c}_f{f}")]
    skip = len(plan) - len(todo)

    print(f"计划 {len(plan)} 个；已完成跳过 {skip} 个；本次要跑 {len(todo)} 个")
    print(f"磁盘剩余 {free_gb():.1f} GB（每个权重 0.72 GB，本次约需 {0.72*len(todo):.1f} GB）\n")
    for i, (f, c) in enumerate(plan, 1):
        name = f"{c}_f{f}"
        mark = "✓ 已完成" if done(name) else "  待跑   "
        print(f"  {i:2d}/{len(plan)}  {mark}  {name:22} "
              f"{' '.join(['--n-folds', str(N_FOLDS), '--fold', str(f)] + CONFIGS[c])}")
    if args.dry_run or not todo:
        print("\n（--dry-run，没有真的训练）" if args.dry_run else "\n全部已完成。")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    all_warn, t0 = [], time.time()
    for i, (f, c) in enumerate(todo, 1):
        name = f"{c}_f{f}"
        if free_gb() < args.min_free_gb:
            sys.exit(f"\n⛔ 磁盘只剩 {free_gb():.1f} GB（< --min-free-gb {args.min_free_gb}）。"
                     f"\n   已完成的 run 都已存档，腾出空间后重跑本命令即可续上。")

        cmd = [PY, os.path.join("src", "models", "train_skullfix.py"),
               "--run-name", name, "--n-folds", str(N_FOLDS), "--fold", str(f)] + CONFIGS[c]
        el = time.time() - t0
        eta = f"，按已用时估计还要 {el / (i - 1) * (len(todo) - i + 1) / 3600:.1f}h" if i > 1 else ""
        print(f"\n{'='*78}\n[{i}/{len(todo)}] {name}   已用 {el/3600:.1f}h{eta}\n"
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
            sys.exit(f"\n⛔ {name} 退出码 {p.returncode} —— 已中止。日志: {log}")

        hard, warn, line = self_check(name)
        print(f"\n  自检 {name}: {line}")
        for w in warn:
            print(f"  ⚠️  {w}")
        all_warn += warn
        if hard:
            sys.exit(f"\n⛔ {hard}\n   前面已完成的 run 都已存档，修好后重跑本命令会自动续上。")
        if warn and args.strict:
            sys.exit("\n⛔ --strict：出现警告即中止。")
        archive(name)
        print(f"  已存档 -> experiments_log/{name}/")

    print(f"\n{'='*78}\n✅ {len(todo)} 个全部完成，用时 {(time.time()-t0)/3600:.1f}h")
    if all_warn:
        print(f"\n⚠️ 共 {len(all_warn)} 条警告（不影响可用性，但读数时要当心）：")
        for w in all_warn:
            print("   -", w)
    print("\n下一步见 KFOLD.md 的「20 个全部跑完之后」。")


if __name__ == "__main__":
    main()
