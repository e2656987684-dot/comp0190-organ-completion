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

ONE MODEL AT A TIME, FIVE FOLDS EACH -- THAT IS THE DEFAULT
    python src/models/run_kfold.py cd_only        # 这一个模型的 5 折，约 4~5 小时
    python src/models/run_kfold.py lr_fix_only    # 跑完上一个，手动切下一个
    python src/models/run_kfold.py rep_w05
    python src/models/run_kfold.py cd_rep05_full

  Five folds of one configuration finish together, so `fold_frame` +
  `fold_summary` read that model's mean +- std the moment it is done -- verified:
  they accept a single configuration, and only `fold_paired` (which compares two)
  needs a second model. So you can look at each model, and run inference on it,
  before committing the next five hours.

  ⚠️ THE TRADE-OFF, STATED HONESTLY. Going config-major means that until the
  SECOND model finishes there is nothing to compare against -- the 2x2 is the
  point of this sweep, and it does not exist yet. Fold-major (`--all`) gives the
  opposite: stop after any fold and all four cells are comparable at that fold,
  but no model is finished. Neither is wrong; config-major is the default because
  the per-model checkpoint is what makes a 20-hour job reviewable.

    python src/models/run_kfold.py --all         # 全部 20 个，按折走（fold-major）
    python src/models/run_kfold.py cd_only --folds 0 1   # 只补跑某几折
    python src/models/run_kfold.py --dry-run cd_only     # 先看它打算干什么
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


def state(name):
    """'done' | 'partial' | 'new'  —— 恢复逻辑全靠这个区分。

    run.json 只在训练**结束时**写，而 ModelCheckpoint / CSVLogger 从第 1 轮就开始写。
    所以一次被打断的训练留下的是「有 best.h5、没有 run.json」——
    而 guard_out_dir 会（正确地）拒绝在它上面重跑。这个函数把那种目录单独认出来，
    否则续跑时拿到的是一句看不懂的 "拒绝启动"。
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


def archive(name, quiet=False):
    """把 run.json + history.csv 复制进 experiments_log/（跟踪进 git 的那份）。

    ⚠️ 幂等，而且**跳过已完成的 run 时也要调一次**：训练结束（run.json 已写）到
    存档之间如果断掉，`state()` 会判成 'done' 而永远跳过它，那条记录就再也进不了 git。
    """
    dst = os.path.join(REPO, "experiments_log", name)
    if all(os.path.exists(os.path.join(dst, f)) for f in ("run.json", "history.csv")):
        return False
    os.makedirs(dst, exist_ok=True)
    for f in ("run.json", "history.csv"):
        shutil.copy2(os.path.join(REPO, OUT_SUB, name, f), os.path.join(dst, f))
    if not quiet:
        print(f"  补存档 -> experiments_log/{name}/（上次跑完没存上）")
    return True


def drop_redundant_last(name):
    """删掉与 best.h5 逐字节相同的 last.h5。⚠️ 不同就保留，绝不盲删。

    `train_skullfix` 每轮结束都写 last.h5，而 `EarlyStopping(restore_best_weights=True)`
    让 fit() 返回时模型里装的已经是最优轮的权重 —— 于是两个文件恒等（8/24 实测 13 个
    run 的 md5 全部相同）。不删的话 20 折要占 28.6 GB 而不是 14.3 GB，**盘会在第
    16~17 个爆掉**。

    保留 last.h5 唯一的价值是"训练被 --minutes / --epochs 截断时能拿到最后状态"，
    而那种 run 本来就要作废（driver 遇到撞上限会直接中止）。所以：md5 相同才删。
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
        print(f"  删掉冗余的 last.h5（与 best.h5 md5 相同，省 {n/1e9:.2f} GB）")
    else:
        print("  ⚠️ last.h5 与 best.h5 不同 —— 保留。这通常意味着这一轮不是自行早停的")


def backup():
    """rsync 到 /workspace。⚠️ /root 是临时盘，重部署会清空 experiments/。"""
    t = time.time()
    r = subprocess.run(["bash", "sync_workspace.sh", "backup"], cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        print(f"  ⚠️ 备份失败（不中止训练）: {r.stderr.strip()[:120]}")
    else:
        print(f"  已备份到 /workspace（{time.time() - t:.0f}s）")


def summarise_model(cfg, names):
    """一个模型五折都跑完之后的小结 —— 让 20 小时的活变成可以中途验收的。"""
    import numpy as np
    import pandas as pd

    print(f"\n{'='*78}\n模型 `{cfg}` 五折完成\n{'='*78}")
    print(f"{'run':22}{'轮数':>6}{'最优':>6}{'LR降':>6}{'末30std':>10}{'best_val_CD_t':>15}")
    vals = []
    for n in names:
        m = json.load(open(os.path.join(REPO, OUT_SUB, n, "run.json")))
        h = pd.read_csv(os.path.join(REPO, OUT_SUB, n, "history.csv"))
        best = int(h["val_loss"].idxmin()) + 1
        drops = sum(1 for i in range(1, len(h)) if h["lr"][i] < h["lr"][i - 1] - 1e-12)
        std = float((h["val_cd_t_metric"] * m["scale_mm"]).tail(30).std())
        v = m["best_val_cd_t_mm"]
        vals.append(v)
        print(f"{n:22}{len(h):>6}{best:>6}{drops:>6}{std:>10.4f}{v:>15.3f}")
    a = np.array(vals)
    print(f"{'五折 均值 ± std':22}{'':>28}{a.mean():>10.3f} ± {a.std(ddof=1):.3f}")

    print(f"""
⚠️ 上面那列是 `run.json` 口径的 CD_t（训练期、数据集平均 scale、全程最优），
   比论文用的逐颅骨口径**系统性低约 0.09mm**，而且 **CD_t 根本不是主指标** ——
   它只能看趋势，判断这五折跑得正不正常。**论文引 `eval_all_runs.csv`。**

▶ 看这个模型的**主指标**（缺损区覆盖）—— 要 GPU，约 5~7 分钟，先 Restart notebook kernel：

  $PY -c "
  import os, sys; sys.path.insert(0, 'src/eval'); import report as rp
  R = os.path.abspath('.')
  runs = rp.load_runs(R, [f'msn_skullfix/{cfg}_f{{f}}' for f in range({N_FOLDS})])
  fdf = rp.fold_frame(rp.eval_runs(R, runs), runs)
  print(rp.fold_summary(fdf).to_string())"

▶ 看一眼它补出来的形状：
  $PY src/eval/mesh_preview.py --run {cfg}_f0 --skull 070 --truth

▶ 跨模型比较（2×2）要等第二个模型跑完 —— `fold_paired` 需要两个配置。""")


def free_gb():
    return shutil.disk_usage(REPO).free / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", choices=list(CONFIGS),
                    help="要跑的模型；一次跑它的 5 折。不给就必须显式 --all")
    ap.add_argument("--all", action="store_true",
                    help="一口气跑全部 20 个，且**按折走**（每跑完一折，四格都可比）")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不训练")
    ap.add_argument("--list", action="store_true", help="只打印四格配置，和 KFOLD.md 对照用")
    ap.add_argument("--strict", action="store_true", help="软警告也中止")
    # ⚠️ 上限，不是预算。train_skullfix 的默认是 600，而 patience 从 20 提到 30 之后
    #    run 会跑得更久 —— cd_only_f0 在 546 轮时仍在创新低，眼看要撞 600。
    #    撞上限的 run 是**不可引用**的（还在下降时被截断），只能作废重跑。
    # ⭐ 提高上限对**没有触到它**的 run 完全没有影响（fit 只是循环上界，
    #    ReduceLROnPlateau 是看平台而非按表衰减，warmup 按步数），所以中途调高是安全的
    #    —— 与 patience 不同，那个改了就是另一个实验。
    ap.add_argument("--epochs", type=int, default=1200,
                    help="--epochs 上限（默认 1200，高于 train_skullfix 自己的 600）。"
                         "只是天花板：没触到它的 run 行为逐位不变，所以中途调高不影响可比性")
    ap.add_argument("--min-free-gb", type=float, default=2.5,
                    help="每一轮开跑前要求的最小剩余磁盘（一个 best.h5 是 0.72 GB）")
    ap.add_argument("--clean-partial", action="store_true",
                    help="删掉被中断的半成品目录后重跑那一轮。⚠️ 删的是权重，单向操作；"
                         "只对「有 best.h5 但没有 run.json」的目录生效")
    ap.add_argument("--no-backup", action="store_true",
                    help="每轮跑完不 rsync 到 /workspace。⚠️ /root 是临时盘，"
                         "重部署会把 experiments/ 清空")
    ap.add_argument("--log-dir", default=os.path.join(REPO, "experiments", "kfold_logs"))
    args = ap.parse_args()

    if args.list:
        print("四格配置（应与 KFOLD.md 完全一致）：")
        for k, v in CONFIGS.items():
            print(f"  {k:16} {' '.join(v)}")
        return

    if not args.model and not args.all:
        sys.exit("请指定一个模型（一次跑它的 5 折），或用 --all 跑全部 20 个：\n"
                 "  python src/models/run_kfold.py cd_only\n"
                 "  python src/models/run_kfold.py --list      # 看四个模型分别是什么")
    if args.model and args.all:
        sys.exit("--all 和指定模型二选一")

    if args.model:
        # 按配置走：这一个模型的 5 折一次跑完，跑完就能看它自己的均值±std
        configs, plan = [args.model], [(f, args.model) for f in args.folds]
    else:
        # --all 按折走：任何时候停下来都有完整可比的整折
        configs = list(CONFIGS)
        plan = [(f, c) for f in args.folds for c in configs]
    # ⚠️ 先看有没有训练在跑。正在写的目录会被 state() 认成 "partial"，
    #    而 --clean-partial 会把**正在训练的权重**删掉。这个必须在任何判断之前拦住。
    running = subprocess.run(["pgrep", "-af", "train_skullfix.py --run-name"],
                             capture_output=True, text=True).stdout.strip()
    if running:
        sys.exit("⛔ 已经有训练在跑，先等它结束（或 kill 掉）再来：\n"
                 + "".join(f"     {l}\n" for l in running.splitlines()) +
                 "   ⚠️ 尤其别在这时候用 --clean-partial —— 它会删掉正在训练的权重。")

    partial = [f"{c}_f{f}" for f, c in plan if state(f"{c}_f{f}") == "partial"]
    if partial and args.clean_partial:
        for n in partial:
            shutil.rmtree(os.path.join(REPO, OUT_SUB, n))
            print(f"已删除中断的半成品目录: {n}")
        partial = []
    elif partial:
        sys.exit(
            "⛔ 这些目录是**被中断的半成品**（有权重但没有 run.json，说明训练没跑完）：\n"
            + "".join(f"     {n}\n" for n in partial) +
            "   train_skullfix 的 guard_out_dir 会拒绝在它们上面重跑（这是对的：\n"
            "   ModelCheckpoint 从第 1 轮就覆盖 best.h5，而 run.json 只在结束时写，\n"
            "   混在一起会让记录和权重描述两次不同的训练）。\n\n"
            "   半成品的权重没有任何用处（不完整、也没有对应的记录），删掉重跑即可：\n"
            "     python src/models/run_kfold.py " + " ".join(sys.argv[1:]) + " --clean-partial")

    # ⚠️ 已完成但没存上档的，先补 —— 训练结束到存档之间断掉的话，
    #    它会被判成 done 而永远跳过，那条记录就再也进不了 git。
    for f, c in plan:
        if done(f"{c}_f{f}"):
            archive(f"{c}_f{f}")

    todo = [(f, c) for f, c in plan if not done(f"{c}_f{f}")]
    skip = len(plan) - len(todo)

    print(f"计划 {len(plan)} 个；已完成跳过 {skip} 个；本次要跑 {len(todo)} 个")
    print(f"磁盘剩余 {free_gb():.1f} GB；本次约需 {0.72*len(todo):.1f} GB"
          f"（每轮留 0.72 GB 的 best.h5；训练结束瞬间会多一个同样大的 last.h5，"
          f"确认 md5 相同后立刻删掉）\n")
    for i, (f, c) in enumerate(plan, 1):
        name = f"{c}_f{f}"
        mark = {"done": "✓ 已完成", "partial": "⚠ 半成品", "new": "  待跑   "}[state(name)]
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
               "--run-name", name, "--n-folds", str(N_FOLDS), "--fold", str(f),
               "--epochs", str(args.epochs)] + CONFIGS[c]
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
        drop_redundant_last(name)      # 先删，备份就少传一半
        archive(name, quiet=True)
        print(f"  已存档 -> experiments_log/{name}/")
        if not args.no_backup:
            backup()

    print(f"\n{'='*78}\n✅ {len(todo)} 个全部完成，用时 {(time.time()-t0)/3600:.1f}h")
    if all_warn:
        print(f"\n⚠️ 共 {len(all_warn)} 条警告（不影响可用性，但读数时要当心）：")
        for w in all_warn:
            print("   -", w)

    for c in configs:
        names = [f"{c}_f{f}" for f in args.folds]
        if all(done(n) for n in names):
            summarise_model(c, names)
    print("\n完整的下一步见 KFOLD.md。")


if __name__ == "__main__":
    main()
