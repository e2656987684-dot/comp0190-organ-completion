#!/bin/bash
# ===== 本地盘 <-> /workspace 网络盘 手动同步 =====
#
# 为什么需要这个脚本：
#   /root 在 overlay 临时盘上（40G），重新部署 pod 会被完全清空；
#   /workspace 是 MooseFS 网络盘，跨部署保留。
#   平时在本地盘干活（网络盘小文件 I/O 慢，会拖慢训练），关机前同步过去。
#
# 为什么 git 不够：
#   .gitignore 排除了 /data/、experiments/、*.h5、*.npz —— 而这些正是
#   丢了最麻烦的东西（experiments/ 4.4G 权重、msn_downloads/ 1.2G 预训练权重）。
#   git 保的是代码和实验记录，这个脚本保的是产物和数据，两者互补，都要做。
#
# 用法：
#   bash sync_workspace.sh backup            关机前：本地 -> workspace
#   bash sync_workspace.sh restore           新机器：workspace -> 本地
#   bash sync_workspace.sh backup --dry-run  只看会传什么，不真传
#   bash sync_workspace.sh backup --delete   严格镜像（本地删了的备份也删）
#
# 新机器上本地仓库还不存在，所以要从 workspace 那份调用：
#   bash /workspace/comp0190-organ-completion/sync_workspace.sh restore

set -euo pipefail

LOCAL="${LOCAL_REPO:-/root/comp0190-organ-completion}"
REMOTE="${WORKSPACE_REPO:-/workspace/comp0190-organ-completion}"

MODE="${1:-}"
shift || true
EXTRA=("$@")

usage() {
    echo "用法: bash sync_workspace.sh {backup|restore} [--dry-run] [--delete]"
    echo "  backup   本地 -> workspace （关机前跑）"
    echo "  restore  workspace -> 本地 （新机器上跑）"
    exit 1
}

[ "$MODE" = "backup" ] || [ "$MODE" = "restore" ] || usage

# 全新 pod 上第一件事就是 restore，那时候镜像里未必有 rsync。
# 与其让它报一句看不懂的 "command not found"，不如直接给出装法。
if ! command -v rsync > /dev/null 2>&1; then
    echo "✗ 没有 rsync。先装一下："
    echo "    apt-get update && apt-get install -y rsync"
    exit 1
fi

# 传输时排除的东西：能瞬间重建，传了纯浪费网络盘 I/O
EXCLUDES=(
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.ipynb_checkpoints/'
    --exclude '.DS_Store'
)

# rsync 选项说明：
#   -a  归档模式：递归 + 保留权限/时间戳/软链接（软链接必须保留，
#       notebooks/demo/MSN_weights3.h5 就是一个指向 msn_downloads/ 的软链）
#   -h  人类可读的大小
#   --info=progress2  整体进度条，而不是刷屏打印每个文件
RSYNC_OPTS=(-a -h --info=progress2 "${EXCLUDES[@]}" "${EXTRA[@]}")

if [ "$MODE" = "backup" ]; then
    SRC="$LOCAL"
    DST="$REMOTE"
    # 安全检查：本地仓库必须看起来是完整的才允许往备份上写。
    # 防的是这种情况：新机器上还没 restore 就误跑了 backup，
    # 用一个空的/残缺的本地目录去覆盖唯一一份好备份。
    for must in src devlog.md .git; do
        if [ ! -e "$LOCAL/$must" ]; then
            echo "✗ 本地仓库不完整：缺少 $LOCAL/$must"
            echo "  拒绝备份 —— 先确认本地目录对不对，或者你其实想跑的是 restore。"
            exit 1
        fi
    done
else
    SRC="$REMOTE"
    DST="$LOCAL"
    if [ ! -d "$REMOTE" ]; then
        echo "✗ 找不到备份目录 $REMOTE"
        exit 1
    fi
fi

echo "=== $MODE: $SRC  ->  $DST ==="
echo "源大小: $(du -sh "$SRC" 2>/dev/null | cut -f1)"
if [[ " ${EXTRA[*]} " == *" --delete "* ]]; then
    echo "⚠ 启用了 --delete：目标端多出来的文件会被删除"
fi
echo ""

mkdir -p "$DST"
# 源路径结尾的斜杠很关键：有斜杠 = 同步目录「内容」，没有 = 把目录本身塞进目标里
rsync "${RSYNC_OPTS[@]}" "$SRC/" "$DST/"

echo ""
echo "=== 完成 ==="
echo "目标大小: $(du -sh "$DST" 2>/dev/null | cut -f1)"

if [ "$MODE" = "backup" ]; then
    # 备份完顺手提醒一下 git 那半边 —— 两者互补，只做一个都不算安全
    cd "$LOCAL"
    dirty=$(git status --porcelain 2>/dev/null | wc -l)
    [ "$dirty" -gt 0 ] && echo "⚠ 还有 $dirty 个文件未 commit"
    if git rev-parse --abbrev-ref '@{upstream}' > /dev/null 2>&1; then
        ahead=$(git rev-list --count '@{upstream}..HEAD' 2>/dev/null || echo 0)
        [ "$ahead" -gt 0 ] && echo "⚠ 有 $ahead 个 commit 未 push"
    else
        echo "⚠ 当前分支没有上游分支 —— 这些 commit 还只存在于磁盘上，没进 GitHub"
    fi
fi
