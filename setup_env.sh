#!/bin/bash
# ===== 一键环境搭建脚本（conda 版，单盘部署，不依赖持久盘）=====
# 用法：clone 仓库后 cd 进去，跑 `bash setup_env.sh`。
# 所有东西（conda、虚拟环境、MSN 权重）都直接装在当前所在的盘上，
# 不做任何 /workspace 持久盘的软链或路径判断。
#
# 只建一个 conda 环境：
#   comp0190-msn    —— 覆盖全部当前代码：explore_skull.ipynb、
#                       src/data/prepare_skullfix.py，以及 notebooks/demo/
#                       下那两个 MSN (PCT+BERT) demo notebook。
# 曾经拆过一个额外的 comp0190（torch）环境，纯粹是为将来 torch 相关代码
# 预留、跟 tensorflow[and-cuda] 的 nvidia-nccl 包（cu12/cu13）互相隔离用的
# ——但当前代码库里没有任何地方真的 import torch，那个环境已删除。如果
# 以后真的要接入 torch，记得重新拆成独立环境，避免两者的 nccl 包装到同一
# 物理路径互相覆盖，导致 "undefined symbol: ncclCommResume"。

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== 项目目录: $PROJECT_ROOT ==="

# 这台 pod 镜像全局把 PIP_CACHE_DIR / HF_HOME 都指向了网络盘 /workspace，
# 小文件读写 + 文件锁在网络盘上巨慢甚至会卡死（实测卡住需要手动中断），
# 跟 GPU/CPU 算力无关。这里统一改到本地盘，pip 安装、HuggingFace 下载都会快很多。
export PIP_CACHE_DIR=/root/.cache/pip
export HF_HOME=/root/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# Claude Code 会话记录 + 登录凭证默认存在本地盘 ~/.claude（以及 ~/.claude.json），
# 机器一销毁就没了。这里把这两个路径本身换成指向 /workspace 网络持久盘的软链接，
# 这样不管 claude 进程是被谁、怎么拉起来的（VS Code 插件走的是 extension host
# 链路，不会 source ~/.bashrc，之前用 CLAUDE_CONFIG_DIR 环境变量的方案在这种
# 场景下不生效——曾经因为这个在换新机器时丢过一次聊天记录），只要它按默认路径
# 读写 ~/.claude*，都会透明落到网络盘上。
# 幂等：已经是软链接就跳过；本地是真目录/文件就搬过去再建软链（网络盘上已有
# 数据 —— 比如换了个新 pod —— 就直接复用，不覆盖）。
CLAUDE_CONFIG_DIR=/workspace/.claude-config
mkdir -p "$CLAUDE_CONFIG_DIR"

_link_claude_path() {
    local target="$1" link_name="$2"
    if [ -L "$link_name" ]; then
        return  # 已经是软链接，跳过
    fi
    if [ -e "$link_name" ]; then
        if [ -e "$target" ]; then
            # 网络盘上已有数据（比如新 pod 上跑这个脚本）：本地的当备份，不覆盖网络盘
            mv "$link_name" "${link_name}.local-backup"
        else
            mv "$link_name" "$target"
        fi
    fi
    ln -s "$target" "$link_name"
}

_link_claude_path "$CLAUDE_CONFIG_DIR" "$HOME/.claude"
_link_claude_path "$CLAUDE_CONFIG_DIR/.claude.json" "$HOME/.claude.json"
unset -f _link_claude_path
echo "已确认 ~/.claude 和 ~/.claude.json 指向 $CLAUDE_CONFIG_DIR"

# ---- 1. git 身份信息（新机器没配过的话自动配一次，不覆盖已有配置）----
if [ -z "$(git config --global user.name 2>/dev/null)" ]; then
    git config --global user.name "jinyu qi"
    echo "已配置 git user.name = jinyu qi"
fi
if [ -z "$(git config --global user.email 2>/dev/null)" ]; then
    git config --global user.email "e2656987684@gmail.com"
    echo "已配置 git user.email = e2656987684@gmail.com"
fi

# ---- 2. 自动下载 MSN 权重（Google Drive）到 msn_downloads/ ----
GDRIVE_FILE_ID="1VBAy9tQ5kProgmdNAS1vcnOIqRkee-9V"
DOWNLOAD_DIR="$PROJECT_ROOT/msn_downloads"
if ! mkdir -p "$DOWNLOAD_DIR" 2>/dev/null; then
    echo "警告：$DOWNLOAD_DIR 不可写，改用 /root/msn_downloads"
    DOWNLOAD_DIR="/root/msn_downloads"
    mkdir -p "$DOWNLOAD_DIR"
fi
WEIGHTS_FILE="$DOWNLOAD_DIR/MSN_weights3.h5"

export PATH="$PATH:$HOME/.local/bin"
if [ ! -s "$WEIGHTS_FILE" ]; then
    echo "=== 下载 MSN_weights3.h5 到 $WEIGHTS_FILE ==="
    python3 -m pip install --user --quiet gdown
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "$WEIGHTS_FILE"
else
    echo "=== MSN_weights3.h5 已存在于 $WEIGHTS_FILE，跳过下载 ==="
fi

# demo notebook 用相对路径 "MSN_weights3.h5" 读权重（需要在 notebooks/demo/ 下能找到），软链过去
mkdir -p "$PROJECT_ROOT/notebooks/demo"
ln -sf "$WEIGHTS_FILE" "$PROJECT_ROOT/notebooks/demo/MSN_weights3.h5"

# ---- 3. 装 miniconda（如果没有）----
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
if ! command -v conda &> /dev/null && [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "=== 安装 miniconda 到 $CONDA_ROOT ==="
    curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# anaconda 默认渠道现在要求先接受服务条款，否则 conda create 会非交互式报错退出
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r > /dev/null 2>&1 || true

# 让新开的 shell（不 source 这个脚本的情况下）也能直接用 conda / 看到这个环境
if ! grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null; then
    "$CONDA_ROOT/bin/conda" init bash > /dev/null
fi

# ---- 4. 项目环境：comp0190-msn（tensorflow + 常规科学计算库）----
if ! conda env list | grep -q "^comp0190-msn "; then
    conda create -y -n comp0190-msn python=3.11
fi
conda activate comp0190-msn
pip install --upgrade pip ipykernel
pip install -r "$PROJECT_ROOT/requirements-msn.txt"
python -m ipykernel install --user --name comp0190-msn --display-name "comp0190-msn (conda)" \
    --env PIP_CACHE_DIR "$PIP_CACHE_DIR" --env HF_HOME "$HF_HOME"
echo "=== 验证 comp0190-msn（tensorflow + TFBertModel）==="
python -c "
import tensorflow as tf
print('tensorflow', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
from transformers import TFBertModel, BertTokenizer
print('TFBertModel import: OK')
"

# ---- 5. 出图依赖：kaleido 要一个无头 Chrome 才能把 plotly 图导成 PNG/SVG/PDF ----
# 2026-09-06 才发现这条是缺的：make_report_figures.py 里每一行 fig.write_image()
# 在没有 Chrome 时都会抛 RuntimeError，而论文的图全靠它。
# ⚠️ Chrome 装在 /root/.local/share/choreographer/ —— /root 是临时盘，重部署会清空，
#    所以这一步每次重建环境都要跑，不能只跑一次。
echo "=== 出图依赖（kaleido + 无头 Chrome）==="
apt-get install -y --no-install-recommends libnss3 libnspr4 >/dev/null 2>&1 || \
    (apt-get update -qq && apt-get install -y --no-install-recommends libnss3 libnspr4)
    # ⚠️ 光下 Chrome 不够：它动态链接 libnss3/libnspr4，容器基础镜像里没有，
    #    缺了会「启动后立刻退出」，报错信息完全不提是缺库。
plotly_get_chrome -y || echo "⚠️ Chrome 安装失败 —— fig.write_image() 会用不了，出图前必须修"
python -c "
import plotly.graph_objects as go, tempfile, os
p = os.path.join(tempfile.mkdtemp(), 't.png')
go.Figure(go.Scatter(x=[1,2], y=[1,2])).write_image(p)
print('出图自检 OK:', os.path.getsize(p), 'bytes')"
conda deactivate

echo ""
echo "=== 完成 ==="
echo "MSN 权重: $WEIGHTS_FILE（已软链到 notebooks/demo/MSN_weights3.h5）"
echo "所有 notebook / 脚本统一选 kernel comp0190-msn (conda)"
echo "如果 conda 命令在新终端里还是找不到，重开一个终端或者 source ~/.bashrc 一次"
