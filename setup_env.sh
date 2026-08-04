#!/bin/bash
# ===== 一键环境搭建脚本（conda 版，单盘部署，不依赖持久盘）=====
# 用法：clone 仓库后 cd 进去，跑 `bash setup_env.sh`。
# 所有东西（conda、两个虚拟环境、MSN 权重）都直接装在当前所在的盘上，
# 不做任何 /workspace 持久盘的软链或路径判断。
#
# 建两个独立的 conda 环境：
#   comp0190        —— 主项目环境（torch + 常规科学计算库），日常用这个
#   comp0190-msn    —— 只给 notebooks/demo/ 里那两个 MSN (PCT+BERT) demo notebook 用
# 拆成两个环境是因为 tensorflow[and-cuda] 和 torch 各自的 nvidia-nccl 包
# （cu12 / cu13）会装到同一个物理路径互相覆盖，导致 torch 报
# "undefined symbol: ncclCommResume"。分开环境后两者不会再共享同一套
# site-packages，这个问题从根上就不会出现，不需要额外修复步骤。

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== 项目目录: $PROJECT_ROOT ==="

# 这台 pod 镜像全局把 PIP_CACHE_DIR / HF_HOME 都指向了网络盘 /workspace，
# 小文件读写 + 文件锁在网络盘上巨慢甚至会卡死（实测卡住需要手动中断），
# 跟 GPU/CPU 算力无关。这里统一改到本地盘，pip 安装、HuggingFace 下载都会快很多。
export PIP_CACHE_DIR=/root/.cache/pip
export HF_HOME=/root/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# Claude Code 会话记录 + 登录凭证默认存在本地盘 ~/.claude，机器一销毁就没了。
# 改成指向 /workspace 网络持久盘，重开机器后聊天记录还在、也不用重新登录，
# 可以直接用 `claude --continue` / `claude --resume` 接着之前的会话聊。
export CLAUDE_CONFIG_DIR=/workspace/.claude-config
mkdir -p "$CLAUDE_CONFIG_DIR"
if ! grep -q "CLAUDE_CONFIG_DIR" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# Claude Code 会话记录持久化：指向 /workspace 网络盘，机器重开后聊天记录/登录状态都还在"
        echo "export CLAUDE_CONFIG_DIR=/workspace/.claude-config"
    } >> "$HOME/.bashrc"
    echo "已写入 CLAUDE_CONFIG_DIR 到 ~/.bashrc"
fi

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

# 让新开的 shell（不 source 这个脚本的情况下）也能直接用 conda / 看到这两个环境
if ! grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null; then
    "$CONDA_ROOT/bin/conda" init bash > /dev/null
fi

# ---- 4. 主项目环境：comp0190（torch）----
if ! conda env list | grep -q "^comp0190 "; then
    conda create -y -n comp0190 python=3.11
fi
conda activate comp0190
pip install --upgrade pip ipykernel
pip install -r "$PROJECT_ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name comp0190 --display-name "comp0190 (conda)" \
    --env PIP_CACHE_DIR "$PIP_CACHE_DIR"
echo "=== 验证 comp0190（torch）==="
python -c "
import torch
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
"
conda deactivate

# ---- 5. MSN demo 环境：comp0190-msn（tensorflow）----
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
conda deactivate

echo ""
echo "=== 完成 ==="
echo "MSN 权重: $WEIGHTS_FILE（已软链到 notebooks/demo/MSN_weights3.h5）"
echo "主项目开发：VS Code / Jupyter 里选 kernel comp0190 (conda)"
echo "跑 MSN demo notebook：选 kernel comp0190-msn (conda)"
echo "如果 conda 命令在新终端里还是找不到，重开一个终端或者 source ~/.bashrc 一次"
