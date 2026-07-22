#!/bin/bash
# ===== 新 pod 一键环境搭建脚本（conda 版）=====
# 用法：把这个仓库 clone 到本地盘（例如 /root/comp0190-organ-completion），
# cd 进去之后跑 `bash setup_env.sh`。
#
# 建两个独立的 conda 环境：
#   comp0190        —— 主项目环境（torch + 常规科学计算库），日常用这个
#   comp0190-msn    —— 只给 notebooks/demo/ 里那两个 MSN (PCT+BERT) demo notebook 用
# 拆成两个环境是因为 tensorflow[and-cuda] 和 torch 各自的 nvidia-nccl 包
# （cu12 / cu13）会装到同一个物理路径互相覆盖，导致 torch 报
# "undefined symbol: ncclCommResume"。分开环境后两者不会再共享同一套
# site-packages，这个问题从根上就不会出现，不需要额外修复步骤。
#
# 背景：本地盘（/root 等，容器 overlay）在同一个 pod 重启/关机重开时是保留的，
# 只有换到新 pod（重新拉镜像）才会清空——这个脚本就是给"换新 pod"准备的。

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 持久盘上存 data/ 和权重文件的位置，换新 pod 时如果网络盘挂载路径不一样，改这里即可
WORKSPACE_DATA_ROOT="${WORKSPACE_DATA_ROOT:-/workspace/comp0190-organ-completion}"

echo "=== 项目目录: $PROJECT_ROOT ==="
echo "=== 持久数据目录: $WORKSPACE_DATA_ROOT ==="

# ---- 1. 软链 data/ 和权重文件到持久盘 ----
if [ -d "$WORKSPACE_DATA_ROOT/data" ]; then
    if [ ! -e "$PROJECT_ROOT/data" ]; then
        ln -s "$WORKSPACE_DATA_ROOT/data" "$PROJECT_ROOT/data"
        echo "已软链 data/ -> $WORKSPACE_DATA_ROOT/data"
    fi
else
    echo "警告：$WORKSPACE_DATA_ROOT/data 不存在，data/ 需要你手动传一份到持久盘上"
fi

WEIGHTS_SRC="$WORKSPACE_DATA_ROOT/notebooks/demo/MSN_weights3.h5"
if [ -f "$WEIGHTS_SRC" ]; then
    if [ ! -e "$PROJECT_ROOT/notebooks/demo/MSN_weights3.h5" ]; then
        ln -s "$WEIGHTS_SRC" "$PROJECT_ROOT/notebooks/demo/MSN_weights3.h5"
        echo "已软链 MSN_weights3.h5 -> $WEIGHTS_SRC"
    fi
else
    echo "警告：$WEIGHTS_SRC 不存在，MSN_weights3.h5 需要你手动传一份到持久盘上"
fi

# ---- 2. git 身份信息软链（供 VS Code Git 面板等不 source 这个脚本的进程使用）----
if [ -f /workspace/.gitconfig ]; then
    ln -sf /workspace/.gitconfig /root/.gitconfig
    echo "已软链 ~/.gitconfig -> /workspace/.gitconfig"
else
    echo "警告：/workspace/.gitconfig 不存在，git 身份信息需要重新配置一次："
    echo '  git config --global user.name "你的名字"'
    echo '  git config --global user.email "你的邮箱"'
    echo "配置完之后手动: cp ~/.gitconfig /workspace/.gitconfig 做一份持久备份"
fi

# ---- 3. 装 miniconda（如果没有）----
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
if ! command -v conda &> /dev/null && [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "=== 安装 miniconda 到 $CONDA_ROOT ==="
    curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# pip 缓存放持久盘，换新 pod 也能省下重新下载 torch/tensorflow 的时间
export PIP_CACHE_DIR=/workspace/.cache/pip

# ---- 4. 主项目环境：comp0190（torch）----
if ! conda env list | grep -q "^comp0190 "; then
    conda create -y -n comp0190 python=3.11
fi
conda activate comp0190
pip install --upgrade pip ipykernel
pip install -r "$PROJECT_ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name comp0190 --display-name "comp0190 (conda)"
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
python -m ipykernel install --user --name comp0190-msn --display-name "comp0190-msn (conda)"
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
echo "主项目开发：VS Code / Jupyter 里选 kernel comp0190 (conda)"
echo "跑 MSN demo notebook：选 kernel comp0190-msn (conda)"
