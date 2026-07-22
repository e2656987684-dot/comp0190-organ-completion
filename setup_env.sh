#!/bin/bash
# ===== 新 pod 一键环境搭建脚本 =====
# 用法：把这个仓库 clone 到本地盘（例如 /root/comp0190-organ-completion），
# cd 进去之后跑 `bash setup_env.sh`。
#
# 背景 / 踩过的坑（详见 requirements.txt 里的注释）：
#   - venv 建在网络盘（/workspace）上装包极慢（几十个包能装 10+ 分钟），所以这次
#     venv 建在本地盘（脚本自己所在目录下的 .venv），本地盘在同一个 pod 内重启不会丢，
#     只有换新 pod 才会清空 —— 换新 pod 就重新跑一遍这个脚本即可，很快。
#   - 必须用 Python 3.11，不能用系统默认的 3.12：PyPI 上 tensorflow<2.16 没有
#     cp312 的 wheel，而 transformers>=5 又彻底砍掉了 TFBertModel（MSN demo 依赖它），
#     两者要求互斥，只有 3.11 能同时满足。
#   - tensorflow[and-cuda] 和 torch 都会装 nvidia-nccl，但 nvidia-nccl-cu12/cu13
#     两个包会装到同一个物理路径 site-packages/nvidia/nccl/，后装的覆盖先装的，
#     导致 torch 报 "undefined symbol: ncclCommResume"。装完后必须强制重装回
#     nvidia-nccl-cu13 才能让 torch 恢复（单卡场景不需要真正的 nccl）。
#   - data/ 和权重文件（MSN_weights3.h5，1.1G）体积大又不好重新获取，留在 /workspace
#     持久盘上，本地盘这边用软链接指过去，代码里的相对路径完全不用改。
#   - git 身份信息要软链 ~/.gitconfig 到 /workspace/.gitconfig，不然 VS Code 自带的
#     Git 面板（不会执行这个脚本的独立进程）读不到，报 "no email was given"。

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

# ---- 3. 建 Python 3.11 venv（本地盘）----
if ! command -v python3.11 &> /dev/null; then
    echo "python3.11 不存在，尝试用 apt 安装..."
    apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv
fi

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    python3.11 -m venv "$PROJECT_ROOT/.venv"
    echo "已建 venv: $PROJECT_ROOT/.venv"
fi

# pip 缓存放持久盘，换新 pod 也能省下重新下载 torch/tensorflow 的时间
export PIP_CACHE_DIR=/workspace/.cache/pip

# ---- 4. 装依赖 ----
"$PROJECT_ROOT/.venv/bin/pip" install --upgrade pip ipykernel -r "$PROJECT_ROOT/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 修复 nvidia-nccl-cu12/cu13 共享安装路径导致 torch 被覆盖的问题
"$PROJECT_ROOT/.venv/bin/pip" install --force-reinstall --no-deps "nvidia-nccl-cu13==2.29.7"

# ---- 5. 注册 Jupyter kernel ----
"$PROJECT_ROOT/.venv/bin/python" -m ipykernel install --user --name comp0190-venv \
    --display-name "comp0190 (.venv)"

# ---- 6. 验证 ----
echo "=== 验证 ==="
"$PROJECT_ROOT/.venv/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
import tensorflow as tf
print('tensorflow', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
from transformers import TFBertModel, BertTokenizer
print('TFBertModel import: OK')
"

echo "=== 完成，VS Code / Jupyter 里选 kernel: comp0190 (.venv) ==="
