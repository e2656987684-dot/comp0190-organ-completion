#!/bin/bash
# Personal pod setup -- NOT tracked in git, and not part of the project.
#
# setup_env.sh sources this if it exists, so everything that is specific to this
# machine lives here and the public script stays runnable by anyone.
#
# Three things, none of which belong in a project repository:
#   1. git identity, so a fresh pod commits under the right name
#   2. Claude Code's config and history, symlinked onto the network disk so a
#      destroyed pod does not take the chat log with it
#   3. pip / HuggingFace caches moved off the network disk, where small-file
#      writes and file locks are slow enough to hang

# ---- 1. git identity (only if this machine has none) ----
if [ -z "$(git config --global user.name 2>/dev/null)" ]; then
    git config --global user.name "jinyu qi"
    echo "git user.name set"
fi
if [ -z "$(git config --global user.email 2>/dev/null)" ]; then
    git config --global user.email "e2656987684@gmail.com"
    echo "git user.email set"
fi

# ---- 2. Claude Code config -> network disk ----
# Symlink the paths themselves rather than setting CLAUDE_CONFIG_DIR: the VS Code
# extension host does not source ~/.bashrc, so an environment variable does not
# reach it, and a chat log was lost that way once. Idempotent: an existing
# symlink is left alone, a real directory is moved across, and data already on
# the network disk wins over a local copy.
CLAUDE_CONFIG_DIR=/workspace/.claude-config
mkdir -p "$CLAUDE_CONFIG_DIR"

_link_claude_path() {
    local target="$1" link_name="$2"
    if [ -L "$link_name" ]; then
        return
    fi
    if [ -e "$link_name" ]; then
        if [ -e "$target" ]; then
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
echo "~/.claude and ~/.claude.json point at $CLAUDE_CONFIG_DIR"

# ---- 3. caches off the network disk ----
# This pod's image points PIP_CACHE_DIR and HF_HOME at /workspace. Small-file
# reads and file locks there are slow enough to hang outright, unrelated to CPU
# or GPU. Moving them to local disk makes pip and HuggingFace downloads fast.
export PIP_CACHE_DIR=/root/.cache/pip
export HF_HOME=/root/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"
