#!/bin/bash
# One-shot environment setup. Clone the repository, cd into it, run this.
#
# Creates a single conda environment (comp0190-msn) covering everything the code
# needs, fetches the released MSN weights used by the baseline comparison, and
# installs the headless browser that plotly needs to export figures.
#
# Assumes Linux with apt and enough privilege to install packages, which is the
# usual case in a GPU container. Everything else is self-contained: conda goes
# under $HOME unless CONDA_ROOT says otherwise, and nothing outside the
# repository is modified.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== project root: $PROJECT_ROOT ==="

# Optional machine-specific setup, not tracked in git. Anything that belongs to
# one particular deployment rather than to the project lives there.
if [ -f "$PROJECT_ROOT/setup_local.sh" ]; then
    echo "=== sourcing setup_local.sh ==="
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/setup_local.sh"
fi

# ---- 1. the released MSN weights ----
# Only the baseline comparison needs these; training does not.
GDRIVE_FILE_ID="1VBAy9tQ5kProgmdNAS1vcnOIqRkee-9V"
DOWNLOAD_DIR="$PROJECT_ROOT/msn_downloads"
if ! mkdir -p "$DOWNLOAD_DIR" 2>/dev/null; then
    echo "warning: $DOWNLOAD_DIR is not writable, using $HOME/msn_downloads"
    DOWNLOAD_DIR="$HOME/msn_downloads"
    mkdir -p "$DOWNLOAD_DIR"
fi
WEIGHTS_FILE="$DOWNLOAD_DIR/MSN_weights3.h5"

export PATH="$PATH:$HOME/.local/bin"
if [ ! -s "$WEIGHTS_FILE" ]; then
    echo "=== downloading MSN_weights3.h5 (1.2 GB) to $WEIGHTS_FILE ==="
    python3 -m pip install --user --quiet gdown
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "$WEIGHTS_FILE"
else
    echo "=== MSN_weights3.h5 already present, skipping download ==="
fi

# The upstream inference notebook loads the weights by the bare relative name, so
# it needs to find them in its own directory.
# Use a RELATIVE symlink: an absolute one is valid only on the machine and at the
# path where it was made, so it breaks for anyone who clones elsewhere -- and the
# failure looks like "the weights did not download". The fallback directory sits
# outside the repository, where only an absolute link is possible.
mkdir -p "$PROJECT_ROOT/notebooks/upstream_msn"
DEMO_LINK="$PROJECT_ROOT/notebooks/upstream_msn/MSN_weights3.h5"
case "$WEIGHTS_FILE" in
    "$PROJECT_ROOT"/*) ln -sfr "$WEIGHTS_FILE" "$DEMO_LINK" ;;
    *)                 ln -sf  "$WEIGHTS_FILE" "$DEMO_LINK" ;;
esac

# ---- 2. miniconda ----
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
if ! command -v conda &> /dev/null && [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "=== installing miniconda into $CONDA_ROOT ==="
    curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# The default channels now require the terms to be accepted, or `conda create`
# fails non-interactively.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r > /dev/null 2>&1 || true

# So a new shell can use conda without sourcing this script.
if ! grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null; then
    "$CONDA_ROOT/bin/conda" init bash > /dev/null
fi

# ---- 3. the project environment ----
if ! conda env list | grep -q "^comp0190-msn "; then
    conda create -y -n comp0190-msn python=3.11
fi
conda activate comp0190-msn
pip install --upgrade pip ipykernel
pip install -r "$PROJECT_ROOT/requirements-msn.txt"
python -m ipykernel install --user --name comp0190-msn --display-name "comp0190-msn (conda)"

echo "=== checking tensorflow and the BERT branch ==="
python -c "
import tensorflow as tf
print('tensorflow', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
from transformers import TFBertModel, BertTokenizer
print('TFBertModel import: OK')
"

# ---- 4. figure export ----
# kaleido renders plotly figures through a headless Chrome, so without one every
# fig.write_image() raises and no figure can be produced.
# Chrome also links against libnss3 and libnspr4, which container base images
# usually lack. Without them it exits immediately after starting, and the error
# says nothing about a missing library.
echo "=== figure export (kaleido + headless Chrome) ==="
apt-get install -y --no-install-recommends libnss3 libnspr4 >/dev/null 2>&1 || \
    (apt-get update -qq && apt-get install -y --no-install-recommends libnss3 libnspr4)
plotly_get_chrome -y || echo "warning: Chrome install failed -- fig.write_image() will not work"
python -c "
import plotly.graph_objects as go, tempfile, os
p = os.path.join(tempfile.mkdtemp(), 't.png')
go.Figure(go.Scatter(x=[1,2], y=[1,2])).write_image(p)
print('figure export works:', os.path.getsize(p), 'bytes')"
conda deactivate

echo ""
echo "=== done ==="
echo "MSN weights: $WEIGHTS_FILE (symlinked into notebooks/upstream_msn/)"
echo "Select the 'comp0190-msn (conda)' kernel in every notebook."
echo "If a new terminal cannot find conda, open another one or source ~/.bashrc."
