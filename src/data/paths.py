"""The one place every data path in this project is written down.

Three paths used to be repeated verbatim: the raw nrrd root in 8 files, the
point-cloud cache in 8 places, the BERT cache in 9. Nothing enforced that the
copies agreed, and a disagreement fails SILENTLY -- two runs evaluated on
different data still produce an ordinary-looking table.

Kept dependency-free on purpose: importing `prepare_skullfix` just to read one
string costs 0.47 s and 1133 modules. Paths are RELATIVE to the repo root
because `report.py` takes `repo` as an argument everywhere -- an absolute
constant would silently discard it, since os.path.join drops anything before an
absolute component. Call sites join: os.path.join(REPO, paths.DATA_CACHE).

`report.py` re-exports these names, so `rp.DATA_CACHE` still works.

TODO: src/models/train_skullfix.py still holds its own copies of DATA_CACHE and
BERT_CACHE. Left alone deliberately while the k-fold sweep was running; fold in
once it finishes.
"""

import os

# The raw SkullFix download. `14161307` is the Figshare article id, i.e. the
# directory name the archive extracts to -- see the Data section of README.md.
RAW_ROOT = os.path.join("data", "14161307", "SkullFix", "training_set")

# What prepare_skullfix.py writes, and what training and evaluation actually
# read. The raw nrrd files above are only touched by the studies that need a
# real surface mesh.
DATA_CACHE = os.path.join("data", "cache", "skullfix_pairs_4096_6144.npz")

# The frozen BERT embedding of the word "skull". One class, so it is a constant
# for every sample of every epoch; train_skullfix.py writes it on the first run.
BERT_CACHE = os.path.join("data", "cache", "bert_skull.npy")

# The released MedShapeNet weights, fetched by setup_env.sh from the authors' own
# Google Drive link. Only the pretrained baseline reads them.
# NOTE: setup_env.sh keeps its own literal copy -- it runs before Python exists,
# so it cannot import this module.
MSN_WEIGHTS = os.path.join("msn_downloads", "MSN_weights3.h5")
