r"""Measure how much attending every attention block in a trained model actually does.

WHAT THIS CHECKS
  The network is sold as a Point Cloud Transformer: sixteen offset-attention
  blocks (eight encoder self-attention, four D1 cross-attention, four D2). An
  attention block that returns near-uniform weights is not attending -- it is
  computing a mean, and `x <- x + LBR(x - mean)` is an affine layer wearing a
  transformer's name. Which of the sixteen are in that state is a claim about
  the architecture, so it has to be measured rather than assumed, and it has to
  be measured from the checkpoint the thesis reports.

  Two DIFFERENT things both produce uniform weights, and the difference matters
  more than the number does:

    STRUCTURAL   every key row is the same vector, so softmax returns 1/n_keys
                 whatever Q and K learned. `key_row_spread` is then exactly 0.
                 This is the four D1 blocks in the published configuration:
                 m1 = tile(global vector), so the block cannot attend even in
                 principle. Nothing was learned or failed to be learned.
    LEARNED      the keys differ, but Q and K never developed enough contrast to
                 separate them -- classic attention collapse. `key_row_spread`
                 is large while `eff_frac` is still ~1. This is what to say
                 something about: 80 training skulls are not enough to learn
                 where to look.

  Reporting them as one number ("12 of 16 collapsed") hides that distinction, so
  the table below keeps them apart and the summary counts them separately.

HOW EACH COLUMN IS DEFINED (state it, because the old temporary script did not)
  eff_keys      exp(H) of the attention row, H the natural-log entropy over the
                key axis, averaged over queries then over skulls. This is the
                perplexity reading of the row: "this query effectively spreads
                itself over N keys". Equals n_keys exactly when uniform.
                ⚠️ Read off the row RENORMALISED to sum to 1 -- see `rows()`. The
                raw rows do not all sum to 1, and entropy on a sub-normalised row
                reads as sharp attention when the truth is the opposite.
  row_mass      what the raw row DOES sum to, averaged over queries. 1.0 is
                normal; below that, this block's own 1e-9 epsilon is suppressing
                queries, and `frac_starved` says how many.
  frac_starved  fraction of query rows carrying less than 0.99 of a unit of
                weight into `att @ V`, i.e. queries the epsilon switched off.
  eff_frac      eff_keys / n_keys. 1.000 is perfectly uniform.
  energy_std    std of the raw q.k scores ACROSS the key axis, averaged over
                queries. It says how much contrast Q and K produced before any
                normalisation, which is the quantity a collapse argument is
                really about -- eff_frac can look uniform simply because the
                rows are long.
  peak_x_unif   max attention weight in a row, in units of the uniform weight.
                1.0 = no key is preferred at all.
  key0_x_unif   the same for key 0 specifically. Only interesting when the key
                sequence begins with the global vector (per_point_attn runs),
                where it answers "did the decoder at least keep looking at the
                global token?" -- measured 0.99, i.e. it did not.
  key_row_spread  std across key ROWS, averaged over channels. A magnitude: how
                different the keys are at all. Read it next to energy_std --
                keys that differ while the scores do not is exactly the learned
                case.
  key_row_range max|row - row_0| over the whole key tensor, and the test that
                decides STRUCTURAL. Exactly 0 when the keys are tiled copies.
                Do not use the std for this: on rows that are bit-for-bit
                identical it still reads ~4e-07, because the mean it subtracts
                is itself rounded.
  ctrl_eff_frac   eff_frac recomputed from the SAME scores with a textbook
                softmax(q.k/sqrt(d), axis=keys) instead of this architecture's
                softmax-over-queries-then-L1. It rules out "you normalised along
                the wrong axis" as the explanation for uniformity: if the scores
                carried contrast, the textbook normalisation would show it.

WHAT IT DOES NOT SHOW
  Not that attention is useless for this task, and not that the encoder is
  wasted. Collapse is a statement about optimisation at this data scale, not a
  verdict on the architecture -- PoinTr / SeedFormer / AdaPoinTr are transformers
  that work, trained on tens of thousands of shapes rather than eighty skulls.
  It also says nothing about whether relieving the collapse would help: the one
  attempt to give D1 something to attend to (`pp_attn`) made everything worse
  BECAUSE the attention stayed uniform and diluted the global vector 2000-fold.

  The numbers are read off ONE checkpoint per run. They are not averaged over
  training seeds and no claim here is a comparison between configurations, so
  they carry no paired statistics -- `std_*` columns are the spread across
  validation skulls, which is the only sampling this measurement does.

⚠️ k 折之后要重跑（每个最终模型各一次），见 src/eval/README.md。
  Collapse is a property of a set of weights, not of the data split, and it has
  reproduced across every checkpoint looked at so far -- but "so far" is three
  checkpoints, and the thesis quotes numbers from whichever model it reports.
  Point --runs at the final models and re-read them.

    python src/eval/attention_collapse.py [--runs A B ...] [--n 3]

  Writes experiments_log/attention_collapse.csv, MERGED on (run, block) so an
  earlier run's rows survive -- which is what lets you analyse one run per
  invocation if three architectures at once exhaust the card.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

OUT_CSV = os.path.join("experiments_log", "attention_collapse.csv")

# A block counts as collapsed when its rows are within 1% of perfectly uniform.
# The measured values are not near this line from either side -- collapsed blocks
# sit at 0.9999+ and the D2 blocks that do attend sit below 0.5 -- so no result
# in this project depends on where exactly it is drawn.
COLLAPSE_FRAC = 0.99

# The three checkpoints whose attention numbers are currently quoted from an
# un-archived script, i.e. the ones whose weights cannot be deleted until this
# has been run: the best configuration, the audit run, and the negative result.
DEFAULT_RUNS = ["msn_skullfix/cd_rep05_full",
                "msn_skullfix/tie_qk",
                "msn_skullfix/pp_attn"]

# What the graph computes, one scalar per block per skull. `eff_frac` is NOT in
# here: it is eff_keys/n_keys, derived after the fact in `_rows_for_run`.
MODEL_STATS = ["eff_keys", "energy_std", "peak_x_unif", "key0_x_unif",
               "key_row_spread", "key_row_range", "ctrl_eff_frac",
               "row_mass", "frac_starved"]


def _stage(block):
    return "encoder" if block.startswith("E-") else block.split("-")[0]


def _blocks(model):
    """Every attention block in the model, in forward order."""
    return [l.name[: -len("_matmul1")] for l in model.layers
            if l.name.endswith("_matmul1")]


def _build_stats_model(tf, model):
    """A model over the same inputs whose outputs are per-sample attention stats.

    Computed inside the graph on purpose: the attention tensors themselves are
    (3072, 3072) per D2 block per skull, and pulling sixteen of them back into
    numpy to take a mean would move ~300 MB per skull for seven scalars.
    """
    from tensorflow.keras import layers as L
    from tensorflow.keras import models as M

    def rows(a):
        """Renormalise each query row to sum to 1 before reading any shape off it.

        Necessary, not cosmetic. This architecture softmaxes along the QUERY axis
        and only then L1-normalises along keys, with an epsilon:

            att = softmax(energy, axis=queries) / (1e-9 + sum over keys)

        A query that loses the query-axis softmax against every key ends up with a
        row whose sum is itself of order 1e-9, at which point the epsilon is half
        the denominator and the row sums to well under 1. Entropy read off such a
        row is not the entropy of a distribution: a row genuinely spread evenly
        over 3072 keys but suppressed to mass 0.48 reads as exp(H) = 67, not 3072,
        which would look like extremely sharp attention while meaning the opposite.
        Measured live: tie_qk's D2-STA3 had mean(max_k att)*n_keys = 0.71, and that
        is < 1 only if the rows do not sum to 1.

        Rows that ARE normal (every encoder and D1 block, where the weights are
        uniform) are unchanged by this, so nothing that reproduced before moves.
        `row_mass` below keeps the suppression itself as a finding rather than
        hiding it.
        """
        return a / (tf.reduce_sum(a, axis=-1, keepdims=True) + 1e-30)

    def eff(a):
        p = tf.clip_by_value(rows(a), 1e-12, 1.0)
        ent = -tf.reduce_sum(p * tf.math.log(p), axis=-1)      # (B, n_query)
        return tf.reduce_mean(tf.exp(ent), axis=-1)            # (B,)

    outputs, index, meta = [], {}, {}
    for name in _blocks(model):
        energy = model.get_layer(name + "_matmul1").output      # (B, n_q, n_k)
        att = model.get_layer(name + "_l1norm").output          # rows sum to 1 over keys
        keys = model.get_layer(name + "_K").input               # (B, n_k, C)
        d = float(model.get_layer(name + "_Q").units)
        n_q, n_k = int(att.shape[1]), int(att.shape[2])
        meta[name] = {"n_queries": n_q, "n_keys": n_k, "head_dim": int(d)}

        stats = {
            "eff_keys": L.Lambda(eff, name=f"st_{name}_eff")(att),
            "energy_std": L.Lambda(
                lambda e: tf.reduce_mean(tf.math.reduce_std(e, axis=-1), axis=-1),
                name=f"st_{name}_estd")(energy),
            "peak_x_unif": L.Lambda(
                lambda a, n=n_k: tf.reduce_mean(tf.reduce_max(rows(a), axis=-1), axis=-1) * n,
                name=f"st_{name}_peak")(att),
            "key0_x_unif": L.Lambda(
                lambda a, n=n_k: tf.reduce_mean(rows(a)[:, :, 0], axis=-1) * n,
                name=f"st_{name}_key0")(att),
            # How much weight each query row actually carries into `att @ V`. 1.0 is
            # a normal row. Anything below says the 1e-9 epsilon in the block's own
            # L1 step is a material part of the denominator, i.e. that query has
            # been switched off -- a property of the architecture, worth reporting.
            "row_mass": L.Lambda(
                lambda a: tf.reduce_mean(tf.reduce_sum(a, axis=-1), axis=-1),
                name=f"st_{name}_mass")(att),
            "frac_starved": L.Lambda(
                lambda a: tf.reduce_mean(
                    tf.cast(tf.reduce_sum(a, axis=-1) < 0.99, tf.float32), axis=-1),
                name=f"st_{name}_starved")(att),
            # How much the keys differ at all, as a magnitude.
            "key_row_spread": L.Lambda(
                lambda k: tf.reduce_mean(tf.math.reduce_std(k, axis=1), axis=-1),
                name=f"st_{name}_keyspread")(keys),
            # The structural test, and it has to be this one rather than the std
            # above: on rows that ARE bit-for-bit identical, reduce_std still
            # returns ~4e-07, because the mean it subtracts is itself rounded.
            # max|row - row_0| is exactly 0 there, measured.
            "key_row_range": L.Lambda(
                lambda k: tf.reduce_max(tf.abs(k - k[:, :1, :]), axis=[1, 2]),
                name=f"st_{name}_keyrange")(keys),
            # Textbook attention on the same scores: scale by 1/sqrt(d), softmax
            # along the KEY axis. If uniformity were an artefact of this
            # architecture's unusual normalisation, this column would disagree.
            "ctrl_eff_frac": L.Lambda(
                lambda e, s=d, n=n_k: eff(tf.nn.softmax(e / np.sqrt(s), axis=-1)) / n,
                name=f"st_{name}_ctrl")(energy),
        }
        for col, tensor in stats.items():
            index[(name, col)] = len(outputs)
            outputs.append(tensor)
    return M.Model(inputs=model.inputs, outputs=outputs), index, meta


def analyse(repo, specs, n_skulls=3, device="/GPU:0"):
    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_skullfix as msn
    import report as rp

    runs = rp.load_runs(repo, specs)
    data = np.load(os.path.join(repo, rp.DATA_CACHE))
    ids, inputs = data["ids"], data["inputs"]
    text_path = os.path.join(repo, "data", "cache", "bert_skull.npy")
    text = np.load(text_path) if os.path.exists(text_path) else None

    # One model per ARCHITECTURE, as in report.eval_runs: TF's allocator does not
    # hand memory back on clear_session(), so rebuilding a 187M model per run
    # stacks and the fourth one exhausts a 24 GB card. Three architectures is the
    # most that has been run in one go; past that, invoke once per run and let the
    # merge-write below join the rows.
    groups = {}
    for r in runs:
        groups.setdefault(r.arch_key, []).append(r)

    rows = []
    with tf.device(device):
        for arch, group in groups.items():
            cfg = rp.arch_config(msn, arch)
            if cfg.use_text and text is None:
                raise FileNotFoundError(
                    f"{group[0].label} was trained with the text branch, but "
                    f"{text_path} is missing")
            model = msn.build_model(cfg)
            stats_model, index, meta = _build_stats_model(tf, model)
            for run in group:
                model.load_weights(run.weights)
                val = run.meta["val_ids"][:n_skulls]
                pos = [int(np.where(ids == sid)[0][0]) for sid in val]
                x = [inputs[pos]]
                if cfg.use_text:
                    x.append(np.tile(text[None], (len(pos), 1)))
                out = stats_model.predict(x, batch_size=1, verbose=0)
                run_rows = _rows_for_run(run, out, index, meta, val)
                report(run, run_rows)
                rows += run_rows
            del stats_model, model
            tf.keras.backend.clear_session()
    return pd.DataFrame(rows)


def _rows_for_run(run, out, index, meta, val):
    """One row per block: mean and spread across the skulls, plus the verdicts.

    `std_*` is the spread across VALIDATION SKULLS, which is the only sampling
    this measurement does -- it is not a training-noise estimate and no claim
    here compares two configurations, so it carries no paired statistics.
    """
    rows = []
    for block, m in meta.items():
        row = {"run": run.label, "arch": run.arch_label,
               "stage": _stage(block), "block": block, **m}
        for col in MODEL_STATS:
            v = np.asarray(out[index[(block, col)]], dtype=np.float64).ravel()
            row[col] = float(v.mean())
            row["std_" + col] = float(v.std())
        row["eff_frac"] = row["eff_keys"] / m["n_keys"]
        row["std_eff_frac"] = row["std_eff_keys"] / m["n_keys"]
        # Exactly 0, not a tolerance: tiled keys really are bit-for-bit copies,
        # so max|row - row_0| is 0 and not merely small. (The std is NOT: it
        # reads ~4e-07 on those same rows.)
        row["keys_identical"] = row["key_row_range"] == 0.0
        row["collapsed"] = row["eff_frac"] > COLLAPSE_FRAC
        row["n_skulls"] = len(val)
        row["skull_ids"] = " ".join(str(s) for s in val)
        rows.append(row)
    return rows


def report(run, rows):
    """Print one run's table, then the count the thesis actually quotes."""
    print(f"\n{'=' * 100}\n{run.label}   ({run.arch_label})   {run.config_str()}\n{'=' * 100}")
    head = (f"{'block':<10}{'n_keys':>8}{'eff_keys':>11}{'eff_frac':>10}"
            f"{'energy_std':>12}{'peak_x_unif':>13}{'row_mass':>10}{'starved':>9}"
            f"{'key_spread':>12}{'ctrl_frac':>11}  verdict")
    print(head)
    print("-" * len(head))
    for r in rows:
        if r["keys_identical"]:
            verdict = "STRUCTURAL (keys identical)"
        elif r["collapsed"]:
            verdict = "collapsed"
        else:
            verdict = "attends"
        print(f"{r['block']:<10}{r['n_keys']:>8}{r['eff_keys']:>11.1f}{r['eff_frac']:>10.4f}"
              f"{r['energy_std']:>12.5f}{r['peak_x_unif']:>13.3f}"
              f"{r['row_mass']:>10.3f}{100 * r['frac_starved']:>8.1f}%"
              f"{r['key_row_spread']:>12.4f}{r['ctrl_eff_frac']:>11.4f}  {verdict}")

    structural = [r for r in rows if r["keys_identical"]]
    learned = [r for r in rows if r["collapsed"] and not r["keys_identical"]]
    attends = [r for r in rows if not r["collapsed"]]
    print(f"\n  {len(structural) + len(learned)}/{len(rows)} blocks uniform "
          f"= {len(structural)} structural ({', '.join(r['block'] for r in structural) or '-'})"
          f" + {len(learned)} collapsed ({', '.join(r['block'] for r in learned) or '-'})")
    print(f"  {len(attends)}/{len(rows)} attend: "
          f"{', '.join(f'''{r['block']} {r['eff_frac']:.3f}''' for r in attends) or '-'}")
    if learned:
        worst = max(learned, key=lambda r: r["energy_std"])
        print(f"  strongest contrast among the collapsed blocks: {worst['block']} "
              f"energy_std {worst['energy_std']:.5f} -- still {worst['eff_frac']:.4f} of uniform")
    # The control exists to be checked, not to be filed away unread.
    bad = [r for r in rows if not r["keys_identical"]
           and r["collapsed"] and r["ctrl_eff_frac"] <= COLLAPSE_FRAC]
    # After renormalisation max_k p >= 1/n_keys holds by construction, so a value
    # below 1 means the rows being read are not the ones being described. That is
    # exactly how the un-renormalised version of this script was caught.
    broken = [r for r in rows if r["peak_x_unif"] < 1.0 - 1e-6]
    if broken:
        print(f"  ⚠️ peak_x_unif < 1 on {', '.join(r['block'] for r in broken)} -- "
              f"impossible for a normalised row; the reading is wrong, do not use it")
    starved = [r for r in rows if r["frac_starved"] > 0.01]
    if starved:
        print("  ⚠️ queries switched off by the block's own 1e-9 epsilon: " + ", ".join(
            f"{r['block']} {100 * r['frac_starved']:.0f}% (mass {r['row_mass']:.2f})"
            for r in starved))
        print("     eff_keys above is read off renormalised rows, so it still measures "
              "spread; these queries contribute almost nothing to the block's output.")
    print("  control: textbook softmax(q.k/sqrt(d)) over the key axis " + (
        f"⚠️ DISAGREES on {', '.join(r['block'] for r in bad)} -- the scores do carry "
        f"contrast and the collapse is in the normalisation, not in Q/K"
        if bad else
        "agrees on every collapsed block -- the scores themselves carry no contrast"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="run directories under experiments/")
    ap.add_argument("--n", type=int, default=3,
                    help="validation skulls to average over (the stats barely move; "
                         "std_* columns record how little)")
    ap.add_argument("--out", default=OUT_CSV, help="CSV to merge the rows into")
    args = ap.parse_args()

    df = analyse(REPO, args.runs, n_skulls=args.n)

    # MERGE, never overwrite: rows for runs whose weights have since been deleted
    # can no longer be recomputed, and this file is the only place they exist.
    # Same rule as eval_all_runs.csv -- see the surface_quality.csv incident.
    out = os.path.join(REPO, args.out)
    if os.path.exists(out):
        old = pd.read_csv(out)
        keep = ~old.set_index(["run", "block"]).index.isin(
            df.set_index(["run", "block"]).index)
        df = pd.concat([old[keep.tolist()], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
