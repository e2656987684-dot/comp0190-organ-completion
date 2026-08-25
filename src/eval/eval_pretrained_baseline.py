"""Re-evaluate the released pretrained weights, repeatedly, with the full metric set.

CLOSES TWO THINGS AT ONCE (TODO 16 and 20).

  20. THE VENDOR MODEL'S INFERENCE IS NOT DETERMINISTIC. `msn_demo_arch`'s
      `UniformSampler` picks its centroids with a STATEFUL `tf.random.uniform`,
      so every forward pass selects a different (and, being sampled with
      replacement, partly duplicated) set of centroids: the same weights on the
      same skull give a different output each call. Every number previously
      reported for this baseline -- CD_t 10.71 mm, DCD 1.42772 -- came from a
      single draw, with no idea how much a second draw would move it. That
      matters most for one specific claim: the paper reports DCD 1.41269 and our
      measurement agreed to within 1%, which is the evidence that the metric
      implementation and the weight loading are both correct. An agreement of
      that size deserves to be quoted with its spread.

      ⚠️ THE FIX IS NOT TO MAKE THE SAMPLER DETERMINISTIC. `msn_demo_arch.py` is
      the published architecture lifted verbatim, and the stochastic sampling is
      how the authors' model actually behaves; replacing it would mean reporting
      a modified version of their work as their baseline. The honest treatment is
      to draw N times and report the distribution, which is what this does.

  16. THE BASELINE WAS MISSING THE PRIMARY METRIC. The archived
      `eval_val20.csv` holds only CD_t / CD_p / DCD -- no defect-region columns,
      no HD95, no F-score -- while the thesis reports defect-region coverage.
      A main table cannot put the baseline next to this project's models without
      them. Metrics here come from `report.metrics_from_points(..., inp=)`, the
      same function used for every other run, so the columns line up exactly.

WHAT IT WRITES
  experiments_log/pretrained_baseline/eval_val20_x{draws}.csv -- one row per
  (draw, skull). The older eval_val20.csv is left alone: it is the archived
  record of the original notebook run, and since each draw samples different
  centroids its numbers were never going to match a re-run anyway.

    python src/eval/eval_pretrained_baseline.py               # 5 draws, 20 skulls
    python src/eval/eval_pretrained_baseline.py --draws 2 --n-skulls 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

PAPER_DCD = 1.41269          # source paper, Table 1
SUMMARY = ["CD_t_mm", "HD95_mm", "F1@0.05", "F1@0.03", "DCD",
           "defect_cov_mm", "defect_HD95_mm", "defect_prec_mm", "defect_F1@0.05"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=5,
                    help="independent inference passes over the whole validation set. Each one "
                         "re-samples the vendor model's centroids, so the spread across draws is "
                         "the sampler's contribution to every reported number.")
    ap.add_argument("--n-skulls", type=int, default=0, help="0 = the whole validation split")
    ap.add_argument("--split-from", default="cd_rep05_full",
                    help="run under experiments_log/ whose val_ids define the split. Every run "
                         "shares the same 20 skulls, and reading it from experiments_log rather "
                         "than experiments/ keeps this working after weights are pruned.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import tensorflow as tf
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    import msn_demo_arch as demo
    import report as rp

    weights = os.path.join(REPO, "msn_downloads", "MSN_weights3.h5")
    data = np.load(os.path.join(REPO, rp.DATA_CACHE))
    ids, inputs, gt, scales = data["ids"], data["inputs"], data["gt"], data["scale_mm"]
    meta = json.load(open(os.path.join(REPO, "experiments_log", args.split_from, "run.json")))
    val = meta["val_ids"][:args.n_skulls] if args.n_skulls else meta["val_ids"]
    pos = [int(np.where(ids == s)[0][0]) for s in val]

    # Fixed global seed: the sampler stays stateful (draws differ from each other,
    # which is the point) but the SEQUENCE of draws is reproducible, so re-running
    # this script gives the same table.
    tf.keras.utils.set_random_seed(42)

    AE = demo.PCT_AE_Multimodal(bert_model=demo.bert_model,
                                PCT_encoder=demo.PCT_encoder,
                                pct_decoder=demo.pct_decoder)
    before = [w.numpy().copy() for w in AE.model.weights[:40]]
    AE.model.load_weights(weights)              # strict: no by_name, no skip_mismatch
    changed = sum(1 for b, w in zip(before, AE.model.weights) if not np.array_equal(b, w.numpy()))
    if changed <= 30:
        raise SystemExit(f"only {changed}/40 weight tensors changed -- wrong architecture module?")
    print(f"权重加载自检   前 40 个张量有 {changed} 个被改写 ✓   "
          f"({AE.model.count_params() / 1e6:.1f}M 参数，含 BERT)")

    tok = demo.BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tok.encode_plus("skull", add_special_tokens=True, max_length=128,
                          padding="max_length", truncation=True, return_tensors="tf")
    n = len(pos)
    x = [inputs[pos],
         np.zeros((n, 1, 1), np.float32),
         np.tile(enc["input_ids"].numpy(), (n, 1)),
         np.tile(enc["attention_mask"].numpy(), (n, 1))]

    rows = []
    for draw in range(args.draws):
        # predict(), not model(x) in a loop: the latter leaks 0.29 GiB per call
        # (measured on this project's own model) and 100 calls would exhaust the card.
        preds = AE.model.predict(x, batch_size=1, verbose=0)
        for sid, i, p in zip(val, pos, preds):
            row = {"draw": draw, "id": sid}
            row.update(rp.metrics_from_points(p, gt[i], float(scales[i]), inp=inputs[i]))
            rows.append(row)
        print(f"  draw {draw + 1}/{args.draws}: {n} skulls")

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(REPO, "experiments_log", "pretrained_baseline",
                                   f"eval_val20_x{args.draws}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    # Two different spreads, and they answer different questions.
    per_draw = df.groupby("draw")[SUMMARY].mean()          # the reported number, per draw
    print(f"\n{n} 颗验证颅骨 × {args.draws} 次独立推理  ->  {os.path.relpath(out, REPO)}")
    print(f"\n{'指标':16}{'均值':>10}{'跨抽样 std':>12}{'跨抽样极差':>12}{'颅骨间 std':>12}")
    print("-" * 62)
    for c in SUMMARY:
        print(f"{c:16}{per_draw[c].mean():>10.4f}{per_draw[c].std(ddof=1):>12.4f}"
              f"{per_draw[c].max() - per_draw[c].min():>12.4f}{df[c].std(ddof=1):>12.4f}")
    print("\n「跨抽样 std」= 重跑一次推理，报告值会动多少（这就是第 20 项问的东西）"
          "\n「颅骨间 std」= 颅骨彼此的差异，与采样器无关，只是提示前一列该拿什么做参照")

    dcd = per_draw["DCD"]
    print(f"\n复现性核对：论文 Table 1 报 DCD = {PAPER_DCD}")
    print(f"  实测 {dcd.mean():.5f} ± {dcd.std(ddof=1):.5f}（{args.draws} 次）"
          f"  ->  差 {abs(dcd.mean() - PAPER_DCD) / PAPER_DCD * 100:.1f}%"
          f"，而采样噪声只有 {dcd.std(ddof=1) / PAPER_DCD * 100:.2f}%")


if __name__ == "__main__":
    main()
