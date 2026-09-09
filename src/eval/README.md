# src/eval

Evaluation and analysis for the skull-completion models: two modules that the
notebooks import, and a set of single-purpose scripts that each write one table
into `experiments_log/`.

## Modules

- **`report.py`** — loading runs, per-skull metrics (`eval_runs`), same-epoch
  comparison (`epoch_matched`), paired statistics, and the figures. For
  cross-validation use `fold_frame` / `fold_summary` / `fold_paired`, which
  replace `paired_stats`: folds validate on disjoint skulls, so per-skull pairing
  does not apply.
- **`mesh_viz.py`** — point cloud to mesh, surface statistics, diagnostic
  figures. `RECON` and the camera are locked constants, because every knob in
  them changes how smooth a surface looks.

## Scripts

| script | what it measures | output | needs GPU |
|---|---|---|---|
| `recompute_eval_all.py` | per-skull metrics for every run | `eval_all_runs.csv` | yes |
| `eval_pretrained_baseline.py` | the released upstream weights on this data | `pretrained_baseline/eval_*.csv` | yes |
| `make_defect_labels.py` | per-point implant ground truth | `defect_mask_labels.npz` | no |
| `defect_mask_audit.py` | that ground truth against the old distance rule | `defect_mask.csv` | no |
| `defect_mask_switch.py` | what changing the defect definition would do | `defect_mask_switch.csv` | yes |
| `sampling_floor.py` | the error two samplings of one surface show, with no model | `sampling_floor.csv` | no |
| `point_to_surface.py` | distances to the reconstructed surface, not to sampled points | `p2s.csv` | yes |
| `normal_quality.py` | whether surface normals can be estimated at all | `normal_quality.csv` | no |
| `roughness.py` | local roughness, prediction against ground truth | `roughness.csv` | yes |
| `attention_collapse.py` | how uniform each attention block's weights are | `attention_collapse.csv` | yes |
| `fold_text_branch.py` | that the text branch folds into four bias vectors | terminal only | yes |
| `mesh_preview.py` | a rendered look at one completion | `reports/preview/*.png` | yes |
| `make_report_figures.py` | the figures for the write-up | `reports/figures/*.png` | yes |

## Usage

Run from the repository root; every script takes `--help`.

```bash
python src/eval/sampling_floor.py
python src/eval/point_to_surface.py --runs cd_rep05_full_f0
python src/eval/mesh_preview.py --run cd_rep05_full_f0 --skull 070 --truth
```

Most take a few minutes. `recompute_eval_all.py` and `eval_pretrained_baseline.py`
take about fifteen. Scripts marked as needing the GPU build a 187M-parameter
model that holds 15.5 GiB, so do not start one while a notebook kernel is holding
the card.

Anything writing a CSV merges into it rather than overwriting, so a partial
re-run cannot destroy rows that can no longer be recomputed.

## Notes

- Read any CSV with an `id` column as `dtype={"id": str}`. Skull ids carry a
  leading zero, and read as integers they make id-keyed merges fail silently.
- The defect region is the implant annotation shipped with the dataset, read from
  `defect_mask_labels.npz`. Evaluation raises if that file is missing rather than
  falling back to the distance rule it replaced.
- `mesh_viz.signed_deviation` and the `dev_*` columns are superseded; use
  `point_to_surface.py`. See that function's docstring for why.
- Reconstructed meshes are for looking at, never for computing a metric.
