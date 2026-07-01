# 3D Organ Shape Completion (COMP0190)

Comparing voxel-based and geometry-aware deep learning for 3D organ shape
completion under a self-built, controlled corruption dataset.

- **Student:** Qi Jinyu   **Supervisor:** Petru Manescu
- **Target organ:** cranial / skull (complete shapes from SkullFix / SkullBreak)
- **Idea:** build a self-constructed corrupted dataset, then compare one voxel
  baseline vs one geometry-aware method under one unified evaluation framework.

## Repo structure
```
src/data/        data loading + conversion to canonical form (mesh / point cloud / voxel)
src/corruption/  self-built, parameterised corruption operators
src/models/      voxel baseline + geometry-aware model
src/eval/        Chamfer / Hausdorff / MSD / Dice metrics
configs/         config files + fixed random seeds
experiments/     outputs (git-ignored)
notebooks/       Colab / Jupyter notebooks
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Data is NOT stored in this repo (see .gitignore). It lives on Google Drive / UCL HPC.
