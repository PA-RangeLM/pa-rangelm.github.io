# PA-RangeLM anonymous release

This package contains the training and evaluation code for PA-RangeLM. It is
distributed without author metadata, experiment logs, local paths, datasets,
checkpoints, or bundled third-party source trees.

## Environment

The reference environment uses Python 3.8, PyTorch 1.11, and CUDA. Install the
packages in `requirements.txt`, then install a compatible PointNet++ operator
package that exposes `pointnet2_ops`.

## Data

Set the paths in `configs/pcn.yaml` for PCN. Dataset files are not included.
The evaluation protocol applies deterministic, independently sampled SO(3)
rotations to the complete 1,200-sample PCN test split with seed 2026.

## Evaluation

Download the Stage-II checkpoint from the project page and run:

```bash
MODEL_PATH=/path/to/pa_rangelm_pcn_rotated_10p20.pth \
  bash scripts/test_pcn_rotated.sh
```

The reported full-test result is CD-L1 = 10.20 (multiplied by 10^3; lower is
better). The script uses eight anchors and 80 iterations of the vectorized
differentiable coordinate-recovery solver.

## Training

Training is split into two stages:

```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
```

Stage I learns the shared representation with the training-only prototype
branch. Stage II freezes the Stage-I trunk and optimizes the dense decoder,
bounded range-correction head, uncertainty head, and coordinate-recovery path.

Optional experiment tracking is disabled unless explicitly enabled through the
training command-line arguments.
