#!/usr/bin/env python3
"""Evaluate one point-completion checkpoint on the PCN test split and write one CSV.

PA-RangeLM passes its learned per-range precision weights to the batched GPU
LM solver. No validation split or auxiliary result file is produced.
"""

import argparse
import csv
import hashlib
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.PCN import get_dataset_and_loader
from datasets.utils import read_yaml
from loss_functions.chamfer_ndim import CDNLoss
from models.pointr.pa_rangelm import PARangeLM
from utils.torch_lm import find_points_from_distance_torch


CATEGORY_NAMES = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03636649": "lamp",
    "04256520": "sofa",
    "04379243": "table",
    "04530566": "watercraft",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--summary_csv",
        default="",
        help="Optional one-row CSV with dataset-level accuracy and profiling results.",
    )
    parser.add_argument("--data_config", default="configs/pcn.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_keypoint", type=int, default=8)
    parser.add_argument("--keypoint", default="curvature_radius")
    parser.add_argument("--curve_k", type=int, default=16)
    parser.add_argument("--curve_radius", type=float, default=0.075)
    parser.add_argument("--curve_thres", type=float, default=0.5)
    parser.add_argument(
        "--random_rotate",
        action="store_true",
        help="Jointly rotate partial, GT, and anchors for every PCN test sample.",
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=2026,
        help="Seed for deterministic per-sample rotations.",
    )
    parser.add_argument(
        "--rotation_mode",
        choices=("independent", "same", "mixed", "uniform_so3"),
        default="independent",
        help=(
            "Independent XYZ Euler angles, one shared Euler angle, a per-sample "
            "mixture, or a Haar-uniform SO(3) rotation."
        ),
    )
    parser.add_argument("--torch_lm_iterations", type=int, default=80)
    parser.add_argument("--torch_lm_damping", type=float, default=0.001)
    parser.add_argument(
        "--torch_lm_dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--torch_lm_target_converged", type=float, default=0.999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV instead of refusing to overwrite it.",
    )
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    model_config = read_yaml(config_path)
    model = PARangeLM(model_config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_best_val" not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no model_state_best_val")

    state = checkpoint["model_state_best_val"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint.get("epoch")


def unpack_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) not in (2, 3):
        raise RuntimeError(f"unexpected PA-RangeLM evaluation output: {type(outputs)}")
    auxiliary = outputs[2] if len(outputs) == 3 else None
    return outputs[1], auxiliary


def get_rotation_angles(taxonomy_id, model_id, seed, mode):
    """Match the reproducible rotation protocol used by the paper."""
    sample_key = f"{seed}:{taxonomy_id}:{model_id}".encode("utf-8")
    sample_seed = int.from_bytes(hashlib.sha256(sample_key).digest()[:8], "little")
    rng = np.random.default_rng(sample_seed)
    use_same_angle = mode == "same" or (mode == "mixed" and rng.random() < 0.5)
    if use_same_angle:
        angle = rng.uniform(-np.pi, np.pi)
        return np.full(3, angle, dtype=np.float32)
    return rng.uniform(-np.pi, np.pi, size=3).astype(np.float32)


def get_uniform_so3_matrix(taxonomy_id, model_id, seed):
    """Return a deterministic Haar-uniform SO(3) matrix for one sample.

    The unit-quaternion construction uses three independent U(0, 1) values.
    The returned matrix acts on column vectors; point rows are multiplied by
    its transpose in ``rotate_batch``.
    """
    sample_key = f"{seed}:{taxonomy_id}:{model_id}".encode("utf-8")
    sample_seed = int.from_bytes(hashlib.sha256(sample_key).digest()[:8], "little")
    rng = np.random.default_rng(sample_seed)
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )
    return rotation


def rotation_matrix_xyz(angles, device, dtype):
    """Build batched Rz @ Ry @ Rx matrices, matching the existing evaluator."""
    angles = torch.as_tensor(np.stack(angles), device=device, dtype=dtype)
    x, y, z = angles.unbind(dim=1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    rotation_x = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=1
    ).reshape(-1, 3, 3)
    rotation_y = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=1
    ).reshape(-1, 3, 3)
    rotation_z = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=1
    ).reshape(-1, 3, 3)
    return torch.bmm(rotation_z, torch.bmm(rotation_y, rotation_x))


def rotate_batch(partial, target, anchors, taxonomy_ids, model_ids, seed, mode):
    """Apply one joint rigid rotation to all geometry belonging to a sample."""
    if mode == "uniform_so3":
        rotations = np.stack(
            [
                get_uniform_so3_matrix(taxonomy_id, model_id, seed)
                for taxonomy_id, model_id in zip(taxonomy_ids, model_ids)
            ]
        )
        rotation = torch.as_tensor(
            rotations, device=partial.device, dtype=partial.dtype
        )
    else:
        angles = [
            get_rotation_angles(taxonomy_id, model_id, seed, mode)
            for taxonomy_id, model_id in zip(taxonomy_ids, model_ids)
        ]
        rotation = rotation_matrix_xyz(angles, partial.device, partial.dtype)
    rotation_t = rotation.transpose(1, 2)
    return (
        torch.bmm(partial, rotation_t),
        torch.bmm(target, rotation_t),
        torch.bmm(anchors, rotation_t),
    )


def per_sample_cd_l1(chamfer, prediction, target):
    """Match CDNLoss.chamfer_loss, while retaining one value per sample."""
    distance_1, distance_2, _, _ = chamfer(prediction, target)
    distance_1 = torch.sqrt(distance_1.clamp_min(1e-9)).mean(dim=1)
    distance_2 = torch.sqrt(distance_2.clamp_min(1e-9)).mean(dim=1)
    return 0.5 * (distance_1 + distance_2)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return "{:02d}:{:02d}:{:06.3f}".format(int(hours), int(minutes), seconds)


def main():
    program_start = time.perf_counter()
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    output_path = Path(args.output_csv)
    summary_path = Path(args.summary_csv) if args.summary_csv else None
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        if summary_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"summary already exists: {summary_path}; pass --overwrite to replace it"
            )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint_epoch = load_model(args.config_path, args.model_path, device)
    model_core = model.module if hasattr(model, "module") else model
    use_uncertainty = bool(
        getattr(model_core, "range_lm_uncertainty_enabled", False)
    )

    data_config = read_yaml(args.data_config)
    curvature_params = {
        "k": args.curve_k,
        "curvature_radius": args.curve_radius,
        "curvature_thres": args.curve_thres,
    }
    dataset, loader = get_dataset_and_loader(
        "test",
        data_config,
        args.batch_size,
        50000,
        shuffle=False,
        drop_last=False,
        num_keypoints=args.num_keypoint,
        keypoint_type=args.keypoint,
        curvature_params=curvature_params,
    )
    if not dataset.file_list:
        raise RuntimeError("PCN test split is empty")

    solve_dtype = (
        torch.float32 if args.torch_lm_dtype == "float32" else torch.float64
    )
    chamfer = CDNLoss(mode="point", metric_fun="cd_l1").chd
    category_counts = Counter()
    total_cd_l1_x1000 = []
    solver_seconds = 0.0
    solver_points = 0
    solver_nonconverged_points = 0.0
    solver_accepted_points = 0.0
    solver_batch_iterations = []
    invalid_xyz_points = 0
    invalid_samples = 0

    cuda_sync(device)
    baseline_gpu_allocated = 0
    baseline_gpu_reserved = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_gpu_allocated = torch.cuda.memory_allocated(device)
        baseline_gpu_reserved = torch.cuda.memory_reserved(device)
    evaluation_start = time.perf_counter()

    fieldnames = [
        "category",
        "category_index",
        "taxonomy_id",
        "model_id",
        "cd_l1_x1000",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for taxonomy_ids, model_ids, data in tqdm(
                loader, desc=f"PCN test -> {output_path.name}"
            ):
                partial = data["partial"].to(device, non_blocking=True)
                target = data["gt"].to(device, non_blocking=True)
                anchors = data["basis_points"].to(device, non_blocking=True)
                if args.random_rotate:
                    partial, target, anchors = rotate_batch(
                        partial,
                        target,
                        anchors,
                        taxonomy_ids,
                        model_ids,
                        args.rotation_seed,
                        args.rotation_mode,
                    )

                dense_distances, auxiliary = unpack_outputs(model(partial, anchors))
                weights = auxiliary if use_uncertainty else None
                cuda_sync(device)
                solver_start = time.perf_counter()
                prediction, diagnostics = find_points_from_distance_torch(
                    dense_distances,
                    anchors,
                    weights=weights,
                    max_iterations=args.torch_lm_iterations,
                    initial_damping=args.torch_lm_damping,
                    solve_dtype=solve_dtype,
                    target_converged_fraction=args.torch_lm_target_converged,
                    return_diagnostics=True,
                )
                cuda_sync(device)
                solver_seconds += time.perf_counter() - solver_start

                batch_solver_points = prediction.shape[0] * prediction.shape[1]
                solver_points += batch_solver_points
                solver_nonconverged_points += (
                    1.0 - diagnostics.converged_fraction
                ) * batch_solver_points
                solver_accepted_points += (
                    diagnostics.accepted_fraction * batch_solver_points
                )
                solver_batch_iterations.append(diagnostics.iterations)

                finite_points = torch.isfinite(prediction).all(dim=-1)
                invalid_xyz_points += int((~finite_points).sum().item())
                invalid_samples += int((~finite_points).any(dim=1).sum().item())
                metric_prediction = torch.nan_to_num(
                    prediction, nan=0.0, posinf=0.0, neginf=0.0
                )
                batch_cd_l1_x1000 = (
                    per_sample_cd_l1(chamfer, metric_prediction, target) * 1000.0
                ).detach().cpu().numpy()

                for taxonomy_id, model_id, cd_l1_x1000 in zip(
                    taxonomy_ids, model_ids, batch_cd_l1_x1000
                ):
                    taxonomy_id = str(taxonomy_id)
                    category_counts[taxonomy_id] += 1
                    value = float(cd_l1_x1000)
                    writer.writerow(
                        {
                            "category": CATEGORY_NAMES.get(taxonomy_id, taxonomy_id),
                            "category_index": category_counts[taxonomy_id],
                            "taxonomy_id": taxonomy_id,
                            "model_id": str(model_id),
                            "cd_l1_x1000": f"{value:.8f}",
                        }
                    )
                    total_cd_l1_x1000.append(value)
                output_file.flush()

    cuda_sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    program_seconds = time.perf_counter() - program_start

    expected_count = len(dataset)
    if len(total_cd_l1_x1000) != expected_count:
        raise RuntimeError(
            f"wrote {len(total_cd_l1_x1000)} rows, expected {expected_count}"
        )
    mean_cd_l1_x1000 = float(np.mean(total_cd_l1_x1000))
    point_solver_failure_rate_pct = (
        100.0 * solver_nonconverged_points / solver_points
        if solver_points
        else float("nan")
    )
    sample_numerical_failure_rate_pct = 100.0 * invalid_samples / expected_count
    accepted_point_rate_pct = (
        100.0 * solver_accepted_points / solver_points
        if solver_points
        else float("nan")
    )
    solver_iterations_mean = float(np.mean(solver_batch_iterations))
    solver_iterations_max = int(max(solver_batch_iterations))

    peak_gpu_allocated = 0
    peak_gpu_reserved = 0
    if device.type == "cuda":
        peak_gpu_allocated = torch.cuda.max_memory_allocated(device)
        peak_gpu_reserved = torch.cuda.max_memory_reserved(device)
    peak_cpu_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    summary = {
        "lm_max_iterations": args.torch_lm_iterations,
        "rotation_mode": args.rotation_mode if args.random_rotate else "none",
        "rotation_seed": args.rotation_seed,
        "samples": expected_count,
        "checkpoint_epoch": checkpoint_epoch,
        "mean_cd_l1_x1000": f"{mean_cd_l1_x1000:.8f}",
        "solver_time_total_s": f"{solver_seconds:.6f}",
        "solver_time_per_sample_ms": f"{1000.0 * solver_seconds / expected_count:.6f}",
        "evaluation_wall_s": f"{evaluation_seconds:.6f}",
        "point_solver_failure_rate_pct": f"{point_solver_failure_rate_pct:.8f}",
        "sample_numerical_failure_rate_pct": f"{sample_numerical_failure_rate_pct:.8f}",
        "invalid_xyz_points": invalid_xyz_points,
        "solver_points": solver_points,
        "solver_nonconverged_points": int(round(solver_nonconverged_points)),
        "accepted_point_rate_pct": f"{accepted_point_rate_pct:.8f}",
        "solver_iterations_mean": f"{solver_iterations_mean:.6f}",
        "solver_iterations_max": solver_iterations_max,
        "peak_gpu_allocated_mb": f"{peak_gpu_allocated / (1024.0 ** 2):.6f}",
        "peak_gpu_reserved_mb": f"{peak_gpu_reserved / (1024.0 ** 2):.6f}",
        "peak_eval_incremental_allocated_mb": f"{max(0, peak_gpu_allocated - baseline_gpu_allocated) / (1024.0 ** 2):.6f}",
        "baseline_gpu_allocated_mb": f"{baseline_gpu_allocated / (1024.0 ** 2):.6f}",
        "baseline_gpu_reserved_mb": f"{baseline_gpu_reserved / (1024.0 ** 2):.6f}",
        "peak_cpu_rss_mb": f"{peak_cpu_rss_mb:.6f}",
    }
    if summary_path is not None:
        with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

    print(
        "Finished: split=test | samples={} | checkpoint_epoch={} | "
        "rotation={} | rotation_mode={} | rotation_seed={} | fast_lm=torch_lm | "
        "uncertainty_weights={} | lm_max_iterations={} | mean_cd_l1_x1000={:.6f}".format(
            expected_count,
            checkpoint_epoch,
            args.random_rotate,
            args.rotation_mode,
            args.rotation_seed,
            use_uncertainty,
            args.torch_lm_iterations,
            mean_cd_l1_x1000,
        )
    )
    print(
        "Solver: points={} | nonconverged_points={} | "
        "point_solver_failure_rate_pct={:.8f} | accepted_point_rate_pct={:.8f} | "
        "iterations_mean={:.6f} | iterations_max={}".format(
            solver_points,
            int(round(solver_nonconverged_points)),
            point_solver_failure_rate_pct,
            accepted_point_rate_pct,
            solver_iterations_mean,
            solver_iterations_max,
        )
    )
    print(
        "Numerics: invalid_samples={} | sample_numerical_failure_rate_pct={:.8f} | "
        "invalid_xyz_points={}".format(
            invalid_samples,
            sample_numerical_failure_rate_pct,
            invalid_xyz_points,
        )
    )
    print(
        "Time: evaluation_wall={} ({:.6f}s) | solver_total={:.6f}s | "
        "solver_per_sample_ms={:.6f} | program_wall={} ({:.6f}s)".format(
            format_seconds(evaluation_seconds),
            evaluation_seconds,
            solver_seconds,
            1000.0 * solver_seconds / expected_count,
            format_seconds(program_seconds),
            program_seconds,
        )
    )
    print(
        "Memory: peak_gpu_allocated_mb={:.6f} | peak_gpu_reserved_mb={:.6f} | "
        "peak_eval_incremental_allocated_mb={:.6f} | peak_cpu_rss_mb={:.6f}".format(
            peak_gpu_allocated / (1024.0 ** 2),
            peak_gpu_reserved / (1024.0 ** 2),
            max(0, peak_gpu_allocated - baseline_gpu_allocated) / (1024.0 ** 2),
            peak_cpu_rss_mb,
        )
    )
    print(f"CSV: {output_path.resolve()}")
    if summary_path is not None:
        print(f"Summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
