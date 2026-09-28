#!/usr/bin/env python3
"""Evaluate whether PA-RangeLM uncertainty ranks reconstruction errors.

The primary analysis is point-level.  A point's uncertainty is the fraction of
the eight anchor measurements that is effectively discounted by the learned
relative precision weights.  Point error is its one-sided nearest-GT Euclidean
distance.  The script reports per-shape Spearman correlation, selective
risk--coverage/AURC, and AUROC for detecting the highest-error 10% of points.

Because the model predicts *relative* per-anchor precision (the eight weights
for every point have a fixed sum), their arithmetic mean is not a valid point
uncertainty.  A secondary measurement-level analysis therefore uses -weight as
the uncertainty score and absolute nearest-GT anchor-range error as the target.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.stats import rankdata, spearmanr

from datasets.PCN import get_dataset_and_loader
from datasets.utils import read_yaml
from evaluate_pcn import CATEGORY_NAMES, load_model, rotate_batch, unpack_outputs
from loss_functions.chamfer_ndim import CDNLoss
from utils.torch_lm import find_points_from_distance_torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_config", default="configs/pcn.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_keypoint", type=int, default=8)
    parser.add_argument("--keypoint", default="curvature_radius")
    parser.add_argument("--curve_k", type=int, default=16)
    parser.add_argument("--curve_radius", type=float, default=0.075)
    parser.add_argument("--curve_thres", type=float, default=0.5)
    parser.add_argument("--random_rotate", action="store_true")
    parser.add_argument("--rotation_seed", type=int, default=2026)
    parser.add_argument(
        "--rotation_mode",
        choices=("independent", "same", "mixed", "uniform_so3"),
        default="independent",
    )
    parser.add_argument("--torch_lm_iterations", type=int, default=80)
    parser.add_argument("--torch_lm_damping", type=float, default=0.001)
    parser.add_argument(
        "--torch_lm_dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--torch_lm_target_converged", type=float, default=0.999)
    parser.add_argument(
        "--high_error_fraction",
        type=float,
        default=0.10,
        help="Within-shape fraction labelled as high-error for AUROC.",
    )
    parser.add_argument(
        "--coverage_min",
        type=float,
        default=0.10,
        help="Smallest displayed risk--coverage value; exact AURC uses all points.",
    )
    parser.add_argument("--coverage_step", type=float, default=0.05)
    parser.add_argument(
        "--range_pairs_per_shape",
        type=int,
        default=16384,
        help="Deterministic point-anchor pairs used for the secondary analysis.",
    )
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_offset(taxonomy_id, model_id, seed, modulus):
    payload = f"{seed}:{taxonomy_id}:{model_id}:uncertainty".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % modulus


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 < args.high_error_fraction < 0.5:
        raise ValueError("high_error_fraction must lie in (0, 0.5)")
    if not 0.0 < args.coverage_min <= 1.0:
        raise ValueError("coverage_min must lie in (0, 1]")
    if not 0.0 < args.coverage_step <= 1.0:
        raise ValueError("coverage_step must lie in (0, 1]")
    if args.range_pairs_per_shape < 1:
        raise ValueError("range_pairs_per_shape must be positive")
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")


def coverage_grid(minimum, step):
    values = np.arange(minimum, 1.0 + step * 0.5, step, dtype=np.float64)
    values = np.clip(values, minimum, 1.0)
    values = np.unique(np.concatenate([values, np.array([1.0])]))
    return values


def point_uncertainty_from_relative_precision(weights):
    """Return 1 - effective-anchor-count/K (higher means less reliable)."""
    weights = weights.float().clamp_min(1e-12)
    count = weights.shape[-1]
    effective = weights.sum(dim=-1).square() / weights.square().sum(dim=-1)
    return (1.0 - effective / float(count)).clamp(0.0, 1.0)


def safe_spearman(score, error):
    if score.size < 2 or np.all(score == score[0]) or np.all(error == error[0]):
        return float("nan")
    result = spearmanr(score, error)
    value = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    return float(value)


def binary_auroc(score, positive):
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = int(positive.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    positive_rank_sum = float(ranks[positive].sum())
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def selective_metrics(score, error, coverages, high_error_fraction):
    score = np.asarray(score, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(score) & np.isfinite(error)
    score = score[finite]
    error = error[finite]
    if score.size < 2:
        raise RuntimeError("fewer than two finite score/error pairs")

    order = np.argsort(score, kind="mergesort")
    ordered_error = error[order]
    cumulative_risk = np.cumsum(ordered_error) / np.arange(1, error.size + 1)
    indices = np.ceil(coverages * error.size).astype(np.int64) - 1
    indices = np.clip(indices, 0, error.size - 1)

    oracle_order = np.argsort(error, kind="mergesort")
    oracle_error = error[oracle_order]
    oracle_cumulative = np.cumsum(oracle_error) / np.arange(1, error.size + 1)

    threshold = float(np.quantile(error, 1.0 - high_error_fraction))
    high_error = error >= threshold
    spearman = safe_spearman(score, error)
    auroc = binary_auroc(score, high_error)
    aurc = float(cumulative_risk.mean())
    oracle_aurc = float(oracle_cumulative.mean())
    random_risk = float(error.mean())
    if not np.isclose(cumulative_risk[-1], random_risk, rtol=1e-10, atol=1e-12):
        raise RuntimeError("100% selective risk does not equal the full-set risk")
    excess_aurc = aurc - oracle_aurc
    aurc_gain_vs_random = (
        (random_risk - aurc) / random_risk if random_risk > 0.0 else float("nan")
    )
    return {
        "count": int(error.size),
        "spearman": spearman,
        "auroc": float(auroc),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": float(excess_aurc),
        "random_risk": random_risk,
        "aurc_gain_vs_random": float(aurc_gain_vs_random),
        "high_error_threshold": threshold,
        "high_error_count": int(high_error.sum()),
        "risk_curve": cumulative_risk[indices],
        "oracle_curve": oracle_cumulative[indices],
        "random_curve": np.full(coverages.shape, random_risk, dtype=np.float64),
    }


def bootstrap_mean_ci(values, resamples, seed, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    finite_rows = np.isfinite(values).all(axis=1)
    values = values[finite_rows]
    if values.shape[0] == 0:
        shape = values.shape[1:]
        nan = np.full(shape, np.nan)
        return (float("nan"), float("nan"), float("nan")) if squeeze else (nan, nan, nan)

    # This RNG only resamples observed shapes for the prespecified bootstrap;
    # it never generates or alters scientific observations.
    rng = np.random.RandomState(seed)
    means = []
    remaining = resamples
    while remaining:
        chunk = min(100, remaining)
        indices = rng.randint(0, values.shape[0], size=(chunk, values.shape[0]))
        means.append(values[indices].mean(axis=1))
        remaining -= chunk
    bootstrap = np.concatenate(means, axis=0)
    alpha = (1.0 - confidence) / 2.0
    mean = values.mean(axis=0)
    low = np.quantile(bootstrap, alpha, axis=0)
    high = np.quantile(bootstrap, 1.0 - alpha, axis=0)
    if squeeze:
        return float(mean[0]), float(low[0]), float(high[0])
    return mean, low, high


def metric_summary(rows, prefix, bootstrap_resamples, bootstrap_seed):
    result = {}
    for metric in (
        "spearman",
        "auroc",
        "aurc",
        "oracle_aurc",
        "excess_aurc",
        "random_risk",
        "aurc_gain_vs_random",
    ):
        values = np.array([row[f"{prefix}_{metric}"] for row in rows], dtype=np.float64)
        mean, low, high = bootstrap_mean_ci(
            values, bootstrap_resamples, bootstrap_seed
        )
        result[metric] = {
            "mean": mean,
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "valid_shapes": int(np.isfinite(values).sum()),
        }
    return result


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    """Convert NumPy values and non-finite floats to standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def render_risk_coverage(output_dir, coverages, point_curves, bootstrap_resamples, seed):
    selected = np.stack([row["risk_curve"] for row in point_curves]) * 1000.0
    oracle = np.stack([row["oracle_curve"] for row in point_curves]) * 1000.0
    random = np.stack([row["random_curve"] for row in point_curves]) * 1000.0
    selected_mean, selected_low, selected_high = bootstrap_mean_ci(
        selected, bootstrap_resamples, seed
    )
    oracle_mean, oracle_low, oracle_high = bootstrap_mean_ci(
        oracle, bootstrap_resamples, seed + 1
    )
    random_mean, random_low, random_high = bootstrap_mean_ci(
        random, bootstrap_resamples, seed + 2
    )

    curve_rows = []
    for index, coverage in enumerate(coverages):
        curve_rows.append(
            {
                "coverage": f"{coverage:.6f}",
                "uncertainty_risk_x1000": f"{selected_mean[index]:.8f}",
                "uncertainty_ci95_low_x1000": f"{selected_low[index]:.8f}",
                "uncertainty_ci95_high_x1000": f"{selected_high[index]:.8f}",
                "oracle_risk_x1000": f"{oracle_mean[index]:.8f}",
                "oracle_ci95_low_x1000": f"{oracle_low[index]:.8f}",
                "oracle_ci95_high_x1000": f"{oracle_high[index]:.8f}",
                "random_risk_x1000": f"{random_mean[index]:.8f}",
                "random_ci95_low_x1000": f"{random_low[index]:.8f}",
                "random_ci95_high_x1000": f"{random_high[index]:.8f}",
            }
        )
    write_csv(
        output_dir / "point_risk_coverage.csv",
        list(curve_rows[0]),
        curve_rows,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axis = plt.subplots(figsize=(3.50, 2.45), constrained_layout=True)
    x = coverages * 100.0
    axis.fill_between(x, selected_low, selected_high, color="#8DBDE6", alpha=0.32, linewidth=0)
    axis.plot(x, selected_mean, color="#1764AB", linewidth=1.7, label="PA-RangeLM uncertainty")
    axis.plot(x, oracle_mean, color="#6F6F6F", linewidth=1.1, linestyle="--", label="Oracle ranking")
    axis.plot(x, random_mean, color="#A7A7A7", linewidth=1.1, linestyle=":", label="Random ranking (expected)")
    axis.set_xlabel("Coverage retained (%)")
    axis.set_ylabel("Mean one-sided point error (×10³)")
    axis.set_xlim(float(x.min()), 100.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    axis.legend(loc="best", frameon=False)
    base = output_dir / "point_risk_coverage"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    start_time = time.perf_counter()
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        output_dir / "summary.json",
        output_dir / "summary.csv",
        output_dir / "per_shape_metrics.csv",
    ]
    existing = [str(path) for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "result files already exist; pass --overwrite to replace: " + ", ".join(existing)
        )

    device = torch.device(args.device)
    model, checkpoint_epoch = load_model(args.config_path, args.model_path, device)
    if not getattr(model, "range_lm_uncertainty_enabled", False):
        raise RuntimeError("checkpoint/config does not enable the RangeLM uncertainty head")
    if args.num_keypoint != model.num_basis:
        raise ValueError(
            f"num_keypoint={args.num_keypoint} but model.num_basis={model.num_basis}"
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
    if len(dataset) != 1200:
        raise RuntimeError(f"expected the complete 1,200-shape PCN test split, got {len(dataset)}")
    audit_counts = Counter(str(item["taxonomy_id"]) for item in dataset.file_list)
    expected_taxonomies = set(CATEGORY_NAMES)
    if set(audit_counts) != expected_taxonomies or any(
        audit_counts[taxonomy_id] != 150 for taxonomy_id in expected_taxonomies
    ):
        raise RuntimeError(
            "PCN test sample-set audit failed: expected 150 shapes in each of "
            f"{len(expected_taxonomies)} categories, got {dict(sorted(audit_counts.items()))}"
        )

    coverages = coverage_grid(args.coverage_min, args.coverage_step)
    solve_dtype = torch.float32 if args.torch_lm_dtype == "float32" else torch.float64
    chamfer = CDNLoss(mode="point", metric_fun="cd_l1").chd
    per_shape_rows = []
    point_curves = []
    category_rows = defaultdict(list)
    total_solver_points = 0
    nonconverged_solver_points = 0.0
    invalid_xyz_points = 0
    weight_sum_deviation_max = 0.0

    print(
        "START PA-RangeLM uncertainty-quality evaluation | "
        f"samples={len(dataset)} | rotation={'none' if not args.random_rotate else args.rotation_mode} | "
        f"seed={args.rotation_seed} | LM={args.torch_lm_iterations} | dtype={args.torch_lm_dtype} | "
        f"high_error_fraction={args.high_error_fraction:.3f}"
    )
    print(
        "Sample-set audit passed: 1,200 PCN test shapes | "
        "8 categories x 150 | checkpoint_epoch={}".format(checkpoint_epoch)
    )
    print(
        "Primary point score: 1 - effective_anchor_count / K | "
        "point error: nearest-GT Euclidean distance | aggregation: per-shape macro mean"
    )
    print("Batch progress is intentionally omitted; detailed results are written to CSV/JSON.")

    shape_index = 0
    with torch.no_grad():
        for taxonomy_ids, model_ids, data in loader:
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

            dense_distances, weights = unpack_outputs(model(partial, anchors))
            if weights is None:
                raise RuntimeError("model did not return uncertainty weights")
            expected_sum = float(weights.shape[-1])
            deviation = (weights.sum(dim=-1) - expected_sum).abs().max().item()
            weight_sum_deviation_max = max(weight_sum_deviation_max, float(deviation))
            if not torch.isfinite(weights).all():
                raise RuntimeError("non-finite uncertainty weights encountered")

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
            solver_points = prediction.shape[0] * prediction.shape[1]
            total_solver_points += solver_points
            nonconverged_solver_points += (1.0 - diagnostics.converged_fraction) * solver_points
            finite_xyz = torch.isfinite(prediction).all(dim=-1)
            invalid_xyz_points += int((~finite_xyz).sum().item())
            prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

            squared_point_error, _, nearest_gt_indices, _ = chamfer(prediction, target)
            point_error = torch.sqrt(squared_point_error.clamp_min(1e-9))
            nearest_gt = torch.gather(
                target,
                dim=1,
                index=nearest_gt_indices.long().unsqueeze(-1).expand(-1, -1, 3),
            )
            target_ranges = torch.linalg.vector_norm(
                nearest_gt.unsqueeze(2) - anchors.unsqueeze(1), dim=-1
            )
            range_error = (dense_distances - target_ranges).abs()
            point_uncertainty = point_uncertainty_from_relative_precision(weights)

            for local_index, (taxonomy_id, model_id) in enumerate(zip(taxonomy_ids, model_ids)):
                taxonomy_id = str(taxonomy_id)
                model_id = str(model_id)
                point_score_np = point_uncertainty[local_index].cpu().numpy()
                point_error_np = point_error[local_index].cpu().numpy()
                point_result = selective_metrics(
                    point_score_np,
                    point_error_np,
                    coverages,
                    args.high_error_fraction,
                )

                n_points, n_anchors = weights[local_index].shape
                pair_count = min(args.range_pairs_per_shape, n_points * n_anchors)
                flat_indices = np.arange(pair_count, dtype=np.int64)
                offset = stable_offset(
                    taxonomy_id, model_id, args.bootstrap_seed, n_points * n_anchors
                )
                flat_indices = (flat_indices * 104729 + offset) % (n_points * n_anchors)
                flat_indices_t = torch.as_tensor(flat_indices, device=device, dtype=torch.long)
                measurement_score = (-weights[local_index].reshape(-1)).index_select(
                    0, flat_indices_t
                ).cpu().numpy()
                measurement_error = range_error[local_index].reshape(-1).index_select(
                    0, flat_indices_t
                ).cpu().numpy()
                range_result = selective_metrics(
                    measurement_score,
                    measurement_error,
                    coverages,
                    args.high_error_fraction,
                )

                row = {
                    "shape_index": shape_index,
                    "category": CATEGORY_NAMES.get(taxonomy_id, taxonomy_id),
                    "taxonomy_id": taxonomy_id,
                    "model_id": model_id,
                    "point_count": point_result["count"],
                    "point_spearman": point_result["spearman"],
                    "point_auroc": point_result["auroc"],
                    "point_aurc": point_result["aurc"],
                    "point_oracle_aurc": point_result["oracle_aurc"],
                    "point_excess_aurc": point_result["excess_aurc"],
                    "point_random_risk": point_result["random_risk"],
                    "point_aurc_gain_vs_random": point_result["aurc_gain_vs_random"],
                    "point_high_error_threshold": point_result["high_error_threshold"],
                    "point_high_error_count": point_result["high_error_count"],
                    "range_pair_count": range_result["count"],
                    "range_spearman": range_result["spearman"],
                    "range_auroc": range_result["auroc"],
                    "range_aurc": range_result["aurc"],
                    "range_oracle_aurc": range_result["oracle_aurc"],
                    "range_excess_aurc": range_result["excess_aurc"],
                    "range_random_risk": range_result["random_risk"],
                    "range_aurc_gain_vs_random": range_result["aurc_gain_vs_random"],
                    "range_high_error_threshold": range_result["high_error_threshold"],
                    "range_high_error_count": range_result["high_error_count"],
                }
                per_shape_rows.append(row)
                point_curves.append(point_result)
                category_rows[taxonomy_id].append(row)
                shape_index += 1

    if len(per_shape_rows) != len(dataset):
        raise RuntimeError(
            f"computed {len(per_shape_rows)} per-shape rows, expected {len(dataset)}"
        )
    if weight_sum_deviation_max > 1e-3:
        raise RuntimeError(
            "relative precision weights do not have the expected fixed sum; "
            f"maximum deviation={weight_sum_deviation_max:.6e}"
        )

    per_shape_path = output_dir / "per_shape_metrics.csv"
    write_csv(per_shape_path, list(per_shape_rows[0]), per_shape_rows)

    point_summary = metric_summary(
        per_shape_rows, "point", args.bootstrap_resamples, args.bootstrap_seed
    )
    range_summary = metric_summary(
        per_shape_rows, "range", args.bootstrap_resamples, args.bootstrap_seed + 100
    )
    category_summary_rows = []
    for taxonomy_id in sorted(category_rows):
        rows = category_rows[taxonomy_id]
        for level in ("point", "range"):
            metrics = metric_summary(
                rows,
                level,
                args.bootstrap_resamples,
                args.bootstrap_seed + int(taxonomy_id[-4:]),
            )
            category_summary_rows.append(
                {
                    "category": CATEGORY_NAMES.get(taxonomy_id, taxonomy_id),
                    "taxonomy_id": taxonomy_id,
                    "analysis_level": level,
                    "shapes": len(rows),
                    "spearman_mean": metrics["spearman"]["mean"],
                    "spearman_ci95_low": metrics["spearman"]["bootstrap_95ci_low"],
                    "spearman_ci95_high": metrics["spearman"]["bootstrap_95ci_high"],
                    "auroc_mean": metrics["auroc"]["mean"],
                    "auroc_ci95_low": metrics["auroc"]["bootstrap_95ci_low"],
                    "auroc_ci95_high": metrics["auroc"]["bootstrap_95ci_high"],
                    "aurc_mean_x1000": 1000.0 * metrics["aurc"]["mean"],
                    "aurc_ci95_low_x1000": 1000.0 * metrics["aurc"]["bootstrap_95ci_low"],
                    "aurc_ci95_high_x1000": 1000.0 * metrics["aurc"]["bootstrap_95ci_high"],
                    "aurc_gain_vs_random_mean": metrics["aurc_gain_vs_random"]["mean"],
                }
            )
    write_csv(
        output_dir / "category_summary.csv",
        list(category_summary_rows[0]),
        category_summary_rows,
    )

    render_risk_coverage(
        output_dir,
        coverages,
        point_curves,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )

    elapsed = time.perf_counter() - start_time
    summary = {
        "experiment": "PA-RangeLM uncertainty quality on rotated PCN test",
        "checkpoint": str(Path(args.model_path).resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "config": str(Path(args.config_path).resolve()),
        "data_config": str(Path(args.data_config).resolve()),
        "samples": len(per_shape_rows),
        "categories": len(category_rows),
        "rotation": {
            "enabled": bool(args.random_rotate),
            "mode": args.rotation_mode if args.random_rotate else "none",
            "seed": args.rotation_seed,
        },
        "solver": {
            "name": "batched PyTorch LM",
            "max_iterations": args.torch_lm_iterations,
            "damping": args.torch_lm_damping,
            "dtype": args.torch_lm_dtype,
            "target_converged_fraction": args.torch_lm_target_converged,
            "points": total_solver_points,
            "nonconverged_points_estimate": nonconverged_solver_points,
            "nonconverged_fraction": nonconverged_solver_points / total_solver_points,
            "invalid_xyz_points": invalid_xyz_points,
        },
        "definitions": {
            "point_uncertainty": "1 - effective_anchor_count / K, computed from K relative precision weights",
            "point_error": "one-sided nearest-GT Euclidean distance for each recovered point",
            "measurement_uncertainty": "negative learned relative precision weight (higher means less reliable)",
            "measurement_error": "absolute corrected-range error to the nearest-GT matched point",
            "spearman": "per-shape Spearman rank correlation, then macro-averaged across shapes",
            "risk_coverage": "retain lowest-uncertainty outputs; risk is mean one-sided error",
            "aurc": "mean selective risk over every attainable coverage from 1/N to 1",
            "auroc": f"within-shape detection of the highest-error {100.0 * args.high_error_fraction:.1f}%",
            "confidence_interval": f"95% percentile bootstrap across shapes ({args.bootstrap_resamples} resamples)",
        },
        "point_primary": point_summary,
        "range_secondary": range_summary,
        "range_pairs_per_shape": min(args.range_pairs_per_shape, model.num_points * model.num_basis),
        "weight_sum_deviation_max": weight_sum_deviation_max,
        "wall_seconds": elapsed,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False, allow_nan=False)

    summary_rows = []
    for level, metrics in (("point_primary", point_summary), ("range_secondary", range_summary)):
        summary_rows.append(
            {
                "analysis_level": level,
                "samples": len(per_shape_rows),
                "spearman_mean": metrics["spearman"]["mean"],
                "spearman_ci95_low": metrics["spearman"]["bootstrap_95ci_low"],
                "spearman_ci95_high": metrics["spearman"]["bootstrap_95ci_high"],
                "auroc_mean": metrics["auroc"]["mean"],
                "auroc_ci95_low": metrics["auroc"]["bootstrap_95ci_low"],
                "auroc_ci95_high": metrics["auroc"]["bootstrap_95ci_high"],
                "aurc_mean_x1000": 1000.0 * metrics["aurc"]["mean"],
                "aurc_ci95_low_x1000": 1000.0 * metrics["aurc"]["bootstrap_95ci_low"],
                "aurc_ci95_high_x1000": 1000.0 * metrics["aurc"]["bootstrap_95ci_high"],
                "aurc_gain_vs_random_mean_pct": 100.0 * metrics["aurc_gain_vs_random"]["mean"],
            }
        )
    write_csv(output_dir / "summary.csv", list(summary_rows[0]), summary_rows)

    point_spearman = point_summary["spearman"]
    point_auroc = point_summary["auroc"]
    point_aurc = point_summary["aurc"]
    point_gain = point_summary["aurc_gain_vs_random"]
    print("=" * 78)
    print("FINAL point-level uncertainty quality (macro mean across 1,200 shapes)")
    print(
        "Spearman: {:.6f} [95% CI {:.6f}, {:.6f}] (higher is better)".format(
            point_spearman["mean"],
            point_spearman["bootstrap_95ci_low"],
            point_spearman["bootstrap_95ci_high"],
        )
    )
    print(
        "High-error AUROC: {:.6f} [95% CI {:.6f}, {:.6f}] (higher is better)".format(
            point_auroc["mean"],
            point_auroc["bootstrap_95ci_low"],
            point_auroc["bootstrap_95ci_high"],
        )
    )
    print(
        "AURC x1e3: {:.6f} [95% CI {:.6f}, {:.6f}] (lower is better)".format(
            1000.0 * point_aurc["mean"],
            1000.0 * point_aurc["bootstrap_95ci_low"],
            1000.0 * point_aurc["bootstrap_95ci_high"],
        )
    )
    print(
        "AURC gain vs random: {:.3f}% [95% CI {:.3f}%, {:.3f}%]".format(
            100.0 * point_gain["mean"],
            100.0 * point_gain["bootstrap_95ci_low"],
            100.0 * point_gain["bootstrap_95ci_high"],
        )
    )
    print(
        "Numerics: invalid_xyz_points={} | estimated_nonconverged_fraction={:.8f} | "
        "weight_sum_deviation_max={:.3e}".format(
            invalid_xyz_points,
            nonconverged_solver_points / total_solver_points,
            weight_sum_deviation_max,
        )
    )
    print(f"Artifacts: {output_dir.resolve()}")
    print(f"END | exit=0 | wall_seconds={elapsed:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
