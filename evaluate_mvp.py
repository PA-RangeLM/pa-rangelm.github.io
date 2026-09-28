"""Zero-shot PCN-to-MVP evaluation for selectable MVP category scopes."""

import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from datasets.MVP import MVP_CATEGORY_NAMES, get_mvp_test_loader
from datasets.utils import read_yaml
from loss_functions.chamfer_ndim import CDNLoss
from models.pointr.pa_rangelm import PARangeLM
from utils.optimization import find_points_from_distance
from utils.torch_lm import find_points_from_distance_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PCN-trained PA-RangeLM checkpoint on MVP."
    )
    parser.add_argument(
        "--data_path",
        default="data/MVP_Benchmark/Completion/MVP_Test_CP.h5",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_keypoint", type=int, default=8)
    parser.add_argument(
        "--keypoint",
        choices=("basis", "curvature_radius"),
        default="curvature_radius",
    )
    parser.add_argument("--curve_k", type=int, default=16)
    parser.add_argument("--curve_radius", type=float, default=0.075)
    parser.add_argument(
        "--keypoint_cache_dir",
        default="cache/mvp_keypoints",
        help="Set to an empty string to disable the reusable anchor cache.",
    )
    parser.add_argument(
        "--views",
        default="all",
        help="'all' for the final 26-view test, or comma-separated IDs such as 0.",
    )
    parser.add_argument(
        "--category_scope",
        choices=("overlap8", "unseen8", "all16"),
        default="overlap8",
        help=(
            "MVP labels to evaluate: PCN-overlapping labels 0--7, unseen labels "
            "8--15, or all 16 categories."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Balanced smoke-test subset. Keep 0 for the final result.",
    )
    parser.add_argument("--random_rotate", action="store_true")
    parser.add_argument("--rotation_seed", type=int, default=2026)
    parser.add_argument(
        "--rotation_mode",
        choices=("independent", "same", "mixed"),
        default="independent",
    )
    parser.add_argument(
        "--solver", choices=("torch_lm", "scipy"), default="torch_lm"
    )
    parser.add_argument(
        "--torch_lm_iterations",
        type=int,
        default=80,
        help="Matches the final PCN CSV evaluation protocol.",
    )
    parser.add_argument("--torch_lm_damping", type=float, default=1e-3)
    parser.add_argument(
        "--torch_lm_dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--torch_lm_target_converged", type=float, default=0.999
    )
    return parser.parse_args()


def parse_views(value):
    if value.lower() == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_rotation_angles(category_name, model_id, seed, mode):
    sample_key = "{}:{}:{}".format(seed, category_name, model_id).encode("utf-8")
    sample_seed = int.from_bytes(hashlib.sha256(sample_key).digest()[:8], "little")
    rng = np.random.default_rng(sample_seed)
    use_same_angle = mode == "same" or (mode == "mixed" and rng.random() < 0.5)
    if use_same_angle:
        angle = rng.uniform(-np.pi, np.pi)
        return np.full(3, angle, dtype=np.float32)
    return rng.uniform(-np.pi, np.pi, size=3).astype(np.float32)


def rotation_matrix_xyz(angles, device, dtype):
    angles = torch.as_tensor(np.stack(angles), device=device, dtype=dtype)
    x, y, z = angles.unbind(dim=1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    rx = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=1
    ).reshape(-1, 3, 3)
    ry = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=1
    ).reshape(-1, 3, 3)
    rz = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=1
    ).reshape(-1, 3, 3)
    return torch.bmm(rz, torch.bmm(ry, rx))


def rotate_batch(data, category_names, model_ids, seed, mode):
    angles = [
        get_rotation_angles(category_name, model_id, seed, mode)
        for category_name, model_id in zip(category_names, model_ids)
    ]
    rotation = rotation_matrix_xyz(
        angles, data["partial"].device, data["partial"].dtype
    )
    rotation_t = rotation.transpose(1, 2)
    for key in ("partial", "gt", "basis_points"):
        data[key] = torch.bmm(data[key], rotation_t)


def unwrap_outputs(outputs):
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1], None
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    raise RuntimeError("Unexpected model output structure")


def checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_best_val", "model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Cannot find a model state dictionary in checkpoint")


def adapt_state_prefix(state, model):
    model_has_module = next(iter(model.state_dict())).startswith("module.")
    state_has_module = next(iter(state)).startswith("module.")
    if model_has_module == state_has_module:
        return state
    if state_has_module:
        return {key[len("module.") :]: value for key, value in state.items()}
    return {"module." + key: value for key, value in state.items()}


def batched_cd_l1(chamfer, prediction, target):
    d1, d2, _, _ = chamfer.chd(prediction, target)
    return 0.5 * (
        torch.sqrt(d1.clamp_min(1e-9)).mean(dim=1)
        + torch.sqrt(d2.clamp_min(1e-9)).mean(dim=1)
    )


def recover_points(
    dense_distances,
    basis_points,
    uncertainty_weights,
    args,
):
    if args.solver == "torch_lm":
        solve_dtype = (
            torch.float32 if args.torch_lm_dtype == "float32" else torch.float64
        )
        return find_points_from_distance_torch(
            dense_distances,
            basis_points,
            weights=uncertainty_weights,
            max_iterations=args.torch_lm_iterations,
            initial_damping=args.torch_lm_damping,
            solve_dtype=solve_dtype,
            target_converged_fraction=args.torch_lm_target_converged,
        )

    recovered = []
    for batch_index in range(dense_distances.shape[0]):
        points = find_points_from_distance(
            dense_distances[batch_index].detach().cpu().numpy(),
            basis_points[batch_index].detach().cpu().numpy(),
        )
        recovered.append(torch.from_numpy(points.astype(np.float32)))
    return torch.stack(recovered, dim=0).to(dense_distances.device)


def summarize(rows):
    per_category = {}
    active_labels = sorted({int(row["label"]) for row in rows})
    for label in active_labels:
        category = MVP_CATEGORY_NAMES[label]
        values = [row["cd_l1_x1000"] for row in rows if row["label"] == label]
        if values:
            per_category[category] = {
                "count": len(values),
                "mean_cd_l1_x1000": float(np.mean(values)),
                "median_cd_l1_x1000": float(np.median(values)),
            }
    category_means = [item["mean_cd_l1_x1000"] for item in per_category.values()]
    return {
        "sample_count": len(rows),
        "micro_mean_cd_l1_x1000": float(
            np.mean([row["cd_l1_x1000"] for row in rows])
        ),
        "macro_mean_cd_l1_x1000": float(np.mean(category_means)),
        "per_category": per_category,
    }


def write_per_category_csv(path, summary):
    fieldnames = (
        "label",
        "category",
        "count",
        "mean_cd_l1_x1000",
        "median_cd_l1_x1000",
    )
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label in sorted(
            int(value) for value in summary["active_labels"]
        ):
            category = MVP_CATEGORY_NAMES[label]
            metrics = summary["per_category"].get(category)
            if metrics is None:
                continue
            writer.writerow(
                {
                    "label": label,
                    "category": category,
                    "count": metrics["count"],
                    "mean_cd_l1_x1000": "{:.8f}".format(
                        metrics["mean_cd_l1_x1000"]
                    ),
                    "median_cd_l1_x1000": "{:.8f}".format(
                        metrics["median_cd_l1_x1000"]
                    ),
                }
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for PA-RangeLM MVP evaluation. Activate the GPU environment "
            "before running this script."
        )
    device = torch.device("cuda")
    views = parse_views(args.views)
    if args.keypoint == "curvature_radius" and args.num_workers != 0:
        print(
            "WARNING: forcing num_workers=0 because Open3D curvature "
            "estimation can segfault in forked DataLoader workers."
        )
        args.num_workers = 0

    dataset, loader = get_mvp_test_loader(
        h5_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_keypoints=args.num_keypoint,
        keypoint_type=args.keypoint,
        curvature_k=args.curve_k,
        curvature_radius=args.curve_radius,
        overlap_only=args.category_scope == "overlap8",
        category_scope=args.category_scope,
        views=views,
        max_samples=args.max_samples,
        sampling_seed=args.rotation_seed,
        keypoint_cache_dir=args.keypoint_cache_dir,
    )

    print("MVP file: {}".format(os.path.abspath(args.data_path)))
    print("MVP arrays: partial={} | complete={}".format(
        dataset.partial_shape, dataset.complete_shape
    ))
    print("Evaluation scope: {} | samples={} | views={}".format(
        args.category_scope, len(dataset), "all 26" if views is None else views
    ))
    print("Rotation: {} | seed={} | mode={}".format(
        "enabled" if args.random_rotate else "disabled",
        args.rotation_seed,
        args.rotation_mode,
    ))
    if args.max_samples:
        print("WARNING: max_samples is nonzero; this is a smoke test, not a final result.")

    model_config = read_yaml(args.config_path)
    model = PARangeLM(model_config.model).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.model_path, map_location=device)
    state = adapt_state_prefix(checkpoint_state(checkpoint), model)
    model.load_state_dict(state, strict=True)
    model.eval()

    loss_model = model.module if hasattr(model, "module") else model
    uncertainty_enabled = bool(
        getattr(loss_model, "range_lm_uncertainty_enabled", False)
    )
    chamfer = CDNLoss(mode="point", metric_fun="cd_l1")
    rows = []
    running_loss_sum = 0.0
    running_sample_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        progress = tqdm(loader, desc="PCN->MVP {}".format(args.category_scope))
        for category_names, model_ids, data in progress:
            for key in ("partial", "gt", "basis_points"):
                data[key] = data[key].to(device, non_blocking=True)
            if args.random_rotate:
                rotate_batch(
                    data,
                    category_names,
                    model_ids,
                    args.rotation_seed,
                    args.rotation_mode,
                )

            outputs = model(data["partial"], data["basis_points"])
            _, dense_distances, auxiliary = unwrap_outputs(outputs)
            uncertainty_weights = auxiliary if uncertainty_enabled else None
            prediction = recover_points(
                dense_distances,
                data["basis_points"],
                uncertainty_weights,
                args,
            )
            losses = batched_cd_l1(chamfer, prediction, data["gt"]) * 1000.0

            labels = data["label"].cpu().numpy()
            original_indices = data["original_index"].cpu().numpy()
            batch_losses = losses.detach().cpu().numpy()
            running_loss_sum += float(batch_losses.sum())
            running_sample_count += int(len(batch_losses))
            for batch_index, loss in enumerate(batch_losses):
                rows.append(
                    {
                        "original_index": int(original_indices[batch_index]),
                        "complete_index": int(original_indices[batch_index] // 26),
                        "view_id": int(original_indices[batch_index] % 26),
                        "label": int(labels[batch_index]),
                        "category": category_names[batch_index],
                        "model_id": model_ids[batch_index],
                        "rotated": int(args.random_rotate),
                        "rotation_seed": args.rotation_seed if args.random_rotate else "",
                        "rotation_mode": args.rotation_mode if args.random_rotate else "",
                        "cd_l1_x1000": float(loss),
                    }
                )
            progress.set_postfix(
                cd_l1_x1000="{:.4f}".format(
                    running_loss_sum / running_sample_count
                )
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["active_labels"] = sorted(
        {int(row["label"]) for row in rows}
    )
    summary.update(
        {
            "data_path": os.path.abspath(args.data_path),
            "model_path": os.path.abspath(args.model_path),
            "config_path": os.path.abspath(args.config_path),
            "rotated": bool(args.random_rotate),
            "rotation_seed": args.rotation_seed if args.random_rotate else None,
            "rotation_mode": args.rotation_mode if args.random_rotate else None,
            "views": "all" if views is None else views,
            "max_samples": args.max_samples,
            "solver": args.solver,
            "category_scope": args.category_scope,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    )
    summary_path = args.summary_json or os.path.splitext(args.output_csv)[0] + ".json"
    with open(summary_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)
    per_category_path = (
        os.path.splitext(args.output_csv)[0] + "_per_category.csv"
    )
    write_per_category_csv(per_category_path, summary)

    print("\nFinal PCN->MVP {} result".format(args.category_scope))
    for category, metrics in summary["per_category"].items():
        print(
            "  {:12s} n={:5d} mean={:.4f} median={:.4f}".format(
                category,
                metrics["count"],
                metrics["mean_cd_l1_x1000"],
                metrics["median_cd_l1_x1000"],
            )
        )
    print("  micro mean CD-L1 x1000: {:.4f}".format(
        summary["micro_mean_cd_l1_x1000"]
    ))
    print("  macro mean CD-L1 x1000: {:.4f}".format(
        summary["macro_mean_cd_l1_x1000"]
    ))
    print("Per-sample CSV: {}".format(os.path.abspath(args.output_csv)))
    print("Per-category CSV: {}".format(os.path.abspath(per_category_path)))
    print("Summary JSON: {}".format(os.path.abspath(summary_path)))


if __name__ == "__main__":
    main()
