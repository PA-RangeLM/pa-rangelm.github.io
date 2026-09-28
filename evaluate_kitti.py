"""Evaluate a PCN-car-trained checkpoint on the rotated KITTI car benchmark.

The loader follows the normalization used by PCN/PoinTr: each extracted car is
centered, yaw-aligned, scaled by its 3-D bounding box, and axis-swapped before
inference. KITTI has no complete target, so the directly available metric is
Fidelity (one-sided squared CD-L2 from the observed input to the completion).
The optional exact MMD pass first maps each rotated completion back to the PCN
canonical frame, then compares it with all complete PCN car references.  Both
metrics can be produced by one invocation; MMD progress is stored per sample so
an interrupted run can resume without repeating finished comparisons.
"""

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.curvature import compute_curvature_normal
from datasets.data_transforms import deterministic_fps
from datasets.utils import read_pcd, read_yaml, to_o3d_pcd
from loss_functions.chamfer_ndim import CDNLoss
from models.pointr.pa_rangelm import PARangeLM
from utils.torch_lm import find_points_from_distance_torch

# Flat/degenerate local neighborhoods can make curvature normalization have a
# zero denominator.  The evaluator already detects the resulting non-finite
# values and falls back to FPS, so suppress this harmless warning to keep the
# two evaluation progress bars readable.
warnings.filterwarnings(
    'ignore',
    message=r'invalid value encountered in (scalar )?divide',
    category=RuntimeWarning,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root', default='data/KITTI')
    parser.add_argument('--pcn_root', default='data/PCN')
    parser.add_argument('--category_file', default='data/KITTI/KITTI.json')
    parser.add_argument('--config_path', default='configs/pa_rangelm_stage2.yaml')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    # Open3D is not fork-safe on this environment; multiprocessing workers can
    # segfault while reading PCD files. Keep KITTI loading single-process.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--rotation_mode',
        choices=['none', 'single_axis'],
        default='none',
        help='KITTI input rotation protocol used by the rotated evaluation.',
    )
    parser.add_argument(
        '--rotation_axis',
        choices=['x', 'y', 'z'],
        default='y',
        help='Axis used by single-axis rotation. After PCN/PoinTr normalization, y is the vertical axis.',
    )
    parser.add_argument('--rotation_min_deg', type=float, default=-180.0)
    parser.add_argument('--rotation_max_deg', type=float, default=180.0)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_keypoints', type=int, default=8)
    parser.add_argument('--curve_k', type=int, default=16)
    parser.add_argument('--curve_radius', type=float, default=0.075)
    parser.add_argument('--torch_lm_iterations', type=int, default=80)
    parser.add_argument('--torch_lm_damping', type=float, default=0.001)
    parser.add_argument('--torch_lm_dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--torch_lm_target_converged', type=float, default=0.999)
    parser.add_argument(
        '--solver',
        choices=['torch_lm'],
        default='torch_lm',
        help=(
            'XYZ recovery solver. PA-RangeLM must use torch_lm so its learned '
            'relative uncertainty weights participate in coordinate recovery.'
        ),
    )
    parser.add_argument('--compute_mmd', action='store_true')
    parser.add_argument('--mmd_batch_size', type=int, default=16)
    parser.add_argument('--skip_inference', action='store_true',
                        help='Reuse existing predictions/fidelity and resume MMD without overwriting them.')
    return parser.parse_args()


def stable_seed(seed, sample_id):
    digest = hashlib.sha256(f'{seed}:{sample_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def single_axis_rotation_matrix(axis, angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    if axis == 'x':
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ], dtype=np.float32)
    if axis == 'y':
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float32)
    if axis == 'z':
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    raise ValueError(f'unsupported rotation axis: {axis}')


def rotate_kitti_points(points, sample_id, seed, mode, axis, min_deg, max_deg):
    """Apply a deterministic per-sample single-axis rotation.

    Rotation is applied after PCN/PoinTr KITTI pose normalization and before
    anchor selection / input sampling, so points and anchors remain in the same
    rotated coordinate frame.
    """
    if mode == 'none':
        return points.astype(np.float32, copy=False), 0.0
    if mode != 'single_axis':
        raise ValueError(f'unsupported rotation mode: {mode}')
    if max_deg < min_deg:
        raise ValueError(f'rotation_max_deg ({max_deg}) must be >= rotation_min_deg ({min_deg})')

    rng = np.random.default_rng(stable_seed(seed, f'rotation:{sample_id}'))
    angle_deg = float(rng.uniform(min_deg, max_deg))
    rotation = single_axis_rotation_matrix(axis, np.deg2rad(angle_deg))
    rotated = np.dot(points, rotation.T)
    return rotated.astype(np.float32), angle_deg


def normalize_object_pose(points, bbox):
    """PCN/PoinTr KITTI normalization, kept byte-for-byte equivalent in math."""
    bbox = bbox.copy()
    center = (bbox.min(0) + bbox.max(0)) / 2
    bbox -= center
    yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    bbox = np.dot(bbox, rotation)
    scale = bbox[3, 0] - bbox[0, 0]
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError(f'invalid KITTI bounding-box scale: {scale}')
    points = np.dot(points - center, rotation) / scale
    points = np.dot(points, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    return points.astype(np.float32)


def curvature_radius_basis(points, n_keypoints, k, radius):
    """Match the training-time curvature-radius anchor selection."""
    fps_basis = deterministic_fps(points, n_keypoints)
    if points.shape[0] < 3:
        return fps_basis.astype(np.float32)

    try:
        pcd = to_o3d_pcd(points)
        curvatures, _ = compute_curvature_normal(pcd, min(k, points.shape[0]))
        if not np.all(np.isfinite(curvatures)):
            return fps_basis.astype(np.float32)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return fps_basis.astype(np.float32)

    distances = np.sqrt(np.sum(
        (fps_basis[:, None, :] - points[None, :, :]) ** 2,
        axis=-1,
    ))
    outside = distances > radius
    selected = np.zeros((n_keypoints, 3), dtype=np.float32)
    for index in range(n_keypoints):
        local_curvature = curvatures.copy()
        local_curvature[outside[index]] = -1.0
        selected[index] = points[int(np.argmax(local_curvature))]
    return selected


class KITTICars(Dataset):
    def __init__(
        self,
        root,
        category_file,
        n_points,
        n_keypoints,
        curve_k,
        curve_radius,
        seed,
        rotation_mode='none',
        rotation_axis='y',
        rotation_min_deg=-180.0,
        rotation_max_deg=180.0,
    ):
        self.root = Path(root)
        self.n_points = n_points
        self.n_keypoints = n_keypoints
        self.curve_k = curve_k
        self.curve_radius = curve_radius
        self.seed = seed
        self.rotation_mode = rotation_mode
        self.rotation_axis = rotation_axis
        self.rotation_min_deg = rotation_min_deg
        self.rotation_max_deg = rotation_max_deg

        with open(category_file, 'r', encoding='utf-8') as handle:
            categories = json.load(handle)
        self.samples = [sample for category in categories for sample in category.get('test', [])]
        self.samples.sort()
        if not self.samples:
            raise RuntimeError(f'no KITTI test samples listed in {category_file}')

        missing_cars = [sample for sample in self.samples if not (self.root / 'cars' / f'{sample}.pcd').is_file()]
        missing_bboxes = [sample for sample in self.samples if not (self.root / 'bboxes' / f'{sample}.txt').is_file()]
        if missing_cars or missing_bboxes:
            details = []
            if missing_cars:
                details.append(f'missing cars: {len(missing_cars)} (first: {missing_cars[0]})')
            if missing_bboxes:
                details.append(f'missing bboxes: {len(missing_bboxes)} (first: {missing_bboxes[0]})')
            raise FileNotFoundError('; '.join(details))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        points = np.asarray(
            o3d.io.read_point_cloud(str(self.root / 'cars' / f'{sample_id}.pcd')).points,
            dtype=np.float32,
        )
        bbox = np.loadtxt(self.root / 'bboxes' / f'{sample_id}.txt').astype(np.float32)
        points = normalize_object_pose(points, bbox)
        points, rotation_deg = rotate_kitti_points(
            points,
            sample_id=sample_id,
            seed=self.seed,
            mode=self.rotation_mode,
            axis=self.rotation_axis,
            min_deg=self.rotation_min_deg,
            max_deg=self.rotation_max_deg,
        )
        if points.shape[0] == 0:
            raise ValueError(f'empty point cloud: {sample_id}')

        basis = curvature_radius_basis(
            points,
            self.n_keypoints,
            self.curve_k,
            self.curve_radius,
        )
        rng = np.random.default_rng(stable_seed(self.seed, sample_id))
        sampled = points[rng.permutation(points.shape[0])[:self.n_points]]
        valid_count = sampled.shape[0]
        padded = np.zeros((self.n_points, 3), dtype=np.float32)
        padded[:valid_count] = sampled
        return sample_id, torch.from_numpy(padded), torch.from_numpy(basis), valid_count, rotation_deg


class PCNCarReferences(Dataset):
    def __init__(self, pcn_root):
        root = Path(pcn_root)
        self.paths = []
        for split in ('train', 'val', 'test'):
            self.paths.extend(sorted((root / split / 'complete' / '02958343').glob('*.pcd')))
        if not self.paths:
            raise FileNotFoundError(f'no PCN car references found below {pcn_root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.from_numpy(read_pcd(str(self.paths[index])).astype(np.float32))

    def load_all(self):
        """Load references once; avoids rereading 5927 PCD files per KITTI sample."""
        arrays = []
        expected_shape = None
        for path in self.paths:
            points = read_pcd(str(path)).astype(np.float32)
            if expected_shape is None:
                expected_shape = points.shape
            if points.shape != expected_shape:
                raise ValueError(
                    f'PCN reference shape mismatch: {path} has {points.shape}, expected {expected_shape}'
                )
            arrays.append(points)
        return torch.from_numpy(np.stack(arrays, axis=0))


def load_model(config_path, checkpoint_path, device):
    model_config = read_yaml(config_path)
    model = PARangeLM(model_config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model_state_best_val']
    has_module_prefix = next(iter(state)).startswith('module.')
    if has_module_prefix:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state)
    model.eval()
    return model


def unpack_outputs(outputs):
    if isinstance(outputs, (tuple, list)) and len(outputs) in (2, 3):
        auxiliary = outputs[2] if len(outputs) == 3 else None
        return outputs[0], outputs[1], auxiliary
    raise RuntimeError(f'unexpected PA-RangeLM output: {type(outputs)}')


def save_summary(output_dir, summary):
    path = Path(output_dir) / 'metrics.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')


def run_inference(args, model, loader, device):
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / 'fidelity_per_sample.jsonl'
    rotation_path = output_dir / 'rotation_per_sample.jsonl'
    chamfer = CDNLoss(mode='point', metric_fun='cd_l1').chd
    fidelity_values = []
    model_core = model.module if hasattr(model, 'module') else model
    use_uncertainty = bool(
        getattr(model_core, 'range_lm_uncertainty_enabled', False)
    )
    solve_dtype = (
        torch.float32 if args.torch_lm_dtype == 'float32' else torch.float64
    )
    solver_diagnostics = []

    print(
        'Coordinate recovery: PA-RangeLM batched PyTorch LM | '
        f'iterations={args.torch_lm_iterations} | '
        f'damping={args.torch_lm_damping} | dtype={args.torch_lm_dtype} | '
        f'target_converged={args.torch_lm_target_converged} | '
        f'uncertainty_weights={use_uncertainty}'
    )

    with open(per_sample_path, 'w', encoding='utf-8') as fidelity_file, \
            open(rotation_path, 'w', encoding='utf-8') as rotation_file, \
            torch.no_grad():
        for sample_ids, partial, basis, valid_counts, rotation_degs in tqdm(loader, desc='KITTI inference'):
            partial = partial.to(device, non_blocking=True)
            basis = basis.to(device, non_blocking=True)
            outputs = model(partial, basis)
            _, dense_distances, auxiliary = unpack_outputs(outputs)

            weights = auxiliary if use_uncertainty else None
            predictions, diagnostics = find_points_from_distance_torch(
                dense_distances,
                basis,
                weights=weights,
                max_iterations=args.torch_lm_iterations,
                initial_damping=args.torch_lm_damping,
                solve_dtype=solve_dtype,
                target_converged_fraction=args.torch_lm_target_converged,
                return_diagnostics=True,
            )
            solver_diagnostics.append(diagnostics)

            if not torch.isfinite(predictions).all():
                invalid_count = int((~torch.isfinite(predictions).all(dim=-1)).sum().item())
                raise RuntimeError(
                    f'coordinate recovery produced {invalid_count} non-finite points'
                )

            for batch_index, sample_id in enumerate(sample_ids):
                valid_count = int(valid_counts[batch_index])
                observed = partial[batch_index:batch_index + 1, :valid_count]
                predicted = predictions[batch_index:batch_index + 1].float()
                input_to_pred, _, _, _ = chamfer(observed, predicted)
                fidelity = float(input_to_pred.mean().item())
                fidelity_values.append(fidelity)

                sample_dir = predictions_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / 'input.npy', observed[0].cpu().numpy())
                np.save(sample_dir / 'pred.npy', predicted[0].cpu().numpy())
                fidelity_file.write(json.dumps({'sample_id': sample_id, 'fidelity_cd_l2': fidelity}) + '\n')
                fidelity_file.flush()
                rotation_file.write(json.dumps({
                    'sample_id': sample_id,
                    'rotation_mode': args.rotation_mode,
                    'rotation_axis': args.rotation_axis,
                    'rotation_deg': float(rotation_degs[batch_index]),
                }) + '\n')
                rotation_file.flush()

    diagnostics_summary = None
    if solver_diagnostics:
        diagnostics_summary = {
            'lm_batch_iterations_mean': float(np.mean([
                item.iterations for item in solver_diagnostics
            ])),
            'lm_batch_iterations_max': int(max(
                item.iterations for item in solver_diagnostics
            )),
            'lm_converged_fraction_mean': float(np.mean([
                item.converged_fraction for item in solver_diagnostics
            ])),
            'lm_accepted_fraction_mean': float(np.mean([
                item.accepted_fraction for item in solver_diagnostics
            ])),
            'lm_range_residual_mse_mean': float(np.mean([
                item.mean_squared_range_residual for item in solver_diagnostics
            ])),
            'lm_range_residual_max_squared': float(max(
                item.max_squared_range_residual for item in solver_diagnostics
            )),
        }
    return (
        float(np.mean(fidelity_values)),
        len(fidelity_values),
        use_uncertainty,
        diagnostics_summary,
    )


def run_exact_mmd(args, device):
    """PoinTr's exact MMD protocol in the canonical PCN coordinate frame.

    KITTI inference is intentionally performed in the randomly rotated frame,
    and Fidelity is measured in that same frame.  PCN references, however, are
    stored in their canonical frame.  Predictions must therefore be inverse
    rotated before their symmetric CD-L2 is compared with those references.
    """
    predictions_dir = Path(args.output_dir) / 'predictions'
    sample_dirs = sorted(path for path in predictions_dir.iterdir() if path.is_dir())
    rotation_path = Path(args.output_dir) / 'rotation_per_sample.jsonl'
    if not rotation_path.is_file():
        raise FileNotFoundError(
            f'cannot compute canonical-frame MMD; missing {rotation_path}'
        )
    rotation_records = {}
    with open(rotation_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record['sample_id']
            if sample_id in rotation_records:
                raise ValueError(f'duplicate rotation record for KITTI sample {sample_id}')
            rotation_records[sample_id] = record
    prediction_ids = {path.name for path in sample_dirs}
    rotation_ids = set(rotation_records)
    if prediction_ids != rotation_ids:
        missing = sorted(prediction_ids - rotation_ids)
        orphaned = sorted(rotation_ids - prediction_ids)
        raise ValueError(
            'prediction/rotation sample mismatch: '
            f'missing_rotation={len(missing)}, orphaned_rotation={len(orphaned)}'
        )

    references = PCNCarReferences(args.pcn_root)
    reference_tensor = references.load_all()
    print(
        'Exact MMD reference set: '
        f'{len(references)} complete PCN cars from train+val+test'
    )
    print('MMD coordinate frame: inverse-rotated predictions vs canonical PCN references')
    chamfer = CDNLoss(mode='point', metric_fun='cd_l1').chd
    # Keep the former rotated-vs-canonical values untouched.  They are not
    # resumable into this corrected metric because they use a different frame.
    progress_path = Path(args.output_dir) / 'mmd_canonical_per_sample.jsonl'
    completed = {}
    if progress_path.is_file():
        with open(progress_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record.get('prediction_frame') != 'pcn_canonical_inverse_rotated':
                    raise ValueError(
                        f'incompatible MMD progress record in {progress_path}: '
                        f'{record.get("prediction_frame")!r}'
                    )
                completed[record['sample_id']] = record['mmd_cd_l2']

    with open(progress_path, 'a', encoding='utf-8') as progress_file, torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc='Exact KITTI MMD'):
            sample_id = sample_dir.name
            if sample_id in completed:
                continue
            prediction = torch.from_numpy(np.load(sample_dir / 'pred.npy')).to(device).unsqueeze(0)
            rotation_record = rotation_records[sample_id]
            rotation_mode = rotation_record['rotation_mode']
            rotation_axis = rotation_record['rotation_axis']
            rotation_deg = float(rotation_record['rotation_deg'])
            if rotation_mode == 'single_axis':
                rotation = torch.from_numpy(single_axis_rotation_matrix(
                    rotation_axis,
                    np.deg2rad(rotation_deg),
                )).to(device=device, dtype=prediction.dtype)
                # Forward rotation uses row vectors: p_rot = p_canonical @ R.T.
                # Its inverse is therefore p_canonical = p_rot @ R.
                prediction = torch.matmul(prediction, rotation)
            elif rotation_mode != 'none':
                raise ValueError(
                    f'unsupported saved rotation mode for {sample_id}: {rotation_mode}'
                )
            best = float('inf')
            for references_batch in reference_tensor.split(args.mmd_batch_size, dim=0):
                references_batch = references_batch.to(device, non_blocking=True)
                expanded_prediction = prediction.expand(references_batch.shape[0], -1, -1).contiguous()
                pred_to_ref, ref_to_pred, _, _ = chamfer(expanded_prediction, references_batch)
                batch_cd = pred_to_ref.mean(dim=1) + ref_to_pred.mean(dim=1)
                best = min(best, float(batch_cd.min().item()))
            completed[sample_id] = best
            progress_file.write(json.dumps({
                'sample_id': sample_id,
                'mmd_cd_l2': best,
                'prediction_frame': 'pcn_canonical_inverse_rotated',
                'rotation_mode': rotation_mode,
                'rotation_axis': rotation_axis,
                'rotation_deg': rotation_deg,
            }) + '\n')
            progress_file.flush()
    mmd_values = [completed[path.name] for path in sample_dirs]
    return float(np.mean(mmd_values)), len(references)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('KITTI evaluation requires a CUDA GPU for PA-RangeLM and Chamfer Distance')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = KITTICars(
        args.kitti_root,
        args.category_file,
        args.num_input_points,
        args.num_keypoints,
        args.curve_k,
        args.curve_radius,
        args.seed,
        rotation_mode=args.rotation_mode,
        rotation_axis=args.rotation_axis,
        rotation_min_deg=args.rotation_min_deg,
        rotation_max_deg=args.rotation_max_deg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(
        'KITTI rotation protocol: '
        f'mode={args.rotation_mode} | axis={args.rotation_axis} | '
        f'range=[{args.rotation_min_deg:.1f},{args.rotation_max_deg:.1f}] deg | '
        f'seed={args.seed}'
    )
    model = load_model(args.config_path, args.model_path, device)
    model_core = model.module if hasattr(model, 'module') else model
    use_uncertainty = bool(
        getattr(model_core, 'range_lm_uncertainty_enabled', False)
    )
    if args.skip_inference:
        fidelity_path = Path(args.output_dir) / 'fidelity_per_sample.jsonl'
        if not fidelity_path.is_file():
            raise FileNotFoundError(f'cannot skip inference; missing {fidelity_path}')
        metrics_path = Path(args.output_dir) / 'metrics.json'
        previous_summary = {}
        if metrics_path.is_file():
            with open(metrics_path, 'r', encoding='utf-8') as handle:
                previous_summary = json.load(handle)
        with open(fidelity_path, 'r', encoding='utf-8') as handle:
            fidelity_values = [json.loads(line)['fidelity_cd_l2'] for line in handle if line.strip()]
        fidelity = float(np.mean(fidelity_values))
        sample_count = len(fidelity_values)
        solver_summary = None
        print(f'Reusing existing predictions ({sample_count} samples); inference skipped.')
    else:
        previous_summary = {}
        fidelity, sample_count, use_uncertainty, solver_summary = run_inference(
            args, model, loader, device
        )
    # Preserve inference diagnostics when only the expensive MMD pass is being
    # rerun, but never carry an older MMD value into a different protocol.
    summary = {
        key: value for key, value in previous_summary.items()
        if not key.startswith('mmd_')
    }
    summary.update({
        'protocol': 'Rotated KITTI; PoinTr Fidelity/MMD CD-L2',
        'sample_count': sample_count,
        'model_path': str(Path(args.model_path).resolve()),
        'config_path': str(Path(args.config_path).resolve()),
        'rotation_mode': args.rotation_mode,
        'rotation_axis': args.rotation_axis,
        'rotation_min_deg': args.rotation_min_deg,
        'rotation_max_deg': args.rotation_max_deg,
        'rotation_seed': args.seed,
        'coordinate_solver': args.solver,
        'uncertainty_weights': use_uncertainty,
        'fidelity_cd_l2': fidelity,
        'fidelity_cd_l2_x1e3': fidelity * 1000.0,
    })
    summary.update({
        'torch_lm_iterations': args.torch_lm_iterations,
        'torch_lm_damping': args.torch_lm_damping,
        'torch_lm_dtype': args.torch_lm_dtype,
        'torch_lm_target_converged': args.torch_lm_target_converged,
    })
    if solver_summary is not None:
        summary.update(solver_summary)
    if args.compute_mmd:
        summary['mmd_status'] = 'in_progress'
        summary['mmd_prediction_frame'] = 'pcn_canonical_inverse_rotated'
        summary['mmd_progress_file'] = 'mmd_canonical_per_sample.jsonl'
    save_summary(args.output_dir, summary)
    print(f'Fidelity CD-L2: {fidelity:.8f} ({fidelity * 1000.0:.4f} x1e3)')

    if args.compute_mmd:
        mmd, reference_count = run_exact_mmd(args, device)
        summary['mmd_cd_l2'] = mmd
        summary['mmd_cd_l2_x1e3'] = mmd * 1000.0
        summary['mmd_reference_count'] = reference_count
        summary['mmd_prediction_frame'] = 'pcn_canonical_inverse_rotated'
        summary['mmd_progress_file'] = 'mmd_canonical_per_sample.jsonl'
        summary['mmd_status'] = 'complete'
        save_summary(args.output_dir, summary)
        print(f'MMD CD-L2: {mmd:.8f} ({mmd * 1000.0:.4f} x1e3)')


if __name__ == '__main__':
    main()
