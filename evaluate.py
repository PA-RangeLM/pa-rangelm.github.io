import torch
import numpy as np
from datasets.utils import read_yaml
from datasets.PCN import get_dataset_and_loader, get_dataset_and_loader_for_car
from models.pointr.pa_rangelm import PARangeLM
from tqdm import tqdm
from utils.training import var_or_cuda, var_keys_for_gpu
from loss_functions.chamfer_ndim import CDNLoss
from utils.optimization import find_points_from_distance
from utils.torch_lm import find_points_from_distance_torch
import argparse
import hashlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('-n_k', '--num_keypoint', type=int, default=8)
parser.add_argument('-cp', '--config_path', type=str, default='configs/pa_rangelm_stage1.yaml')
parser.add_argument('-mp', '--model_path', type=str, default='')
parser.add_argument('-k', '--keypoint', type=str, default='curvature_radius')
parser.add_argument('-c_k', '--curve_k', type=int, default=16)
parser.add_argument('-c_r', '--curve_radius', type=float, default=0.075)
parser.add_argument('-c_t', '--curve_thres', type=float, default=0.5)
parser.add_argument('-r', '--random_rotate', action='store_true',
                    help='Randomly rotate partial, GT, and basis points during evaluation.')
parser.add_argument('-rs', '--rotation_seed', type=int, default=2026,
                    help='Seed used to generate a reproducible rotation for each sample.')
parser.add_argument('-rm', '--rotation_mode', type=str, default='independent',
                    choices=['independent', 'same', 'mixed'],
                    help='Use independent XYZ angles, one shared angle, or a per-sample mixture.')
parser.add_argument('--solver', type=str, default='scipy', choices=['scipy', 'torch_lm'],
                    help='XYZ recovery solver. torch_lm is the batched PyTorch parity implementation.')
parser.add_argument('--torch_lm_iterations', type=int, default=50,
                    help='Maximum iterations used by the PyTorch LM solver.')
parser.add_argument('--torch_lm_damping', type=float, default=1e-3,
                    help='Initial damping used by the PyTorch LM solver.')
parser.add_argument('--torch_lm_dtype', type=str, default='float64',
                    choices=['float32', 'float64'],
                    help='Linear-solve precision. float64 is for strict parity; float32 is for full evaluation.')
parser.add_argument('--torch_lm_target_converged', type=float, default=1.0,
                    help='Stop a PyTorch LM batch after this fraction of points has converged.')
parser.add_argument('--diagnose_uncertainty', action='store_true',
                    help='Report uncertainty-weight calibration against final LM range residuals.')
parser.add_argument('--uncertainty_diag_samples_per_batch', type=int, default=2048,
                    help='Number of point-anchor pairs retained per batch for percentile/correlation diagnostics.')
parser.add_argument('--skip_distance_eval', action='store_true',
                    help='Skip the separate distance-space pass; useful for a faster LM-only diagnostic run.')
parser.add_argument('--car_only', action='store_true',
                    help='Evaluate only the PCN car category (taxonomy 02958343).')


def get_rotation_angles(taxonomy_id, model_id, seed, mode):
    """Return deterministic XYZ Euler angles in [-pi, pi] for one sample."""
    sample_key = f'{seed}:{taxonomy_id}:{model_id}'.encode('utf-8')
    sample_seed = int.from_bytes(hashlib.sha256(sample_key).digest()[:8], 'little')
    rng = np.random.default_rng(sample_seed)

    use_same_angle = mode == 'same' or (mode == 'mixed' and rng.random() < 0.5)
    if use_same_angle:
        angle = rng.uniform(-np.pi, np.pi)
        return np.full(3, angle, dtype=np.float32)
    return rng.uniform(-np.pi, np.pi, size=3).astype(np.float32)


def rotation_matrix_xyz(angles, device, dtype):
    """Build batched Rz @ Ry @ Rx rotation matrices for column vectors."""
    angles = torch.as_tensor(np.stack(angles), device=device, dtype=dtype)
    x, y, z = angles.unbind(dim=1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack([
        ones, zeros, zeros,
        zeros, cx, -sx,
        zeros, sx, cx,
    ], dim=1).reshape(-1, 3, 3)
    ry = torch.stack([
        cy, zeros, sy,
        zeros, ones, zeros,
        -sy, zeros, cy,
    ], dim=1).reshape(-1, 3, 3)
    rz = torch.stack([
        cz, -sz, zeros,
        sz, cz, zeros,
        zeros, zeros, ones,
    ], dim=1).reshape(-1, 3, 3)
    return torch.bmm(rz, torch.bmm(ry, rx))


def rotate_eval_batch(data, taxonomy_ids, model_ids, seed, mode):
    angles = [
        get_rotation_angles(taxonomy_id, model_id, seed, mode)
        for taxonomy_id, model_id in zip(taxonomy_ids, model_ids)
    ]
    reference = data['partial']
    rotation = rotation_matrix_xyz(angles, reference.device, reference.dtype)
    rotation_t = rotation.transpose(1, 2)
    for key in ('partial', 'gt', 'basis_points'):
        data[key] = torch.bmm(data[key], rotation_t)


def unpack_eval_outputs(ret):
    if isinstance(ret, (list, tuple)):
        if len(ret) == 2:
            return ret[0], ret[1], None
        if len(ret) == 3:
            return ret[0], ret[1], ret[2]
    raise RuntimeError(f"Unexpected eval output structure: {type(ret)} / {len(ret) if isinstance(ret, (list, tuple)) else 'NA'}")


class UncertaintyDiagnostics:
    """Streaming diagnostics for learned per-range LM precision weights."""

    def __init__(self, samples_per_batch=2048):
        self.samples_per_batch = max(1, int(samples_per_batch))
        self.batch_index = 0
        self.total_count = 0
        self.below_01 = 0
        self.above_4 = 0
        self.above_6 = 0
        self.weight_samples = []
        self.lm_residual_samples = []
        self.target_error_samples = []
        self.effective_anchor_samples = []
        self.anchor_weight_sum = None
        self.anchor_lm_residual_sum = None
        self.anchor_target_error_sum = None
        self.anchor_count = 0

    def update(
        self,
        weights,
        corrected_distances,
        recovered_xyz,
        anchors,
        gt_xyz,
        nearest_gt_indices,
    ):
        # This is the exact nonlinear measurement residual optimized by LM:
        # | ||x_hat-a_k||_2 - D_corr,k |.  It does not assume a point-wise
        # correspondence between the unordered prediction and GT point clouds.
        solve_dtype = recovered_xyz.dtype
        corrected_distances = corrected_distances.to(dtype=solve_dtype)
        weights = weights.to(dtype=solve_dtype)
        recovered_ranges = torch.linalg.vector_norm(
            recovered_xyz.unsqueeze(2) - anchors.to(dtype=solve_dtype).unsqueeze(1),
            dim=-1,
        )
        lm_abs_residual = (recovered_ranges - corrected_distances).abs()

        # Point clouds are unordered, so D_corr cannot be compared with GT by
        # tensor position.  Match every recovered point to its nearest GT point
        # (the prediction-to-GT Chamfer correspondence), then compare its eight
        # corrected ranges with that target point's true anchor ranges.
        nearest_gt_indices = nearest_gt_indices.long()
        nearest_gt_xyz = torch.gather(
            gt_xyz,
            dim=1,
            index=nearest_gt_indices.unsqueeze(-1).expand(-1, -1, 3),
        ).to(dtype=solve_dtype)
        target_ranges = torch.linalg.vector_norm(
            nearest_gt_xyz.unsqueeze(2)
            - anchors.to(dtype=solve_dtype).unsqueeze(1),
            dim=-1,
        )
        target_abs_error = (corrected_distances - target_ranges).abs()

        finite = (
            torch.isfinite(weights)
            & torch.isfinite(lm_abs_residual)
            & torch.isfinite(target_abs_error)
        )
        finite_weights = weights[finite]
        self.total_count += int(finite_weights.numel())
        self.below_01 += int((finite_weights < 0.1).sum().item())
        self.above_4 += int((finite_weights > 4.0).sum().item())
        self.above_6 += int((finite_weights > 6.0).sum().item())

        # Exact per-anchor means over the full split.
        masked_weights = torch.where(finite, weights, torch.zeros_like(weights))
        masked_lm_residuals = torch.where(
            finite, lm_abs_residual, torch.zeros_like(lm_abs_residual)
        )
        masked_target_errors = torch.where(
            finite, target_abs_error, torch.zeros_like(target_abs_error)
        )
        anchor_weight_sum = masked_weights.sum(dim=(0, 1)).double().cpu().numpy()
        anchor_lm_residual_sum = masked_lm_residuals.sum(dim=(0, 1)).double().cpu().numpy()
        anchor_target_error_sum = masked_target_errors.sum(dim=(0, 1)).double().cpu().numpy()
        anchor_counts = finite.sum(dim=(0, 1)).double().cpu().numpy()
        if self.anchor_weight_sum is None:
            self.anchor_weight_sum = np.zeros_like(anchor_weight_sum)
            self.anchor_lm_residual_sum = np.zeros_like(anchor_lm_residual_sum)
            self.anchor_target_error_sum = np.zeros_like(anchor_target_error_sum)
            self.anchor_count = np.zeros_like(anchor_counts)
        self.anchor_weight_sum += anchor_weight_sum
        self.anchor_lm_residual_sum += anchor_lm_residual_sum
        self.anchor_target_error_sum += anchor_target_error_sum
        self.anchor_count += anchor_counts

        # Deterministic subsampling keeps memory bounded while preserving the
        # joint weight/residual pairs needed by Pearson and Spearman tests.
        flat_weights = weights.reshape(-1)
        flat_lm_residuals = lm_abs_residual.reshape(-1)
        flat_target_errors = target_abs_error.reshape(-1)
        sample_count = min(self.samples_per_batch, flat_weights.numel())
        sample_indices = (
            torch.arange(sample_count, device=weights.device, dtype=torch.long)
            * 104729
            + self.batch_index * 8191
        ) % flat_weights.numel()
        sampled_weights = flat_weights.index_select(0, sample_indices)
        sampled_lm_residuals = flat_lm_residuals.index_select(0, sample_indices)
        sampled_target_errors = flat_target_errors.index_select(0, sample_indices)
        sampled_finite = (
            torch.isfinite(sampled_weights)
            & torch.isfinite(sampled_lm_residuals)
            & torch.isfinite(sampled_target_errors)
        )
        self.weight_samples.append(
            sampled_weights[sampled_finite].float().cpu().numpy()
        )
        self.lm_residual_samples.append(
            sampled_lm_residuals[sampled_finite].float().cpu().numpy()
        )
        self.target_error_samples.append(
            sampled_target_errors[sampled_finite].float().cpu().numpy()
        )

        # Effective anchor count equals 8 for uniform weights and approaches 1
        # when one anchor dominates.  Sample points rather than individual ranges.
        effective_count = weights.sum(dim=-1).square() / weights.square().sum(dim=-1).clamp_min(1e-12)
        flat_effective = effective_count.reshape(-1)
        point_sample_count = min(max(1, sample_count // weights.size(-1)), flat_effective.numel())
        point_indices = (
            torch.arange(point_sample_count, device=weights.device, dtype=torch.long)
            * 65537
            + self.batch_index * 4099
        ) % flat_effective.numel()
        self.effective_anchor_samples.append(
            flat_effective.index_select(0, point_indices).float().cpu().numpy()
        )
        self.batch_index += 1

    @staticmethod
    def _format_quantiles(values, quantiles):
        results = np.percentile(values, quantiles)
        return " | ".join(
            "P{}={:.4f}".format(q, value)
            for q, value in zip(quantiles, results)
        )

    def report(self):
        if not self.weight_samples or self.total_count == 0:
            print("Uncertainty calibration diagnostics: no finite samples")
            return

        weights = np.concatenate(self.weight_samples).astype(np.float64)
        lm_residuals = np.concatenate(self.lm_residual_samples).astype(np.float64)
        target_errors = np.concatenate(self.target_error_samples).astype(np.float64)
        effective = np.concatenate(self.effective_anchor_samples).astype(np.float64)
        quantiles = [1, 5, 25, 50, 75, 95, 99]
        target_pearson = float(np.corrcoef(weights, target_errors)[0, 1])
        lm_pearson = float(np.corrcoef(weights, lm_residuals)[0, 1])
        try:
            from scipy.stats import spearmanr
            target_spearman_result = spearmanr(weights, target_errors)
            target_spearman = float(
                getattr(
                    target_spearman_result,
                    'statistic',
                    getattr(target_spearman_result, 'correlation', target_spearman_result[0]),
                )
            )
            lm_spearman_result = spearmanr(weights, lm_residuals)
            lm_spearman = float(
                getattr(
                    lm_spearman_result,
                    'statistic',
                    getattr(lm_spearman_result, 'correlation', lm_spearman_result[0]),
                )
            )
        except (ImportError, AttributeError, TypeError):
            target_spearman = float('nan')
            lm_spearman = float('nan')

        error_p10, error_p90 = np.percentile(target_errors, [10, 90])
        low_error_weight = float(weights[target_errors <= error_p10].mean())
        high_error_weight = float(weights[target_errors >= error_p90].mean())
        weight_p10, weight_p90 = np.percentile(weights, [10, 90])
        low_weight_error = float(target_errors[weights <= weight_p10].mean())
        high_weight_error = float(target_errors[weights >= weight_p90].mean())
        # A merely negative sign is too weak with hundreds of thousands of
        # samples.  Require a meaningful monotonic effect for PASS; reserve
        # PARTIAL for useful separation that only appears in the extremes.
        strong_monotonic = (
            target_spearman <= -0.10
            and high_error_weight <= 0.95 * low_error_weight
        )
        extreme_separation = low_weight_error >= 1.05 * high_weight_error
        if strong_monotonic and extreme_separation:
            mechanism_status = "PASS"
        elif target_spearman < 0.0 and extreme_separation:
            mechanism_status = "PARTIAL"
        else:
            mechanism_status = "CHECK"

        print("=" * 72)
        print("Uncertainty calibration diagnostics (read-only)")
        print(
            "sampled_pairs={} | full_pairs={} | batches={}".format(
                weights.size, self.total_count, self.batch_index
            )
        )
        print("weight quantiles: " + self._format_quantiles(weights, quantiles))
        print("nearest-GT range error quantiles: " + self._format_quantiles(target_errors, quantiles))
        print("|LM range residual| quantiles: " + self._format_quantiles(lm_residuals, quantiles))
        print(
            "extreme weight fractions: w<0.1={:.4%} | w>4={:.4%} | w>6={:.4%}".format(
                self.below_01 / self.total_count,
                self.above_4 / self.total_count,
                self.above_6 / self.total_count,
            )
        )
        print(
            "effective anchors: mean={:.3f} | P5={:.3f} | P50={:.3f} | P95={:.3f} (uniform=8)".format(
                effective.mean(), *np.percentile(effective, [5, 50, 95])
            )
        )
        print(
            "weight vs nearest-GT range error: Pearson={:.4f} | Spearman={:.4f} (desired: negative)".format(
                target_pearson, target_spearman
            )
        )
        print(
            "weight vs |LM residual|: Pearson={:.4f} | Spearman={:.4f} (secondary)".format(
                lm_pearson, lm_spearman
            )
        )
        print(
            "mean weight by target error: easiest 10%={:.4f} | hardest 10%={:.4f}".format(
                low_error_weight, high_error_weight
            )
        )
        print(
            "mean target error by weight: lowest 10%={:.6e} | highest 10%={:.6e}".format(
                low_weight_error, high_weight_error
            )
        )
        print("per-anchor mean weight / target error / |LM residual|:")
        for anchor_index, (weight_sum, target_error_sum, lm_residual_sum, count) in enumerate(zip(
            self.anchor_weight_sum,
            self.anchor_target_error_sum,
            self.anchor_lm_residual_sum,
            self.anchor_count,
        )):
            print(
                "  anchor {}: weight={:.4f} | target_error={:.6e} | lm_residual={:.6e}".format(
                    anchor_index,
                    weight_sum / max(count, 1.0),
                    target_error_sum / max(count, 1.0),
                    lm_residual_sum / max(count, 1.0),
                )
            )
        print(
            "mechanism check: {} (PASS requires Spearman<=-0.10 and >=5% decile separation)".format(
                mechanism_status
            )
        )
        print(
            "Note: target error uses each recovered point's nearest GT Chamfer match; LM residual is reported separately because it is influenced by weighting."
        )
        print("=" * 72)


def test_single_epoch(model, dataloader, loss_object, rotation_options=None):
    model.eval()
    epoch_losses = []

    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in tqdm(dataloader):

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            if rotation_options is not None:
                rotate_eval_batch(data, taxonomy_ids, model_ids, **rotation_options)

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])
            coarse_points, dense_points, _ = unpack_eval_outputs(ret)

            sparse_loss = loss_object.chamfer_loss(coarse_points, gt_dists)
            dense_loss = loss_object.chamfer_loss(dense_points, gt_dists)
            loss_vals = [sparse_loss.item() * 1000, dense_loss.item() * 1000]

            epoch_losses.append(loss_vals)

    return np.mean(epoch_losses, axis=0)


def optimize_points_loader(
    model,
    dataloader,
    loss_object,
    rotation_options=None,
    solver='scipy',
    torch_lm_iterations=50,
    torch_lm_damping=1e-3,
    torch_lm_dtype='float64',
    torch_lm_target_converged=1.0,
    diagnose_uncertainty=False,
    uncertainty_diag_samples_per_batch=2048,
):
    losses = []
    model.eval()
    results_per_cat = {}
    torch_lm_diagnostics = []
    uncertainty_weight_stats = []
    loss_model = model.module if hasattr(model, 'module') else model
    uncertainty_enabled = getattr(
        loss_model, 'range_lm_uncertainty_enabled', False
    )
    uncertainty_diagnostics = (
        UncertaintyDiagnostics(uncertainty_diag_samples_per_batch)
        if diagnose_uncertainty and uncertainty_enabled else None
    )
    progress = tqdm(dataloader)
    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in progress:

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            if rotation_options is not None:
                rotate_eval_batch(data, taxonomy_ids, model_ids, **rotation_options)

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])
            coarse_batch, dense_batch, eval_aux = unpack_eval_outputs(ret)
            uncertainty_weights = eval_aux if uncertainty_enabled else None

            optimized_batch = None
            diagnostic_point_losses = None
            if solver == 'torch_lm':
                lm_start = time.perf_counter()
                solve_dtype = (
                    torch.float32 if torch_lm_dtype == 'float32' else torch.float64
                )
                optimized_batch, diagnostics = find_points_from_distance_torch(
                    dense_batch,
                    data['basis_points'],
                    weights=uncertainty_weights,
                    max_iterations=torch_lm_iterations,
                    initial_damping=torch_lm_damping,
                    solve_dtype=solve_dtype,
                    target_converged_fraction=torch_lm_target_converged,
                    return_diagnostics=True,
                )
                torch_lm_diagnostics.append(diagnostics)
                if uncertainty_weights is not None:
                    uncertainty_weight_stats.append((
                        float(uncertainty_weights.min().item()),
                        float(uncertainty_weights.max().item()),
                        float(uncertainty_weights.std().item()),
                    ))
                    if uncertainty_diagnostics is not None:
                        # Reuse this batched Chamfer result both for the GT
                        # correspondence diagnostic and the reported point loss.
                        chamfer_d1, chamfer_d2, nearest_gt_indices, _ = loss_object.chd(
                            optimized_batch, data['gt']
                        )
                        diagnostic_point_losses = 0.5 * (
                            torch.sqrt(chamfer_d1.clamp_min(1e-9)).mean(dim=1)
                            + torch.sqrt(chamfer_d2.clamp_min(1e-9)).mean(dim=1)
                        )
                        uncertainty_diagnostics.update(
                            uncertainty_weights,
                            dense_batch,
                            optimized_batch,
                            data['basis_points'],
                            data['gt'],
                            nearest_gt_indices,
                        )
                progress.set_postfix(
                    lm_seconds='{:.2f}'.format(time.perf_counter() - lm_start),
                    lm_iterations=diagnostics.iterations,
                    lm_converged='{:.4f}'.format(diagnostics.converged_fraction),
                )

            for i in range(coarse_batch.shape[0]):
                category_id = taxonomy_ids[i]

                if solver == 'scipy':
                    basis_points_np = data['basis_points'][i].cpu().numpy()
                    distances_np = dense_batch[i].cpu().numpy()
                    optimized_points_np = find_points_from_distance(
                        distances_np, basis_points_np
                    )
                    optimized_points = torch.from_numpy(
                        optimized_points_np.astype('float32')
                    ).unsqueeze(0).to(data['gt'].device)
                else:
                    optimized_points = optimized_batch[i:i + 1]

                if diagnostic_point_losses is not None:
                    loss_val = float(diagnostic_point_losses[i].item())
                else:
                    loss = loss_object.chamfer_loss(
                        optimized_points,
                        data['gt'][i:i + 1],
                    )
                    loss_val = loss.item()
                losses.append(loss_val)

                if category_id not in results_per_cat:
                    results_per_cat[category_id] = []

                results_per_cat[category_id].append(loss_val * 1000)

    if torch_lm_diagnostics:
        print(
            "PyTorch LM diagnostics: iterations_max={} | converged_mean={:.4f} | "
            "accepted_mean={:.4f} | range_mse_mean={:.3e} | range_max_sq_max={:.3e}".format(
                max(item.iterations for item in torch_lm_diagnostics),
                np.mean([item.converged_fraction for item in torch_lm_diagnostics]),
                np.mean([item.accepted_fraction for item in torch_lm_diagnostics]),
                np.mean([item.mean_squared_range_residual for item in torch_lm_diagnostics]),
                max(item.max_squared_range_residual for item in torch_lm_diagnostics),
            )
        )
    if uncertainty_weight_stats:
        print(
            "Uncertainty weights: min={:.4f} | max={:.4f} | std_mean={:.4f}".format(
                min(item[0] for item in uncertainty_weight_stats),
                max(item[1] for item in uncertainty_weight_stats),
                np.mean([item[2] for item in uncertainty_weight_stats]),
            )
        )
    if uncertainty_diagnostics is not None:
        uncertainty_diagnostics.report()

    return losses, results_per_cat


# ['02691156', '02933112', '02958343',  '03001627',  '03636649',  '04256520',  '04379243', '04530566']
# ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']


def get_curvature_exp_name(curv_dict):
    return "k_{}_cr_{}_th_{}".format(
        curv_dict['k'],
        curv_dict['curvature_radius'],
        curv_dict['curvature_thres']
    )
if __name__ == '__main__':

    args = parser.parse_args()
    keypoint_number = args.num_keypoint
    keypoint_type = args.keypoint
    model_config_path = args.config_path
    model_save_path = args.model_path
    curvature_params = {}
    curvature_params['k'] =  args.curve_k
    curvature_params['curvature_radius'] = args.curve_radius
    curvature_params['curvature_thres'] = args.curve_thres

    rotation_options = None
    if args.random_rotate:
        rotation_options = {'seed': args.rotation_seed, 'mode': args.rotation_mode}

    print("Keypoint Type: ", keypoint_type ,  " Model config: ", curvature_params)
    print("XYZ recovery solver: {}".format(args.solver))
    if args.solver == 'torch_lm':
        print(
            "PyTorch LM config: iterations={} | initial_damping={} | dtype={} | "
            "target_converged={}".format(
                args.torch_lm_iterations,
                args.torch_lm_damping,
                args.torch_lm_dtype,
                args.torch_lm_target_converged,
            )
        )
    if rotation_options is not None:
        print("Random XYZ rotation: enabled | range: [-180, 180] degrees | mode: {} | seed: {}".format(
            args.rotation_mode, args.rotation_seed
        ))

    config_path = './configs/pcn.yaml'
    config = read_yaml(config_path)
    dataset_loader = get_dataset_and_loader_for_car if args.car_only else get_dataset_and_loader
    if args.car_only:
        print('Dataset category scope: car only | taxonomy_id=02958343')

    model_config = read_yaml(model_config_path)

    model = PARangeLM(model_config.model).cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    saved_model_config = torch.load(model_save_path)

    model.load_state_dict(saved_model_config['model_state_best_val'])

    print("VALIDATION")

    sample_num = 50000
    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1')
    if args.skip_distance_eval:
        print(" distance-space pass: skipped")
    else:
        val_dataset, val_loader = dataset_loader('val', config, 1, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
        val_score = test_single_epoch(model, val_loader, loss_obj, rotation_options)
        print(" val loss: ", val_score)

    opt_val_dataset, opt_val_loader = dataset_loader('val', config, 4, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)

    point_losses, opt_res_per_cat = optimize_points_loader(
        model,
        opt_val_loader,
        loss_obj,
        rotation_options,
        solver=args.solver,
        torch_lm_iterations=args.torch_lm_iterations,
        torch_lm_damping=args.torch_lm_damping,
        torch_lm_dtype=args.torch_lm_dtype,
        torch_lm_target_converged=args.torch_lm_target_converged,
        diagnose_uncertainty=args.diagnose_uncertainty,
        uncertainty_diag_samples_per_batch=args.uncertainty_diag_samples_per_batch,
    )

    mean_loss = np.mean(point_losses)
    print("Optimized point loss: %.2f " % (mean_loss*1e3))

    for cat in opt_res_per_cat:
        print("Distance loss %.4f for %s" % (np.mean(opt_res_per_cat[cat]), cat))

    print("TEST")

    sample_num = 50000
    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1')
    if args.skip_distance_eval:
        print(" distance-space pass: skipped")
    else:
        val_dataset, val_loader = dataset_loader('test', config, 1, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
        val_score = test_single_epoch(model, val_loader, loss_obj, rotation_options)
        print(" test loss: ", val_score)

    opt_val_dataset, opt_val_loader = dataset_loader('test', config, 4, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)

    point_losses, opt_res_per_cat = optimize_points_loader(
        model,
        opt_val_loader,
        loss_obj,
        rotation_options,
        solver=args.solver,
        torch_lm_iterations=args.torch_lm_iterations,
        torch_lm_damping=args.torch_lm_damping,
        torch_lm_dtype=args.torch_lm_dtype,
        torch_lm_target_converged=args.torch_lm_target_converged,
        diagnose_uncertainty=args.diagnose_uncertainty,
        uncertainty_diag_samples_per_batch=args.uncertainty_diag_samples_per_batch,
    )

    mean_loss = np.mean(point_losses)
    print("Optimized point loss: %.2f " % (mean_loss*1e3))

    for cat in opt_res_per_cat:
        print("Distance loss %.4f for %s" % (np.mean(opt_res_per_cat[cat]), cat))
