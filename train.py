import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from datasets.utils import read_yaml
from datasets.PCN import get_dataset_and_loader, get_dataset_and_loader_for_car
from models.pointr.pa_rangelm import PARangeLM
from tqdm import tqdm
from utils.training import var_or_cuda, var_keys_for_gpu, save_model
from loss_functions.chamfer_ndim import CDNLoss
from utils.visualization import save_loss_graphs
from utils.pointr import build_optimizer, build_scheduler
from models.pointr.misc import set_random_seed
import argparse

try:
    import wandb
except ImportError:
    wandb = None


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

parser = argparse.ArgumentParser()
parser.add_argument('-n_k', '--num_keypoint', type=int, default=8)
parser.add_argument('-cp', '--config_path', type=str, default='configs/pa_rangelm_stage1.yaml')
parser.add_argument('-tcp', '--train_config_path', type=str, default='configs/pcn.yaml')
parser.add_argument('-k', '--keypoint', type=str, default='curvature_radius')
parser.add_argument('-c_k', '--curve_k', type=int, default=16)
parser.add_argument('-c_r', '--curve_radius', type=float, default=0.075)
parser.add_argument('-c_t', '--curve_thres', type=float, default=0.5)
parser.add_argument('-c_n', '--curvature_neighbor', type=int, default=16)
parser.add_argument('-sr', '--sample_ratio', type=float, default=1.0)
parser.add_argument('--balanced_train_subset', action='store_true',
                    help='Sample the same number of training objects from every PCN category.')
parser.add_argument('--subset_seed', type=int, default=2026,
                    help='Seed for the reproducible training subset selection.')
parser.add_argument('--seed', type=int, default=2026,
                    help='Random seed for model initialization and training.')
parser.add_argument('--experiment_name', type=str, default='',
                    help='Optional output directory name. Use this to isolate subset experiments.')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='PA-RangeLM')
parser.add_argument('--wandb_run_name', type=str, default='')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
parser.add_argument('--wandb_entity', type=str, default='')
parser.add_argument('--pretrained_path', type=str, default='',
                    help='Warm-start weights, e.g. the prototype-only val_best.pth.')
parser.add_argument('--car_only', action='store_true',
                    help='Train and validate only the PCN car category (taxonomy 02958343).')
parser.add_argument(
    '--range_lm_gradient_iterations',
    type=int,
    default=None,
    help='Override only range_lm_config.gradient_iterations for a controlled sensitivity run.',
)
parser.add_argument(
    '--profile_run_resources',
    action='store_true',
    help='Record training-loop wall time and peak CUDA memory once per run.',
)


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_{timestamp}.log')
    latest_log_path = os.path.join(log_dir, 'train_latest.log')

    logger = logging.getLogger('pa_rangelm_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    latest_file_handler = logging.FileHandler(latest_log_path, mode='w', encoding='utf-8')
    latest_file_handler.setLevel(logging.INFO)
    latest_file_handler.setFormatter(formatter)

    stream_handler = TqdmLoggingHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(latest_file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


def init_wandb(args, exp_base_path, model_config, data_config, logger):
    if not args.use_wandb:
        return None
    if wandb is None:
        logger.warning('wandb is not installed. Continue without wandb logging.')
        return None

    run_name = args.wandb_run_name if args.wandb_run_name else os.path.basename(exp_base_path)
    wandb_kwargs = dict(
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        dir=exp_base_path,
        config={
            'args': vars(args),
            'model_config': dict(model_config),
            'data_config': dict(data_config),
        },
    )
    if args.wandb_entity:
        wandb_kwargs['entity'] = args.wandb_entity

    run = wandb.init(**wandb_kwargs)
    logger.info(f'Initialized W&B: project={args.wandb_project} | run={run_name} | mode={args.wandb_mode}')
    return run


def save_subset_manifest(dataset, output_dir):
    manifest_path = os.path.join(output_dir, 'train_subset_manifest.json')
    full_total = sum(dataset.full_category_counts.values())
    selected_total = len(dataset)
    manifest = {
        'sampling_ratio_requested': dataset.subset_ratio,
        'sampling_ratio_actual': selected_total / full_total if full_total else 0.0,
        'balanced_sampling': dataset.balanced_sampling,
        'sampling_seed': dataset.sampling_seed,
        'full_total': full_total,
        'selected_total': selected_total,
        'full_category_counts': dataset.full_category_counts,
        'selected_category_counts': dataset.selected_category_counts,
        'samples': [
            {
                'taxonomy_id': sample['taxonomy_id'],
                'model_id': sample['model_id'],
            }
            for sample in dataset.file_list
        ],
    }
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    return manifest_path, manifest


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}


def load_weights_safely(model, state_dict, logger=None, strict_shapes=True):
    target = unwrap_model(model)
    current = target.state_dict()
    source = strip_module_prefix(state_dict)

    matched = {}
    skipped = []
    for k, v in source.items():
        if k not in current:
            skipped.append((k, 'missing_in_current'))
            continue
        if strict_shapes and tuple(current[k].shape) != tuple(v.shape):
            skipped.append((k, f'shape_mismatch {tuple(v.shape)} -> {tuple(current[k].shape)}'))
            continue
        matched[k] = v

    current.update(matched)
    target.load_state_dict(current, strict=False)
    if logger is not None:
        logger.info(f'Loaded {len(matched)} matched parameters; skipped {len(skipped)} incompatible parameters.')
    return len(matched), skipped


def safe_load_training_checkpoint(model, optim, schedular, checkpoint_path, logger):
    if not os.path.exists(checkpoint_path):
        return 0, None, None

    logger.info('Found final training checkpoint. Trying to resume...')
    ckpt = torch.load(checkpoint_path)
    initial_epoch = int(ckpt.get('epoch', 0))
    best_val_losses = ckpt.get('best_val_loss', None)
    best_losses = ckpt.get('best_loss', None)

    model_state = ckpt.get('model_state', None)
    if model_state is None:
        logger.warning('Checkpoint has no model_state. Start from scratch.')
        return 0, None, None

    load_weights_safely(model, model_state, logger=logger)

    resumed_optim = False
    if 'optimizer_state' in ckpt:
        try:
            optim.load_state_dict(ckpt['optimizer_state'])
            resumed_optim = True
            logger.info('Optimizer state resumed successfully.')
        except Exception as e:
            logger.warning(f'Optimizer state is incompatible and will be ignored: {e}')

    if resumed_optim and 'scheduler_state' in ckpt:
        try:
            if not isinstance(schedular, list):
                schedular.load_state_dict(ckpt['scheduler_state'])
            elif isinstance(ckpt['scheduler_state'], list):
                for sched_item, state in zip(schedular, ckpt['scheduler_state']):
                    if hasattr(sched_item, 'load_state_dict') and state is not None:
                        sched_item.load_state_dict(state)
            logger.info('Scheduler state resumed successfully.')
        except Exception as e:
            logger.warning(f'Scheduler state is incompatible and will be ignored: {e}')

    if resumed_optim:
        return initial_epoch, best_losses, best_val_losses

    logger.info('Resume fallback: weights loaded, but optimizer/scheduler states were not resumed. Restarting from epoch 0.')
    return 0, best_losses, best_val_losses


def unpack_eval_outputs(ret):
    if isinstance(ret, (list, tuple)):
        if len(ret) == 2:
            return ret[0], ret[1], None
        if len(ret) == 3:
            return ret[0], ret[1], ret[2]
    raise RuntimeError(f"Unexpected eval output structure: {type(ret)} / {len(ret) if isinstance(ret, (list, tuple)) else 'NA'}")



def test_single_epoch(model, dataloader, loss_object):
    """ 。

    non-English text removed/non-English text removed CD-L1；non-English text removed
    non-English text removed LM non-English text removed XYZ，non-English text removed XYZ CD-L1。non-English text removed
    torch.no_grad()，non-English text removed。
    """
    model.eval()
    epoch_losses = []
    loss_model = unwrap_model(model)
    range_lm_enabled = getattr(loss_model, 'range_lm_enabled', False)

    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in dataloader:

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])
            coarse_points, dense_points, eval_aux = unpack_eval_outputs(ret)

            sparse_loss = loss_object.chamfer_loss(coarse_points, gt_dists)
            dense_loss = loss_object.chamfer_loss(dense_points, gt_dists)
            loss_vals = [sparse_loss.item() * 1000, dense_loss.item() * 1000]
            if range_lm_enabled:
                uncertainty_weights = (
                    eval_aux
                    if getattr(loss_model, 'range_lm_uncertainty_enabled', False)
                    else None
                )
                recovered_xyz = loss_model.recover_xyz_for_validation(
                    dense_points,
                    data['basis_points'],
                    uncertainty_weights=uncertainty_weights,
                )
                xyz_loss = loss_object.chamfer_loss(recovered_xyz, data['gt'])
                loss_vals.append(xyz_loss.item() * 1000)
            epoch_losses.append(loss_vals)

    return np.mean(epoch_losses, axis=0)

def train_single_epoch(model, dataloader, optim, current_epoch, loss_object=None, accum_iter=1, use_progress=False):
    """  epoch  。

    non-English text removed head_only_epochs=0，non-English text removed1non-English text removed，
    non-English text removed、non-English text removed；non-English text removed、non-English text removed
    non-English text removed Prototype non-English text removed configure_range_lm_training_phase() non-English text removed。
    """
    model.train()
    epoch_losses = []
    iter_num = 0
    loss_model = unwrap_model(model)
    range_lm_enabled = getattr(loss_model, 'range_lm_enabled', False)
    if range_lm_enabled:
        head_only = current_epoch < loss_model.range_lm_head_only_epochs
        loss_model.configure_range_lm_training_phase(
            head_only=head_only,
            current_epoch=current_epoch,
        )
    if use_progress:
        pbar = tqdm(total=len(dataloader))

    for (taxonomy_ids, model_ids, data) in dataloader:

        for k in data:
            if k in var_keys_for_gpu:
                data[k] = var_or_cuda(data[k])

        gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))
        ret = model(data['partial'], data['basis_points'])

        (
            denoise_loss,
            recon_loss,
            proto_loss,
            xyz_loss,
            delta_loss,
            uncertainty_loss,
        ) = loss_model.get_loss(
            ret,
            gt_dists,
            gt_xyz=data['gt'],
            basis_points=data['basis_points'],
        )
        total_loss = (
            denoise_loss
            + recon_loss
            + proto_loss
            + xyz_loss
            + delta_loss
            + uncertainty_loss
        )

        (total_loss / accum_iter).backward()

        if ((iter_num + 1) % accum_iter == 0) or (iter_num + 1 == len(dataloader)):
            optim.step()
            model.zero_grad()

        loss_vals = [
            denoise_loss.item() * 1000,
            recon_loss.item() * 1000,
            proto_loss.item() * 1000,
            xyz_loss.item() * 1000,
            delta_loss.item() * 1000,
            uncertainty_loss.item() * 1000,
            total_loss.item() * 1000,
        ]
        epoch_losses.append(loss_vals)

        if use_progress:
            mean_epoch_loss = np.mean(epoch_losses, axis=0)
            pbar.set_description('[Batch: %d/%d]' % (iter_num + 1, len(dataloader)))
            pbar.set_postfix(
                denoise='%.2f' % mean_epoch_loss[0],
                recon='%.2f' % mean_epoch_loss[1],
                proto='%.2f' % mean_epoch_loss[2],
                xyz='%.2f' % mean_epoch_loss[3],
                delta='%.4f' % mean_epoch_loss[4],
                uncertainty='%.5f' % mean_epoch_loss[5],
                total='%.2f' % mean_epoch_loss[6],
            )
            pbar.update(1)
        iter_num += 1

    if use_progress:
        pbar.close()

    return np.mean(epoch_losses, axis=0)

def train_model(
    model,
    dataloader,
    optim,
    schedular,
    loss_object,
    epochs,
    val_dataloader=None,
    accum_iter=1,
    use_batch_progress=False,
    model_save_dir="",
    logger=None,
    wandb_run=None,
    early_stopping_patience=50,
):
    best_losses = None
    best_val_losses = None

    training_losses = []
    val_losses = []

    best_train_model = os.path.join(model_save_dir, "train_best.pth")
    best_val_model = os.path.join(model_save_dir, "val_best.pth")
    cur_train_config = os.path.join(model_save_dir, "train_last.pth")

    val_loss_last_improve = 0
    patience = int(early_stopping_patience)

    if logger is None:
        logger = logging.getLogger('pa_rangelm_train')
    initial_epoch, best_losses, best_val_losses = safe_load_training_checkpoint(
        model, optim, schedular, cur_train_config, logger
    )
    loss_model = unwrap_model(model)
    range_lm_enabled = getattr(loss_model, 'range_lm_enabled', False)

    model.zero_grad()
    remaining_epochs = max(0, epochs - initial_epoch)
    pbar = tqdm(total=remaining_epochs)
    for e in range(initial_epoch, epochs):
        if range_lm_enabled:
            if getattr(loss_model, 'range_lm_full_model_training', False):
                warmup_epochs = loss_model.range_lm_xyz_warmup_epochs
                xyz_scale = (
                    min(1.0, float(e + 1) / float(warmup_epochs))
                    if warmup_epochs > 0 else 1.0
                )
                phase = 'full-model-from-scratch (xyz_scale=%.3f)' % (
                    xyz_scale
                )
            elif getattr(loss_model, 'range_lm_uncertainty_only', False):
                phase = 'uncertainty-head-only'
            elif not getattr(loss_model, 'range_lm_correction_enabled', True):
                phase = 'decoder+uncertainty-head (no bounded correction)'
            else:
                phase = 'correction-head-only' if e < loss_model.range_lm_head_only_epochs else 'decoder+RangeLM-heads'
            logger.info('[Epoch %d] RangeLM trainable phase: %s' % (e + 1, phase))
        epoch_loss = train_single_epoch(model, dataloader, optim, e, loss_object, accum_iter=accum_iter, use_progress=use_batch_progress)
        if best_losses is None:
            best_losses = [float('inf')] * len(epoch_loss)
        training_losses.append(epoch_loss)

        if epoch_loss[-1] < best_losses[-1]:
            best_losses = np.minimum(np.array(best_losses), epoch_loss).tolist()
            if model_save_dir != "":
                save_model(model, best_train_model)

        if val_dataloader:
            val_loss = test_single_epoch(model, val_dataloader, loss_object)
            if best_val_losses is None:
                best_val_losses = [float('inf')] * len(val_loss)
            val_losses.append(val_loss)

            if val_loss[-1] < best_val_losses[-1]:
                best_val_losses = np.minimum(np.array(best_val_losses), val_loss).tolist()
                metric_name = 'XYZ CD-L1' if range_lm_enabled else 'dense distance CD-L1'
                logger.info("Epoch %d Best val %s improved to: %.2f" % (e, metric_name, val_loss[-1]))
                val_loss_last_improve = e

                if model_save_dir != "":
                    torch.save(
                        {
                            'epoch': e,
                            'model_state_best_val': model.state_dict(),
                            'best_loss': best_losses,
                            'current_loss': epoch_loss,
                            'best_val_loss': best_val_losses
                        }, best_val_model)

        if isinstance(schedular, list):
            scheduler_state = []
            for item in schedular:
                item.step()
                scheduler_state.append(item.state_dict() if hasattr(item, 'state_dict') else None)
        else:
            schedular.step()
            scheduler_state = schedular.state_dict()

        pbar.set_description('[Epoch %d/%d]' % (e + 1, epochs))
        pbar.set_postfix(
            train_total='%.2f' % epoch_loss[-1],
            train_recon='%.2f' % epoch_loss[1],
            train_proto='%.2f' % epoch_loss[2],
            train_xyz='%.2f' % epoch_loss[3],
            val_metric='%.2f' % (val_losses[-1][-1] if len(val_losses) > 0 else -1),
        )
        logger.info(
            '[Epoch %d] train_denoise=%.4f | train_recon=%.4f | train_proto=%.4f | train_xyz=%.4f | train_delta=%.4f | train_uncertainty=%.6f | train_total=%.4f | val_coarse=%.4f | val_dense=%.4f | val_xyz=%.4f' % (
                e + 1,
                epoch_loss[0],
                epoch_loss[1],
                epoch_loss[2],
                epoch_loss[3],
                epoch_loss[4],
                epoch_loss[5],
                epoch_loss[6],
                (val_losses[-1][0] if len(val_losses) > 0 else -1),
                (val_losses[-1][1] if len(val_losses) > 0 else -1),
                (val_losses[-1][2] if len(val_losses) > 0 and len(val_losses[-1]) > 2 else -1),
            )
        )

        if wandb_run is not None:
            log_dict = {
                'epoch': e + 1,
                'optimizer/lr': optim.param_groups[0]['lr'],
                'train/denoise': epoch_loss[0],
                'train/recon': epoch_loss[1],
                'train/proto': epoch_loss[2],
                'train/xyz': epoch_loss[3],
                'train/delta': epoch_loss[4],
                'train/uncertainty': epoch_loss[5],
                'train/total': epoch_loss[6],
            }
            if range_lm_enabled:
                log_dict['train/rangelm_xyz_scale'] = float(
                    loss_model._range_lm_xyz_loss_scale
                )
            uncertainty_stats = getattr(loss_model, '_last_uncertainty_stats', None)
            if uncertainty_stats is not None:
                log_dict.update({
                    'uncertainty/log_var_mean': uncertainty_stats['log_var_mean'],
                    'uncertainty/log_var_std': uncertainty_stats['log_var_std'],
                    'uncertainty/weight_min': uncertainty_stats['weight_min'],
                    'uncertainty/weight_max': uncertainty_stats['weight_max'],
                    'uncertainty/weight_std': uncertainty_stats['weight_std'],
                })
            if len(val_losses) > 0:
                log_dict.update({
                    'val/coarse': val_losses[-1][0],
                    'val/dense': val_losses[-1][1],
                })
                if len(val_losses[-1]) > 2:
                    log_dict['val/xyz'] = val_losses[-1][2]
            wandb_run.log(log_dict, step=e + 1)

        pbar.update(1)

        if model_save_dir != "":
            torch.save(
                {
                    'epoch': e + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'scheduler_state': scheduler_state,
                    'best_loss': best_losses,
                    'current_loss': epoch_loss,
                    'best_val_loss': best_val_losses
                }, cur_train_config)

        if patience >= 0 and (e - val_loss_last_improve) > patience:
            logger.info("Val loss not improved between %d and %d. Early stopping..." % (val_loss_last_improve, e))
            break
    pbar.close()

    train_arr = np.stack(training_losses, axis=0) if len(training_losses) > 0 else np.zeros((0, 7))
    val_width = 3 if range_lm_enabled else 2
    val_arr = np.stack(val_losses, axis=0) if len(val_losses) > 0 else np.zeros((0, val_width))
    return best_losses, train_arr, val_arr

def get_curvature_exp_name(curv_dict):
    return "k_{}_cr_{}_th_{}_cn_{}".format(
        curv_dict['k'],
        curv_dict['curvature_radius'],
        curv_dict['curvature_thres'],
        curv_dict['curvature_neighbor']
    )


if __name__ == '__main__':
    program_wall_start = time.perf_counter()
    args = parser.parse_args()
    set_random_seed(args.seed, deterministic=False)
    keypoint_number = args.num_keypoint
    keypoint_type = args.keypoint
    model_config_path = args.config_path
    dataset_ratio = args.sample_ratio
    train_config_path = args.train_config_path

    curvature_params = {}
    curvature_params['k'] =  args.curve_k
    curvature_params['curvature_radius'] = args.curve_radius
    curvature_params['curvature_thres'] = args.curve_thres
    curvature_params['curvature_neighbor'] = args.curvature_neighbor


    config = read_yaml(train_config_path)

    model_config = read_yaml(model_config_path)
    if args.range_lm_gradient_iterations is not None:
        if args.range_lm_gradient_iterations < 1:
            raise ValueError('--range_lm_gradient_iterations must be positive')
        override_cfg = getattr(model_config.model, 'range_lm_config', None)
        if override_cfg is None or not getattr(override_cfg, 'enabled', False):
            raise ValueError(
                '--range_lm_gradient_iterations requires an enabled range_lm_config'
            )
        override_cfg.gradient_iterations = int(args.range_lm_gradient_iterations)

    model = PARangeLM(model_config.model).cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    config_stem = Path(model_config_path).stem
    experiment_name = args.experiment_name if args.experiment_name else config_stem
    exp_base_path = os.path.join('train_res', keypoint_type, get_curvature_exp_name(curvature_params),  "k_{}".format(keypoint_number), experiment_name)

    os.makedirs(exp_base_path, exist_ok=True)
    log_dir = os.path.join(exp_base_path, 'logs')
    logger, log_path = setup_logger(log_dir)
    logger.info('Starting training run')
    logger.info(f'Arguments: {vars(args)}')
    logger.info(f'Model config path: {model_config_path}')
    logger.info(f'Train config path: {train_config_path}')
    logger.info(f'Logs will be saved to: {log_path}')
    if args.range_lm_gradient_iterations is not None:
        logger.info(
            'Controlled override: range_lm_config.gradient_iterations=%d' %
            args.range_lm_gradient_iterations
        )

    tr_cache_dir = os.path.join('cache', keypoint_type, "k{}".format(keypoint_number), get_curvature_exp_name(curvature_params), "train")
    os.makedirs(tr_cache_dir, exist_ok=True)

    sample_num = 40000
    num_epochs = int(getattr(model_config, 'max_epoch', 250))
    early_stopping_patience = int(
        getattr(model_config, 'early_stopping_patience', 50)
    )
    use_batch_progress = bool(
        getattr(model_config, 'use_batch_progress', True)
    )
    step_size = 40

    if args.car_only:
        if dataset_ratio != 1.0 or args.balanced_train_subset:
            raise ValueError('--car_only uses the complete car category; use --sample_ratio 1.0 without --balanced_train_subset')
        tr_dataset, tr_loader = get_dataset_and_loader_for_car(
            'train', config, 16, sample_num, shuffle=True, drop_last=True,
            cache_dir=tr_cache_dir, num_keypoints=keypoint_number,
            keypoint_type=keypoint_type, curvature_params=curvature_params,
        )
        val_dataset, val_loader = get_dataset_and_loader_for_car(
            'val', config, 1, sample_num, shuffle=False, drop_last=False,
            num_keypoints=keypoint_number, keypoint_type=keypoint_type,
            curvature_params=curvature_params,
        )
        logger.info('Dataset category scope: car only | taxonomy_id=02958343')
    else:
        tr_dataset, tr_loader = get_dataset_and_loader(
            'train', config, 16, sample_num, shuffle=True, drop_last=True,
            cache_dir=tr_cache_dir, num_keypoints=keypoint_number,
            keypoint_type=keypoint_type, curvature_params=curvature_params,
            dataset_size=dataset_ratio,
            balanced_sampling=args.balanced_train_subset,
            sampling_seed=args.subset_seed,
        )
        val_dataset, val_loader = get_dataset_and_loader(
            'val', config, 1, sample_num, shuffle=False, drop_last=False,
            num_keypoints=keypoint_number, keypoint_type=keypoint_type,
            curvature_params=curvature_params,
        )

    logger.info(f'Train samples: {len(tr_dataset)} | Val samples: {len(val_dataset)}')
    logger.info(f'Train iters/epoch: {len(tr_loader)} | Val iters: {len(val_loader)}')
    manifest_path, subset_manifest = save_subset_manifest(tr_dataset, exp_base_path)
    logger.info(
        'Train subset: requested=%.4f | actual=%.4f | balanced=%s | seed=%d' % (
            dataset_ratio,
            subset_manifest['sampling_ratio_actual'],
            args.balanced_train_subset,
            args.subset_seed,
        )
    )
    logger.info(f"Full train category counts: {subset_manifest['full_category_counts']}")
    logger.info(f"Selected train category counts: {subset_manifest['selected_category_counts']}")
    logger.info(f'Train subset manifest: {manifest_path}')
    proto_cfg = getattr(model_config.model, 'prototype_config', None)
    if proto_cfg is not None and getattr(proto_cfg, 'enabled', False):
        logger.info(
            'Prototype branch mode: %s | proto_loss_weight: %s | target_space: %s' % (
                getattr(proto_cfg, 'mode', 'full'),
                getattr(proto_cfg, 'loss_weight', 0.1),
                getattr(proto_cfg, 'target_space', 'distance'),
            )
        )
    else:
        logger.info('Prototype mode: disabled')

    range_lm_cfg = getattr(model_config.model, 'range_lm_config', None)
    range_lm_enabled = range_lm_cfg is not None and getattr(range_lm_cfg, 'enabled', False)
    range_lm_full_training = (
        range_lm_enabled
        and getattr(range_lm_cfg, 'full_model_training', False)
    )
    if range_lm_full_training and args.pretrained_path:
        raise ValueError(
            'full_model_training requires random initialization; do not pass --pretrained_path'
        )
    if range_lm_enabled:
        logger.info(
            'RangeLM enabled: correction=%s | correction_scale=%s | xyz_weight=%s | delta_weight=%s | '
            'LM=%s no-grad + %s differentiable | train_xyz_points=%s/%s | '
            'val_lm_iterations=%s | head_only_epochs=%s | uncertainty=%s | '
            'uncertainty_only=%s | uncertainty_reg=%s | min_weight=%s | '
            'full_model_training=%s | xyz_loss_warmup_epochs=%s' % (
                getattr(range_lm_cfg, 'correction_enabled', True),
                getattr(range_lm_cfg, 'correction_scale', 0.05),
                getattr(range_lm_cfg, 'xyz_loss_weight', 1.0),
                getattr(range_lm_cfg, 'delta_loss_weight', 0.01),
                getattr(range_lm_cfg, 'warmup_iterations', 20),
                getattr(range_lm_cfg, 'gradient_iterations', 3),
                getattr(range_lm_cfg, 'train_points', model_config.model.num_points),
                model_config.model.num_points,
                getattr(range_lm_cfg, 'val_iterations', 40),
                getattr(range_lm_cfg, 'head_only_epochs', 5),
                getattr(range_lm_cfg, 'uncertainty_enabled', False),
                getattr(range_lm_cfg, 'uncertainty_only', False),
                getattr(range_lm_cfg, 'uncertainty_reg_weight', 1e-4),
                getattr(range_lm_cfg, 'min_weight', 0.05),
                getattr(range_lm_cfg, 'full_model_training', False),
                getattr(range_lm_cfg, 'xyz_loss_warmup_epochs', 0),
            )
        )

    models_save_dir = os.path.join(exp_base_path, "models")
    os.makedirs(models_save_dir, exist_ok=True)
    train_last_path = os.path.join(models_save_dir, 'train_last.pth')
    if args.pretrained_path:
        if os.path.exists(train_last_path):
            logger.info(
                'Existing train_last.pth found; skip pretrained warm-start and resume the current run.'
            )
        else:
            if not os.path.isfile(args.pretrained_path):
                raise FileNotFoundError('Pretrained checkpoint not found: %s' % args.pretrained_path)
            pretrained = torch.load(args.pretrained_path, map_location='cpu')
            pretrained_state = pretrained.get(
                'model_state_best_val',
                pretrained.get('model_state', pretrained),
            )
            matched_count, skipped = load_weights_safely(
                model,
                pretrained_state,
                logger=logger,
                strict_shapes=True,
            )
            logger.info(
                'Warm-started from %s | matched=%d | skipped=%d | unmatched new RangeLM heads keep backward-compatible initialization' % (
                    args.pretrained_path,
                    matched_count,
                    len(skipped),
                )
            )

    optim = build_optimizer(model, model_config)
    scheduler = build_scheduler(model, optim, model_config)
    logger.info(f'Optimizer: {optim.__class__.__name__}')
    logger.info(f'Scheduler: {scheduler.__class__.__name__ if not isinstance(scheduler, list) else [s.__class__.__name__ for s in scheduler]}')
    logger.info(
        'Early stopping: %s' % (
            'disabled' if early_stopping_patience < 0 else 'patience=%d' % early_stopping_patience
        )
    )
    logger.info(
        'Batch progress logging: %s' % (
            'enabled' if use_batch_progress else 'disabled (epoch summaries only)'
        )
    )

    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1')

    wandb_run = init_wandb(args, exp_base_path, model_config, config, logger)
    if wandb_run is not None:
        wandb_subset_config = {
            key: value for key, value in subset_manifest.items() if key != 'samples'
        }
        wandb_run.config.update({'train_subset': wandb_subset_config}, allow_val_change=True)
        wandb_run.summary['train_samples'] = len(tr_dataset)
        wandb_run.summary['full_val_samples'] = len(val_dataset)
        wandb_run.summary['train_subset_manifest'] = manifest_path

    profile_device = next(model.parameters()).device
    profile_baseline_allocated = 0
    profile_baseline_reserved = 0
    if args.profile_run_resources:
        torch.cuda.synchronize(profile_device)
        torch.cuda.reset_peak_memory_stats(profile_device)
        profile_baseline_allocated = torch.cuda.memory_allocated(profile_device)
        profile_baseline_reserved = torch.cuda.memory_reserved(profile_device)
    training_loop_start = time.perf_counter()

    best_loss, training_losses, validation_losses = train_model(model, tr_loader,
                                                            optim, scheduler, loss_obj, num_epochs,
                                                            val_dataloader=val_loader, use_batch_progress=use_batch_progress,
                                                            model_save_dir=models_save_dir, logger=logger, wandb_run=wandb_run,
                                                            early_stopping_patience=early_stopping_patience)

    if args.profile_run_resources:
        torch.cuda.synchronize(profile_device)
    training_loop_wall_seconds = time.perf_counter() - training_loop_start
    profile_peak_allocated = (
        torch.cuda.max_memory_allocated(profile_device)
        if args.profile_run_resources else 0
    )
    profile_peak_reserved = (
        torch.cuda.max_memory_reserved(profile_device)
        if args.profile_run_resources else 0
    )

    logger.info(f'Best loss: {best_loss}')

    save_loss_graphs(training_losses, exp_base_path, "train")
    save_loss_graphs(validation_losses, exp_base_path, "val")

    val_score = test_single_epoch(model, val_loader, loss_obj)
    logger.info(f'Final val loss: {val_score}')

    if args.profile_run_resources:
        completed_epochs = int(training_losses.shape[0])
        profile = {
            'scope': 'train_model including per-epoch validation; excludes setup and final post-training validation',
            'experiment_name': experiment_name,
            'seed': args.seed,
            'subset_seed': args.subset_seed,
            'completed_epochs_in_this_invocation': completed_epochs,
            'configured_max_epoch': num_epochs,
            'warmup_iterations': int(getattr(range_lm_cfg, 'warmup_iterations', 0)),
            'gradient_iterations': int(getattr(range_lm_cfg, 'gradient_iterations', 0)),
            'train_points': int(getattr(range_lm_cfg, 'train_points', 0)),
            'val_iterations': int(getattr(range_lm_cfg, 'val_iterations', 0)),
            'training_loop_wall_seconds': training_loop_wall_seconds,
            'mean_wall_seconds_per_completed_epoch': (
                training_loop_wall_seconds / completed_epochs
                if completed_epochs else None
            ),
            'program_wall_seconds_through_final_validation': (
                time.perf_counter() - program_wall_start
            ),
            'device': torch.cuda.get_device_name(profile_device),
            'baseline_gpu_allocated_mb': profile_baseline_allocated / (1024.0 ** 2),
            'baseline_gpu_reserved_mb': profile_baseline_reserved / (1024.0 ** 2),
            'peak_gpu_allocated_mb': profile_peak_allocated / (1024.0 ** 2),
            'peak_gpu_reserved_mb': profile_peak_reserved / (1024.0 ** 2),
            'peak_training_incremental_allocated_mb': max(
                0, profile_peak_allocated - profile_baseline_allocated
            ) / (1024.0 ** 2),
        }
        profile_path = os.path.join(exp_base_path, 'resource_profile.json')
        with open(profile_path, 'w', encoding='utf-8') as profile_file:
            json.dump(profile, profile_file, ensure_ascii=False, indent=2)
        logger.info(
            'Run resources: training_loop_wall_seconds=%.3f | '
            'mean_seconds_per_epoch=%s | peak_gpu_allocated_mb=%.3f | '
            'peak_gpu_reserved_mb=%.3f | peak_training_incremental_allocated_mb=%.3f' % (
                training_loop_wall_seconds,
                (
                    '%.3f' % profile['mean_wall_seconds_per_completed_epoch']
                    if profile['mean_wall_seconds_per_completed_epoch'] is not None
                    else 'NA'
                ),
                profile['peak_gpu_allocated_mb'],
                profile['peak_gpu_reserved_mb'],
                profile['peak_training_incremental_allocated_mb'],
            )
        )
        logger.info(f'Resource profile: {profile_path}')

    if wandb_run is not None:
        if len(val_score) >= 2:
            wandb_run.summary['final_val_coarse'] = float(val_score[0])
            wandb_run.summary['final_val_dense'] = float(val_score[1])
        if len(val_score) >= 3:
            wandb_run.summary['final_val_xyz'] = float(val_score[2])
        wandb_run.summary['best_loss'] = best_loss
        if args.profile_run_resources:
            for key, value in profile.items():
                if value is not None:
                    wandb_run.summary[f'resources/{key}'] = value
        wandb_run.finish()
