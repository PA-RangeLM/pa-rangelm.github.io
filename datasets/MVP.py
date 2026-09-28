import os

import h5py
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.curvature import compute_curvature_normal
from datasets.data_transforms import deterministic_fps
from datasets.utils import to_o3d_pcd


MVP_CATEGORY_NAMES = (
    "airplane",
    "cabinet",
    "car",
    "chair",
    "lamp",
    "sofa",
    "table",
    "watercraft",
    "bed",
    "bench",
    "bookshelf",
    "bus",
    "guitar",
    "motorbike",
    "pistol",
    "skateboard",
)


class MVPCompletionDataset(Dataset):
    """MVP single-view completion test set for PCN-to-MVP evaluation.

    MVP stores one complete cloud for every 26 consecutive partial views.  The
    dataset keeps the original HDF5 row index so that filtering categories or
    views never breaks the ``complete_index = partial_index // 26`` mapping.
    """

    def __init__(
        self,
        h5_path,
        num_keypoints=8,
        keypoint_type="curvature_radius",
        curvature_k=16,
        curvature_radius=0.075,
        overlap_only=True,
        category_scope=None,
        views=None,
        max_samples=0,
        sampling_seed=2026,
        keypoint_cache_dir="",
    ):
        self.h5_path = os.path.abspath(h5_path)
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError("MVP HDF5 file not found: {}".format(self.h5_path))
        if keypoint_type not in ("basis", "curvature_radius"):
            raise ValueError(
                "MVP evaluation supports keypoint_type 'basis' or "
                "'curvature_radius', got {!r}".format(keypoint_type)
            )

        self.num_keypoints = int(num_keypoints)
        self.keypoint_type = keypoint_type
        self.curvature_k = int(curvature_k)
        self.curvature_radius = float(curvature_radius)
        self.keypoint_cache_dir = keypoint_cache_dir
        self._h5 = None

        with h5py.File(self.h5_path, "r") as h5_file:
            required = {"incomplete_pcds", "complete_pcds", "labels"}
            missing = required.difference(h5_file.keys())
            if missing:
                raise KeyError(
                    "MVP file is missing required fields: {}".format(sorted(missing))
                )
            self.labels = np.asarray(h5_file["labels"][:], dtype=np.int64)
            self.partial_shape = tuple(h5_file["incomplete_pcds"].shape)
            self.complete_shape = tuple(h5_file["complete_pcds"].shape)

        if category_scope is None:
            category_scope = "overlap8" if overlap_only else "all16"
        if category_scope not in ("overlap8", "unseen8", "all16"):
            raise ValueError(
                "category_scope must be 'overlap8', 'unseen8', or 'all16', "
                "got {!r}".format(category_scope)
            )
        self.category_scope = category_scope

        indices = np.arange(len(self.labels), dtype=np.int64)
        if category_scope == "overlap8":
            indices = indices[self.labels[indices] < 8]
        elif category_scope == "unseen8":
            indices = indices[self.labels[indices] >= 8]

        if views is not None:
            selected_views = np.asarray(sorted(set(int(item) for item in views)))
            if selected_views.size == 0 or np.any(selected_views < 0) or np.any(selected_views > 25):
                raise ValueError("views must contain integers in [0, 25]")
            indices = indices[np.isin(indices % 26, selected_views)]

        if max_samples and max_samples < len(indices):
            # A deterministic, category-balanced subset is useful only for a
            # smoke test.  Full paper evaluation must leave max_samples at 0.
            rng = np.random.default_rng(sampling_seed)
            selected = []
            active_labels = sorted(np.unique(self.labels[indices]).tolist())
            base_count, remainder = divmod(int(max_samples), len(active_labels))
            for position, label in enumerate(active_labels):
                category_indices = indices[self.labels[indices] == label]
                category_count = base_count + (1 if position < remainder else 0)
                category_count = min(category_count, len(category_indices))
                if category_count:
                    chosen = rng.choice(category_indices, category_count, replace=False)
                    selected.extend(chosen.tolist())
            indices = np.asarray(sorted(selected), dtype=np.int64)

        self.indices = indices
        if self.keypoint_cache_dir:
            os.makedirs(self.keypoint_cache_dir, exist_ok=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _cache_path(self, original_index):
        if not self.keypoint_cache_dir:
            return ""
        filename = "mvp_{:05d}_{}_k{}_r{:.6f}_n{}.npy".format(
            original_index,
            self.keypoint_type,
            self.curvature_k,
            self.curvature_radius,
            self.num_keypoints,
        )
        return os.path.join(self.keypoint_cache_dir, filename)

    def _curvature_radius_keypoints(self, partial):
        point_cloud = to_o3d_pcd(partial)
        curvatures, _ = compute_curvature_normal(point_cloud, self.curvature_k)
        basis_points = deterministic_fps(partial, self.num_keypoints)
        squared_distances = np.sum(
            (basis_points[:, None, :] - partial[None, :, :]) ** 2,
            axis=-1,
        )
        radius_mask = squared_distances > self.curvature_radius ** 2
        selected = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        for anchor_index in range(self.num_keypoints):
            local_curvatures = curvatures.copy()
            local_curvatures[radius_mask[anchor_index]] = -1.0
            selected[anchor_index] = partial[int(np.argmax(local_curvatures))]
        return selected

    def _keypoints(self, partial, original_index):
        cache_path = self._cache_path(original_index)
        if cache_path and os.path.isfile(cache_path):
            try:
                cached = np.load(cache_path)
                if cached.shape == (self.num_keypoints, 3):
                    return cached.astype(np.float32, copy=False)
            except (OSError, ValueError, EOFError):
                # A worker killed while writing a cache entry may leave a
                # truncated file. Recompute it safely instead of failing the
                # whole evaluation.
                pass

        if self.keypoint_type == "basis":
            keypoints = deterministic_fps(partial, self.num_keypoints).astype(np.float32)
        else:
            keypoints = self._curvature_radius_keypoints(partial)

        if cache_path:
            temporary_path = cache_path + ".tmp.npy"
            np.save(temporary_path, keypoints)
            os.replace(temporary_path, cache_path)
        return keypoints

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        h5_file = self._file()
        partial = np.asarray(
            h5_file["incomplete_pcds"][original_index], dtype=np.float32
        )
        complete_index = original_index // 26
        gt = np.asarray(
            h5_file["complete_pcds"][complete_index], dtype=np.float32
        )
        label = int(self.labels[original_index])
        category_name = MVP_CATEGORY_NAMES[label]
        model_id = "mvp_{:04d}_view_{:02d}".format(
            complete_index, original_index % 26
        )
        basis_points = self._keypoints(partial, original_index)

        data = {
            "partial": torch.from_numpy(partial.copy()),
            "gt": torch.from_numpy(gt.copy()),
            "basis_points": torch.from_numpy(basis_points.copy()),
            "label": torch.tensor(label, dtype=torch.long),
            "original_index": torch.tensor(original_index, dtype=torch.long),
        }
        return category_name, model_id, data

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()


def get_mvp_test_loader(
    h5_path,
    batch_size,
    num_workers=0,
    num_keypoints=8,
    keypoint_type="curvature_radius",
    curvature_k=16,
    curvature_radius=0.075,
    overlap_only=True,
    category_scope=None,
    views=None,
    max_samples=0,
    sampling_seed=2026,
    keypoint_cache_dir="",
):
    dataset = MVPCompletionDataset(
        h5_path=h5_path,
        num_keypoints=num_keypoints,
        keypoint_type=keypoint_type,
        curvature_k=curvature_k,
        curvature_radius=curvature_radius,
        overlap_only=overlap_only,
        category_scope=category_scope,
        views=views,
        max_samples=max_samples,
        sampling_seed=sampling_seed,
        keypoint_cache_dir=keypoint_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader
