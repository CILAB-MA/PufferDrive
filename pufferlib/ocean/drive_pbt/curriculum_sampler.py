import json
import os

import numpy as np


def load_difficulty_types(curriculum_types, curriculum_types_path, num_combination):
    """Load a length-num_combination difficulty type array (lower = easier)."""
    if curriculum_types is not None and curriculum_types != "" and curriculum_types != []:
        types = np.asarray(curriculum_types, dtype=np.int64).reshape(-1)
    elif curriculum_types_path:
        path = str(curriculum_types_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"curriculum_types_path not found: {path}")
        if path.endswith(".npy"):
            types = np.load(path)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                types = np.asarray(json.load(f), dtype=np.int64)
        else:
            raise ValueError(
                f"Unsupported curriculum_types_path extension (use .npy or .json): {path}"
            )
        types = np.asarray(types, dtype=np.int64).reshape(-1)
    else:
        raise ValueError(
            "strategy=curriculum requires curriculum_types (list) or curriculum_types_path"
        )

    if types.shape[0] != int(num_combination):
        raise ValueError(
            f"difficulty_types length {types.shape[0]} != num_combination {num_combination}"
        )
    return types


class CurriculumSampler:
    """Fixed-type curriculum: unlock easy→hard types on an internal sample schedule."""

    def __init__(
        self,
        num_combination,
        difficulty_types,
        curriculum_steps=10000,
        pbt_mode="replay",
        num_maps=0,
        strategy="curriculum",
    ):
        self.num_combination = int(num_combination)
        self.num_maps = int(num_maps)
        self.pbt_mode = pbt_mode
        self.strategy = strategy
        self.curriculum_steps = max(1, int(curriculum_steps))

        self.difficulty_types = np.asarray(difficulty_types, dtype=np.int64).reshape(-1)
        if self.difficulty_types.shape[0] != self.num_combination:
            raise ValueError(
                f"difficulty_types length {self.difficulty_types.shape[0]} "
                f"!= num_combination {self.num_combination}"
            )

        self.unique_types = np.unique(self.difficulty_types)  # ascending
        self.num_types = int(self.unique_types.size)
        if self.num_types < 1:
            raise ValueError("difficulty_types must contain at least one type")

        self._sample_count = 0
        self._pool_by_unlock = self._build_pools()

    def _build_pools(self):
        """pools[k] = indices with type in unique_types[:k+1] (k=0..num_types-1)."""
        pools = []
        for k in range(self.num_types):
            allowed = set(self.unique_types[: k + 1].tolist())
            idx = np.flatnonzero(np.isin(self.difficulty_types, list(allowed))).astype(np.int64)
            if idx.size == 0:
                raise ValueError(f"No population members for unlocked types up to index {k}")
            pools.append(idx)
        return pools

    def _progress(self):
        return min(1.0, self._sample_count / float(self.curriculum_steps))

    def _unlock_level(self, progress):
        # p=0 → 1 type (easiest); p=1 → all types
        n = max(1, int(np.ceil(progress * self.num_types)))
        return min(n, self.num_types)

    def _current_pool(self):
        progress = self._progress()
        n_unlocked = self._unlock_level(progress)
        pool = self._pool_by_unlock[n_unlocked - 1]
        unlocked_types = self.unique_types[:n_unlocked]
        return pool, progress, n_unlocked, unlocked_types

    def sample(self, map_indices):
        map_indices = np.asarray(map_indices, dtype=np.int64).reshape(-1)
        num_sampled_maps = map_indices.size

        pool, progress, n_unlocked, unlocked_types = self._current_pool()
        sampled_population = np.random.choice(
            pool,
            size=num_sampled_maps,
            replace=True,
        ).astype(np.int64)

        self._sample_count += 1
        wandb_metrics = {
            "curriculum_progress": float(progress),
            "unlocked_type_max": float(unlocked_types[-1]),
            "num_unlocked_types": float(n_unlocked),
            "pool_size": float(pool.size),
            "curriculum_sample_count": float(self._sample_count),
        }
        return sampled_population, wandb_metrics

    def update_policy_score(self, score, controlled_entity_idx, population_idx,
                            minimum_distance, map_idx=None):
        """No-op: difficulty is fixed by the provided type list."""
        return {}
