import numpy as np


class AgentSampler:
    """Sample population members and track scores in (map, population) matrices."""

    def __init__(
        self,
        num_population, # replay: num_rollouts; reactive: num_policy_assignments
        strategy="prioritized", # options: ["prioritized", "uniform"]
        pbt_mode="replay", # options: ["replay", "reactive"]
        score_transform="power",
        temperature=1.0,
        eps=0.05,
        rho=0.2,
        alpha=1.0,
        staleness_coef=0.1,
        staleness_transform="power",
        staleness_temperature=1.0,
        num_maps=0, # number of maps
    ):
        self.num_population = int(num_population)
        self.num_maps = int(num_maps)
        self.strategy = strategy
        self.pbt_mode = pbt_mode

        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        self.unseen_map_population_weights = np.ones((self.num_maps, self.num_population), dtype=np.float64)
        self.map_population_scores = np.zeros((self.num_maps, self.num_population), dtype=np.float64)
        self.map_population_staleness = np.zeros((self.num_maps, self.num_population), dtype=np.float64)
        self.distance_threshold = 0.02 # TODO: args로 만들기
        self.new_map_population_scores = np.zeros((self.num_maps, self.num_population), dtype=np.float64)
        self.encountered_maps = np.zeros(self.num_maps, dtype=bool)

    def _distance_filtering(self, score, minimum_distance, population_idx, controlled_entity_idx, map_idx=None):
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        dist = np.asarray(minimum_distance, dtype=np.float64).reshape(-1)
        controlled_entity_idx = np.asarray(controlled_entity_idx, dtype=np.int64).reshape(-1)
        population_idx = np.asarray(population_idx, dtype=np.int64).reshape(-1)

        map_idx = np.asarray(map_idx, dtype=np.int64).reshape(-1)
        population_per_slot = population_idx

        tracked = (controlled_entity_idx >= 0) & np.isfinite(dist)
        keep = tracked & (dist >= self.distance_threshold)

        self.new_map_population_scores.fill(np.nan)
        map_ids, map_scores, map_population = self._aggregate_mean_by_map(
            score, keep, map_idx, population_per_slot
        )
        for i in range(map_ids.size):
            map_id = int(map_ids[i])
            population_id = int(map_population[i])
            if not (0 <= map_id < self.num_maps):
                continue
            if not (0 <= population_id < self.num_population):
                continue
            self.new_map_population_scores[map_id, population_id] = map_scores[i]

    def _aggregate_mean_by_map(self, score, keep, map_idx, population_idx):
        """Aggregate slot scores into parallel map, score, and population arrays."""
        map_idx = np.asarray(map_idx, dtype=np.int64).reshape(-1)
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        population_idx = np.asarray(population_idx, dtype=np.int64).reshape(-1)
        if not (score.shape == map_idx.shape == keep.shape == population_idx.shape):
            raise ValueError(
                "score, keep, map_idx, population_idx must match, "
                f"got {score.shape}, {keep.shape}, {map_idx.shape}, {population_idx.shape}"
            )

        valid = keep & (map_idx >= 0) & (map_idx < self.num_maps)
        if not np.any(valid):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )

        valid_maps = map_idx[valid]
        valid_scores = score[valid]
        valid_population = population_idx[valid]
        map_ids, inv = np.unique(valid_maps, return_inverse=True)
        _, first = np.unique(inv, return_index=True)
        map_scores = np.bincount(inv, weights=valid_scores) / np.bincount(inv)
        map_population = valid_population[first]
        return map_ids, map_scores, map_population

    def _normalize_scores(self):
        """Min-max normalize new map-population scores to [0, 1]."""
        active = np.isfinite(self.new_map_population_scores)
        if not np.any(active):
            return
        vals = self.new_map_population_scores[active]
        lo, hi = vals.min(), vals.max()
        with np.errstate(invalid="ignore"): # if hi == lo, 0/0 = nan skip
            self.new_map_population_scores[active] = (vals - lo) / (hi - lo)

    def update_policy_score(self, score, controlled_entity_idx, population_idx, minimum_distance,
                            map_idx=None):
        if self.strategy == "uniform":
            return {}

        self._distance_filtering(score, minimum_distance, population_idx, controlled_entity_idx, map_idx=map_idx)
        raw_return_metrics = self._raw_return_summary_metrics()
        self._normalize_scores()

        # Only update (g, p) pairs that received a new score — avoids zeroing scores for non-passing slots
        active = np.isfinite(self.new_map_population_scores)
        self.map_population_scores[active] = (
            (1 - self.alpha) * self.map_population_scores[active]
            + self.alpha * self.new_map_population_scores[active]
        )

        # Mark valid (map, population) pairs as seen regardless of distance.
        map_ids = np.asarray(map_idx, dtype=np.int64).reshape(-1)
        population_ids = np.asarray(population_idx, dtype=np.int64).reshape(-1)
        valid = (
            (map_ids >= 0)
            & (map_ids < self.num_maps)
            & (population_ids >= 0)
            & (population_ids < self.num_population)
        )
        self.unseen_map_population_weights[map_ids[valid], population_ids[valid]] = 0.0
        self.encountered_maps[map_ids[valid]] = True

        return raw_return_metrics

    def _raw_return_summary_metrics(self):
        """Raw return metrics for maps scored in this update.

        Must be read before _normalize_scores() overwrites the new scores in place.
        """
        active = np.isfinite(self.new_map_population_scores)
        touched_maps = np.flatnonzero(active.any(axis=1))
        if touched_maps.size == 0:
            return {}
        metrics = {}
        for map_idx in touched_maps:
            vals = self.new_map_population_scores[map_idx][active[map_idx]]
            label = f"map_{int(map_idx)}"
            metrics[f"{label}_raw_return_mean"] = float(vals.mean())
            metrics[f"{label}_raw_return_min"] = float(vals.min())
            metrics[f"{label}_raw_return_max"] = float(vals.max())
        return metrics

    def _update_staleness(self, population_per_map):
        """Update staleness using one population index per map; inactive maps are -1."""
        if self.staleness_coef > 0:
            self.map_population_staleness[self.encountered_maps] += 1
            active_maps = np.flatnonzero(population_per_map >= 0)
            if active_maps.size > 0:
                self.map_population_staleness[
                    active_maps,
                    population_per_map[active_maps],
                ] = 0

    def _sample_replay_policy(self, map_idx):
        weights = self.sample_weights(map_idx)

        if np.isclose(np.sum(weights), 0): # 모든 확률이 0인 경우
            weights = np.ones(self.num_population, dtype=np.float64) / self.num_population

        weights = weights / weights.sum()  # float 오차로 합이 1이 아닐 경우 재정규화
        population_idx = np.random.choice(self.num_population, p=weights)

        return int(population_idx)

    def _sample_unseen_policy(self, map_idx):
        weights = self.unseen_map_population_weights[map_idx].astype(np.float64)
        s = weights.sum()

        if s == 0: # all seen,
            population_idx = np.random.randint(self.num_population)
        else:
            population_idx = np.random.choice(self.num_population, p=weights / s) 

        return int(population_idx)

    def sample(self, map_indices):
        map_indices = np.asarray(map_indices, dtype=np.int64).reshape(-1)
        num_sampled_maps = map_indices.size
        sampled_population = np.full(num_sampled_maps, -1, dtype=np.int64)

        if self.strategy == "uniform":
            sampled_population = np.random.randint(0, self.num_population, num_sampled_maps)
            return sampled_population, {}

        # "prioritized": select one population member for each unique map.
        if self.encountered_maps.any():
            global_proportion_seen = (
                self.unseen_map_population_weights[self.encountered_maps] == 0
            ).mean()
        else:
            global_proportion_seen = 0.0

        population_per_map = np.full(self.num_maps, -1, dtype=np.int64)

        for map_idx in np.unique(map_indices):
            if global_proportion_seen >= self.rho and np.random.rand() < global_proportion_seen:
                population_per_map[map_idx] = self._sample_replay_policy(int(map_idx))
            else:
                population_per_map[map_idx] = self._sample_unseen_policy(int(map_idx))
        sampled_population[:] = population_per_map[map_indices]

        wandb_metrics = {
            "global_proportion_seen": global_proportion_seen,
        }
        wandb_metrics.update(self._sampling_weight_summary_metrics(population_per_map))

        self._update_staleness(population_per_map)
        return sampled_population, wandb_metrics

    def _sampling_weight_summary_metrics(self, population_per_map):
        """Sampling-weight concentration for maps sampled in this update."""
        touched_maps = np.flatnonzero(population_per_map >= 0)
        if touched_maps.size == 0:
            return {}
        metrics = {}
        for map_idx in touched_maps:
            weights = self.sample_weights(int(map_idx))
            label = f"map_{int(map_idx)}"
            metrics[f"{label}_weight_mean"] = float(weights.mean())
            metrics[f"{label}_weight_max"] = float(weights.max())
        return metrics

    def sample_weights(self, map_idx):
        scores = self.map_population_scores[map_idx]  # (num_population,)
        unseen = self.unseen_map_population_weights[map_idx]  # (num_population,)

        weights = self._score_transform(self.score_transform, self.temperature, scores, unseen)
        weights = weights * (1 - unseen)

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness = self.map_population_staleness[map_idx]
            staleness_weights = self._score_transform(
                self.staleness_transform,
                self.staleness_temperature,
                staleness,
                unseen,
            )
            staleness_weights = staleness_weights * (1 - unseen)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores, unseen=None):
        scores = np.asarray(scores, dtype=np.float64)
        if transform == "constant":
            weights = np.ones_like(scores)
        elif transform == "max":
            weights = np.zeros_like(scores)
            masked_scores = scores.copy()
            if unseen is not None:
                masked_scores[unseen > 0] = -float("inf")
            max_val = masked_scores.max()
            candidates = np.flatnonzero(np.isclose(masked_scores, max_val)) if np.isfinite(max_val) else np.arange(len(masked_scores))
            weights[np.random.choice(candidates)] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / self.num_population
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1.0 / temperature)
        elif transform == "rank_low":
            temp = scores.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1.0 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (scores + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(scores / temperature)
        else:
            raise ValueError(f"Unsupported score transform, {transform}")

        return weights
