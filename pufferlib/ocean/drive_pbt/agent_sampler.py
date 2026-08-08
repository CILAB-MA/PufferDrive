import numpy as np


class AgentSampler:
    def __init__(
        self,
        num_population, # when pbt_mode is replay, this is num_rollout. when pbt_mode is reactive, this is num_policies.
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
        num_assignments=0, # when pbt_mode is replay, this is num_maps. when pbt_mode is reactive, this is total_agents.
    ):
        self.num_population = int(num_population)
        self.num_assignments = int(num_assignments)
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

        self.unseen_population_weights = np.ones((self.num_assignments, self.num_population), dtype=np.float64)
        self.population_scores = np.zeros((self.num_assignments, self.num_population), dtype=np.float64)
        self.population_staleness = np.zeros((self.num_assignments, self.num_population), dtype=np.float64) # Todo : population_staleness 의미가 있을까?
        self.distance_threshold = 0.02 # TODO: args로 만들기
        self.new_score = np.zeros((self.num_assignments, self.num_population), dtype=np.float64)
        self.encountered = np.zeros(self.num_assignments, dtype=bool)

    def _distance_filtering(self, score, minimum_distance, assignment_idx, agent_idx, map_idx=None):
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        dist = np.asarray(minimum_distance, dtype=np.float64).reshape(-1)
        corpus_idx = np.asarray(agent_idx, dtype=np.int64).reshape(-1)
        assignment_idx = np.asarray(assignment_idx, dtype=np.int64).reshape(-1)

        if self.pbt_mode == "reactive":
            policy_per_slot = assignment_idx
            if not (score.shape == dist.shape == corpus_idx.shape == policy_per_slot.shape):
                raise ValueError(
                    "score, minimum_distance, minimum_other_global_idx, policy_per_slot must match, "
                    f"got {score.shape}, {dist.shape}, {corpus_idx.shape}, {policy_per_slot.shape}"
                )
        elif self.pbt_mode == "replay":
            map_idx = np.asarray(map_idx, dtype=np.int64).reshape(-1)
            rollout_idx = assignment_idx
            if not (score.shape == dist.shape == corpus_idx.shape == map_idx.shape == rollout_idx.shape):
                raise ValueError(
                    "score, minimum_distance, agent_idx, map_idx, rollout_idx must match, "
                    f"got {score.shape}, {dist.shape}, {corpus_idx.shape}, {map_idx.shape}, {rollout_idx.shape}"
                )

        tracked = (corpus_idx >= 0) & np.isfinite(dist)
        keep = tracked & (dist >= self.distance_threshold)

        self.new_score.fill(np.nan)
        if self.pbt_mode == "reactive":
            for slot in np.flatnonzero(keep):
                p = int(policy_per_slot[slot])
                if not (0 <= p < self.num_population):
                    continue
                g = int(corpus_idx[slot])
                if 0 <= g < self.num_assignments:
                    self.new_score[g, p] = score[slot]
        elif self.pbt_mode == "replay":
            map_ids, map_scores, map_rollouts = self._aggregate_mean_by_map(
                score, keep, map_idx, rollout_idx
            )
            for i in range(map_ids.size):
                m = int(map_ids[i])
                r = int(map_rollouts[i])
                if not (0 <= m < self.num_assignments):
                    continue
                if not (0 <= r < self.num_population):
                    continue
                self.new_score[m, r] = map_scores[i]

    def _aggregate_mean_by_map(self, score, keep, map_idx, rollout_idx):
        """Aggregate kept slot scores to unique maps; returns parallel (map_ids, scores, rollouts)."""
        map_idx = np.asarray(map_idx, dtype=np.int64).reshape(-1)
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        rollout_idx = np.asarray(rollout_idx, dtype=np.int64).reshape(-1)
        if not (score.shape == map_idx.shape == keep.shape == rollout_idx.shape):
            raise ValueError(
                "score, keep, map_idx, rollout_idx must match, "
                f"got {score.shape}, {keep.shape}, {map_idx.shape}, {rollout_idx.shape}"
            )

        valid = keep & (map_idx >= 0) & (map_idx < self.num_assignments)
        if not np.any(valid):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )

        vm = map_idx[valid]
        vs = score[valid]
        vr = rollout_idx[valid]
        map_ids, inv = np.unique(vm, return_inverse=True)
        _, first = np.unique(inv, return_index=True)
        map_scores = np.bincount(inv, weights=vs) / np.bincount(inv)
        map_rollouts = vr[first]
        return map_ids, map_scores, map_rollouts

    def _normalize_scores(self):
        """Min-max normalize new_score to [0, 1]. hi==lo → 0/0=nan → no EMA update."""
        active = np.isfinite(self.new_score)
        if not np.any(active):
            return
        vals = self.new_score[active]
        lo, hi = vals.min(), vals.max()
        with np.errstate(invalid="ignore"): # if hi == lo, 0/0 = nan skip
            self.new_score[active] = (vals - lo) / (hi - lo)

    def update_policy_score(self, score, agent_idx, policy_idx, minimum_distance,
                            map_idx=None):
        if self.strategy == "uniform":
            return {}

        if self.pbt_mode == "replay" and rollout_idx is None:
            rollout_idx = policy_idx
        self._distance_filtering(score, minimum_distance, policy_idx, agent_idx, map_idx=map_idx)
        raw_return_metrics = self._raw_return_summary_metrics()
        self._normalize_scores()

        # Only update (g, p) pairs that received a new score — avoids zeroing scores for non-passing slots
        active = np.isfinite(self.new_score)
        self.population_scores[active] = (1 - self.alpha) * self.population_scores[active] + self.alpha * self.new_score[active]

        # Mark as seen for all valid corpus entities that played this episode (distance와 무관)
        if self.pbt_mode == "reactive":
            a = np.asarray(agent_idx, dtype=np.int64).reshape(-1)
            p = np.asarray(policy_idx, dtype=np.int64).reshape(-1)
            valid = (a >= 0) & (a < self.num_assignments) & (p >= 0) & (p < self.num_population)
            self.unseen_population_weights[a[valid], p[valid]] = 0.0
            self.encountered[a[valid]] = True
        elif self.pbt_mode == "replay":
            m = np.asarray(map_idx, dtype=np.int64).reshape(-1)
            r = np.asarray(rollout_idx, dtype=np.int64).reshape(-1)
            valid = (m >= 0) & (m < self.num_assignments) & (r >= 0) & (r < self.num_population)
            self.unseen_population_weights[m[valid], r[valid]] = 0.0
            self.encountered[m[valid]] = True

        return raw_return_metrics

    def _raw_return_summary_metrics(self):
        """Per-corpus raw (pre-normalization) return, aggregated across policies, for corpus entities scored this step.

        Must be read from self.new_score before _normalize_scores() overwrites it in place.
        """
        active = np.isfinite(self.new_score)
        touched = np.flatnonzero(active.any(axis=1))
        if touched.size == 0:
            return {}
        default_prefix = "map" if self.pbt_mode == "replay" else "agent"
        metrics = {}
        for g in touched:
            vals = self.new_score[g][active[g]]
            label = f"{default_prefix}_{int(g)}"
            metrics[f"{label}_raw_return_mean"] = float(vals.mean())
            metrics[f"{label}_raw_return_min"] = float(vals.min())
            metrics[f"{label}_raw_return_max"] = float(vals.max())
        return metrics

    def _update_staleness(self, assignment_per_corpus):
        """assignment_per_corpus: (num_assignments,) with -1 for corpus entities not active this episode."""
        if self.staleness_coef > 0:
            self.population_staleness[self.encountered] += 1
            assigned = np.flatnonzero(assignment_per_corpus >= 0)
            if assigned.size > 0:
                self.population_staleness[assigned, assignment_per_corpus[assigned]] = 0

    def _sample_replay_policy(self, agent_idx):
        weights = self.sample_weights(agent_idx)

        if np.isclose(np.sum(weights), 0):
            weights = np.ones(self.num_population, dtype=np.float64) / self.num_population

        weights = weights / weights.sum()  # float 오차로 합이 1이 아닐 경우 재정규화
        policy_idx = np.random.choice(self.num_population, p=weights)

        return int(policy_idx)

    def _sample_unseen_policy(self, agent_idx):
        weights = self.unseen_population_weights[agent_idx].astype(np.float64)
        s = weights.sum()

        if s == 0:
            policy_idx = np.random.randint(self.num_population)
        else:
            policy_idx = np.random.choice(self.num_population, p=weights / s)

        return int(policy_idx)

    def sample(self, corpus_idx_per_slot):
        corpus_idx_per_slot = np.asarray(corpus_idx_per_slot, dtype=np.int64).reshape(-1)
        n_other = corpus_idx_per_slot.size
        flat = np.full(n_other, -1, dtype=np.int64)

        if self.strategy == "uniform":
            flat = np.tile(np.arange(self.num_population), -(-n_other // self.num_population))[:n_other]
            np.random.shuffle(flat)
            return flat, {}

        # "prioritized": PLR-based sampling per corpus entity
        valid = (corpus_idx_per_slot >= 0) & (corpus_idx_per_slot < self.num_assignments)
        flat[~valid] = np.random.randint(self.num_population, size=int((~valid).sum()))

        if self.encountered.any():
            global_proportion_seen = (self.unseen_population_weights[self.encountered] == 0).mean()
        else:
            global_proportion_seen = 0.0
        assignment_per_corpus = np.full(self.num_assignments, -1, dtype=np.int64)
        for g in np.unique(corpus_idx_per_slot[valid]):
            if global_proportion_seen >= self.rho and np.random.rand() < global_proportion_seen:
                assignment_per_corpus[g] = self._sample_replay_policy(int(g))
            else:
                assignment_per_corpus[g] = self._sample_unseen_policy(int(g))
        flat[valid] = assignment_per_corpus[corpus_idx_per_slot[valid]]
        wandb_metrics = {
            "global_proportion_seen": global_proportion_seen,
        }
        wandb_metrics.update(self._sampling_weight_summary_metrics(assignment_per_corpus))
        self._update_staleness(assignment_per_corpus)
        return flat, wandb_metrics

    def _sampling_weight_summary_metrics(self, assignment_per_corpus):
        """Per-corpus sampling-weight concentration (mean/max across policies) for corpus entities assigned this step."""
        touched = np.flatnonzero(assignment_per_corpus >= 0)
        if touched.size == 0:
            return {}
        default_prefix = "map" if self.pbt_mode == "replay" else "agent"
        metrics = {}
        for g in touched:
            weights = self.sample_weights(int(g))
            label = f"{default_prefix}_{int(g)}"
            metrics[f"{label}_weight_mean"] = float(weights.mean())
            metrics[f"{label}_weight_max"] = float(weights.max())
        return metrics

    def sample_weights(self, agent_idx):
        scores = self.population_scores[agent_idx]  # (num_population,)
        unseen = self.unseen_population_weights[agent_idx]  # (num_population,)

        weights = self._score_transform(self.score_transform, self.temperature, scores, unseen)
        weights = weights * (1 - unseen)

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness = self.population_staleness[agent_idx]
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
