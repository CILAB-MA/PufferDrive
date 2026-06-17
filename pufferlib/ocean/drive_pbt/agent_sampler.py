import numpy as np


class AgentSampler:
    def __init__(
        self,
        num_policies,
        strategy="prioritized",
        score_transform="power",
        temperature=1.0,
        eps=0.05,
        rho=0.2,
        alpha=1.0,
        staleness_coef=0,
        staleness_transform="power",
        staleness_temperature=1.0,
        total_agents=0,
    ):
        self.num_policies = int(num_policies)
        self.total_agents = int(total_agents)
        self.strategy = strategy

        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        self.unseen_policy_weights = np.ones((self.total_agents, self.num_policies), dtype=np.float64)
        self.policy_scores = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)
        self.policy_staleness = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)
        self.distance_threshold = 0.02 # TODO: args로 만들기
        self.new_score = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)
        
    def _distance_filtering(self, score, minimum_distance, policy_idx, agent_idx):
        """Filter by distance, scatter per-slot scores into policy x corpus columns."""
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        dist = np.asarray(minimum_distance, dtype=np.float64).reshape(-1)
        corpus_idx = np.asarray(agent_idx, dtype=np.int64).reshape(-1)
        policy_per_slot = np.asarray(policy_idx, dtype=np.int64).reshape(-1)
        if not (score.shape == dist.shape == corpus_idx.shape == policy_per_slot.shape):
            raise ValueError(
                "score, minimum_distance, minimum_other_global_idx, policy_per_slot must match, "
                f"got {score.shape}, {dist.shape}, {corpus_idx.shape}, {policy_per_slot.shape}"
            )

        tracked = (corpus_idx >= 0) & np.isfinite(dist)
        keep = tracked & (dist >= self.distance_threshold)

        self.new_score.fill(0.0)
        for slot in np.flatnonzero(keep):
            p = int(policy_per_slot[slot])
            if not (0 <= p < self.num_policies):
                continue
            g = int(corpus_idx[slot])
            if 0 <= g < self.total_agents:
                self.new_score[g, p] = score[slot]

    def update_policy_score(self, score, agent_idx, policy_idx, minimum_distance):
        self._distance_filtering(score, minimum_distance, policy_idx, agent_idx)

        # Only update (g, p) pairs that received a new score — avoids zeroing scores for non-passing slots
        active = self.new_score != 0
        self.policy_scores[active] = (1 - self.alpha) * self.policy_scores[active] + self.alpha * self.new_score[active]

        # Mark as seen for all valid corpus entities that played this episode (distance와 무관)
        a = np.asarray(agent_idx, dtype=np.int64).reshape(-1)
        p = np.asarray(policy_idx, dtype=np.int64).reshape(-1)
        valid = (a >= 0) & (a < self.total_agents) & (p >= 0) & (p < self.num_policies)
        self.unseen_policy_weights[a[valid], p[valid]] = 0.0

    def _update_staleness(self, policy_per_corpus):
        """policy_per_corpus: (total_agents,) with -1 for corpus entities not active this episode."""
        if self.staleness_coef > 0:
            self.policy_staleness += 1
            assigned = np.flatnonzero(policy_per_corpus >= 0)
            if assigned.size > 0:
                self.policy_staleness[assigned, policy_per_corpus[assigned]] = 0

    def _sample_replay_policy(self, agent_idx):
        weights = self.sample_weights(agent_idx)

        if np.isclose(np.sum(weights), 0):
            weights = np.ones(self.num_policies, dtype=np.float64) / self.num_policies

        weights = weights / weights.sum()  # float 오차로 합이 1이 아닐 경우 재정규화
        policy_idx = np.random.choice(self.num_policies, p=weights)

        return int(policy_idx)

    def _sample_unseen_policy(self, agent_idx):
        weights = self.unseen_policy_weights[agent_idx].astype(np.float64)
        s = weights.sum()

        if s == 0:
            policy_idx = np.random.randint(self.num_policies)
        else:
            policy_idx = np.random.choice(self.num_policies, p=weights / s)

        return int(policy_idx)

    def sample(self, corpus_idx_per_slot):
        corpus_idx_per_slot = np.asarray(corpus_idx_per_slot, dtype=np.int64).reshape(-1)
        n_other = corpus_idx_per_slot.size
        flat = np.full(n_other, -1, dtype=np.int64)

        if self.strategy == "uniform":
            for policy_idx, slots in enumerate(
                np.array_split(np.random.permutation(n_other), self.num_policies)
            ):
                flat[slots] = policy_idx
            return flat

        # "prioritized": PLR-based sampling per corpus entity
        policy_per_corpus = np.full(self.total_agents, -1, dtype=np.int64)
        for slot in range(n_other):
            g = int(corpus_idx_per_slot[slot])
            if not (0 <= g < self.total_agents):
                flat[slot] = np.random.randint(self.num_policies)
                continue
            if policy_per_corpus[g] < 0:  # sample once per corpus entity per episode
                policy_unseen = self.unseen_policy_weights[g] > 0
                num_unseen = policy_unseen.sum()
                proportion_seen = (self.num_policies - num_unseen) / self.num_policies
                if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                    policy_per_corpus[g] = self._sample_replay_policy(g)
                else:
                    policy_per_corpus[g] = self._sample_unseen_policy(g)
            flat[slot] = policy_per_corpus[g]

        self._update_staleness(policy_per_corpus)
        return flat

    def sample_weights(self, agent_idx):
        scores = self.policy_scores[agent_idx]  # (num_policies,)
        unseen = self.unseen_policy_weights[agent_idx]  # (num_policies,)

        weights = self._score_transform(self.score_transform, self.temperature, scores, unseen)
        weights = weights * (1 - unseen)

        z = np.sum(weights)
        if z > 0:
            weights /= z

        if self.staleness_coef > 0:
            staleness = self.policy_staleness[agent_idx]
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
            argmax = np.random.choice(np.flatnonzero(np.isclose(masked_scores, masked_scores.max())))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / self.num_policies
        elif transform == "rank":
            temp = np.flip(scores.argsort())
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