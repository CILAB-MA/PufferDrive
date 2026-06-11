import numpy as np


class AgentSampler:
    def __init__(
        self,
        num_policies,
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
        
    def _distance_filtering(self, score, minimum_distance, minimum_other_idx, policy_per_slot, agent_idx, policy_idx):
        """Filter by distance, scatter per-slot scores into policy x corpus columns."""
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        dist = np.asarray(minimum_distance, dtype=np.float64).reshape(-1)
        corpus_idx = np.asarray(minimum_other_idx, dtype=np.int64).reshape(-1)
        policy_per_slot = np.asarray(policy_per_slot, dtype=np.int64).reshape(-1)
        if not (score.shape == dist.shape == corpus_idx.shape == policy_per_slot.shape):
            raise ValueError(
                "score, minimum_distance, minimum_other_idx, policy_per_slot must match, "
                f"got {score.shape}, {dist.shape}, {corpus_idx.shape}, {policy_per_slot.shape}"
            )

        tracked = (corpus_idx >= 0) & np.isfinite(dist)
        keep = tracked & (dist >= self.distance_threshold)

        new_score = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)
        new_score.fill(0.0)
        for slot in np.flatnonzero(keep):
            policy_idx = int(policy_per_slot[slot])
            if policy_idx < 0:
                continue
            g = int(corpus_idx[slot])
            if 0 <= g < self.total_agents:
                new_score[g, policy_idx] = score[slot]

        return new_score

    def update_policy_score(self, score, agent_idx, policy_idx, minimum_distance, minimum_other_idx, policy_per_slot):

        new_score = self._distance_filtering(score, minimum_distance, minimum_other_idx, policy_per_slot, agent_idx, policy_idx) # (total_agents, num_policies)
        new_score_col = new_score[:, policy_idx]

        self.unseen_policy_weights[agent_idx, policy_idx] = 0.0  #  no longer unseen -> 0 to unseen_policy_weights

        old_score = self.policy_scores[:, policy_idx]

        self.policy_scores[:, policy_idx] = (1 - self.alpha) * old_score + self.alpha * new_score_col

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.policy_staleness = self.policy_staleness + 1 # update_staleness to all idx
            self.policy_staleness[np.arange(self.total_agents), selected_idx] = 0

    def _sample_replay_policy(self, agent_idx):
        weights = self.sample_weights(agent_idx)

        if np.isclose(np.sum(weights), 0):
            weights = np.ones(self.num_policies, dtype=np.float64) / self.num_policies

        policy_idx = np.random.choice(self.num_policies, p=weights)

        return int(policy_idx)

    def _sample_unseen_policy(self, agent_idx):
        weights = self.unseen_policy_weights[agent_idx].astype(np.float64)

        if weights.sum() == 0:
            policy_idx = np.random.randint(self.num_policies)
        else:
            probs = weights / weights.sum()
            policy_idx = np.random.choice(self.num_policies, p=probs)

        return int(policy_idx)

    def sample(self):
        policy_idx = np.empty(self.total_agents, dtype=np.int64)

        for agent_idx in range(self.total_agents):
            policy_unseen = self.unseen_policy_weights[agent_idx] > 0
            num_unseen = policy_unseen.sum()

            proportion_seen = (self.num_policies - num_unseen) / self.num_policies

            if proportion_seen >= self.rho and np.random.rand() < proportion_seen: # proportional prioritization sampling (seen more, replay more)
                policy_idx[agent_idx] = self._sample_replay_policy(agent_idx)
            else:
                policy_idx[agent_idx] = self._sample_unseen_policy(agent_idx)
        
        self._update_staleness(policy_idx)

        return policy_idx

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