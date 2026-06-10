import numpy as np
import torch


class AgentSampler:
    def __init__(
        self,
        num_policies,
        strategy="random",
        replay_schedule="fixed",
        score_transform="power",
        temperature=1.0,
        eps=0.05,
        rho=0.2,
        nu=0.5,
        alpha=1.0,
        staleness_coef=0,
        staleness_transform="power",
        staleness_temperature=1.0,
        total_agents=0,
    ):
        self.num_policies = int(num_policies)
        self.total_agents = int(total_agents)

        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        self.unseen_policy_weights = np.ones(self.total_agents, self.num_policies, dtype=np.float64)
        self.policy_scores = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)
        self.policy_staleness = np.zeros(self.total_agents, self.num_policies, dtype=np.float64)
        self.distance_threshold = 0.02 # TODO: args로 만들기
        self.new_score = np.zeros((self.total_agents, self.num_policies), dtype=np.float64)

    def _distance_filtering(self, score, minimum_distance, minimum_other_idx, policy_per_slot):
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

        self.new_score.fill(0.0)
        for slot in np.flatnonzero(keep):
            policy_idx = int(policy_per_slot[slot])
            if policy_idx < 0:
                continue
            g = int(corpus_idx[slot])
            if 0 <= g < self.total_agents:
                self.new_score[g, policy_idx] = score[slot]
        return self.new_score

    @staticmethod
    def _agent_map_idx(agent_idx, agent_offsets):
        ao = np.asarray(agent_offsets, dtype=np.int64)
        return int(np.searchsorted(ao[1:], agent_idx, side="right"))

        self._update_with_rollouts(rollouts, score_function)

    def update_policy_score(self, score, agent_idx, policy_idx, minimum_distance, minimum_other_idx, policy_per_slot):

        score = __distance_filtering(self, score, minimum_distance, minimum_other_idx, policy_per_slot)

        self.unseen_policy_weights[agent_idx][policy_idx] = 0.0  # score (total_agent x N_policy), no longer unseen

        old_score = self.policy_scores[:, policy_idx]
        self.policy_scores[:, policy_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def _update_staleness(self, agent_idx, selected_idx):
        if self.staleness_coef > 0:
            self.policy_staleness = self.policy_staleness + 1 # update_staleness to all idx
            self.policy_staleness[agent_idx, selected_idx] = 0

    def _sample_replay_policy(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones(self.num_policies, dtype=np.float64) / self.num_policies

        policy_idx = np.random.choice(self.num_policies, 1, p=sample_weights)[0]
        self._update_staleness(agent_idx, policy_idx)

        return int(policy_idx)

    def _sample_unseen_policy(self):

        weights = self.unseen_policy_weights.astype(float)
        flat_weights = weights.flatten()

        probs = flat_weights / flat_weights.sum()

        flat_idx = np.random.choice(len(flat_weights), p=probs)
        agent_idx, policy_idx = np.unravel_index(flat_idx, weights.shape)

        self._update_staleness(agent_idx, policy_idx)
        
        return agent_idx, policy_idx

    def sample(self):

        policy_unseen = (self.unseen_policy_weights > 0).any(axis=0)
        num_unseen = policy_unseen.sum()

        proportion_seen = (self.num_policies - num_unseen) / self.num_policies

        if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
            return self._sample_replay_policy()
        else:
            return self._sample_unseen_policy()

    # TODO need to fix consider the dimension
    def sample_weights(self):
        policy_scores = np.mean(self.policy_scores, axis=0)
        weights = self._score_transform(self.score_transform, self.temperature, policy_scores)
        weights = weights * (1 - self.unseen_policy_weights)

        z = np.sum(weights)
        if z > 0:
            weights /= z

        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform,
                self.staleness_temperature,
                self.policy_staleness,
            )
            staleness_weights = staleness_weights * (1 - self.unseen_policy_weights)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        elif transform == "max":
            weights = np.zeros_like(scores)
            scores = scores.copy()
            scores[self.unseen_policy_weights > 0] = -float("inf")
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
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
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)
        else:
            raise ValueError(f"Unsupported score transform, {transform}")

        return weights