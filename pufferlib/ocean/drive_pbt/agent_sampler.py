import numpy as np
import torch


class AgentSampler:
    def __init__(
        self,
        num_policies,
        num_agents_per_map,
        num_actors=1,
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
    ):
        self.num_policies = int(num_policies)
        self.num_agents_per_map = np.asarray(num_agents_per_map, dtype=np.float64)
        if self.num_agents_per_map.ndim != 1:
            raise ValueError("num_agents_per_map must be a 1-D array")
        if np.any(self.num_agents_per_map <= 0):
            raise ValueError("num_agents_per_map must be positive")

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

        self.unseen_policy_weights = np.ones(self.num_policies, dtype=np.float64)
        self.policy_scores = np.zeros(self.num_policies, dtype=np.float64)
        self.partial_policy_scores = np.zeros((num_actors, self.num_policies), dtype=np.float64)
        self.partial_policy_steps = np.zeros((num_actors, self.num_policies), dtype=np.int64)
        self.policy_staleness = np.zeros(self.num_policies, dtype=np.float64)

        self.next_policy_index = 0

    def _distance_filtering(self, score, map_idx):
        pass
        
    def _normalize_score(self, score, map_idx):
        return float(score) / self.num_agents_per_map[map_idx]

    @staticmethod
    def _agent_map_idx(agent_idx, agent_offsets):
        ao = np.asarray(agent_offsets, dtype=np.int64)
        return int(np.searchsorted(ao[1:], agent_idx, side="right"))

    def update_with_rollouts(self, rollouts):
        if self.strategy == "random":
            return

        if self.strategy == "policy_entropy":
            score_function = self._average_entropy
        elif self.strategy == "least_confidence":
            score_function = self._average_least_confidence
        elif self.strategy == "min_margin":
            score_function = self._average_min_margin
        elif self.strategy == "gae":
            score_function = self._average_gae
        elif self.strategy == "value_l1":
            score_function = self._average_value_l1
        elif self.strategy == "one_step_td_error":
            score_function = self._one_step_td_error
        else:
            raise ValueError(f"Unsupported strategy, {self.strategy}")

        self._update_with_rollouts(rollouts, score_function)

    def update_policy_score(self, actor_index, policy_idx, score, num_steps, map_idx):
        score = self._normalize_score(score, map_idx)
        score = self._partial_update_policy_score(actor_index, policy_idx, score, num_steps, done=True)

        self.unseen_policy_weights[policy_idx] = 0.0

        old_score = self.policy_scores[policy_idx]
        self.policy_scores[policy_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def _partial_update_policy_score(self, actor_index, policy_idx, score, num_steps, done=False):
        partial_score = self.partial_policy_scores[actor_index][policy_idx]
        partial_num_steps = self.partial_policy_steps[actor_index][policy_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps / float(running_num_steps)

        if done:
            self.partial_policy_scores[actor_index][policy_idx] = 0.0
            self.partial_policy_steps[actor_index][policy_idx] = 0
        else:
            self.partial_policy_scores[actor_index][policy_idx] = merged_score
            self.partial_policy_steps[actor_index][policy_idx] = running_num_steps

        return merged_score

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        num_actions = kwargs["num_actions"]
        max_entropy = -(1.0 / num_actions) * np.log(1.0 / num_actions) * num_actions
        return (-torch.exp(episode_logits) * episode_logits).sum(-1).mean().item() / max_entropy

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        return (1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])).mean().item()

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        return 1 - (top2_confidence[:, 0] - top2_confidence[:, 1]).mean().item()

    def _average_gae(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        advantages = returns - value_preds
        return advantages.mean().item()

    def _average_value_l1(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        advantages = returns - value_preds
        return advantages.abs().mean().item()

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs["rewards"]
        value_preds = kwargs["value_preds"]
        max_t = len(rewards)
        td_errors = (rewards[:-1] + value_preds[: max_t - 1] - value_preds[1:max_t]).abs()
        return td_errors.abs().mean().item()

    @property
    def requires_value_buffers(self):
        return self.strategy in ["gae", "value_l1", "one_step_td_error"]

    def _update_with_rollouts(self, rollouts, score_function):
        policy_ids = rollouts.policy_ids
        map_ids = rollouts.map_ids
        policy_logits = rollouts.action_log_dist
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = policy_logits.shape[:2]
        num_actions = policy_logits.shape[-1]

        for actor_index in range(num_actors):
            done_steps = done[:, actor_index].nonzero()[:total_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps:
                    break

                if t == 0:
                    continue

                policy_idx = int(policy_ids[start_t, actor_index].item())
                map_idx = int(map_ids[start_t, actor_index].item())

                score_function_kwargs = {
                    "episode_logits": torch.log_softmax(policy_logits[start_t:t, actor_index], -1),
                    "num_actions": num_actions,
                }
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = rollouts.returns[start_t:t, actor_index]
                    score_function_kwargs["rewards"] = rollouts.rewards[start_t:t, actor_index]
                    score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:t, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = t - start_t
                self.update_policy_score(actor_index, policy_idx, score, num_steps, map_idx)

                start_t = t.item()

            if start_t < total_steps:
                policy_idx = int(policy_ids[start_t, actor_index].item())
                map_idx = int(map_ids[start_t, actor_index].item())

                score_function_kwargs = {
                    "episode_logits": torch.log_softmax(policy_logits[start_t:, actor_index], -1),
                    "num_actions": num_actions,
                }
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = rollouts.returns[start_t:, actor_index]
                    score_function_kwargs["rewards"] = rollouts.rewards[start_t:, actor_index]
                    score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = total_steps - start_t
                self._partial_update_policy_score(
                    actor_index,
                    policy_idx,
                    self._normalize_score(score, map_idx),
                    num_steps,
                )

    def after_update(self):
        for actor_index in range(self.partial_policy_scores.shape[0]):
            for policy_idx in range(self.partial_policy_scores.shape[1]):
                partial = self.partial_policy_scores[actor_index][policy_idx]
                if partial != 0:
                    old_score = self.policy_scores[policy_idx]
                    self.policy_scores[policy_idx] = (1 - self.alpha) * old_score + self.alpha * partial
                    self.unseen_policy_weights[policy_idx] = 0.0
        self.partial_policy_scores.fill(0)
        self.partial_policy_steps.fill(0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.policy_staleness = self.policy_staleness + 1
            self.policy_staleness[selected_idx] = 0

    def _sample_replay_policy(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones(self.num_policies, dtype=np.float64) / self.num_policies

        policy_idx = np.random.choice(self.num_policies, 1, p=sample_weights)[0]
        self._update_staleness(policy_idx)
        return int(policy_idx)

    def _sample_unseen_policy(self):
        sample_weights = self.unseen_policy_weights / self.unseen_policy_weights.sum()
        policy_idx = np.random.choice(self.num_policies, 1, p=sample_weights)[0]
        self._update_staleness(policy_idx)
        return int(policy_idx)

    def sample(self, strategy=None):
        if not strategy:
            strategy = self.strategy

        if strategy == "random":
            return int(np.random.choice(self.num_policies))

        if strategy == "sequential":
            policy_idx = self.next_policy_index
            self.next_policy_index = (self.next_policy_index + 1) % self.num_policies
            return int(policy_idx)

        num_unseen = (self.unseen_policy_weights > 0).sum()
        proportion_seen = (self.num_policies - num_unseen) / self.num_policies

        if self.replay_schedule == "fixed":
            if proportion_seen >= self.rho:
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_policy()
            return self._sample_unseen_policy()

        if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
            return self._sample_replay_policy()
        return self._sample_unseen_policy()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.policy_scores)
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

    def allocate_other_indices(self, num_agents, ego_indices, agent_offsets):
        """Assign non-ego agents to partner policies using sampler weights."""
        ego_indices = np.asarray(ego_indices, dtype=np.int64)
        other_indices = np.setdiff1d(np.arange(num_agents), ego_indices, assume_unique=False)

        if self.strategy == "random":
            np.random.shuffle(other_indices)
            splits = np.array_split(other_indices, self.num_policies)
            return [s.astype(np.int64, copy=False) for s in splits]

        weights = self.sample_weights()
        if np.isclose(weights.sum(), 0):
            weights = np.ones(self.num_policies, dtype=np.float64) / self.num_policies

        counts = np.random.multinomial(len(other_indices), weights)
        other_indices = np.random.permutation(other_indices)
        splits = []
        start = 0
        for count in counts:
            end = start + count
            splits.append(other_indices[start:end].astype(np.int64, copy=False))
            start = end
        return splits
