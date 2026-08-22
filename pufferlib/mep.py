
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


Logits = Union[torch.Tensor, Sequence[torch.Tensor]]


def logits_to_joint_probs(logits: Logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return F.softmax(logits, dim=-1)


def population_mixture_probs(member_probs: Sequence[torch.Tensor]) -> torch.Tensor:
    """bar_pi(a|s) = (1/K) sum_j pi_j(a|s). ``member_probs[j]`` is [B, A]."""
    stacked = torch.stack(list(member_probs), dim=0)
    return stacked.mean(dim=0)


def mep_bonus_from_mixture(
    mixture_probs: torch.Tensor,
    actions: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """r_MEP = -log(bar_pi(a_i|s) + eps). ``mixture_probs`` [B, A], ``actions`` [B] or [B, 1]."""
    idx = actions.long().reshape(-1, 1)
    if idx.shape[0] != mixture_probs.shape[0]:
        raise ValueError(
            f"actions batch {idx.shape[0]} != mixture batch {mixture_probs.shape[0]}"
        )
    chosen = mixture_probs.gather(dim=-1, index=idx).squeeze(-1)
    return -torch.log(chosen.clamp_min(float(eps)))


def mixture_entropy(mixture_probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """H(bar_pi) per row, nats. ``mixture_probs`` [B, A] → [B]."""
    p = mixture_probs.clamp_min(float(eps))
    return -(mixture_probs * p.log()).sum(dim=-1)


def pairwise_js_divergence(
    member_probs: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean pairwise Jensen–Shannon divergence over members, per batch row.

    Diagnostic only. ``member_probs[j]`` is [B, A] → returns [B].
    """
    if len(member_probs) < 2:
        ref = member_probs[0] if member_probs else None
        if ref is None:
            raise ValueError("pairwise_js_divergence needs probabilities")
        return torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)

    stacked = torch.stack(list(member_probs), dim=0)  # [K, B, A]
    k = stacked.shape[0]
    acc = torch.zeros(stacked.shape[1], device=stacked.device, dtype=stacked.dtype)
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            m = 0.5 * (stacked[i] + stacked[j])
            log_m = m.clamp_min(float(eps)).log()
            kl_im = (stacked[i] * (stacked[i].clamp_min(float(eps)).log() - log_m)).sum(-1)
            kl_jm = (stacked[j] * (stacked[j].clamp_min(float(eps)).log() - log_m)).sum(-1)
            acc = acc + 0.5 * (kl_im + kl_jm)
            n_pairs += 1
    return acc / max(n_pairs, 1)


def mep_reward_components(
    current_logits: Logits,
    reference_logits: Sequence[Logits],
    actions: torch.Tensor,
    env_reward: torch.Tensor,
    *,
    mep_entropy_coef: float,
    mep_eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build augmented reward and diagnostic tensors.

    Mixture includes the current policy (ENT_VERSION=3 ``action_probs_np_new``).
    When ``mep_entropy_coef == 0``, returns ``env_reward`` unchanged (no-op).
    """
    coef = float(mep_entropy_coef)
    if coef == 0.0:
        stats = {
            "env_reward": env_reward.detach(),
            "mep_bonus": torch.zeros_like(env_reward),
            "augmented_reward": env_reward.detach(),
            "mixture_entropy": torch.zeros_like(env_reward),
            "pairwise_js": torch.zeros_like(env_reward),
        }
        return env_reward, stats

    probs = [logits_to_joint_probs(current_logits)]
    for lg in reference_logits:
        probs.append(logits_to_joint_probs(lg))
    mixture = population_mixture_probs(probs)
    bonus = mep_bonus_from_mixture(mixture, actions, eps=mep_eps)
    # Broadcast bonus onto env_reward if env_reward has extra dims
    while bonus.ndim < env_reward.ndim:
        bonus = bonus.unsqueeze(-1)
    aug = env_reward + coef * bonus
    stats = {
        "env_reward": env_reward.detach(),
        "mep_bonus": bonus.detach(),
        "augmented_reward": aug.detach(),
        "mixture_entropy": mixture_entropy(mixture, eps=mep_eps).detach(),
        "pairwise_js": pairwise_js_divergence(probs, eps=mep_eps).detach(),
    }
    return aug, stats


def population_round_robin_orders(
    population_size: int,
    num_iterations: int,
    rng: Optional[torch.Generator] = None,
) -> list[list[int]]:
    """Each iteration: a permutation of ``0..K-1`` (every member updated once)."""
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    orders: list[list[int]] = []
    device = torch.device("cpu")
    for _ in range(int(num_iterations)):
        perm = torch.randperm(population_size, generator=rng, device=device)
        orders.append(perm.tolist())
    return orders
