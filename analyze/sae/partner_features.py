"""Partner-encoder slot embeddings used as SAE input activations.

Policy graph (puffer Drive + LSTMWrapper)::

    per-other partner_encoder  ──┐
    ego_encoder / road_encoder ──┼→ shared_embedding → LSTM → policy head → π, V
                                  │
    SAE hook (default): interrupt │
      partner slot z → z' = D(f+α e_j) before pooling,
      then the *remaining* network including LSTM.

Hook points
-----------
1. ``per_other`` (implemented): intervene on one partner slot embedding,
   then pool → shared_embedding → LSTM → head.
2. ``lstm_hidden`` (optional later): intervene after LSTM; stronger for
   temporal-belief claims, but loses per-vehicle attribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# drive.h ACCELERATION_VALUES / STEERING_VALUES (joint action = accel*13 + steer)
ACCEL_VALUES = torch.tensor(
    [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=torch.float32
)
STEER_VALUES = torch.tensor(
    [
        -1.000,
        -0.833,
        -0.667,
        -0.500,
        -0.333,
        -0.167,
        0.000,
        0.167,
        0.333,
        0.500,
        0.667,
        0.833,
        1.000,
    ],
    dtype=torch.float32,
)
N_STEER = 13


def drive_backbone(policy: torch.nn.Module) -> torch.nn.Module:
    return policy.policy if hasattr(policy, "policy") else policy


def has_lstm(policy: torch.nn.Module) -> bool:
    return hasattr(policy, "cell") and hasattr(policy, "policy")


@torch.no_grad()
def tracked_slot_is_maxpool_winner(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
    *,
    min_channel_frac: float = 0.0,
) -> torch.Tensor:
    """True where tracked slot wins ≥``min_channel_frac`` of partner max-pool channels.

    Drive pools with element-wise ``max`` over partners, so each embedding dim can
    select a different slot. A tracked slot receives gradient on dims it wins.
    Default ``min_channel_frac=0`` keeps rows where the slot wins *any* channel.
    """
    net = drive_backbone(policy)
    ego_dim = net.ego_dim
    partner_features = net.partner_features
    max_partners = net.max_partner_objects
    partner_dim = max_partners * partner_features
    partner_obs = obs[:, ego_dim : ego_dim + partner_dim]
    partner_objects = partner_obs.view(-1, max_partners, partner_features)
    partner_encoded = net.partner_encoder(partner_objects)
    # (B, D) — winning partner index per channel
    winner = partner_encoded.max(dim=1).indices
    slots = partner_slot.long().view(-1, 1).expand_as(winner)
    frac = (winner == slots).float().mean(dim=-1)
    return frac > float(min_channel_frac)


@torch.no_grad()
def extract_partner_context(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
) -> torch.Tensor:
    """Partner-slot embedding from partner_encoder, before max-pool."""
    net = drive_backbone(policy)
    ego_dim = net.ego_dim
    partner_features = net.partner_features
    max_partners = net.max_partner_objects

    partner_start = ego_dim
    partner_end = partner_start + max_partners * partner_features
    partner_obs = obs[:, partner_start:partner_end]
    partner_objects = partner_obs.view(-1, max_partners, partner_features)

    partner_encoded = net.partner_encoder(partner_objects)
    slot = partner_slot.long().view(-1, 1, 1).expand(-1, 1, partner_encoded.size(-1))
    return partner_encoded.gather(1, slot).squeeze(1)


def _partner_pooled_from_override(
    net: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
    slot_embedding: torch.Tensor,
    *,
    pool_mode: str,
    baseline_slot_embedding: torch.Tensor | None,
) -> torch.Tensor:
    ego_dim = net.ego_dim
    partner_features = net.partner_features
    max_partners = net.max_partner_objects
    partner_dim = max_partners * partner_features
    partner_obs = obs[:, ego_dim : ego_dim + partner_dim]
    partner_objects = partner_obs.view(-1, max_partners, partner_features)
    partner_encoded = net.partner_encoder(partner_objects).clone()
    b = torch.arange(obs.shape[0], device=obs.device)
    slots = partner_slot.long().view(-1)
    partner_encoded[b, slots] = slot_embedding.to(dtype=partner_encoded.dtype)
    partner_pooled, _ = partner_encoded.max(dim=1)
    if pool_mode == "slot_only":
        return slot_embedding.to(dtype=partner_pooled.dtype)
    if pool_mode == "delta":
        if baseline_slot_embedding is None:
            raise ValueError("pool_mode=delta requires baseline_slot_embedding")
        return partner_pooled + (
            slot_embedding.to(dtype=partner_pooled.dtype)
            - baseline_slot_embedding.to(dtype=partner_pooled.dtype)
        )
    if pool_mode != "max":
        raise ValueError(pool_mode)
    return partner_pooled


def encode_observations_with_slot_override(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
    slot_embedding: torch.Tensor,
    *,
    pool_mode: str = "max",
    baseline_slot_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drive.encode_observations with one partner slot replaced (pre-LSTM)."""
    net = drive_backbone(policy)
    ego_dim = net.ego_dim
    partner_features = net.partner_features
    max_partners = net.max_partner_objects
    road_features = net.road_features
    max_road = net.max_road_objects

    partner_dim = max_partners * partner_features
    road_dim = max_road * road_features
    ego_obs = obs[:, :ego_dim]
    road_obs = obs[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

    partner_pooled = _partner_pooled_from_override(
        net,
        obs,
        partner_slot,
        slot_embedding,
        pool_mode=pool_mode,
        baseline_slot_embedding=baseline_slot_embedding,
    )

    road_objects = road_obs.view(-1, max_road, road_features)
    road_continuous = road_objects[:, :, : road_features - 1]
    road_categorical = road_objects[:, :, road_features - 1]
    road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
    road_objects = torch.cat([road_continuous, road_onehot], dim=2)

    ego_features = net.ego_encoder(ego_obs)
    road_pooled, _ = net.road_encoder(road_objects).max(dim=1)
    concat = torch.cat([ego_features, road_pooled, partner_pooled], dim=1)
    return F.relu(net.shared_embedding(concat))


def apply_lstm(
    policy: torch.nn.Module,
    encoded: torch.Tensor,
    *,
    lstm_h: torch.Tensor | None = None,
    lstm_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-step LSTMCell. ``None`` state ⇒ zeros (fresh episode / single frame).

    Returns ``(hidden, h_next, c_next)``.
    """
    if not has_lstm(policy):
        return encoded, encoded, torch.zeros_like(encoded)

    b = encoded.shape[0]
    hsz = int(policy.hidden_size)
    if lstm_h is None:
        lstm_h = torch.zeros(b, hsz, device=encoded.device, dtype=encoded.dtype)
    if lstm_c is None:
        lstm_c = torch.zeros(b, hsz, device=encoded.device, dtype=encoded.dtype)
    hidden, c_next = policy.cell(encoded, (lstm_h, lstm_c))
    return hidden, hidden, c_next


def encode_with_slot_override(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
    slot_embedding: torch.Tensor,
    *,
    pool_mode: str = "max",
    baseline_slot_embedding: torch.Tensor | None = None,
    lstm_h: torch.Tensor | None = None,
    lstm_c: torch.Tensor | None = None,
    skip_lstm: bool = False,
):
    """Intervene on per-other embedding, then run the *remaining* policy network.

    Path::

        z' (slot) → pool → shared_embedding → [LSTMCell] → decode_actions → π, V

    ``skip_lstm=True`` reproduces the old (incorrect) path for ablation only.
    Default LSTM state is zeros — single-frame sensitivity. Pass real ``lstm_h/c``
    when you have rolled-out temporal belief states.

    Returns ``(hidden_post_lstm, actions, value)``.
    """
    net = drive_backbone(policy)
    encoded = encode_observations_with_slot_override(
        policy,
        obs,
        partner_slot,
        slot_embedding,
        pool_mode=pool_mode,
        baseline_slot_embedding=baseline_slot_embedding,
    )
    if skip_lstm:
        hidden = encoded
    else:
        hidden, _, _ = apply_lstm(policy, encoded, lstm_h=lstm_h, lstm_c=lstm_c)
    actions, value = net.decode_actions(hidden)
    return hidden, actions, value


def encode_with_slot_override_step(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    partner_slot: torch.Tensor,
    slot_embedding: torch.Tensor,
    *,
    pool_mode: str = "max",
    baseline_slot_embedding: torch.Tensor | None = None,
    lstm_h: torch.Tensor | None = None,
    lstm_c: torch.Tensor | None = None,
    skip_lstm: bool = False,
):
    """One recurrent step with explicit next LSTM state (for rollouts / impulse tests).

    Returns ``(hidden_post_lstm, actions, value, h_next, c_next)``.
    """
    net = drive_backbone(policy)
    encoded = encode_observations_with_slot_override(
        policy,
        obs,
        partner_slot,
        slot_embedding,
        pool_mode=pool_mode,
        baseline_slot_embedding=baseline_slot_embedding,
    )
    if skip_lstm:
        hidden = encoded
        h_next = encoded
        c_next = torch.zeros_like(encoded)
    else:
        hidden, h_next, c_next = apply_lstm(
            policy, encoded, lstm_h=lstm_h, lstm_c=lstm_c
        )
    actions, value = net.decode_actions(hidden)
    return hidden, actions, value, h_next, c_next


def encode_with_lstm_hidden_override(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    *,
    hidden_override: torch.Tensor,
):
    """Hook point 3/4 scaffold: replace post-LSTM hidden, then policy head only.

    Use when intervening on temporal belief / decision representation directly.
    ``obs`` is unused for the head but kept for API symmetry.
    """
    del obs  # head only
    net = drive_backbone(policy)
    actions, value = net.decode_actions(hidden_override)
    return hidden_override, actions, value


def action_stats_from_logits(actions) -> dict[str, torch.Tensor]:
    """Differentiable brake / throttle / steer / accel / entropy from policy logits."""
    if isinstance(actions, torch.distributions.Normal):
        accel = actions.loc[:, 0].float()
        steer = actions.loc[:, 1].float() if actions.loc.shape[-1] > 1 else torch.zeros_like(accel)
        # differential entropy of Normal (sum over dims)
        entropy = actions.entropy().sum(dim=-1).float()
        p_brake = torch.sigmoid(-accel)
        p_throttle = torch.sigmoid(accel)
        return {
            "accel": accel,
            "steer": steer,
            "steer_mag": steer.abs(),
            "p_brake": p_brake,
            "p_throttle": p_throttle,
            "p_neg_accel": torch.sigmoid(-accel),
            "p_strong_brake": torch.sigmoid(-accel - 1.0),
            "brake_proxy": (-accel),
            "log_p_brake": torch.nn.functional.logsigmoid(-accel),
            "brake_minus_throttle": p_brake - p_throttle,
            "entropy": entropy,
        }

    if isinstance(actions, (tuple, list)):
        logit = actions[0]
    elif torch.is_tensor(actions):
        logit = actions
    else:
        raise TypeError(f"unsupported action type: {type(actions)}")

    if logit.shape[-1] != 91:
        probs = torch.softmax(logit.float(), dim=-1)
        idx = torch.arange(logit.shape[-1], device=logit.device, dtype=torch.float32)
        expected = (probs * idx).sum(dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)
        half = max(1, logit.shape[-1] // 2)
        p_brake = probs[:, :half].sum(-1)
        p_throttle = probs[:, half:].sum(-1)
        return {
            "accel": -expected,
            "steer": torch.zeros_like(expected),
            "steer_mag": torch.zeros_like(expected),
            "p_brake": p_brake,
            "p_throttle": p_throttle,
            "p_neg_accel": p_brake,
            "p_strong_brake": p_brake,
            "brake_proxy": -expected,
            "log_p_brake": p_brake.clamp_min(1e-8).log(),
            "brake_minus_throttle": p_brake - p_throttle,
            "entropy": entropy,
        }

    probs = torch.softmax(logit.float(), dim=-1)  # (B, 91)
    device = logit.device
    accel_tbl = ACCEL_VALUES.to(device)
    steer_tbl = STEER_VALUES.to(device)
    a_idx = torch.arange(91, device=device) // N_STEER
    s_idx = torch.arange(91, device=device) % N_STEER
    accel = (probs * accel_tbl[a_idx]).sum(dim=-1)
    steer = (probs * steer_tbl[s_idx]).sum(dim=-1)
    p_brake = probs[:, a_idx < 3].sum(dim=-1)
    p_throttle = probs[:, a_idx > 3].sum(dim=-1)
    p_neg_accel = probs[:, accel_tbl[a_idx] < 0].sum(dim=-1)
    p_strong_brake = probs[:, a_idx == 0].sum(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return {
        "accel": accel,
        "steer": steer,
        "steer_mag": steer.abs(),
        "p_brake": p_brake,
        "p_throttle": p_throttle,
        "p_neg_accel": p_neg_accel,
        "p_strong_brake": p_strong_brake,
        "brake_proxy": -accel,
        "log_p_brake": p_brake.clamp_min(1e-8).log(),
        "brake_minus_throttle": p_brake - p_throttle,
        "entropy": entropy,
    }


def brake_proxy_from_actions(actions) -> torch.Tensor:
    """Higher ⇒ more braking (negative expected accel)."""
    return action_stats_from_logits(actions)["brake_proxy"]


@torch.no_grad()
def batch_partner_context(
    policy: torch.nn.Module,
    obs: np.ndarray,
    partner_slot: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Stacked partner contexts for numpy obs. Shape: (N, context_dim)."""
    if obs.shape[0] == 0:
        net = drive_backbone(policy)
        try:
            dim = int(net.partner_encoder[-1].out_features)
        except Exception:
            dim = int(getattr(net, "input_size", 128))
        return np.zeros((0, dim), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for start in range(0, obs.shape[0], batch_size):
        stop = start + batch_size
        ob = torch.from_numpy(obs[start:stop].astype(np.float32)).to(device)
        slot = torch.from_numpy(partner_slot[start:stop].astype(np.int64)).to(device)
        ctx = extract_partner_context(policy, ob, slot)
        chunks.append(ctx.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)
