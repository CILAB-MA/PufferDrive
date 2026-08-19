import argparse
import json
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


_registered_heatmap_font: Optional[str] = None


def _heatmap_serif_font_name() -> str:
    """Prefer ``times.ttf`` next to this script; otherwise DejaVu Serif."""
    global _registered_heatmap_font
    if _registered_heatmap_font is not None:
        return _registered_heatmap_font
    ttf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "times.ttf")
    if os.path.isfile(ttf):
        try:
            font_manager.fontManager.addfont(ttf)
            _registered_heatmap_font = font_manager.FontProperties(
                fname=ttf
            ).get_name()
            return _registered_heatmap_font
        except (OSError, ValueError, RuntimeError):
            pass
    _registered_heatmap_font = "DejaVu Serif"
    return _registered_heatmap_font


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smooth_edge(x, window=101):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    window = min(window, len(x))
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    x_pad = np.pad(x, (pad_l, pad_r), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x_pad, kernel, mode="valid")


DEFAULT_RISKY_FAMILIES = {
    "very_conservative": np.array([0.2, 0.25, 0.3], dtype=np.float64),
    "conservative": np.array([0.35, 0.4, 0.45], dtype=np.float64),
    "medium": np.array([0.45, 0.5, 0.55], dtype=np.float64),
    "aggressive": np.array([0.5, 0.55, 0.6], dtype=np.float64),
    "very_aggressive": np.array([0.6, 0.65, 0.7], dtype=np.float64),
}


TYPE_NAMES = ["type_0", "type_1", "type_2"]
STATE_NAMES = ["safe", "risky"]

# Heatmap layout (shared by compute / plot / cache)
_HEATMAP_MODE_ORDER = ["replay", "reactive"]
_HEATMAP_LAYOUTS = {
    "1x3": [
        ["risky_pgo", "risky_return_fixed", "risky_collision_fixed"],
    ],
    "2x3": [
        ["safe_pgo", "risky_return_fixed", "risky_collision_fixed"],
        ["risky_pgo", "risky_return_reactive", "risky_collision_reactive"],
    ],
}
_HEATMAP_COL_GROUP_TITLES = {
    "1x3": ["P(Go)", "Risky state Return", "Risky state Collision"],
    "2x3": ["P(Go)", "Risky state Return", "Risky state Collision"],
}
DEFAULT_HEATMAP_LAYOUT = "1x3"
DEFAULT_RISKY_SCALES = (0.10, 0.15, 0.20, 0.25)
HEATMAP_METRICS_CACHE = "heatmap_metrics_cache.pkl"


def _heatmap_layout(layout_key):
    key = str(layout_key).strip().lower()
    if key not in _HEATMAP_LAYOUTS:
        raise ValueError(
            f"Unknown heatmap layout {layout_key!r}. "
            f"Available: {list(_HEATMAP_LAYOUTS)}"
        )
    return _HEATMAP_LAYOUTS[key]


def _heatmap_metric_order(layout_key=None):
    if layout_key is None:
        keys = []
        seen = set()
        for layout in _HEATMAP_LAYOUTS.values():
            for k in (cell for row in layout for cell in row):
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return keys
    return [k for row in _heatmap_layout(layout_key) for k in row]


def parse_risky_scales(text, scale_steps=11):
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if parts:
        return [float(x) for x in parts]
    n_s = max(2, int(scale_steps))
    return list(np.linspace(0.0, 1.0, n_s, dtype=np.float64))


def select_heatmap_scale_slice(metrics, src_scales, dst_scales, atol=1e-8):
    """Subset (or linearly interpolate) heatmap metric grids onto ``dst_scales``."""
    src = np.asarray(src_scales, dtype=np.float64).reshape(-1)
    dst = np.asarray(dst_scales, dtype=np.float64).reshape(-1)
    if src.size == 0:
        raise ValueError("Heatmap cache has no α scales to plot.")
    idxs = []
    exact = True
    for a in dst:
        hits = np.where(np.isclose(src, a, atol=atol, rtol=0.0))[0]
        if hits.size == 0:
            exact = False
            break
        idxs.append(int(hits[0]))
    out = {}
    if exact:
        for mode, md in metrics.items():
            out[mode] = {
                k: np.ascontiguousarray(np.asarray(v)[:, idxs]) for k, v in md.items()
            }
        return out, [float(x) for x in dst], False
    order = np.argsort(src)
    src_sorted = src[order]
    for mode, md in metrics.items():
        out[mode] = {}
        for k, v in md.items():
            arr = np.asarray(v, dtype=np.float64)
            out[mode][k] = np.stack(
                [np.interp(dst, src_sorted, arr[i, order]) for i in range(arr.shape[0])],
                axis=0,
            )
    return out, [float(x) for x in dst], True


def _family_display_name_plot(name: str) -> str:
    """Plot labels: very_conservative → Very Conservative."""
    return str(name).replace("_", " ").strip().title()


@dataclass
class Config:
    steps: int = 4000
    eval_episodes: int = 4000
    dataset_size: int = 4000
    n_seeds: int = 16
    lr: float = 0.05
    baseline_momentum: float = 0.97
    output_dir: str = "toy_population_outputs"

    state_probs: tuple = (0.55, 0.45)  # safe, risky
    type_probs: tuple = (0.30, 0.40, 0.30)

    # OR game base probabilities
    or_safe_base: tuple = (0.80, 0.92, 0.98)
    risky_population_families: dict = field(default_factory=lambda: {
        k: v.copy() for k, v in DEFAULT_RISKY_FAMILIES.items()
    })

    # reactive coupling
    or_safe_scale: float = 0.05
    or_risky_scale: float = 0.60

    # Risky-state OR matrix (align with tmp.py / generate_pareto_risky_families.py)
    risky_gg: float = -1.0
    risky_gy: float = 1.0
    risky_yg: float = 0.0
    risky_yy: float = -0.2

    smooth_window: int = 101


def get_or_payoff(state, ego, other, cfg: Config):
    # action 1 = Go, 0 = Yield
    if state == 0:  # safe
        #          other
        #          Y     G
        # ego Y    0     0
        # ego G    1     2
        if ego == 1 and other == 1:
            return 2.0
        if ego == 1 and other == 0:
            return 1.0
        return 0.0
    else:  # risky
        if ego == 1 and other == 1:
            return float(cfg.risky_gg)
        if ego == 1 and other == 0:
            return float(cfg.risky_gy)
        if ego == 0 and other == 1:
            return float(cfg.risky_yg)
        return float(cfg.risky_yy)


def get_risky_base(cfg, family_name):
    return np.array(cfg.risky_population_families[family_name], dtype=np.float64)


def load_risky_families_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"Family {k!r} must have 3 risky Go probs, got shape {arr.shape}")
        out[str(k)] = arr
    return out


def get_base_probs(cfg, family_name):
    safe = np.array(cfg.or_safe_base, dtype=np.float64)
    risky = get_risky_base(cfg, family_name)
    return np.stack([safe, risky], axis=0)  # [state, type]


def sample_state_and_type(cfg, rng):
    s = rng.choice(2, p=np.array(cfg.state_probs, dtype=np.float64))
    t = rng.choice(3, p=np.array(cfg.type_probs, dtype=np.float64))
    return s, t


def build_replay_dataset(cfg, family_name, rng):
    base = get_base_probs(cfg, family_name)
    states = np.zeros(cfg.dataset_size, dtype=np.int64)
    others = np.zeros(cfg.dataset_size, dtype=np.int64)

    for i in range(cfg.dataset_size):
        s, t = sample_state_and_type(cfg, rng)
        p_other = base[s, t]
        states[i] = s
        others[i] = rng.binomial(1, p_other)

    return {
        "states": states,
        "other_actions": others,
    }


def sample_other_fixed(cfg, family_name, state, rng):
    _, t = sample_state_and_type(cfg, rng)
    base = get_base_probs(cfg, family_name)
    p_other = base[state, t]
    return rng.binomial(1, p_other)


def sample_other_reactive(cfg, family_name, state, ego_go_prob, rng):
    _, t = sample_state_and_type(cfg, rng)
    base = get_base_probs(cfg, family_name)
    p_base = base[state, t]
    scale = cfg.or_safe_scale if state == 0 else cfg.or_risky_scale
    p_other = np.clip(p_base - scale * ego_go_prob, 0.01, 0.99)
    return rng.binomial(1, p_other)


def train_one(cfg, family_name, mode, seed):
    rng = np.random.default_rng(seed)
    theta = np.zeros(2, dtype=np.float64)  # safe, risky
    baseline = 0.0

    replay_data = None
    replay_perm = None
    replay_idx = 0
    if mode == "replay":
        replay_data = build_replay_dataset(cfg, family_name, rng)
        replay_perm = rng.permutation(cfg.dataset_size)

    hist_reward = []
    hist_safe_go = []
    hist_risky_go = []

    for _ in range(cfg.steps):
        state, _ = sample_state_and_type(cfg, rng)
        p_ego = sigmoid(theta[state])
        ego_action = rng.binomial(1, p_ego)

        if mode == "replay":
            if replay_idx >= cfg.dataset_size:
                replay_perm = rng.permutation(cfg.dataset_size)
                replay_idx = 0

            idx = replay_perm[replay_idx]
            state = replay_data["states"][idx]
            p_ego = sigmoid(theta[state])
            ego_action = rng.binomial(1, p_ego)
            other_action = replay_data["other_actions"][idx]
            replay_idx += 1
        else:
            other_action = sample_other_reactive(cfg, family_name, state, p_ego, rng)

        r = get_or_payoff(state, ego_action, other_action, cfg)

        grad_log_pi = ego_action - p_ego
        theta[state] += cfg.lr * (r - baseline) * grad_log_pi
        baseline = cfg.baseline_momentum * baseline + (1.0 - cfg.baseline_momentum) * r

        hist_reward.append(r)
        hist_safe_go.append(sigmoid(theta[0]))
        hist_risky_go.append(sigmoid(theta[1]))

    return {
        "theta": theta.copy(),
        "reward": np.asarray(hist_reward, dtype=np.float64),
        "safe_go": np.asarray(hist_safe_go, dtype=np.float64),
        "risky_go": np.asarray(hist_risky_go, dtype=np.float64),
    }


def evaluate_one(cfg, family_name, theta, eval_mode, seed):
    rng = np.random.default_rng(seed)

    rewards = []
    risky_collisions = []
    safe_success = []
    risky_success = []  # risky: ego Go & other Yield (no collision; Go–Yield cell)
    state_returns = {0: [], 1: []}
    state_go = {0: [], 1: []}

    for _ in range(cfg.eval_episodes):
        state, _ = sample_state_and_type(cfg, rng)
        p_ego = sigmoid(theta[state])
        ego_action = rng.binomial(1, p_ego)

        if eval_mode == "fixed":
            other_action = sample_other_fixed(cfg, family_name, state, rng)
        elif eval_mode == "reactive":
            other_action = sample_other_reactive(cfg, family_name, state, p_ego, rng)
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        r = get_or_payoff(state, ego_action, other_action, cfg)

        rewards.append(r)
        state_returns[state].append(r)
        state_go[state].append(ego_action)

        if state == 1:
            risky_collisions.append(int((ego_action == 1) and (other_action == 1)))
            risky_success.append(int((ego_action == 1) and (other_action == 0)))
        if state == 0:
            safe_success.append(int(ego_action == 1))

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) > 0 else np.nan

    return {
        "return": mean_or_nan(rewards),
        "safe_return": mean_or_nan(state_returns[0]),
        "risky_return": mean_or_nan(state_returns[1]),
        "safe_go_rate": mean_or_nan(state_go[0]),
        "risky_go_rate": mean_or_nan(state_go[1]),
        "risky_collision_rate": mean_or_nan(risky_collisions),
        "safe_success_rate": mean_or_nan(safe_success),
        "risky_success_rate": mean_or_nan(risky_success),
    }


def evaluate_population_policy(cfg, family_name, state, p_go, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    returns = []
    collisions = []
    safe_success = []

    base = get_base_probs(cfg, family_name)

    for _ in range(n):
        _, t = sample_state_and_type(cfg, rng)
        p_other = base[state, t]
        other = rng.binomial(1, p_other)
        ego = rng.binomial(1, p_go)
        r = get_or_payoff(state, ego, other, cfg)
        returns.append(r)

        if state == 1:
            collisions.append(int((ego == 1) and (other == 1)))
        else:
            safe_success.append(int(ego == 1))

    return {
        "return": float(np.mean(returns)),
        "collision": float(np.mean(collisions)) if state == 1 else np.nan,
        "success": float(np.mean(safe_success)) if state == 0 else np.nan,
    }


def population_diagnostics(cfg, family_names):
    rows = []

    for family_name in family_names:
        risky_base = get_risky_base(cfg, family_name)

        for ti, pop_name in enumerate(TYPE_NAMES):
            safe_p = float(cfg.or_safe_base[ti])
            risky_p = float(risky_base[ti])

            safe_diag = evaluate_population_policy(cfg, family_name, state=0, p_go=safe_p, seed=100 + ti)
            risky_diag = evaluate_population_policy(cfg, family_name, state=1, p_go=risky_p, seed=200 + ti)

            total_ret = (
                cfg.state_probs[0] * safe_diag["return"] +
                cfg.state_probs[1] * risky_diag["return"]
            )

            rows.append({
                "family": family_name,
                "population": pop_name,
                "avg_return": total_ret,
                "safe_return": safe_diag["return"],
                "risky_return": risky_diag["return"],
                "safe_go_rate": safe_p,
                "risky_go_rate": risky_p,
                "safe_success_rate": safe_diag["success"],
                "risky_collision_rate": risky_diag["collision"],
            })

        safe_mix = float(np.sum(np.array(cfg.or_safe_base) * np.array(cfg.type_probs)))
        risky_mix = float(np.sum(risky_base * np.array(cfg.type_probs)))

        safe_diag = evaluate_population_policy(cfg, family_name, state=0, p_go=safe_mix, seed=300)
        risky_diag = evaluate_population_policy(cfg, family_name, state=1, p_go=risky_mix, seed=400)

        total_ret = (
            cfg.state_probs[0] * safe_diag["return"] +
            cfg.state_probs[1] * risky_diag["return"]
        )

        rows.append({
            "family": family_name,
            "population": "mixture",
            "avg_return": total_ret,
            "safe_return": safe_diag["return"],
            "risky_return": risky_diag["return"],
            "safe_go_rate": safe_mix,
            "risky_go_rate": risky_mix,
            "safe_success_rate": safe_diag["success"],
            "risky_collision_rate": risky_diag["collision"],
        })

        # Oracle: safe=Go, risky=Yield
        safe_diag = evaluate_population_policy(cfg, family_name, state=0, p_go=1.0, seed=500)
        risky_diag = evaluate_population_policy(cfg, family_name, state=1, p_go=0.0, seed=600)

        total_ret = (
            cfg.state_probs[0] * safe_diag["return"] +
            cfg.state_probs[1] * risky_diag["return"]
        )

        rows.append({
            "family": family_name,
            "population": "oracle",
            "avg_return": total_ret,
            "safe_return": safe_diag["return"],
            "risky_return": risky_diag["return"],
            "safe_go_rate": 1.0,
            "risky_go_rate": 0.0,
            "safe_success_rate": safe_diag["success"],
            "risky_collision_rate": risky_diag["collision"],
        })

    return pd.DataFrame(rows)


def plot_local_population_metrics(df_family, family_name, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("avg_return", "avg return"),
        ("safe_return", "safe return"),
        ("risky_return", "risky return"),
        ("safe_success_rate", "safe success"),
        ("risky_collision_rate", "risky collision"),
        ("risky_go_rate", "risky go rate"),
    ]

    pop_order = ["type_0", "type_1", "type_2", "mixture", "oracle"]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        sub = df_family.set_index("population").reindex(pop_order).reset_index()
        ax.bar(sub["population"], sub[metric])
        ax.set_title(f"{_family_display_name_plot(family_name)}: {title}")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_or_family_sweep(cfg, family_names):
    root = os.path.join(cfg.output_dir, "or", "risky_population_family")
    os.makedirs(root, exist_ok=True)

    summary_rows = []

    for family_name in family_names:
        family_dir = os.path.join(root, family_name)
        os.makedirs(family_dir, exist_ok=True)

        histories = {"replay": [], "reactive": []}
        thetas = {"replay": [], "reactive": []}

        for mode in ["replay", "reactive"]:
            for seed in range(cfg.n_seeds):
                out = train_one(cfg, family_name, mode, seed)
                histories[mode].append(out)
                thetas[mode].append(out["theta"])

        # Learning curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"replay": "tab:blue", "reactive": "tab:orange"}

        for mode in ["replay", "reactive"]:
            color = colors[mode]

            rewards = np.stack([h["reward"] for h in histories[mode]], axis=0)
            mean_r = smooth_edge(rewards.mean(0), cfg.smooth_window)
            std_r = smooth_edge(rewards.std(0), cfg.smooth_window)
            x = np.arange(len(mean_r))

            axes[0].plot(x, mean_r, label=mode, color=color)
            axes[0].fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.20, color=color)

            safe_go = np.stack([h["safe_go"] for h in histories[mode]], axis=0)
            risky_go = np.stack([h["risky_go"] for h in histories[mode]], axis=0)

            mean_safe = smooth_edge(safe_go.mean(0), cfg.smooth_window)
            mean_risky = smooth_edge(risky_go.mean(0), cfg.smooth_window)
            std_safe = smooth_edge(safe_go.std(0), cfg.smooth_window)
            std_risky = smooth_edge(risky_go.std(0), cfg.smooth_window)

            axes[1].plot(x, mean_safe, label=f"{mode} | safe", color=color)
            axes[1].fill_between(x, mean_safe - std_safe, mean_safe + std_safe, alpha=0.12, color=color)
            axes[1].plot(x, mean_risky, label=f"{mode} | risky", linestyle="--", color=color)
            axes[1].fill_between(x, mean_risky - std_risky, mean_risky + std_risky, alpha=0.12, color=color)

        axes[0].set_title(f"OR game: training reward | {_family_display_name_plot(family_name)}")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("reward")
        axes[0].legend()

        axes[1].set_title(f"OR game: ego policy | {_family_display_name_plot(family_name)}")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("P(Go)")
        axes[1].legend()

        plt.tight_layout()
        # Guard against transient/nonexistent parent errors on some runs.
        os.makedirs(family_dir, exist_ok=True)
        plt.savefig(os.path.join(family_dir, "learning_curves.pdf"), dpi=300)
        plt.close()

        # Train/eval summary
        local_rows = []
        for mode in ["replay", "reactive"]:
            for seed, theta in enumerate(thetas[mode]):
                for eval_mode in ["fixed", "reactive"]:
                    metrics = evaluate_one(cfg, family_name, theta, eval_mode, seed=10000 + seed)
                    row = {
                        "family": family_name,
                        "train_mode": mode,
                        "eval_mode": eval_mode,
                        "seed": seed,
                    }
                    row.update(metrics)
                    local_rows.append(row)
                    summary_rows.append(row)

        df_local = pd.DataFrame(local_rows)
        os.makedirs(family_dir, exist_ok=True)
        df_local.to_csv(os.path.join(family_dir, "train_eval_summary.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cfg.output_dir, "summary.csv"), index=False)
    return summary_df


def plot_replay_reactive_family_sweep(summary_df, family_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    replay_adv = []
    collision_reduction = []

    for family_name in family_names:
        sub = summary_df[
            (summary_df["family"] == family_name) &
            (summary_df["eval_mode"] == "fixed")
        ]

        replay_ret = sub[sub["train_mode"] == "replay"]["return"].mean()
        reactive_ret = sub[sub["train_mode"] == "reactive"]["return"].mean()
        replay_col = sub[sub["train_mode"] == "replay"]["risky_collision_rate"].mean()
        reactive_col = sub[sub["train_mode"] == "reactive"]["risky_collision_rate"].mean()

        replay_adv.append(replay_ret - reactive_ret)
        collision_reduction.append(reactive_col - replay_col)

    xs = np.arange(len(family_names))

    axes[0].plot(xs, replay_adv, marker="o")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([_family_display_name_plot(f) for f in family_names], rotation=20)
    axes[0].set_title("Non-reactive eval: return advantage")
    axes[0].set_ylabel("recorded-play return - reactive return")

    axes[1].plot(xs, collision_reduction, marker="o")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([_family_display_name_plot(f) for f in family_names], rotation=20)
    axes[1].set_title("Non-reactive eval: collision reduction")
    axes[1].set_ylabel("reactive collision - replay collision")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_population_family_sweep(pop_df, family_names, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("avg_return", "Population avg return"),
        ("safe_return", "Population safe return"),
        ("risky_return", "Population risky return"),
        ("safe_success_rate", "Population safe success"),
        ("risky_collision_rate", "Population risky collision"),
        ("risky_go_rate", "Population risky go rate"),
    ]
    pop_order = ["type_0", "type_1", "type_2", "mixture", "oracle"]
    xs = np.arange(len(family_names))

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for pop in pop_order:
            ys = []
            for family_name in family_names:
                val = pop_df[
                    (pop_df["family"] == family_name) &
                    (pop_df["population"] == pop)
                ][metric].mean()
                ys.append(val)

            ax.plot(xs, ys, marker="o", label=pop)

        ax.set_xticks(xs)
        ax.set_xticklabels([_family_display_name_plot(f) for f in family_names], rotation=20)
        ax.set_title(title)
        ax.set_ylabel(metric)

    axes[0, 0].legend()
    fig.suptitle("Population performance across risky population types", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_metric_grouped_heatmaps(cfg, family_names, risky_scales):
    """Train/eval grid for heatmaps (expensive).

    Each column ``α`` in ``risky_scales`` sets both reactive couplings:
    ``or_safe_scale = or_risky_scale = α``.
    """
    n_rows = len(family_names)
    n_cols = len(risky_scales)
    metric_order = _heatmap_metric_order()
    metrics = {
        mode: {metric: np.zeros((n_rows, n_cols), dtype=np.float64) for metric in metric_order}
        for mode in _HEATMAP_MODE_ORDER
    }

    for row_i, family_name in enumerate(family_names):
        for col_i, alpha in enumerate(risky_scales):
            alpha = float(alpha)
            local_cfg = Config(
                steps=cfg.steps,
                eval_episodes=cfg.eval_episodes,
                dataset_size=cfg.dataset_size,
                n_seeds=cfg.n_seeds,
                lr=cfg.lr,
                baseline_momentum=cfg.baseline_momentum,
                output_dir=cfg.output_dir,
                state_probs=cfg.state_probs,
                type_probs=cfg.type_probs,
                or_safe_base=cfg.or_safe_base,
                risky_population_families={
                    k: np.array(v, dtype=np.float64).copy()
                    for k, v in cfg.risky_population_families.items()
                },
                or_safe_scale=alpha,
                or_risky_scale=alpha,
                smooth_window=cfg.smooth_window,
                risky_gg=cfg.risky_gg,
                risky_gy=cfg.risky_gy,
                risky_yg=cfg.risky_yg,
                risky_yy=cfg.risky_yy,
            )

            for mode in _HEATMAP_MODE_ORDER:
                final_safe_all = []
                final_risky_all = []
                eval_risky_return_fixed = []
                eval_risky_collision_fixed = []
                eval_risky_return_reactive = []
                eval_risky_collision_reactive = []

                for seed in range(local_cfg.n_seeds):
                    out = train_one(local_cfg, family_name, mode, seed)
                    theta = out["theta"]
                    final_safe_all.append(float(sigmoid(theta[0])))
                    final_risky_all.append(float(sigmoid(theta[1])))
                    eval_fixed = evaluate_one(local_cfg, family_name, theta, "fixed", seed=10000 + seed)
                    eval_reactive = evaluate_one(local_cfg, family_name, theta, "reactive", seed=20000 + seed)
                    eval_risky_return_fixed.append(float(eval_fixed["risky_return"]))
                    eval_risky_collision_fixed.append(float(eval_fixed["risky_collision_rate"]))
                    eval_risky_return_reactive.append(float(eval_reactive["risky_return"]))
                    eval_risky_collision_reactive.append(float(eval_reactive["risky_collision_rate"]))

                metrics[mode]["safe_pgo"][row_i, col_i] = np.mean(final_safe_all)
                metrics[mode]["risky_pgo"][row_i, col_i] = np.mean(final_risky_all)
                metrics[mode]["risky_return_fixed"][row_i, col_i] = np.mean(eval_risky_return_fixed)
                metrics[mode]["risky_collision_fixed"][row_i, col_i] = np.mean(eval_risky_collision_fixed)
                metrics[mode]["risky_return_reactive"][row_i, col_i] = np.mean(eval_risky_return_reactive)
                metrics[mode]["risky_collision_reactive"][row_i, col_i] = np.mean(
                    eval_risky_collision_reactive
                )

    return metrics


def plot_metric_grouped_heatmaps_from_metrics(
    metrics, family_names, risky_scales, out_path, layout_key=DEFAULT_HEATMAP_LAYOUT
):
    """Render heatmap PDF from precomputed ``metrics`` (replay minus reactive per cell)."""
    n_rows = len(family_names)
    n_cols = len(risky_scales)
    font = _heatmap_serif_font_name()
    fs_title = 15
    fs_colgroup = 18
    fs_axis = 15
    fs_tick = 13
    fs_cell = 12
    fs_cbar = 13

    metric_layout = _heatmap_layout(layout_key)
    needed = _heatmap_metric_order(layout_key)
    missing = [
        k for k in needed
        if k not in metrics.get("replay", {}) or k not in metrics.get("reactive", {})
    ]
    if missing:
        raise KeyError(
            f"Heatmap layout {layout_key!r} needs metrics {missing}; "
            "re-run the full pipeline so heatmap_metrics_cache.pkl has them."
        )
    n_plot_rows = len(metric_layout)
    n_plot_cols = len(metric_layout[0])
    metric_titles = {
        "safe_pgo": r"$P(\mathrm{Go} \mid \mathtt{safe})$",
        "risky_pgo": r"$P(\mathrm{Go} \mid \mathtt{risky})$",
        "risky_return_fixed": r"Mean return | $\mathtt{risky}$ (non-reactive)",
        "risky_collision_fixed": r"Collision rate | $\mathtt{risky}$ (non-reactive)",
        "risky_return_reactive": r"Mean return | $\mathtt{risky}$ (reactive)",
        "risky_collision_reactive": r"Collision rate | $\mathtt{risky}$ (reactive)",
    }

    rc = {
        "font.family": font,
        "font.size": fs_tick,
        "axes.titlesize": fs_title,
        "axes.labelsize": fs_axis,
        "xtick.labelsize": fs_tick,
        "ytick.labelsize": fs_tick,
    }
    with plt.rc_context(rc):
        fig_h = 5.2 if n_plot_rows == 1 else 9.2
        fig = plt.figure(figsize=(16.5, fig_h), constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=n_plot_rows,
            ncols=2 * n_plot_cols,
            width_ratios=[1.0, 0.018] * n_plot_cols,
            wspace=0.05,
            hspace=0.09,
        )
        axes = np.empty((n_plot_rows, n_plot_cols), dtype=object)
        cbar_axes = np.empty((n_plot_rows, n_plot_cols), dtype=object)
        for r in range(n_plot_rows):
            for c in range(n_plot_cols):
                axes[r, c] = fig.add_subplot(gs[r, 2 * c])
                cbar_axes[r, c] = fig.add_subplot(gs[r, 2 * c + 1])

        col_group_titles = _HEATMAP_COL_GROUP_TITLES.get(str(layout_key).strip().lower(), [])
        for r in range(n_plot_rows):
            for c in range(n_plot_cols):
                metric_key = metric_layout[r][c]
                ax = axes[r, c]
                data = metrics["replay"][metric_key] - metrics["reactive"][metric_key]
                vmax = float(np.max(np.abs(data)))
                if vmax < 1e-12:
                    vmax = 1e-12
                vmin = -vmax
                im = ax.imshow(
                    data,
                    aspect="auto",
                    origin="upper",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="coolwarm",
                )
                norm = im.norm
                cmap = im.cmap
                ax.set_title(
                    f"{metric_titles[metric_key]}",
                    fontsize=fs_title,
                )
                ax.set_xticks(np.arange(n_cols))
                ax.set_xticklabels(
                    [f"{s:.2f}" for s in risky_scales],
                    rotation=45,
                    ha="right",
                    fontsize=fs_tick,
                )
                ax.set_yticks(np.arange(n_rows))
                if c == 0:
                    ax.set_yticklabels(
                        [_family_display_name_plot(f) for f in family_names],
                        fontsize=fs_tick,
                    )
                    ax.set_ylabel("Population type", fontsize=fs_axis)
                else:
                    ax.set_yticklabels([])
                ax.set_xlabel("α scale", fontsize=fs_axis)
                if r == 0 and c < len(col_group_titles):
                    ax.text(
                        0.5,
                        1.12,
                        col_group_titles[c],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=fs_colgroup,
                        fontweight="bold",
                    )
                for i in range(n_rows):
                    for j in range(n_cols):
                        rgba = cmap(norm(data[i, j]))
                        luminance = (
                            0.2126 * rgba[0]
                            + 0.7152 * rgba[1]
                            + 0.0722 * rgba[2]
                        )
                        txt_color = "white" if luminance < 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{data[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=fs_cell,
                            color=txt_color,
                        )
                cbar = fig.colorbar(im, cax=cbar_axes[r, c])
                cbar.ax.tick_params(labelsize=fs_cbar)

        plt.savefig(out_path, dpi=300)
        plt.close()


def save_heatmap_metrics_cache(output_dir, family_names, risky_scales, metrics):
    path = os.path.join(output_dir, HEATMAP_METRICS_CACHE)
    payload = {
        "family_names": list(family_names),
        "risky_scales": np.asarray(risky_scales, dtype=np.float64),
        "metrics": metrics,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_heatmap_metrics_cache(output_dir):
    path = os.path.join(output_dir, HEATMAP_METRICS_CACHE)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_metric_grouped_heatmaps(
    cfg, family_names, risky_scales, out_path, layout_key=DEFAULT_HEATMAP_LAYOUT
):
    metrics = compute_metric_grouped_heatmaps(cfg, family_names, risky_scales)
    plot_metric_grouped_heatmaps_from_metrics(
        metrics, family_names, risky_scales, out_path, layout_key=layout_key
    )
    save_heatmap_metrics_cache(cfg.output_dir, family_names, risky_scales, metrics)


def _family_order_from_summary(summary_df, requested):
    seen = []
    present = set(summary_df["family"].astype(str))
    if requested:
        for f in requested:
            if f in present and f not in seen:
                seen.append(f)
        missing = [f for f in requested if f not in present]
        if missing:
            raise ValueError(f"--families not found in summary.csv: {missing}")
        return seen
    for f in summary_df["family"].astype(str):
        if f not in seen:
            seen.append(f)
    return seen


def regenerate_plots_from_disk(
    output_dir,
    families_filter,
    heatmap_layout=DEFAULT_HEATMAP_LAYOUT,
    risky_scales=None,
):
    """Rebuild figures from ``summary.csv`` + ``population_diagnostics.csv`` (+ optional heatmap cache)."""
    summary_path = os.path.join(output_dir, "summary.csv")
    pop_path = os.path.join(output_dir, "population_diagnostics.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing {summary_path} (run full pipeline once or pass --output-dir).")
    if not os.path.isfile(pop_path):
        raise FileNotFoundError(f"Missing {pop_path}")

    summary_df = pd.read_csv(summary_path)
    pop_df = pd.read_csv(pop_path)
    family_names = _family_order_from_summary(summary_df, families_filter)

    plot_replay_reactive_family_sweep(
        summary_df,
        family_names,
        os.path.join(output_dir, "replay_reactive_family_sweep.pdf"),
    )
    plot_population_family_sweep(
        pop_df,
        family_names,
        os.path.join(output_dir, "population_family_sweep.pdf"),
    )

    cache = load_heatmap_metrics_cache(output_dir)
    if cache is not None:
        c_families = list(cache["family_names"])
        c_scales = [float(x) for x in np.asarray(cache["risky_scales"]).reshape(-1)]
        if set(c_families) != set(family_names):
            print(
                "Warning: heatmap cache families differ from summary.csv families; "
                "heatmap uses cached family order."
            )
        plot_scales = list(DEFAULT_RISKY_SCALES) if risky_scales is None else list(risky_scales)
        metrics, plot_scales, interpolated = select_heatmap_scale_slice(
            cache["metrics"], c_scales, plot_scales
        )
        if interpolated:
            print(
                "Heatmap α slice interpolated from cache scales "
                f"{[round(x, 4) for x in c_scales]} -> {[round(x, 4) for x in plot_scales]}"
            )
        plot_metric_grouped_heatmaps_from_metrics(
            metrics,
            c_families,
            plot_scales,
            os.path.join(output_dir, "policy_metric_heatmaps_family_x_risky_scale.pdf"),
            layout_key=heatmap_layout,
        )
    else:
        print(
            f"No {HEATMAP_METRICS_CACHE} in {output_dir}; skipping heatmap "
            "(run full pipeline once to write the cache)."
        )

    root = os.path.join(output_dir, "or", "risky_population_family")
    for family_name in family_names:
        family_dir = os.path.join(root, family_name)
        os.makedirs(family_dir, exist_ok=True)
        pop_local = pop_df[pop_df["family"] == family_name].copy()
        pop_local.to_csv(os.path.join(family_dir, "population_summary.csv"), index=False)
        plot_local_population_metrics(
            pop_local,
            family_name,
            os.path.join(family_dir, "population_metrics.pdf"),
        )

    print(f"plots-only: regenerated figures under {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip simulation/training; rebuild PDFs from summary.csv and "
        "population_diagnostics.csv under --output-dir. Heatmap needs "
        f"{HEATMAP_METRICS_CACHE} from a prior full run.",
    )
    parser.add_argument("--output-dir", type=str, default="toy_population_outputs")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=4000)
    parser.add_argument("--dataset-size", type=int, default=4000)
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--or-risky-scale", type=float, default=0.60)
    parser.add_argument(
        "--heatmap-layout",
        type=str,
        default=DEFAULT_HEATMAP_LAYOUT,
        choices=sorted(_HEATMAP_LAYOUTS),
        help="Heatmap panel grid key: 1x3 (risky P(Go) + non-reactive return/collision) "
        "or 2x3 (safe/risky P(Go) + fixed/reactive return & collision).",
    )
    parser.add_argument(
        "--risky-scales",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RISKY_SCALES),
        help="Comma-separated α scales for heatmaps (or_safe_scale = or_risky_scale = α). "
        f"Default: {', '.join(str(x) for x in DEFAULT_RISKY_SCALES)}. "
        "Empty = np.linspace(0,1, --risky-scale-steps).",
    )
    parser.add_argument(
        "--risky-scale-steps",
        type=int,
        default=11,
        help="When --risky-scales is empty, use np.linspace(0,1, this many α values for both scales).",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="very_conservative,conservative,medium,aggressive,very_aggressive",
        help="Comma-separated subset of family names. Empty with --risky-families-json uses all keys in file order. "
        "With --plots-only, use 'auto' (or empty) to take family order from summary.csv.",
    )
    parser.add_argument(
        "--risky-families-json",
        type=str,
        default="",
        help="JSON object mapping family name -> [q0,q1,q2] risky-state opponent Go probs; replaces built-in families.",
    )
    parser.add_argument(
        "--risky-gg",
        type=float,
        default=-1.0,
        help="Risky payoff (Go, Go); match tmp.py --gg when using Pareto families.",
    )
    parser.add_argument("--risky-gy", type=float, default=1.0, help="Risky payoff (Go, Yield).")
    parser.add_argument("--risky-yg", type=float, default=0.0, help="Risky payoff (Yield, Go).")
    parser.add_argument("--risky-yy", type=float, default=-0.2, help="Risky payoff (Yield, Yield).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.plots_only:
        fam = args.families.strip()
        if fam.lower() in ("", "auto"):
            requested = []
        else:
            requested = [x.strip() for x in args.families.split(",") if x.strip()]
        regenerate_plots_from_disk(
            args.output_dir,
            requested,
            heatmap_layout=args.heatmap_layout,
            risky_scales=parse_risky_scales(args.risky_scales, args.risky_scale_steps),
        )
        return

    cfg = Config(
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        dataset_size=args.dataset_size,
        n_seeds=args.n_seeds,
        lr=args.lr,
        output_dir=args.output_dir,
        or_risky_scale=args.or_risky_scale,
        risky_gg=args.risky_gg,
        risky_gy=args.risky_gy,
        risky_yg=args.risky_yg,
        risky_yy=args.risky_yy,
    )

    if args.risky_families_json:
        loaded = load_risky_families_json(args.risky_families_json)
        cfg = replace(cfg, risky_population_families=loaded)

    requested_families = [x.strip() for x in args.families.split(",") if x.strip()]
    if not requested_families:
        if args.risky_families_json:
            requested_families = list(cfg.risky_population_families.keys())
        else:
            raise SystemExit("No --families given; pass names or use --risky-families-json with empty --families for all keys.")

    for name in requested_families:
        if name not in cfg.risky_population_families:
            raise ValueError(
                f"Unknown family '{name}'. "
                f"Available: {list(cfg.risky_population_families.keys())}"
            )

    family_names = requested_families
    os.makedirs(cfg.output_dir, exist_ok=True)

    risky_scales = parse_risky_scales(args.risky_scales, args.risky_scale_steps)

    # Replay/reactive training + eval
    summary_df = run_or_family_sweep(cfg, family_names)
    plot_replay_reactive_family_sweep(
        summary_df,
        family_names,
        os.path.join(cfg.output_dir, "replay_reactive_family_sweep.pdf"),
    )

    # Population diagnostics
    pop_df = population_diagnostics(cfg, family_names)
    pop_df.to_csv(os.path.join(cfg.output_dir, "population_diagnostics.csv"), index=False)
    plot_population_family_sweep(
        pop_df,
        family_names,
        os.path.join(cfg.output_dir, "population_family_sweep.pdf"),
    )

    plot_metric_grouped_heatmaps(
        cfg=cfg,
        family_names=family_names,
        risky_scales=risky_scales,
        out_path=os.path.join(cfg.output_dir, "policy_metric_heatmaps_family_x_risky_scale.pdf"),
        layout_key=args.heatmap_layout,
    )

    root = os.path.join(cfg.output_dir, "or", "risky_population_family")
    for family_name in family_names:
        family_dir = os.path.join(root, family_name)
        os.makedirs(family_dir, exist_ok=True)

        pop_local = pop_df[pop_df["family"] == family_name].copy()
        pop_local.to_csv(os.path.join(family_dir, "population_summary.csv"), index=False)
        plot_local_population_metrics(
            pop_local,
            family_name,
            os.path.join(family_dir, "population_metrics.pdf"),
        )

    print(f"saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()