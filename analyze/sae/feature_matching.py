"""Cross-model SAE feature matching (normalize → similarity → Hungarian).

Assumes per-experiment analysis dirs already have::

    <analysis>/<exp>/feature_acts.npy   # [N, d_sae] float16
    <analysis>/<exp>/sample_keys.npz
    <analysis>/<exp>/feature_stats.npz
    <analysis>/<exp>/top_activations.npz

Default model aliases (ReCord paper naming)::

    record   → replay_0.25
    reactive → reactive_0.25
    selfplay → selfplay

Usage::

    python analyze/sae/feature_matching.py \\
      --analysis-dir /data/puffer/sae/runs/topk_exp16_k32_step1908/analysis/top_activations \\
      --normalize rank --metric spearman \\
      --match-threshold 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

# Paper / claim naming → on-disk experiment folder name
MODEL_ALIASES = {
    "record": "replay_0.25",
    "rec": "replay_0.25",
    "replay": "replay_0.25",
    "replay_0.25": "replay_0.25",
    "reactive": "reactive_0.25",
    "reactive_0.25": "reactive_0.25",
    "selfplay": "selfplay",
    "sp": "selfplay",
}

DEFAULT_TRIPLET = ("record", "reactive", "selfplay")


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        # allow raw folder names
        return name
    return MODEL_ALIASES[key]


def load_feature_acts(exp_dir: Path) -> np.ndarray:
    path = exp_dir / "feature_acts.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; re-run top_activation_retrieval without --no-full-acts"
        )
    return np.load(path, mmap_mode="r")


def load_alive_mask(exp_dir: Path, *, min_density: float = 0.0) -> np.ndarray:
    stats = np.load(exp_dir / "feature_stats.npz")
    dens = stats["density"]
    alive = dens > min_density
    if "alive" in stats.files:
        alive = alive & (stats["alive"].astype(bool))
    return alive


# --------------------------------------------------------------------------- normalize
def normalize_features(
    feats: np.ndarray,
    *,
    method: str = "rank",
    chunk_features: int = 64,
) -> np.ndarray:
    """Normalize each feature column across samples.

    Parameters
    ----------
    feats:
        ``[N, d]`` (may be memmap / float16).
    method:
        ``rank`` (recommended), ``zscore``, or ``none``.
    """
    n, d = feats.shape
    if method == "none":
        return np.asarray(feats, dtype=np.float32)

    out = np.empty((n, d), dtype=np.float32)
    for start in range(0, d, chunk_features):
        stop = min(start + chunk_features, d)
        block = np.asarray(feats[:, start:stop], dtype=np.float32)
        if method == "rank":
            # Average ranks for ties; scale to [0, 1].
            for j in range(block.shape[1]):
                col = block[:, j]
                out[:, start + j] = scipy_stats.rankdata(col, method="average") / n
        elif method == "zscore":
            mu = block.mean(axis=0, keepdims=True)
            sd = block.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-8, 1.0, sd)
            out[:, start:stop] = (block - mu) / sd
        else:
            raise ValueError(f"unknown normalize method: {method}")
    return out


# --------------------------------------------------------------------------- similarity
def _spearman_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman corr for all column pairs. ``a[N,da]``, ``b[N,db]`` → ``[da,db]``.

    Equivalent to Pearson on rank-normalized columns.
    """
    a_r = a - a.mean(axis=0, keepdims=True)
    b_r = b - b.mean(axis=0, keepdims=True)
    a_n = np.linalg.norm(a_r, axis=0)
    b_n = np.linalg.norm(b_r, axis=0)
    a_n = np.where(a_n < 1e-12, 1.0, a_n)
    b_n = np.where(b_n < 1e-12, 1.0, b_n)
    return (a_r.T @ b_r) / np.outer(a_n, b_n)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = np.linalg.norm(a, axis=0)
    b_n = np.linalg.norm(b, axis=0)
    a_n = np.where(a_n < 1e-12, 1.0, a_n)
    b_n = np.where(b_n < 1e-12, 1.0, b_n)
    return (a.T @ b) / np.outer(a_n, b_n)


def _pearson_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    return _cosine_matrix(a_c, b_c)


def top_quantile_index_matrix(feats: np.ndarray, *, quantile: float = 0.99) -> np.ndarray:
    """Return ``(d, k)`` row indices of top quantile per feature."""
    n, d = feats.shape
    k = max(int(np.ceil(n * (1.0 - quantile))), 1)
    out = np.empty((d, k), dtype=np.int64)
    for j in range(d):
        col = np.asarray(feats[:, j], dtype=np.float32)
        out[j] = np.argpartition(-col, kth=k - 1)[:k]
    return out


def jaccard_matrix_from_indices(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Jaccard overlap of top-index sets. ``idx_*`` shape ``(d, k)``."""
    da, ka = idx_a.shape
    db, kb = idx_b.shape
    # Sort once for faster intersect via searchsorted-ish set; use python sets on k~1e3
    sets_a = [set(idx_a[i].tolist()) for i in range(da)]
    sets_b = [set(idx_b[j].tolist()) for j in range(db)]
    out = np.zeros((da, db), dtype=np.float32)
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            inter = len(sa & sb)
            union = len(sa | sb)
            out[i, j] = inter / union if union else 0.0
    return out


def compute_similarity_bundle(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    *,
    normalize: str = "rank",
    top_quantile: float = 0.99,
    alive_a: np.ndarray | None = None,
    alive_b: np.ndarray | None = None,
    compute_jaccard: bool = True,
) -> dict[str, np.ndarray]:
    """Similarity matrices between two SAE feature banks on shared observations."""
    print(f"  normalize={normalize}  shapes A={feats_a.shape} B={feats_b.shape}")
    a = normalize_features(feats_a, method=normalize)
    b = normalize_features(feats_b, method=normalize)

    spearman = _spearman_matrix(a, b).astype(np.float32)
    cosine = _cosine_matrix(a, b).astype(np.float32)
    # Pearson on raw (not rank) activations
    raw_a = np.asarray(feats_a, dtype=np.float32)
    raw_b = np.asarray(feats_b, dtype=np.float32)
    pearson = _pearson_matrix(raw_a, raw_b).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "spearman": spearman,
        "cosine": cosine,
        "pearson": pearson,
        "top_quantile": np.float32(top_quantile),
    }

    if compute_jaccard:
        print(f"  top-{int(round((1 - top_quantile) * 100))}% Jaccard (alive features)…")
        if alive_a is None:
            alive_a = np.ones(feats_a.shape[1], dtype=bool)
        if alive_b is None:
            alive_b = np.ones(feats_b.shape[1], dtype=bool)
        ia = np.flatnonzero(alive_a)
        ib = np.flatnonzero(alive_b)
        idx_a = top_quantile_index_matrix(raw_a[:, ia], quantile=top_quantile)
        idx_b = top_quantile_index_matrix(raw_b[:, ib], quantile=top_quantile)
        sub = jaccard_matrix_from_indices(idx_a, idx_b)
        jaccard = np.zeros((feats_a.shape[1], feats_b.shape[1]), dtype=np.float32)
        jaccard[np.ix_(ia, ib)] = sub
        out["jaccard_topq"] = jaccard

    if alive_a is not None:
        out["alive_a"] = np.asarray(alive_a, dtype=bool)
    if alive_b is not None:
        out["alive_b"] = np.asarray(alive_b, dtype=bool)
    return out


# --------------------------------------------------------------------------- matching
def hungarian_match(
    sim: np.ndarray,
    *,
    alive_a: np.ndarray | None = None,
    alive_b: np.ndarray | None = None,
    threshold: float = 0.25,
) -> dict:
    """1:1 match maximizing similarity; drop pairs below ``threshold``.

    Returns matched pairs + unmatched ids for A and B.
    Kept for diagnostics; prefer ``mutual_nn_match`` for main analysis.
    """
    sim = np.asarray(sim, dtype=np.float64)
    da, db = sim.shape
    if alive_a is None:
        alive_a = np.ones(da, dtype=bool)
    if alive_b is None:
        alive_b = np.ones(db, dtype=bool)

    idx_a = np.flatnonzero(alive_a)
    idx_b = np.flatnonzero(alive_b)
    if idx_a.size == 0 or idx_b.size == 0:
        return {
            "matched_a": np.zeros(0, np.int64),
            "matched_b": np.zeros(0, np.int64),
            "matched_score": np.zeros(0, np.float32),
            "unmatched_a": idx_a.astype(np.int64),
            "unmatched_b": idx_b.astype(np.int64),
            "threshold": float(threshold),
            "method": "hungarian",
        }

    sub = sim[np.ix_(idx_a, idx_b)]
    r, c = linear_sum_assignment(-sub)
    scores = sub[r, c]
    keep = scores >= threshold

    matched_a = idx_a[r[keep]].astype(np.int64)
    matched_b = idx_b[c[keep]].astype(np.int64)
    matched_score = scores[keep].astype(np.float32)

    unmatched_a = np.setdiff1d(idx_a, matched_a, assume_unique=False)
    unmatched_b = np.setdiff1d(idx_b, matched_b, assume_unique=False)

    return {
        "matched_a": matched_a,
        "matched_b": matched_b,
        "matched_score": matched_score,
        "unmatched_a": unmatched_a.astype(np.int64),
        "unmatched_b": unmatched_b.astype(np.int64),
        "threshold": float(threshold),
        "method": "hungarian",
        "best_b_for_a": idx_b[sub.argmax(axis=1)].astype(np.int64),
        "best_score_a": sub.max(axis=1).astype(np.float32),
        "best_a_for_b": idx_a[sub.argmax(axis=0)].astype(np.int64),
        "best_score_b": sub.max(axis=0).astype(np.float32),
        "alive_a_ids": idx_a.astype(np.int64),
        "alive_b_ids": idx_b.astype(np.int64),
    }


def mutual_nn_match(
    sim: np.ndarray,
    *,
    alive_a: np.ndarray | None = None,
    alive_b: np.ndarray | None = None,
    threshold: float = 0.25,
) -> dict:
    """Mutual nearest-neighbor matches with similarity threshold.

    Feature ``a`` matches ``b`` only if::

        argmax_b' sim[a,b'] = b  AND  argmax_a' sim[a',b] = a  AND  sim[a,b] >= thr
    """
    sim = np.asarray(sim, dtype=np.float64)
    da, db = sim.shape
    if alive_a is None:
        alive_a = np.ones(da, dtype=bool)
    if alive_b is None:
        alive_b = np.ones(db, dtype=bool)

    idx_a = np.flatnonzero(alive_a)
    idx_b = np.flatnonzero(alive_b)
    empty = {
        "matched_a": np.zeros(0, np.int64),
        "matched_b": np.zeros(0, np.int64),
        "matched_score": np.zeros(0, np.float32),
        "unmatched_a": idx_a.astype(np.int64),
        "unmatched_b": idx_b.astype(np.int64),
        "threshold": float(threshold),
        "method": "mutual_nn",
        "alive_a_ids": idx_a.astype(np.int64),
        "alive_b_ids": idx_b.astype(np.int64),
    }
    if idx_a.size == 0 or idx_b.size == 0:
        return empty

    sub = sim[np.ix_(idx_a, idx_b)]
    best_b_local = sub.argmax(axis=1)
    best_a_local = sub.argmax(axis=0)
    best_score_a = sub.max(axis=1)
    best_score_b = sub.max(axis=0)

    matched_a_list: list[int] = []
    matched_b_list: list[int] = []
    matched_s_list: list[float] = []
    for i, j_loc in enumerate(best_b_local.tolist()):
        if best_a_local[j_loc] != i:
            continue
        s = float(sub[i, j_loc])
        if s < threshold:
            continue
        matched_a_list.append(int(idx_a[i]))
        matched_b_list.append(int(idx_b[j_loc]))
        matched_s_list.append(s)

    matched_a = np.asarray(matched_a_list, dtype=np.int64)
    matched_b = np.asarray(matched_b_list, dtype=np.int64)
    matched_score = np.asarray(matched_s_list, dtype=np.float32)
    unmatched_a = np.setdiff1d(idx_a, matched_a, assume_unique=False)
    unmatched_b = np.setdiff1d(idx_b, matched_b, assume_unique=False)

    return {
        "matched_a": matched_a,
        "matched_b": matched_b,
        "matched_score": matched_score,
        "unmatched_a": unmatched_a.astype(np.int64),
        "unmatched_b": unmatched_b.astype(np.int64),
        "threshold": float(threshold),
        "method": "mutual_nn",
        "best_b_for_a": idx_b[best_b_local].astype(np.int64),
        "best_score_a": best_score_a.astype(np.float32),
        "best_a_for_b": idx_a[best_a_local].astype(np.int64),
        "best_score_b": best_score_b.astype(np.float32),
        "alive_a_ids": idx_a.astype(np.int64),
        "alive_b_ids": idx_b.astype(np.int64),
    }


def match_quality_table(
    sim: np.ndarray,
    match: dict,
    *,
    jaccard: np.ndarray | None = None,
) -> list[dict]:
    """Per-match quality: Spearman, Jaccard, MNN flag, shuffled baseline percentile."""
    idx_a = match["alive_a_ids"]
    idx_b = match["alive_b_ids"]
    sub = sim[np.ix_(idx_a, idx_b)]
    flat = sub.ravel()
    flat_sorted = np.sort(flat)

    best_b = {
        int(a): int(b)
        for a, b in zip(match["alive_a_ids"], match["best_b_for_a"])
    }
    best_a = {
        int(b): int(a)
        for b, a in zip(match["alive_b_ids"], match["best_a_for_b"])
    }

    rows = []
    for a, b, s in zip(match["matched_a"], match["matched_b"], match["matched_score"]):
        a_i, b_i = int(a), int(b)
        # percentile among all alive pairwise similarities
        pct = float(np.searchsorted(flat_sorted, float(s), side="right") / max(flat_sorted.size, 1))
        mnn = best_b.get(a_i) == b_i and best_a.get(b_i) == a_i
        jac = None
        if jaccard is not None:
            jac = float(jaccard[a_i, b_i])
        rows.append(
            {
                "a": a_i,
                "b": b_i,
                "spearman": float(s),
                "jaccard_top1pct": jac,
                "mutual_nn": bool(mnn),
                "shuffled_baseline_percentile": pct,
            }
        )
    rows.sort(key=lambda r: -r["spearman"])
    return rows


def same_observation_activations(
    feats_by_model: dict[str, np.ndarray],
    match_triplet: list[dict],
    *,
    row_indices: np.ndarray | None = None,
    max_rows: int = 5000,
) -> dict:
    """For each matched triple, collect activation traces on shared rows.

    ``match_triplet`` entries::
        {"record": j, "reactive": k, "selfplay": m, "scores": {...}}
    """
    models = list(feats_by_model.keys())
    n = feats_by_model[models[0]].shape[0]
    if row_indices is None:
        rng = np.random.default_rng(0)
        row_indices = rng.choice(n, size=min(max_rows, n), replace=False)
        row_indices.sort()
    rows = np.asarray(row_indices, dtype=np.int64)

    out = {
        "row_indices": rows,
        "n_pairs": np.int32(len(match_triplet)),
    }
    for model, feats in feats_by_model.items():
        # [n_pairs, n_rows]
        mat = np.zeros((len(match_triplet), rows.size), dtype=np.float32)
        for i, trip in enumerate(match_triplet):
            if model not in trip:
                continue
            j = int(trip[model])
            mat[i] = np.asarray(feats[rows, j], dtype=np.float32)
        out[f"acts_{model}"] = mat
    # pack feature ids
    for model in models:
        out[f"feature_id_{model}"] = np.asarray(
            [int(t.get(model, -1)) for t in match_triplet], dtype=np.int64
        )
    return out


def build_triplet_matches(
    match_rec_rea: dict,
    match_rec_sp: dict,
    *,
    score_key: str = "matched_score",
) -> list[dict]:
    """Join record↔reactive and record↔selfplay matches on shared record ids."""
    rec_to_rea = {
        int(a): (int(b), float(s))
        for a, b, s in zip(
            match_rec_rea["matched_a"],
            match_rec_rea["matched_b"],
            match_rec_rea[score_key],
        )
    }
    rec_to_sp = {
        int(a): (int(b), float(s))
        for a, b, s in zip(
            match_rec_sp["matched_a"],
            match_rec_sp["matched_b"],
            match_rec_sp[score_key],
        )
    }
    triples = []
    for rec, (rea, s_rea) in rec_to_rea.items():
        if rec not in rec_to_sp:
            continue
        sp, s_sp = rec_to_sp[rec]
        triples.append(
            {
                "record": rec,
                "reactive": rea,
                "selfplay": sp,
                "score_record_reactive": s_rea,
                "score_record_selfplay": s_sp,
                "score_min": min(s_rea, s_sp),
            }
        )
    triples.sort(key=lambda t: -t["score_min"])
    return triples


def record_specific_report(
    match_rec_rea: dict,
    match_rec_sp: dict,
    *,
    purity: dict[int, float] | None = None,
    density: np.ndarray | None = None,
) -> list[dict]:
    """Features matched poorly to both Reactive and SP (ReCord-specific)."""
    # Use best_score over all alive (including below threshold unmatched)
    best_rea = {
        int(a): float(s)
        for a, s in zip(match_rec_rea["alive_a_ids"], match_rec_rea["best_score_a"])
    }
    best_sp = {
        int(a): float(s)
        for a, s in zip(match_rec_sp["alive_a_ids"], match_rec_sp["best_score_a"])
    }
    unmatched = set(int(x) for x in match_rec_rea["unmatched_a"]) & set(
        int(x) for x in match_rec_sp["unmatched_a"]
    )
    rows = []
    for fid in sorted(unmatched):
        rows.append(
            {
                "record_feature": fid,
                "best_reactive_sim": best_rea.get(fid),
                "best_selfplay_sim": best_sp.get(fid),
                "density": float(density[fid]) if density is not None else None,
                "top_scene_conflict_purity": purity.get(fid) if purity else None,
            }
        )
    rows.sort(
        key=lambda r: (
            -((r["top_scene_conflict_purity"] or 0.0)),
            (r["best_reactive_sim"] or 0.0) + (r["best_selfplay_sim"] or 0.0),
        )
    )
    return rows


def run_pair(
    analysis_dir: Path,
    model_a: str,
    model_b: str,
    *,
    normalize: str,
    metric: str,
    threshold: float,
    top_quantile: float,
    min_density: float,
    out_dir: Path,
    match_method: str = "mutual_nn",
) -> dict:
    exp_a = resolve_model_name(model_a)
    exp_b = resolve_model_name(model_b)
    dir_a = analysis_dir / exp_a
    dir_b = analysis_dir / exp_b
    print(f"\n=== {model_a}({exp_a}) ↔ {model_b}({exp_b}) ===")

    feats_a = load_feature_acts(dir_a)
    feats_b = load_feature_acts(dir_b)
    if feats_a.shape[0] != feats_b.shape[0]:
        raise RuntimeError(
            f"row count mismatch: {exp_a} N={feats_a.shape[0]} vs {exp_b} N={feats_b.shape[0]}"
        )
    alive_a = load_alive_mask(dir_a, min_density=min_density)
    alive_b = load_alive_mask(dir_b, min_density=min_density)
    print(f"  alive: {alive_a.sum()}/{alive_a.size} vs {alive_b.sum()}/{alive_b.size}")

    bundle = compute_similarity_bundle(
        feats_a,
        feats_b,
        normalize=normalize,
        top_quantile=top_quantile,
        alive_a=alive_a,
        alive_b=alive_b,
        compute_jaccard=True,
    )
    if metric not in bundle:
        raise KeyError(f"metric {metric} not in similarity bundle keys {list(bundle)}")
    sim = bundle[metric]
    if match_method == "hungarian":
        match = hungarian_match(sim, alive_a=alive_a, alive_b=alive_b, threshold=threshold)
    else:
        match = mutual_nn_match(sim, alive_a=alive_a, alive_b=alive_b, threshold=threshold)

    quality = match_quality_table(
        sim, match, jaccard=bundle.get("jaccard_topq")
    )

    pair_name = f"{exp_a}__vs__{exp_b}"
    pair_dir = out_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pair_dir / "similarity.npz", **bundle)
    np.savez_compressed(pair_dir / "matching.npz", **{k: v for k, v in match.items() if not isinstance(v, str)})
    (pair_dir / "match_quality.json").write_text(json.dumps(quality, indent=2))

    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "exp_a": exp_a,
        "exp_b": exp_b,
        "normalize": normalize,
        "metric": metric,
        "match_method": match_method,
        "threshold": threshold,
        "n_alive_a": int(alive_a.sum()),
        "n_alive_b": int(alive_b.sum()),
        "n_matched": int(match["matched_a"].shape[0]),
        "n_unmatched_a": int(match["unmatched_a"].shape[0]),
        "n_unmatched_b": int(match["unmatched_b"].shape[0]),
        "matched_score_mean": float(match["matched_score"].mean())
        if match["matched_score"].size
        else None,
        "matched_score_median": float(np.median(match["matched_score"]))
        if match["matched_score"].size
        else None,
        "matched_jaccard_mean": float(
            np.nanmean([r["jaccard_top1pct"] for r in quality if r["jaccard_top1pct"] is not None])
        )
        if quality
        else None,
        "matched_mnn_frac": float(np.mean([r["mutual_nn"] for r in quality])) if quality else None,
        "matched_baseline_pct_mean": float(
            np.mean([r["shuffled_baseline_percentile"] for r in quality])
        )
        if quality
        else None,
        "top_matches": quality[:50],
    }
    (pair_dir / "matching_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  method={match_method} matched={summary['n_matched']}  "
        f"unmatched_a={summary['n_unmatched_a']} unmatched_b={summary['n_unmatched_b']}  "
        f"mean_spearman={summary['matched_score_mean']}  "
        f"mean_jaccard={summary['matched_jaccard_mean']}  "
        f"mean_baseline_pct={summary['matched_baseline_pct_mean']}"
    )
    return {"summary": summary, "match": match, "bundle": bundle, "pair_dir": pair_dir, "quality": quality}


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-model SAE feature matching")
    p.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help="Dir with <exp>/feature_acts.npy from top_activation_retrieval",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--normalize", choices=("rank", "zscore", "none"), default="rank")
    p.add_argument(
        "--metric",
        choices=("spearman", "cosine", "pearson", "jaccard_topq"),
        default="spearman",
    )
    p.add_argument("--match-threshold", type=float, default=0.25)
    p.add_argument(
        "--match-method",
        choices=("mutual_nn", "hungarian"),
        default="mutual_nn",
        help="Main analysis uses mutual nearest-neighbor + threshold",
    )
    p.add_argument("--top-quantile", type=float, default=0.99)
    p.add_argument("--min-density", type=float, default=0.0)
    p.add_argument(
        "--pairs",
        type=str,
        default="record,reactive;record,selfplay;reactive,selfplay",
        help="Semicolon-separated pairs of model aliases",
    )
    args = p.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_specs = []
    for chunk in args.pairs.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = [x.strip() for x in chunk.split(",")]
        pair_specs.append((a, b))

    results = {}
    for a, b in pair_specs:
        key = f"{a}__{b}"
        results[key] = run_pair(
            analysis_dir,
            a,
            b,
            normalize=args.normalize,
            metric=args.metric,
            threshold=args.match_threshold,
            top_quantile=args.top_quantile,
            min_density=args.min_density,
            out_dir=out_dir,
            match_method=args.match_method,
        )

    # Triplet join when both record pairs exist
    if "record__reactive" in results and "record__selfplay" in results:
        triples = build_triplet_matches(
            results["record__reactive"]["match"],
            results["record__selfplay"]["match"],
        )
        (out_dir / "triplet_matches.json").write_text(json.dumps(triples, indent=2))
        print(f"\nTriplet matches (record↔reactive↔selfplay, {args.match_method}): {len(triples)}")

        dens = None
        dens_path = analysis_dir / resolve_model_name("record") / "feature_stats.npz"
        if dens_path.is_file():
            dens = np.load(dens_path)["density"]
        specific = record_specific_report(
            results["record__reactive"]["match"],
            results["record__selfplay"]["match"],
            density=dens,
        )
        (out_dir / "record_specific_features.json").write_text(
            json.dumps(specific, indent=2)
        )
        print(f"ReCord-specific (unmatched to both under {args.match_method}): {len(specific)}")

        # Same-observation activation pack for top triples
        feats = {
            "record": load_feature_acts(analysis_dir / resolve_model_name("record")),
            "reactive": load_feature_acts(analysis_dir / resolve_model_name("reactive")),
            "selfplay": load_feature_acts(analysis_dir / resolve_model_name("selfplay")),
        }
        feats_norm = {
            k: normalize_features(v, method=args.normalize) for k, v in feats.items()
        }
        top_trips = triples[: min(100, len(triples))]
        same = same_observation_activations(feats_norm, top_trips)
        np.savez_compressed(out_dir / "same_obs_matched_acts.npz", **same)
        print(f"Wrote same_obs_matched_acts.npz for {len(top_trips)} triples")

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "normalize": args.normalize,
                "metric": args.metric,
                "match_method": args.match_method,
                "match_threshold": args.match_threshold,
                "pairs": pair_specs,
                "aliases": {k: MODEL_ALIASES[k] for k in ("record", "reactive", "selfplay")},
            },
            indent=2,
        )
    )
    print(f"\nMatching outputs → {out_dir}")


if __name__ == "__main__":
    main()
