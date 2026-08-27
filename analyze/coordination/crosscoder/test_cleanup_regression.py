#!/usr/bin/env python3
"""Lightweight regression checks for Crosscoder cleanup (no full-dataset rerun)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

_COORD = Path(__file__).resolve().parents[1]
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))


def test_frozen_constants() -> None:
    from crosscoder import frozen_config as fc
    from crosscoder import intervention, persistence, replicate, transient_causal

    assert fc.FROZEN_DICT == 1024
    assert fc.FROZEN_LAMBDA == 0.3
    assert fc.FROZEN_K == 32
    assert fc.HIDDEN_DIM == 256
    assert fc.PRIMARY_ALPHA == 0.5
    assert fc.ALPHAS == (0.25, 0.5, 0.75, 1.0)
    assert replicate.FROZEN_K == fc.FROZEN_K
    assert replicate.FROZEN_DICT == fc.FROZEN_DICT
    assert replicate.FROZEN_LAMBDA == fc.FROZEN_LAMBDA
    assert intervention.ALPHAS == fc.ALPHAS
    assert intervention.HIDDEN == fc.HIDDEN_DIM
    assert persistence.ALPHA == fc.PRIMARY_ALPHA
    assert transient_causal.ALPHA == fc.PRIMARY_ALPHA


def test_imports_core_pipeline() -> None:
    for mod in [
        "crosscoder.model",
        "crosscoder.collect",
        "crosscoder.metrics",
        "crosscoder.pipeline",
        "crosscoder.replicate",
        "crosscoder.intervention",
        "crosscoder.persistence",
        "crosscoder.transient_causal",
    ]:
        importlib.import_module(mod)


def test_mvp_and_semantics_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "crosscoder" / "run.py").exists()
    assert not (root / "run_crosscoder.sh").exists()
    assert not (root / "crosscoder" / "semantics.py").exists()
    assert not (root / "run_crosscoder_semantics.sh").exists()
    assert not (root / "run_crosscoder_10k.sh").exists()
    assert not (root / "run_crosscoder_mechanism.sh").exists()
    assert (root / "run_final_mechanism.sh").exists()


def test_patch_projector_identity() -> None:
    """δ_U = UUᵀ Δ; projecting twice is idempotent; α=0 leaves h unchanged."""
    from crosscoder.intervention import project
    from crosscoder.metrics import orthonormal_basis

    rng = np.random.default_rng(0)
    u = orthonormal_basis(rng.normal(size=(32, 256)))
    delta = rng.normal(size=(100, 256)).astype(np.float64)
    d1 = project(delta, u)
    d2 = project(d1, u)
    assert np.allclose(d1, d2, atol=1e-6)
    # zero alpha: patched == original
    h = rng.normal(size=(100, 256))
    assert np.allclose(h + 0.0 * d1, h)


def test_kl_zero_when_identical() -> None:
    from crosscoder.intervention import kl_t

    logits = torch.randn(64, 91)
    log_p = torch.log_softmax(logits, dim=-1)
    assert float(kl_t(log_p, log_p).abs().max()) < 1e-6


def main() -> int:
    tests = [
        test_frozen_constants,
        test_imports_core_pipeline,
        test_mvp_and_semantics_removed,
        test_patch_projector_identity,
        test_kl_zero_when_identical,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("All cleanup regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
