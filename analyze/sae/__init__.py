"""Sparse autoencoders over Drive partner-encoder representations."""

from .sae_model import SAEConfig, SparseAutoencoder, TopK, build_sae

__all__ = ["SAEConfig", "SparseAutoencoder", "TopK", "build_sae"]

# Collect entrypoint lives in collect_sae_activations.py (heavy LP deps).
