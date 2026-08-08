"""Sparse Autoencoder (SAE) for Drive representations.

Self-contained PyTorch module inspired by SAELens
(https://github.com/decoderesearch/SAELens): same encode/decode/forward
interface and Standard / TopK architectures, without TransformerLens hooks.

Intended input is the partner-encoder slot embedding saved as
``activation__*`` in SAE ``activations.npz`` files
(see ``analyze/sae/collect_sae_activations.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Architecture = Literal["standard", "topk"]


@dataclass
class SAEConfig:
    """Configuration for :class:`SparseAutoencoder`.

    Attributes match the SAELens naming where practical:
    ``d_in``, ``d_sae``, ``architecture``, ``apply_b_dec_to_input``, etc.
    """

    d_in: int
    d_sae: int
    architecture: Architecture = "standard"
    # Standard SAE: ReLU + L1 on decoder-norm-weighted activations
    l1_coefficient: float = 1e-3
    lp_norm: float = 1.0
    # TopK SAE: keep only the top-k pre-activations after ReLU
    k: int = 32
    aux_loss_coefficient: float = 1.0
    # Shared
    apply_b_dec_to_input: bool = True
    normalize_decoder: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    # Optional metadata (not used by the module itself)
    repr_layer: str = "partner_encoder_slot"
    experiment: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SAEConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class TopK(nn.Module):
    """Keep the top-k values along the last dim; ReLU then scatter back.

    Matches SAELens ``TopK`` (dense variant).
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = min(self.k, x.shape[-1])
        values, indices = torch.topk(x, k=k, dim=-1, sorted=False)
        values = values.relu()
        out = torch.zeros_like(x)
        out.scatter_(-1, indices, values)
        return out


class SparseAutoencoder(nn.Module):
    """Standard or TopK sparse autoencoder (SAELens-style).

    Parameters
    ----------
    cfg:
        :class:`SAEConfig`. ``architecture="standard"`` uses ReLU + L1;
        ``architecture="topk"`` uses structural TopK sparsity.

    Notes
    -----
    Weight layout follows SAELens:

    - ``W_enc``: ``(d_in, d_sae)``
    - ``W_dec``: ``(d_sae, d_in)``  (rows are feature dictionaries)
    - ``b_enc``: ``(d_sae,)``
    - ``b_dec``: ``(d_in,)``

    Forward:

    .. code-block:: python

        x_hat = sae(x)                 # encode + decode
        feats = sae.encode(x)          # sparse features
        x_hat = sae.decode(feats)
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        if cfg.d_in <= 0 or cfg.d_sae <= 0:
            raise ValueError(f"d_in and d_sae must be > 0, got {cfg.d_in=}, {cfg.d_sae=}")
        if cfg.architecture == "topk" and cfg.k <= 0:
            raise ValueError(f"topk SAE requires k > 0, got {cfg.k=}")

        self.cfg = cfg
        dtype = cfg.torch_dtype()
        device = torch.device(cfg.device)

        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae, dtype=dtype, device=device))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=dtype, device=device))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, dtype=dtype, device=device))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, dtype=dtype, device=device))

        if cfg.architecture == "topk":
            self.activation_fn: nn.Module = TopK(cfg.k)
        else:
            self.activation_fn = nn.ReLU()

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        """SAELens-style init: unit-norm decoder rows, encoder ≈ decoder^T."""
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)
            self.W_enc.data = self.W_dec.data.T.clone()
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder rows to unit L2 norm (and rescale encoder / b_enc)."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)
        self.W_enc.mul_(norms.T)
        self.b_enc.mul_(norms.squeeze(1))

    # -------------------------------------------------------------- encode/decode
    def process_sae_in(self, x: torch.Tensor) -> torch.Tensor:
        """Cast + optional ``x - b_dec`` centering (SAELens ``process_sae_in``)."""
        x = x.to(dtype=self.W_enc.dtype, device=self.W_enc.device)
        if self.cfg.apply_b_dec_to_input:
            x = x - self.b_dec
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activations ``(..., d_in)`` → sparse features ``(..., d_sae)``."""
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return self.activation_fn(hidden_pre)

    def encode_with_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(feature_acts, hidden_pre)`` for training / diagnostics."""
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return self.activation_fn(hidden_pre), hidden_pre

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Map sparse features ``(..., d_sae)`` → reconstruction ``(..., d_in)``."""
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencode: ``decode(encode(x))``."""
        return self.decode(self.encode(x))

    # --------------------------------------------------------------- training
    def training_loss(
        self,
        x: torch.Tensor,
        *,
        l1_coefficient: float | None = None,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction + sparsity (or TopK aux) losses.

        Returns a dict with at least ``loss``, ``mse_loss``, ``l0``, and
        architecture-specific terms (``l1_loss`` or ``aux_loss``).
        """
        feature_acts, hidden_pre = self.encode_with_pre(x)
        # encode may center with b_dec; reconstruction target is the original x.
        x_hat = self.decode(feature_acts)

        mse = F.mse_loss(x_hat, x)
        l0 = (feature_acts > 0).float().sum(dim=-1).mean()

        out: dict[str, torch.Tensor] = {
            "mse_loss": mse,
            "l0": l0,
            "feature_acts": feature_acts,
            "hidden_pre": hidden_pre,
            "x_hat": x_hat,
        }

        if self.cfg.architecture == "standard":
            coef = self.cfg.l1_coefficient if l1_coefficient is None else l1_coefficient
            # Decoder-norm-weighted L1 (SAELens standard aux loss).
            weighted = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted.norm(p=self.cfg.lp_norm, dim=-1).mean()
            l1_loss = coef * sparsity
            out["l1_loss"] = l1_loss
            out["loss"] = mse + l1_loss
        else:
            aux = self._topk_aux_loss(
                x=x,
                x_hat=x_hat,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            out["aux_loss"] = aux
            out["loss"] = mse + aux

        return out

    def _topk_aux_loss(
        self,
        *,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dead-neuron auxiliary reconstruction (SAELens TopK aux)."""
        if dead_neuron_mask is None or int(dead_neuron_mask.sum()) == 0:
            return x_hat.new_tensor(0.0)

        residual = (x - x_hat).detach()
        num_dead = int(dead_neuron_mask.sum())
        k_aux = max(x.shape[-1] // 2, 1)
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Broadcast mask over batch: (d_sae,) -> (..., d_sae)
        mask = dead_neuron_mask.to(device=hidden_pre.device, dtype=torch.bool)
        masked = torch.where(mask, hidden_pre, torch.full_like(hidden_pre, -float("inf")))
        topk = masked.topk(k_aux, sorted=False)
        aux_acts = torch.zeros_like(hidden_pre)
        aux_acts.scatter_(-1, topk.indices, topk.values.relu())
        # No b_dec on aux decode (SAELens Appendix A.2)
        recons = aux_acts @ self.W_dec
        aux = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * aux

    # ------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def feature_density(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Per-feature fraction of samples with activation > 0. Shape ``(d_sae,)``."""
        return (feature_acts > 0).float().mean(dim=0)

    @torch.no_grad()
    def explained_variance(self, x: torch.Tensor, x_hat: torch.Tensor | None = None) -> torch.Tensor:
        """``1 - Var(x - x_hat) / Var(x)`` over the last dim, then mean."""
        if x_hat is None:
            x_hat = self.forward(x)
        resid = x - x_hat
        total_var = x.var(dim=-1).clamp_min(1e-12)
        resid_var = resid.var(dim=-1)
        return (1.0 - resid_var / total_var).mean()

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> SparseAutoencoder:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = SAEConfig.from_dict(ckpt["cfg"])
        if map_location is not None:
            cfg.device = str(map_location)
        sae = cls(cfg)
        sae.load_state_dict(ckpt["state_dict"])
        return sae

    def extra_repr(self) -> str:
        c = self.cfg
        extra = f", k={c.k}" if c.architecture == "topk" else f", l1={c.l1_coefficient}"
        return f"arch={c.architecture}, d_in={c.d_in}, d_sae={c.d_sae}{extra}"


def build_sae(
    d_in: int,
    d_sae: int | None = None,
    *,
    expansion: int = 16,
    architecture: Architecture = "standard",
    k: int = 32,
    l1_coefficient: float = 1e-3,
    device: str = "cpu",
    **kwargs: Any,
) -> SparseAutoencoder:
    """Convenience constructor. Default ``d_sae = expansion * d_in``."""
    if d_sae is None:
        d_sae = expansion * d_in
    cfg = SAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        architecture=architecture,
        k=k,
        l1_coefficient=l1_coefficient,
        device=device,
        **kwargs,
    )
    return SparseAutoencoder(cfg)
