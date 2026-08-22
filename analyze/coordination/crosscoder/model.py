"""Label-free sparse Crosscoder and per-policy sparse coding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Crosscoder(nn.Module):
    """Paired-stream dictionary.

    train_mode:
      shared   — z = ReLU(W_r h_r + W_a h_a + b); both streams reconstructed from z
      separate — z_r = ReLU(W_r h_r + b), z_a = ReLU(W_a h_a + b); independent recon
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dict_size: int = 1024,
        train_mode: str = "shared",
        topk: int | None = None,
    ):
        super().__init__()
        if train_mode not in ("shared", "separate"):
            raise ValueError(f"train_mode must be shared|separate, got {train_mode!r}")
        self.hidden_dim = int(hidden_dim)
        self.dict_size = int(dict_size)
        self.train_mode = train_mode
        self.topk = int(topk) if topk is not None else None
        self.enc_r = nn.Linear(hidden_dim, dict_size, bias=False)
        self.enc_a = nn.Linear(hidden_dim, dict_size, bias=False)
        self.bias_e = nn.Parameter(torch.zeros(dict_size))
        self.bias_e_a = nn.Parameter(torch.zeros(dict_size))
        self.dec_r = nn.Linear(dict_size, hidden_dim, bias=True)
        self.dec_a = nn.Linear(dict_size, hidden_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.enc_r.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.enc_a.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.dec_r.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.dec_a.weight, a=5**0.5)
        nn.init.zeros_(self.dec_r.bias)
        nn.init.zeros_(self.dec_a.bias)
        nn.init.zeros_(self.bias_e)
        nn.init.zeros_(self.bias_e_a)

    @staticmethod
    def _topk(z: torch.Tensor, k: int | None) -> torch.Tensor:
        if k is None or k >= z.shape[-1]:
            return z
        vals, idx = torch.topk(z, int(k), dim=-1)
        out = torch.zeros_like(z)
        return out.scatter(-1, idx, vals)

    def encode_shared(self, h_r: torch.Tensor, h_a: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.enc_r(h_r) + self.enc_a(h_a) + self.bias_e)
        return self._topk(z, self.topk)

    def encode_separate(self, h: torch.Tensor, which: str) -> torch.Tensor:
        if which == "record":
            z = F.relu(self.enc_r(h) + self.bias_e)
        else:
            z = F.relu(self.enc_a(h) + self.bias_e_a)
        return self._topk(z, self.topk)

    def encode(self, h_r: torch.Tensor, h_a: torch.Tensor) -> torch.Tensor:
        return self.encode_shared(h_r, h_a)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dec_r(z), self.dec_a(z)

    def forward(self, h_r: torch.Tensor, h_a: torch.Tensor):
        if self.train_mode == "separate":
            z_r = self.encode_separate(h_r, "record")
            z_a = self.encode_separate(h_a, "reactive")
            return z_r, z_a, self.dec_r(z_r), self.dec_a(z_a)
        z = self.encode_shared(h_r, h_a)
        rec_r, rec_a = self.decode(z)
        return z, z, rec_r, rec_a

    def loss(
        self,
        h_r: torch.Tensor,
        h_a: torch.Tensor,
        *,
        l1_coeff: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z_r, z_a, rec_r, rec_a = self.forward(h_r, h_a)
        recon_r = (h_r - rec_r).pow(2).mean()
        recon_a = (h_a - rec_a).pow(2).mean()
        recon = recon_r + recon_a
        l1 = 0.5 * (z_r.abs().mean() + z_a.abs().mean())
        total = recon + float(l1_coeff) * l1
        with torch.no_grad():
            z_cat = torch.stack([z_r, z_a], dim=0)
            active = (z_cat > 0).float().mean()
            l0 = 0.5 * ((z_r > 0).float().sum(dim=-1).mean() + (z_a > 0).float().sum(dim=-1).mean())
        stats = {
            "loss": float(total.detach()),
            "recon": float(recon.detach()),
            "recon_record": float(recon_r.detach()),
            "recon_reactive": float(recon_a.detach()),
            "l1": float(l1.detach()),
            "frac_active": float(active),
            "l0": float(l0),
        }
        return total, stats

    @torch.no_grad()
    def normalize_decoders(self) -> None:
        """Unit-column decoders; rescale matching encoder rows."""
        for enc, dec in ((self.enc_r, self.dec_r), (self.enc_a, self.dec_a)):
            col_norm = dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            dec.weight.div_(col_norm)
            enc.weight.mul_(col_norm.T)

    @torch.no_grad()
    def resample_dead_features(self, fired: torch.Tensor, h_r: torch.Tensor, h_a: torch.Tensor) -> int:
        """Reinit decoder columns that did not fire this epoch. fired: (dict_size,) bool."""
        dead = ~fired.bool()
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        n = h_r.shape[0]
        pick = torch.randint(0, n, (n_dead,), device=h_r.device)
        for which, h in (("record", h_r), ("reactive", h_a)):
            dec = self.dec_r if which == "record" else self.dec_a
            enc = self.enc_r if which == "record" else self.enc_a
            vecs = h[pick]
            vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dec.weight[:, dead] = vecs.T
            enc.weight[dead] = vecs
        self.normalize_decoders()
        return n_dead

    def decoder_columns(self, which: str) -> torch.Tensor:
        """Feature vectors as (dict_size, hidden_dim)."""
        layer = self.dec_r if which == "record" else self.dec_a
        return layer.weight.detach().T.contiguous()

    def decoder_bias(self, which: str) -> torch.Tensor:
        layer = self.dec_r if which == "record" else self.dec_a
        return layer.bias.detach()


def reconstruct_from_codes(
    codes: torch.Tensor,
    decoder_cols: torch.Tensor,
    decoder_bias: torch.Tensor | None,
) -> torch.Tensor:
    rec = codes @ decoder_cols
    if decoder_bias is not None:
        rec = rec + decoder_bias.view(1, -1)
    return rec


@torch.no_grad()
def ista_sparse_code(
    h: torch.Tensor,
    decoder_cols: torch.Tensor,
    decoder_bias: torch.Tensor | None,
    *,
    l1_coeff: float,
    n_iters: int = 80,
    step_size: float | None = None,
) -> torch.Tensor:
    """FISTA-style ISTA: min ||h - D c - b||^2 + λ||c||_1, c >= 0.

    decoder_cols: (dict_size, hidden_dim)
    h: (N, hidden_dim)
    """
    d = decoder_cols.float()
    n, _hid = h.shape
    k = d.shape[0]
    residual_h = h.float()
    if decoder_bias is not None:
        residual_h = residual_h - decoder_bias.float().view(1, -1)
    gram = d @ d.T
    lip = torch.linalg.eigvalsh(gram).max().clamp_min(1e-4)
    lr = float(step_size) if step_size is not None else float(1.0 / (2.2 * lip.item()))
    c = torch.zeros(n, k, device=h.device, dtype=torch.float32)
    y = c.clone()
    t_prev = 1.0
    dt = d.T
    thresh = float(l1_coeff) * lr
    for _ in range(int(n_iters)):
        rec = y @ d
        grad = (rec - residual_h) @ dt
        c_next = F.relu(y - lr * grad - thresh)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_prev**2) ** 0.5)
        y = c_next + ((t_prev - 1.0) / t_next) * (c_next - c)
        c = c_next
        t_prev = t_next
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).clamp(max=50.0)
