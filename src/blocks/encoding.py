import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Added to the token embeddings once, at the input of the model.
    Input/Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # (ceil(D/2),)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[: d_model // 2]
        )  # handles odd d_model

        # Not a parameter (no gradients); persistent=False keeps it out of state_dict
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].to(x.dtype)
        return self.dropout(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and map (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotates query/key vectors by a position-dependent angle, so the attention
    score q_m . k_n depends only on the relative offset (m - n).

    Unlike sinusoidal encoding, it is applied inside every attention layer to
    q and k (never to v), and adds no learned parameters.

    Expected shapes: q, k = (batch, n_heads, seq_len, head_dim)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)  # (L, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """
        offset: position of the first token, e.g. the KV-cache length during
        incremental decoding.
        """
        seq_len = q.size(-2)
        if offset + seq_len > self.cos_cached.size(0):
            self._build_cache(offset + seq_len)

        cos = self.cos_cached[offset : offset + seq_len].to(q.dtype)  # (T, Dh)
        sin = self.sin_cached[offset : offset + seq_len].to(q.dtype)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot
