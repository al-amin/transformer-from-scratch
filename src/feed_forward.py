"""
Feed-forward layer.

Plain-language idea:
Attention mixes information ACROSS tokens (each token looks at others and blends their
Values in). The feed-forward layer does the opposite: it processes each token entirely
on its own, with no mixing across tokens. It's a small 2-layer neural net: expand the
vector to a bigger size (usually 4x), apply a non-linearity (ReLU/GELU -- lets the model
learn more complex functions than a plain linear map could), then shrink back down to
the original size. Think of attention as "gathering information from your neighbors"
and feed-forward as "privately thinking over what you just gathered."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, sequence_len, embed_dim)
        returns: same shape -- every token processed independently, no cross-token mixing
        """
        return self.net(x)


if __name__ == "__main__":
    embed_dim = 32
    ff = FeedForward(embed_dim)

    x = torch.randn(1, 13, embed_dim)  # pretend this came from attention
    out = ff(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (same as input -- one token in, one token out)")

    hidden_dim = embed_dim * 4
    print(f"Internal hidden size: {hidden_dim} (expanded {embed_dim} -> {hidden_dim} -> {embed_dim})")

    num_params = sum(p.numel() for p in ff.parameters())
    print(f"Learnable parameters in this layer: {num_params:,}")
