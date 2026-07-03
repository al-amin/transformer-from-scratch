"""
Transformer block: the repeating unit that gets stacked N times to make the full model.

Plain-language idea:
One block = attention (gather info from other tokens) + feed-forward (each token thinks
it over on its own), each wrapped with two things:
- LAYER NORM before it runs ("pre-norm") -- rescales the input to a consistent, well-
  behaved range so training stays stable as we stack many blocks.
- A RESIDUAL CONNECTION around it (output = input + sublayer(normalized_input)) -- a
  direct shortcut so gradients (and information) can flow straight through during
  training, even through a deep stack of blocks, instead of getting diluted at every
  layer.

Stacking several of these blocks is what lets the model build up increasingly rich
understanding: block 1 might learn simple local patterns, block 4 might combine those
into more abstract relationships.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_sequence_len: int):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, max_sequence_len)
        self.feed_forward = FeedForward(embed_dim)

        # One LayerNorm before attention, one before feed-forward ("pre-norm" style --
        # the modern standard used since GPT-2, more stable to train than the original
        # 2017 "Attention Is All You Need" paper's post-norm design).
        self.norm_before_attention = nn.LayerNorm(embed_dim)
        self.norm_before_feedforward = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, sequence_len, embed_dim)
        returns: same shape -- this is what lets us stack blocks on top of each other
        """
        # Residual connection #1: normalize, run attention, add the ORIGINAL x back on.
        x = x + self.attention(self.norm_before_attention(x))

        # Residual connection #2: normalize, run feed-forward, add x back on again.
        x = x + self.feed_forward(self.norm_before_feedforward(x))

        return x


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import CharTokenizer
    from embeddings import TokenAndPositionEmbedding

    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")

    embed_dim = 32
    num_heads = 4
    max_seq_len = 16

    embedder = TokenAndPositionEmbedding(tok.vocab_size, max_seq_len, embed_dim)
    block = TransformerBlock(embed_dim, num_heads, max_seq_len)

    sample = "To be, or not"
    ids = torch.tensor([tok.encode(sample)])

    x = embedder(ids)
    out = block(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (same shape -- proves blocks can be stacked)")

    # Prove blocks genuinely stack: run the output through a SECOND, independent block.
    block2 = TransformerBlock(embed_dim, num_heads, max_seq_len)
    out2 = block2(out)
    print(f"After a 2nd block: {tuple(out2.shape)}  (still the same shape)")

    num_params = sum(p.numel() for p in block.parameters())
    print(f"Learnable parameters in one block: {num_params:,}")
