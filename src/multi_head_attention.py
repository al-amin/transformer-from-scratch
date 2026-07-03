"""
Multi-head self-attention.

Plain-language idea:
A single attention head can only learn one "kind" of relevance at a time (e.g. maybe
just word order). Multi-head attention splits the same total vector size into several
smaller, independent heads that run in parallel -- one might learn to track
subject/verb relationships, another punctuation patterns, another something no human
label quite captures. Each head does exactly the same Query/Key/Value math as
SingleHeadAttention, just on a smaller slice of the vector, and at the end we
concatenate all the heads' outputs back together into one vector of the original size.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_sequence_len: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly across heads"
        head_size = embed_dim // num_heads

        # One independent SingleHeadAttention per head, each working on a head_size-
        # sized slice. nn.ModuleList so PyTorch tracks all their parameters correctly.
        self.heads = nn.ModuleList(
            [SingleHeadAttention(embed_dim, head_size, max_sequence_len) for _ in range(num_heads)]
        )

        # After concatenating all heads back together, one more learned linear layer
        # to let the model mix information across heads before passing it onward.
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, sequence_len, embed_dim)
        returns: shape (batch_size, sequence_len, embed_dim) -- same shape as input
        """
        # Run every head on the full input, then concatenate their outputs along the
        # last dimension: num_heads * head_size = embed_dim again.
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        return self.output_projection(combined)


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
    mha = MultiHeadAttention(embed_dim, num_heads, max_seq_len)

    sample = "To be, or not"
    ids = torch.tensor([tok.encode(sample)])

    x = embedder(ids)
    out = mha(x)

    print(f"Input text: {sample!r} ({len(sample)} characters)")
    print(f"Embedding shape:        {tuple(x.shape)}  (batch, seq_len, embed_dim)")
    print(f"Multi-head output shape: {tuple(out.shape)}  (batch, seq_len, embed_dim)")
    print(f"Note: output shape matches input shape ({embed_dim}) -- this is what lets us")
    print(f"stack multiple attention blocks on top of each other later.")
    print(f"Number of heads: {num_heads}, each working on {embed_dim // num_heads} dimensions.")
