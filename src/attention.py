"""
Single-head self-attention (the core mechanism, hand-rolled).

Plain-language recap:
- Every token produces three vectors from the same input: Query (what am I looking
  for?), Key (what do I contain?), Value (what info do I pass along if picked?).
- We compare every token's Query to every OTHER token's Key (dot product) to get a
  relevance score per pair.
- We scale the scores down (divide by sqrt(head_size)) so they don't get too large and
  destabilize training as the vector size grows.
- For a GPT-style model, a token must never see FUTURE tokens (that would be like
  reading the answer before the question). We enforce this with a "causal mask": scores
  pointing to future tokens are set to -infinity BEFORE softmax, so after softmax they
  become exactly 0% attention.
- softmax turns the scores into percentages that sum to 100% per token ("how much
  attention goes to each earlier token").
- We use those percentages to blend the Value vectors together -- that blend is this
  token's attention output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, max_sequence_len: int):
        super().__init__()
        # Three separate learned projections of the same input embedding.
        # bias=False matches the standard transformer attention formulation.
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.head_size = head_size

        # A fixed (non-learned) lower-triangular matrix of 1s and 0s -- our causal mask
        # blueprint. register_buffer means it moves with the model to GPU/MPS but is not
        # a trainable parameter.
        causal_mask = torch.tril(torch.ones(max_sequence_len, max_sequence_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, sequence_len, embed_dim)
        returns: shape (batch_size, sequence_len, head_size)
        """
        batch_size, sequence_len, _ = x.shape

        q = self.query(x)  # (batch, seq_len, head_size) -- "what am I looking for"
        k = self.key(x)    # (batch, seq_len, head_size) -- "what do I contain"
        v = self.value(x)  # (batch, seq_len, head_size) -- "what do I pass along"

        # Compare every Query to every Key: (batch, seq_len, seq_len) relevance scores.
        # k.transpose swaps the last two dims so the matrix multiply lines up correctly.
        scores = q @ k.transpose(-2, -1)

        # Scale down so scores don't explode in magnitude as head_size grows.
        scores = scores / math.sqrt(self.head_size)

        # Apply the causal mask: anywhere the mask is 0 (a future position), set the
        # score to -infinity so softmax turns it into exactly 0.
        mask = self.causal_mask[:sequence_len, :sequence_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Turn scores into percentages that sum to 1 per row (per query token).
        attention_weights = F.softmax(scores, dim=-1)

        # Blend the Value vectors together, weighted by those percentages.
        return attention_weights @ v


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import CharTokenizer
    from embeddings import TokenAndPositionEmbedding

    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")

    embed_dim = 32
    head_size = 16
    max_seq_len = 16

    embedder = TokenAndPositionEmbedding(tok.vocab_size, max_seq_len, embed_dim)
    attention = SingleHeadAttention(embed_dim, head_size, max_seq_len)

    sample = "To be, or not"
    ids = torch.tensor([tok.encode(sample)])

    x = embedder(ids)
    out = attention(x)

    print(f"Input text: {sample!r} ({len(sample)} characters)")
    print(f"Embedding shape: {tuple(x.shape)}  (batch, seq_len, embed_dim)")
    print(f"Attention output shape: {tuple(out.shape)}  (batch, seq_len, head_size)")

    # Sanity check the causal mask is actually working: print the attention weights
    # for the LAST token and confirm they sum to 1 and every value is non-negative.
    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        mask = attention.causal_mask[: len(sample), : len(sample)]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

    last_token_weights = weights[0, -1]
    print()
    print(f"Attention weights for the last token ('{sample[-1]}'), over all {len(sample)} tokens:")
    print(last_token_weights)
    print(f"Sum of weights (should be ~1.0): {last_token_weights.sum().item():.6f}")

    first_token_weights = weights[0, 0]
    print()
    print(f"Attention weights for the FIRST token ('{sample[0]}') -- should be [1, 0, 0, ...]")
    print("since it can only attend to itself (nothing came before it):")
    print(first_token_weights)
