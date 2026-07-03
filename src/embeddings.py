"""
Embeddings: token embedding + positional embedding.

Plain-language idea:
- A tokenizer gives us integer IDs (e.g. 'h' -> 7). But 7 by itself carries no
  meaning -- it's just an arbitrary label. A TOKEN EMBEDDING replaces each ID with
  a small vector (a list of numbers, e.g. 32 numbers) that the model learns during
  training. Over time, characters that behave similarly in the text end up with
  similar vectors.
- Attention (which we build next) compares every token to every other token, but it
  has no built-in sense of ORDER -- it would treat "ab" and "ba" as the same set of
  characters. A POSITIONAL EMBEDDING fixes this: it's a second learned vector, one
  per position (position 0, position 1, position 2, ...), added on top of the token
  embedding, so the model can tell "this is the 1st character" from "this is the
  5th character".
- We add the two vectors together (token meaning + position) to get the final input
  the rest of the model works with.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_sequence_len: int, embed_dim: int):
        super().__init__()
        # One learned vector per possible character ID.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # One learned vector per possible position in the sequence.
        self.position_embedding = nn.Embedding(max_sequence_len, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: shape (batch_size, sequence_len) -- integer IDs
        returns:   shape (batch_size, sequence_len, embed_dim) -- vectors
        """
        batch_size, sequence_len = token_ids.shape

        # Look up the "meaning" vector for each character ID.
        token_vecs = self.token_embedding(token_ids)  # (batch, seq_len, embed_dim)

        # Look up the "position" vector for positions 0, 1, 2, ..., sequence_len-1.
        positions = torch.arange(sequence_len, device=token_ids.device)
        position_vecs = self.position_embedding(positions)  # (seq_len, embed_dim)

        # Add them together. Broadcasting handles the batch dimension automatically:
        # every sequence in the batch gets the same position vectors added.
        return token_vecs + position_vecs


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import CharTokenizer

    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")

    embed_dim = 32       # how many numbers describe each character's "meaning"
    max_seq_len = 16      # longest sequence we'll support for this demo

    embedder = TokenAndPositionEmbedding(
        vocab_size=tok.vocab_size, max_sequence_len=max_seq_len, embed_dim=embed_dim
    )

    sample = "To be, or not"  # 13 characters
    ids = torch.tensor([tok.encode(sample)])  # shape (1, 13) -- batch of 1

    vectors = embedder(ids)

    print(f"Input text: {sample!r} ({len(sample)} characters)")
    print(f"Token ID tensor shape:  {tuple(ids.shape)}  (batch_size, sequence_len)")
    print(f"Embedding output shape: {tuple(vectors.shape)}  (batch_size, sequence_len, embed_dim)")
    print()
    print(f"First character {sample[0]!r} is now a vector of {embed_dim} numbers, e.g.:")
    print(vectors[0, 0, :8], "... (showing first 8 of", embed_dim, "numbers)")
