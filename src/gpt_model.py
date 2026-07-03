"""
The full tiny GPT model: stack of transformer blocks + final output layer.

Plain-language idea:
Everything we've built (embeddings, attention, feed-forward, transformer blocks) is
plumbing that leads to one question: "given the characters so far, what's the probability
of each possible next character?" This file assembles the full pipeline:

  text -> tokenize -> embed (meaning + position) -> N transformer blocks (each refining
  the understanding) -> final LayerNorm -> Linear layer -> one score per vocabulary
  character, for every position in the sequence.

Those final scores are called "logits" -- raw, un-normalized numbers. Running them
through softmax (usually during training/generation, not stored on the model) turns them
into actual probabilities that sum to 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import TokenAndPositionEmbedding
from transformer_block import TransformerBlock


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        max_sequence_len: int,
    ):
        super().__init__()
        self.max_sequence_len = max_sequence_len

        self.embed = TokenAndPositionEmbedding(vocab_size, max_sequence_len, embed_dim)

        # Stack N independent transformer blocks -- each one refines the representation
        # a bit further. This is literally just "repeat the same block shape N times."
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, max_sequence_len) for _ in range(num_blocks)]
        )

        # One final normalization before the output layer (standard GPT-2+ practice).
        self.final_norm = nn.LayerNorm(embed_dim)

        # Turn each token's final vector into one score per possible character.
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        token_ids: shape (batch_size, sequence_len)
        targets:   optional, same shape as token_ids -- the "correct next character" at
                   each position, used to compute a loss during training.

        returns: (logits, loss) -- loss is None if targets weren't provided.
        """
        x = self.embed(token_ids)  # (batch, seq_len, embed_dim)

        for block in self.blocks:
            x = block(x)  # still (batch, seq_len, embed_dim) -- proven stackable earlier

        x = self.final_norm(x)
        logits = self.output_layer(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, num_classes) and (N,), so we flatten the
            # batch and sequence dimensions together.
            batch_size, sequence_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_len, vocab_size),
                targets.view(batch_size * sequence_len),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Given a starting sequence, generate max_new_tokens more characters, one at a
        time. Each new character is SAMPLED from the model's predicted probability
        distribution (not just the single most likely one), so output varies run to run.
        """
        for _ in range(max_new_tokens):
            # The model only has learned position embeddings up to max_sequence_len,
            # so if the sequence is already too long, only feed it the most recent chunk.
            context = token_ids[:, -self.max_sequence_len :]

            logits, _ = self(context)

            # We only care about the prediction for the NEXT character, i.e. the last
            # position's logits.
            last_logits = logits[:, -1, :]  # (batch, vocab_size)

            probabilities = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # (batch, 1)

            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import CharTokenizer

    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")

    model = TinyGPT(
        vocab_size=tok.vocab_size, embed_dim=32, num_heads=4, num_blocks=2, max_sequence_len=16
    )

    sample = "To be, or not"
    ids = torch.tensor([tok.encode(sample)])

    logits, loss = model(ids)
    print(f"Input shape:  {tuple(ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size={tok.vocab_size})")
    print(f"Loss without targets: {loss}  (None, as expected -- no targets given)")

    # Now WITH targets, to confirm the loss computation actually runs.
    targets = torch.tensor([tok.encode("o be, or not ")])  # shifted by one character
    _, loss = model(ids, targets)
    print(f"Loss with targets: {loss.item():.4f}")
    print(f"(Untrained model, so this should be roughly ln({tok.vocab_size}) = "
          f"{torch.log(torch.tensor(float(tok.vocab_size))).item():.4f} -- pure random guessing.)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal learnable parameters in this tiny model: {num_params:,}")

    # Generate from an untrained model -- should be complete gibberish, which is the
    # correct/expected result before any training has happened.
    generated = model.generate(ids, max_new_tokens=20)
    print(f"\nUntrained generation (expected: gibberish):")
    print(repr(tok.decode(generated[0].tolist())))
