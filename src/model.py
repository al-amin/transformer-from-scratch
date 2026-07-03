"""
TinyGPT: the full model, assembled from the pieces we've already built and verified.

Plain-language idea:
1. Turn input token IDs into vectors (token + position embeddings).
2. Pass them through N transformer blocks stacked in a row -- each block lets tokens
   share information (attention) and then think it over individually (feed-forward).
3. One final LayerNorm to keep the output well-behaved.
4. A linear "output head" that turns each token's final vector into a score (called a
   LOGIT) for every character in the vocabulary -- "how likely is each possible next
   character, according to the model, before we've even looked at probabilities yet".

Running softmax on those logits turns them into real probabilities that sum to 100%,
which we can sample from to actually generate text one character at a time.
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
        num_layers: int,
        max_sequence_len: int,
    ):
        super().__init__()
        self.max_sequence_len = max_sequence_len

        self.embed = TokenAndPositionEmbedding(vocab_size, max_sequence_len, embed_dim)

        # Stack num_layers identical (but independently-learned) transformer blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, max_sequence_len) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        token_ids: shape (batch_size, sequence_len) -- integer IDs
        targets:   optional, same shape as token_ids -- the "correct next character" for
                   each position, used to compute the training loss

        returns: (logits, loss) -- loss is None if targets weren't provided
        """
        x = self.embed(token_ids)  # (batch, seq_len, embed_dim)

        for block in self.blocks:
            x = block(x)  # shape stays (batch, seq_len, embed_dim) at every block

        x = self.final_norm(x)
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy expects (N, num_classes) and (N,), so we flatten the
            # batch and sequence dimensions together.
            batch_size, sequence_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_len, vocab_size),
                targets.view(batch_size * sequence_len),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, start_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressive generation: repeatedly predict the next character, append it, and
        feed the whole thing back in for the next prediction. This is exactly how GPT-
        style models generate text one token at a time.

        start_ids: shape (batch_size, sequence_len) -- the "prompt" to continue from
        returns:   shape (batch_size, sequence_len + max_new_tokens)
        """
        ids = start_ids
        for _ in range(max_new_tokens):
            # The model only has position embeddings up to max_sequence_len, so if the
            # sequence has grown longer than that, only feed it the most recent chunk.
            ids_cropped = ids[:, -self.max_sequence_len :]

            logits, _ = self(ids_cropped)

            # We only care about the prediction for the NEXT character, which comes
            # from the last position's logits.
            last_logits = logits[:, -1, :]  # (batch, vocab_size)

            probs = F.softmax(last_logits, dim=-1)

            # Sample one character according to the probability distribution (not just
            # always picking the single most likely one -- that would make output
            # repetitive/deterministic).
            next_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            ids = torch.cat([ids, next_id], dim=1)

        return ids


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import CharTokenizer

    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")

    model = TinyGPT(
        vocab_size=tok.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        max_sequence_len=64,
    )

    sample = "To be, or not to be"
    ids = torch.tensor([tok.encode(sample)])

    logits, loss = model(ids)
    print(f"Input shape:  {tuple(ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")
    print(f"Loss (no targets given): {loss}")

    # Now with targets, to prove the loss computation actually works.
    targets = torch.tensor([tok.encode(sample[1:] + ".")])  # shifted by one character
    _, loss = model(ids, targets)
    print(f"Loss (untrained model, with targets): {loss.item():.4f}")
    print(f"(For reference: a completely random guess over {tok.vocab_size} characters")
    print(f"would give a loss around {torch.log(torch.tensor(float(tok.vocab_size))).item():.4f} -- ")
    print(f"an untrained model's loss should be close to this, since it hasn't learned anything yet.)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal learnable parameters: {num_params:,}")

    # Generate from an untrained model -- should be complete gibberish, proving the
    # generation loop runs correctly even before any training has happened.
    print("\nGenerating 40 characters from an UNTRAINED model (expect gibberish):")
    start = torch.tensor([tok.encode("To ")])
    generated = model.generate(start, max_new_tokens=40)
    print(repr(tok.decode(generated[0].tolist())))
