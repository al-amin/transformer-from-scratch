"""
Character-level tokenizer.

Plain-language idea: a neural network only understands numbers. So before we can
feed text into a model, we need a way to turn text into numbers, and a way to turn
the model's number output back into text. That's all a tokenizer does.

We use the simplest possible scheme: character-level. Every unique character that
appears anywhere in our dataset (a, b, c, ..., !, ?, newline, etc.) gets its own
integer ID. "hello" might become [7, 4, 11, 11, 14] for example, where 7 = 'h',
4 = 'e', and so on. This is simpler than the subword tokenizers real LLMs use
(like GPT's BPE), but it's easier to understand and works fine for a small model
on a small dataset like ours.
"""

from __future__ import annotations

from pathlib import Path


class CharTokenizer:
    """Encodes text to a list of integers and decodes integers back to text."""

    def __init__(self, text: str):
        # sorted() so the vocabulary (and therefore the IDs) is always built the
        # same way given the same text -- this makes training reproducible.
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # Two lookup tables, one in each direction.
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Text -> list of integer IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """List of integer IDs -> text."""
        return "".join(self.id_to_char[i] for i in ids)

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        text = Path(path).read_text(encoding="utf-8")
        return cls(text)


if __name__ == "__main__":
    # Quick manual check: build the tokenizer from our dataset and round-trip a
    # sample string through encode -> decode to prove it comes back unchanged.
    tok = CharTokenizer.from_file(Path(__file__).parent.parent / "data" / "tinyshakespeare.txt")
    print(f"Vocabulary size: {tok.vocab_size} unique characters")

    sample = "To be, or not to be"
    ids = tok.encode(sample)
    back = tok.decode(ids)

    print(f"Original: {sample!r}")
    print(f"Encoded:  {ids}")
    print(f"Decoded:  {back!r}")
    assert back == sample, "round-trip failed -- decode(encode(x)) should equal x"
    print("Round-trip check passed.")
