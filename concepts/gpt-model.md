# Full GPT Model (Stacking Blocks)

## Plain-language explanation
Everything built so far (embeddings, attention, feed-forward, transformer block) is
plumbing that leads to one question: "given the characters so far, what's the
probability of each possible next character?" The full model assembles the pipeline:

```
text -> tokenize -> embed (meaning + position) -> N transformer blocks (stacked) ->
final LayerNorm -> Linear layer -> one score per vocabulary character, per position
```

Stacking is literally just "run the same block shape N times in a row" — each pass
refines the representation further.

**Generation (sampling)**: to produce new text, feed the model its own output back in as
input, one character at a time. At each step: get the probability distribution over the
next character, SAMPLE one from it (not just always pick the most likely), append it,
repeat.

## Why it matters
This is literally what "GPT" means architecturally at its core — a stack of these
identical blocks. GPT-2 small uses 12 blocks, GPT-3 uses 96 — same shape, just deeper
and wider.

## Where it's implemented
[`src/gpt_model.py`](../src/gpt_model.py) — verified two ways before any training: (1) an
untrained model's loss on real targets was 4.4551, close to the theoretical
random-guessing baseline `ln(65) = 4.1744`; (2) untrained generation produced pure
gibberish, as expected before any learning happens.
