# Concepts — Overview

> The ideas behind this project, explained simply. Each links to the exact code that
> implements it.

## The big picture, in one plain sentence
A transformer turns text into numbers, figures out which earlier words matter most for
predicting the next one, and repeats that reasoning a few times before making its guess.

![Transformer overview](diagrams/overview.svg)

## The pipeline, step by step
1. **[Tokenization](tokenization.md)** — turn text into numbers (one number per character).
2. **[Embeddings](embeddings.md)** — turn those numbers into learned vectors that carry
   both *meaning* (what the character is) and *order* (where it sits in the sequence).
3. **[Attention](attention.md)** — for each word, decide which earlier words are most
   relevant to predicting what comes next.
4. **[Feed-Forward layer & Transformer Block](transformer-block.md)** — process each
   word's result a bit further on its own, wrapped in residual connections + layer norm.
5. **[Stacking into a full GPT model](gpt-model.md)** — repeat the block N times, add a
   final output layer, and sample from it to generate text.
6. **[Training loop](training-loop.md)** — how the model actually learns from data:
   loss, optimizer, train/val split.

## Status: complete — real trained model
An 824,897-parameter version of this exact pipeline was trained end-to-end on Tiny
Shakespeare and produced genuinely learned (if small-scale) results — see the
[training loop](training-loop.md) entry and the main [project README](../README.md) for
real numbers and a real generated sample.
