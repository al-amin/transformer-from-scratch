# Transformer From Scratch

> Status: ✅ Complete — Tier 0 foundations project.

## Hook
A GPT-style language model built from first principles — no pre-built transformer library —
to understand exactly what's happening inside attention, and to have a working, trained
model to show for it.

## Problem
Most people who "use" LLMs have never implemented the mechanism that makes them work.
This project implements a small transformer (tokenization → embeddings → positional
encoding → multi-head self-attention → feed-forward → training loop) from scratch in
PyTorch, then trains it on a small text corpus to generate text in that style.

## Architecture

In plain terms: text goes in as tokens, gets turned into numbers (embeddings), then for
each word the model asks "which earlier words matter most for predicting what comes
next?" (that question-asking step is called **attention**), blends the important ones
together, processes the result a bit more, and outputs a prediction for the next word.

![Attention flow architecture](docs/architecture-attention-flow.svg)

- **Query / Key / Value** are three different "views" of the same input: Query = "what am
  I looking for", Key = "what do I contain", Value = "what info do I pass along if picked".
- **Attention Scores** = comparing every Query against every Key to get a relevance score
  per word-pair.
- **Weighted Sum** = blending the Values together, weighted by those relevance scores —
  this is the actual "attending" step.

(Diagram is animated in the source SVG — open `docs/architecture-attention-flow.svg`
directly, or view it on GitHub, to see the data flow in motion.)

**📚 Want the concepts explained one at a time, in plain language, with their own
diagrams?** See [`concepts/`](concepts/README.md).

## Metrics

Trained an 824,897-parameter model on Tiny Shakespeare (~1M characters) for 3,000
iterations on an Apple Silicon Mac (MPS GPU), in **89.4 seconds**.

| | Start (iter 0) | End (iter 3000) |
|---|---|---|
| Train loss | 4.3845 | **1.5006** |
| Validation loss | 4.3870 | **1.6923** |

(Starting loss of ~4.38 matches the theoretical random-guessing baseline of
`ln(65 characters) ≈ 4.17` — confirms the model started from genuine random weights,
not a trivial shortcut.)

![Training loss curve](docs/loss_curve.png)

**Sample generation from the trained model** (temperature-free multinomial sampling,
starting from a newline character):

```
Have how all Commonten his a proloud.

Romeo in RomeBEet:
Stand am, cond you thou goods' ranged ear.

LEONTES:
My pribusiseniss me, shall some.
Dis may, I mean another will death a headful to thy court you.
I lo, give so accimpardes uppeaks ouch.

ISABELLA:

Lord Non'never:

WARWICK, HENREY:
You you
```

Not coherent English, but genuinely learned Shakespeare-*shaped* structure: character
names in ALL CAPS followed by a colon (matching the play-script format), plausible verb
endings, and punctuation patterns — all emergent from raw character prediction, no
hardcoded rules. Full raw output: [`checkpoints/sample_generation.txt`](checkpoints/sample_generation.txt),
full per-iteration numbers: [`checkpoints/loss_history.csv`](checkpoints/loss_history.csv).

## Tradeoffs

- **Character-level tokenization** means the model has to learn spelling from scratch
  (no subword priors) — part of why output isn't real words yet at this scale/training
  budget. A production LLM's BPE tokenizer sidesteps this.
- **824K parameters and 3,000 iterations is tiny** by any real standard (GPT-2 small is
  124M parameters, trained for far longer) — intentional, since the goal was
  understanding the mechanism, not chasing benchmark numbers.
- **Train/val gap** (1.50 vs 1.69) shows mild memorization starting to set in; more
  data, dropout, or fewer iterations would narrow this, but wasn't the focus here.
- **Sampling is unconstrained multinomial** (no temperature scaling, no top-k/top-p) —
  simplest possible decoding strategy, intentionally, to keep focus on the model itself.

## Link
[Full source, `src/train.py` for the training loop, `src/gpt_model.py` for the model class](https://github.com/al-amin/transformer-from-scratch)
