# Training Loop (Loss, Optimizer, Train/Val Split)

## Plain-language explanation
Training teaches the model by repeatedly showing it examples and nudging its weights
toward better predictions. The loop:

1. **Chunk the data**: cut the training text into overlapping windows. The "input" is
   the window itself, and the "target" is the SAME window shifted one character to the
   right — i.e. "predict the next character at every position."
2. **Forward pass**: feed a batch of chunks through the model, get predicted
   probabilities for the next character at every position.
3. **Loss**: **cross-entropy loss** measures how wrong those predictions were vs. the
   actual next characters.
4. **Backward pass + optimizer step**: compute how much each weight contributed to the
   loss, nudge every weight slightly to reduce it. We use **AdamW**, a standard,
   well-behaved variant of gradient descent.
5. Repeat thousands of times. Loss going down = the model is learning real patterns.

**Train/validation split**: 10% of the data is held out and never trained on. Checking
loss on this held-out set tells us honestly whether the model is learning general
patterns (both losses drop together) or just memorizing (train loss drops while
validation loss stalls/rises).

![Training loss curve — real numbers from an actual training run](diagrams/loss-curve.png)

## Why it matters
This loop — chunk, predict, measure error, adjust weights, repeat — is the exact same
pattern used to train every neural network, from this tiny model to GPT-4-scale systems.

## Where it's implemented
[`src/train.py`](../src/train.py) — real results, not simulated: trained an
824,897-parameter model for 3,000 iterations in 89.4 seconds on an Apple Silicon Mac
(MPS GPU). Train loss dropped 4.3845 → 1.5006, validation loss 4.3870 → 1.6923. Full
numbers: [loss_history.csv](../checkpoints/loss_history.csv).
