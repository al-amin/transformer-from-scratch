"""
Training loop.

Plain-language idea:
- Cut the training text into overlapping chunks. For each chunk, the "input" is the
  chunk itself, and the "target" is the SAME chunk shifted one character to the right --
  i.e. "predict the next character at every position, given everything before it."
- Feed a batch of these chunks through the model, get the cross-entropy loss (how wrong
  the model's predicted probabilities were vs. the actual next characters).
- Use an optimizer (AdamW -- a standard, well-behaved variant of gradient descent) to
  nudge every learnable weight slightly in the direction that would have reduced the loss.
- Repeat thousands of times. Loss going down over time = the model is actually learning
  real patterns in the text, not just random guessing.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from tokenizer import CharTokenizer
from gpt_model import TinyGPT

# ---- Hyperparameters (kept small on purpose -- this is a learning project, not a
# ---- production model. Even at this size it learns real character-level patterns.) ----
BLOCK_SIZE = 128        # how many characters of context the model sees at once
BATCH_SIZE = 64         # how many chunks we train on simultaneously
EMBED_DIM = 128
NUM_HEADS = 4
NUM_BLOCKS = 4
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 300     # how often to check validation loss
EVAL_ITERS = 50         # how many batches to average for a stable validation loss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    """Sample a random batch of (input, target) chunks from the data."""
    # Pick random starting points, leaving room for a block_size-long chunk plus 1
    # extra character (the target is shifted by one).
    start_indices = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[i : i + block_size] for i in start_indices])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    """Average the loss over several batches for a stable (less noisy) estimate."""
    model.eval()
    losses = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            inputs, targets = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            _, loss = model(inputs, targets)
            split_losses[i] = loss.item()
        losses[split_name] = split_losses.mean().item()
    model.train()
    return losses


def main():
    device = get_device()
    print(f"Using device: {device}")

    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    tok = CharTokenizer.from_file(data_path)
    text = data_path.read_text(encoding="utf-8")

    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # 90/10 train/validation split -- validation data the model never trains on, so we
    # can honestly check whether it's learning general patterns vs. just memorizing.
    split_point = int(0.9 * len(data))
    train_data = data[:split_point]
    val_data = data[split_point:]

    model = TinyGPT(
        vocab_size=tok.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_sequence_len=BLOCK_SIZE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Training data: {len(train_data):,} characters, Validation data: {len(val_data):,} characters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []  # (iteration, train_loss, val_loss) -- real numbers, logged as we go

    start_time = time.time()
    for iteration in range(MAX_ITERS + 1):
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - start_time
            print(
                f"iter {iteration:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )
            history.append((iteration, losses["train"], losses["val"]))

        inputs, targets = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)
        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({MAX_ITERS} iterations).")

    # Save the trained model and the loss history so we can report REAL numbers, not
    # estimates, in the README.
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "tinygpt.pt")

    with open(checkpoint_dir / "loss_history.csv", "w") as f:
        f.write("iteration,train_loss,val_loss\n")
        for it, tr, va in history:
            f.write(f"{it},{tr:.4f},{va:.4f}\n")

    # Generate a real sample from the TRAINED model, starting from a newline character
    # (a natural place to start generating Shakespeare-style text from).
    start_ids = torch.tensor([[tok.encode("\n")[0]]], device=device)
    generated = model.generate(start_ids, max_new_tokens=300)
    generated_text = tok.decode(generated[0].tolist())

    with open(checkpoint_dir / "sample_generation.txt", "w") as f:
        f.write(generated_text)

    print("\n--- Sample generation from the TRAINED model ---")
    print(generated_text)


if __name__ == "__main__":
    main()
