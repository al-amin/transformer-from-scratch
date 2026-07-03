"""
Training loop.

Plain-language idea:
1. Cut random chunks out of the training text.
2. For each chunk, the "input" is the chunk itself, and the "target" is the SAME chunk
   shifted one character to the right -- i.e. at every position, the target is simply
   "what character actually comes next in the real text".
3. Run the chunk through the model, get a loss (cross-entropy between predicted
   probabilities and the real next character).
4. Backpropagate that loss and take one optimizer step -- nudge every learnable number
   in the model slightly toward reducing this loss.
5. Repeat thousands of times. Over time, the model's predictions get closer to what
   Shakespeare actually wrote.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CharTokenizer
from model import TinyGPT


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample a random batch of (input, target) chunks from the data."""
    start_indices = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[i : i + block_size] for i in start_indices])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    return inputs.to(device), targets.to(device)


def main():
    project_root = Path(__file__).parent.parent

    # --- Config (small, deliberately -- this is a learning project on a laptop, not a
    # production training run) ---
    embed_dim = 64
    num_heads = 4
    num_layers = 4
    block_size = 64          # how many characters of context the model sees at once
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 3000
    eval_interval = 300      # how often to measure and print progress
    eval_iters = 50          # how many batches to average over when measuring loss

    # --- Device: use the Mac's GPU (MPS) if available, otherwise fall back to CPU ---
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- Data ---
    tok = CharTokenizer.from_file(project_root / "data" / "tinyshakespeare.txt")
    text = (project_root / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # 90/10 train/validation split -- validation loss tells us if the model is actually
    # generalizing, or just memorizing the exact training text.
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # --- Model ---
    model = TinyGPT(
        vocab_size=tok.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_sequence_len=block_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        """Average loss over several random batches, for both train and val splits --
        a single batch's loss is noisy, averaging gives a much more honest reading."""
        model.eval()
        results = {}
        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(split_data, block_size, batch_size, device)
                _, loss = model(x, y)
                losses[i] = loss.item()
            results[split_name] = losses.mean().item()
        model.train()
        return results

    # --- Training loop ---
    history = []  # (step, train_loss, val_loss) -- for real loss-curve plotting later
    start_time = time.time()

    for step in range(max_iters + 1):
        if step % eval_interval == 0 or step == max_iters:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(
                f"step {step:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})

        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({max_iters} steps).")

    # --- Save the real training history to disk, so the README can report real numbers
    # instead of anyone having to trust an unverifiable claim. ---
    history_path = project_root / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved training history to {history_path}")

    # --- Save model weights locally (gitignored -- checkpoints aren't committed) ---
    checkpoint_path = project_root / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path} (not committed -- see .gitignore)")

    # --- Generate real sample text from the TRAINED model ---
    model.eval()
    start = torch.tensor([tok.encode("To ")], device=device)
    generated = model.generate(start, max_new_tokens=300)
    generated_text = tok.decode(generated[0].tolist())

    print("\n--- Sample generation from the TRAINED model (300 characters) ---")
    print(generated_text)

    sample_path = project_root / "sample_generation.txt"
    sample_path.write_text(generated_text)
    print(f"\nSaved sample generation to {sample_path}")


if __name__ == "__main__":
    main()
