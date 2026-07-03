"""
Plot the real training loss curve saved by train.py -- no fabricated numbers, this
reads directly from training_history.json which was written during the actual run.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
history = json.loads((project_root / "training_history.json").read_text())

steps = [h["step"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]

plt.figure(figsize=(8, 5))
plt.plot(steps, train_loss, label="Train loss", marker="o", markersize=3)
plt.plot(steps, val_loss, label="Validation loss", marker="o", markersize=3)
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("TinyGPT training on Tiny Shakespeare (real run, 211,777 parameters)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out_path = project_root / "docs" / "loss_curve.png"
plt.savefig(out_path, dpi=150)
print(f"Saved loss curve to {out_path}")
