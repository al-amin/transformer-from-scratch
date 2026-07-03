# Transformer From Scratch

> Status: 🚧 In progress — Tier 0 foundations project.

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
_Diagram coming as we build — will include an animated view of how attention flows
token-to-token._

## Metrics
_TBD once training runs — will report loss curves and sample generations, not just claims._

## Tradeoffs
_TBD — will document scale limits (small model, small dataset, CPU/MPS-only) honestly._

## Link
_Demo/notebook link coming once built._
