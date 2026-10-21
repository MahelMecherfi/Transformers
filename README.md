Project: MiniGPT Dialogue Model (Homer Simpson Personalities)
1. Project Overview
This project implements a custom-designed MiniGPT autoregressive language model trained specifically on dataset dialogues from Homer Simpson. The goal is to build, regularize, and optimize a generative Transformer model from scratch to simulate character-specific dialogue lines using PyTorch and Hugging Face tools.

2. Architecture: Rebuilding the Original Transformer
Our model architecture directly inherits and adapts the foundational decoder-only Transformer design introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017).

Key Components Implemented From Scratch:
Causal Self-Attention: A custom multi-head causal attention mechanism that utilizes lower-triangular masking to ensure autoregressive tokens can only attend to past positions.
Feed-Forward Networks (FFN): Multi-layer linear projections with GELU activation functions and dropout layers for feature extraction.
Transformer Blocks: Modular blocks combining Layer Normalization, Residual Connections, Self-Attention, and Feed-Forward subnetworks.
Byte-Pair Encoding (BPE) Tokenization: A custom BPE tokenizer trained directly on our dataset corpus to optimize vocabulary utilization and handle special markers (<HUMAN>, <ASSISTANT>).
3. Model Scaling & Regularization
To adapt the classic Transformer to a custom-sized dataset (~15k conversational turns) while maintaining training stability and preventing overfitting, the following configurations were applied:

Scaled Parameters: Configured to ~13.7M parameters with D_MODEL = 384, NUM_LAYERS = 6, NUM_HEADS = 6, and D_FF = 1536.
Regularization & Optimization:
Increased Dropout to 0.2 across self-attention blocks and token embedding layers.
Coupled the AdamW Optimizer with L2 Weight Decay (0.01).
Integrated a custom Cosine Annealing Learning Rate Scheduler with a linear warmup phase.
Implemented Gradient Norm Clipping (1.0) and Early Stopping based on validation loss monitoring.
