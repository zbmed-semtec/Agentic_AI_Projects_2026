# SapiensCorev.4

- Model ID: aashish10807as/sapienscorev-4
- License: Apache 2.0
- Tasks: india, sampling, computer science, advanced, text, pytorch, transformer, english
- Frameworks: Transformers
- Shared by: Aashish Kumar
- Source: https://www.kaggle.com/models/aashish10807as/sapienscorev-4

## Description

🧠 SapienCore (v4) — Custom Trained Language Model

AspienCore is a personally engineered transformer-based language model designed and trained from scratch using a custom pipeline focused on efficiency, experimentation, and full control over training dynamics.

It represents an independent effort to build a functional LLM system without relying on pre-trained APIs or external large-scale model backbones.

⚙️ Core Architecture
Transformer-based decoder language model
Custom implementation optimized for experimental flexibility
Token-level autoregressive generation
Designed for scalable depth and width expansion during training iterations
🧪 Training System

AspienCore is trained using a fully custom pipeline that includes:

Mixed dataset ingestion (text + structured corpora)
Parquet-based dataset streaming for efficiency
Custom tokenizer (BPETokenizer)
AdamW optimizer with full state tracking
Gradient-based learning with checkpoint recovery system

Training artifacts include:

model.pt — learned parameters
optim.pt — optimizer state for continuation training
temporary checkpointing system for safe recovery during long runs
📊 Data Pipeline

The model is trained on a mixed dataset composed of:

Story-style corpora (e.g., TinyStories)
Instruction-style datasets (Claude-style structured text)
Custom curated text files for domain adaptation

All datasets are processed through a unified loading and batching system optimized for Kaggle GPU constraints.

🧠 Key Features
🧩 Fully custom training loop (no high-level trainer dependency)
🔁 Resume-capable checkpoint system
⚡ Optimized for limited compute environments (Kaggle GPU T4 x2)
🧠 Custom tokenizer integration (BPE-based)
📦 Efficient dataset handling via Parquet + streaming loaders
🛠️ Modular architecture for experimentation (layer scaling, vocab scaling, etc.)
🚀 Design Philosophy

AspienCore is built with a focus on:

Understanding how LLMs learn internally
Experimenting with architecture scaling (depth, width, vocab size)
Avoiding dependency on black-box APIs
Enabling iterative research-driven improvements
📌 Current Status
Actively trained and iterated on Kaggle GPU environments
Supports checkpoint saving and resumption
Early-stage but structurally scalable LLM prototype
🔮 Future Direction
Scaling model depth and embedding size
Improving tokenizer efficiency
Adding attention optimizations (Flash / grouped attention)
Transitioning to multi-GPU distributed training
Improving generalization across instruction task

## References

Most of the data are available in HuggingFace
