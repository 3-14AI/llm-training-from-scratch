# Training Pipeline & Architecture Checklist

This document contains actionable checklist items for advancing the training pipeline and overall project architecture. Each checkbox is sized appropriately for a single, focused pull request.

## Data & Tokenization
- [x] **Integrate BPE/SentencePiece Tokenizer**
  - **Description**: Replace the simplistic word-level tokenization in `data_preparation/tokenizer.py` with a standard subword tokenizer like BPE or SentencePiece (using `tokenizers` or `sentencepiece` library).
  - **Why**: Word-level tokenization leads to Out-Of-Vocabulary (OOV) errors and bloated vocabularies. Subword tokenization is standard practice for modern LLMs.

## Evaluation & Monitoring
- [x] **Implement Validation & Evaluation Metrics**
  - **Description**: Extend `pretrain.py` and `finetune.py` to support a held-out validation set. Log validation loss and perplexity at fixed steps using the existing Weights & Biases integration.
  - **Why**: Currently, the models only log training loss. We need validation metrics to prevent overfitting and properly compare experiment configurations.

- [x] **Integrate LM Evaluation Harness**
  - **Description**: Add a script to interface the generated models (via `generate.py` or standard HuggingFace wrappers) with EleutherAI's LM Evaluation Harness for zero-shot testing on standard benchmarks (e.g., MMLU, HellaSwag).
  - **Why**: Standardized benchmarking is required to assess model quality and measure real progress.

## Distributed Training & Scaling
- [ ] **Implement Multi-GPU Training via DeepSpeed/FSDP**
  - **Description**: Wrap the training loop in `pretrain.py` with DeepSpeed (`deepspeed.initialize`) or PyTorch FSDP. Ensure data loading uses `DistributedSampler`.
  - **Why**: Training large configurations (as in `configs/large_config.py`) requires multi-GPU scaling due to VRAM limitations on single GPUs.

## Checkpointing & State Management
- [ ] **Enhance Checkpointing Mechanism**
  - **Description**: Update the training scripts to save not just model weights (`model.state_dict()`), but also the optimizer state, learning rate scheduler state, current epoch, and global step.
  - **Why**: In case of failures or resource preemption (especially in cloud environments), robust checkpointing allows resuming training without losing progress.
