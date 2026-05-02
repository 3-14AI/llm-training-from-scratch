# Agent Instructions

Welcome to the LLM Training From Scratch project! This document outlines rules, guidelines, and commands for AI agents and developers working on this codebase.

## Codebase Overview
- **`data_preparation/`**: Tokenization and PyTorch dataset logic.
- **`model_architecture/`**: Core PyTorch implementations of the Transformer, Context Compressor, etc.
- **`pretraining/` & `fine_tuning/`**: Training scripts.
- **`inference/`**: Scripts to generate text.
- **`configs/`**: Hyperparameter files.
- **`monitoring/`**: Integration with wandb or similar systems.
- **`run_e2e_tests.py`**: E2E test suite running experiments on subsets of data.

## Rules and Guidelines
1. **Testing**: Before submitting any architectural change, ensure that models can compile and forward propagate correctly. Always run `python test_transformer.py` (if applicable) or create a small smoke test file to catch shape errors.
2. **E2E Testing**: Make use of `run_e2e_tests.py` to test major training pipelines. For a fast test run without requiring heavy resources, run: `python run_e2e_tests.py --mode e2e --data dummy_pretrain_data.txt`
3. **Environment**: If dependencies are missing, install them via `pip install -r requirements.txt`. For full reproduction, you can use the provided Dockerfile.
4. **Architecture Philosophy**: Keep components modular. Modifications to `Transformer` and `ContextCompressor` should retain original method signatures where possible and properly document shape changes.

## Development Tasks
Follow tasks specified in `TODO.md` when expanding the project architecture and training pipeline. Each subtask must be fully complete, checked, and testable before opening a pull request.
