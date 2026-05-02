
# LLM Training From Scratch

This repository provides a comprehensive framework for training Large Language Models (LLMs) from scratch using PyTorch. It includes scripts for data preparation, model architecture definition, pre-training, fine-tuning (SFT, LoRA), inference, configuration management, and training monitoring.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Pre-training](#pre-training)
  - [Fine-tuning](#fine-tuning)
  - [Inference](#inference)
  - [Configuration](#configuration)
  - [Monitoring](#monitoring)
- [DeepSpeed/FSDP (Multi-GPU Training)](#deepspeedfsdp-multi-gpu-training)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to provide a clear and functional codebase for understanding and implementing the core components of LLM training. From tokenization to model inference, each step is designed to be modular and extensible, allowing researchers and developers to experiment with different aspects of LLM development.

## Architecture

The LLM architecture implemented here is a **Transformer model** built from fundamental PyTorch modules. It consists of an Encoder and a Decoder, following the original Transformer paper's design principles. Key components include:

- **Multi-Head Self-Attention**: Enables the model to weigh the importance of different words in the input sequence.
- **Feed-Forward Networks**: Position-wise fully connected layers.
- **Positional Encoding**: Injects information about the relative or absolute position of tokens in the sequence.
- **Layer Normalization and Dropout**: Regularization techniques to improve training stability and generalization.

### Context Compression

To handle longer contexts more efficiently, a **ContextCompressor** module has been introduced. This module is a smaller transformer that learns to condense a chunk of input tokens into a single vector representation. The main transformer can then process a sequence of these compressed vectors, effectively increasing its receptive field without a linear increase in computational cost. The `CompressedTransformer` integrates this compressor, allowing for joint training of both models.

### Directory Structure

```
llm-training-from-scratch/
├── data_preparation/
│   ├── tokenizer.py             # Script for basic tokenization and vocabulary management
│   └── dataset_creator.py       # Script for creating PyTorch datasets and dataloaders
├── model_architecture/
│   └── transformer.py           # PyTorch implementation of the Transformer model, including ContextCompressor
├── pretraining/
│   └── pretrain.py              # Script for pre-training the LLM on a large corpus
├── fine_tuning/
│   └── finetune.py              # Script for fine-tuning the LLM (SFT, LoRA)
├── inference/
│   └── generate.py              # Script for text generation and inference
├── configs/
│   ├── small_config.py          # Configuration for a small LLM
│   ├── medium_config.py         # Configuration for a medium LLM
│   └── large_config.py          # Configuration for a large LLM
├── monitoring/
│   └── train_monitor.py         # Integration with Weights & Biases (wandb) for logging
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── Dockerfile                   # Dockerfile for reproducible environment setup
```

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support (recommended for training)
- Docker (for reproducible environment)
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/3-14AI/llm-training-from-scratch.git
    cd llm-training-from-scratch
    ```

2.  **Create a Python virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Docker Setup

For a reproducible and isolated environment, it is highly recommended to use Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t llm-from-scratch .
    ```

2.  **Run the Docker container (with GPU support):**
    ```bash
    docker run --gpus all -it --rm -v $(pwd):/app llm-from-scratch bash
    ```
    This command will start an interactive bash session inside the container, with your local project directory mounted at `/app`.

## Usage

### Data Preparation

The `data_preparation` directory contains scripts for tokenization and dataset creation.

-   **`tokenizer.py`**: Implements a `SimpleTokenizer` for basic word-level tokenization. You can extend this to use more advanced tokenization methods (e.g., BPE, WordPiece) as needed.

    To train a tokenizer and save its vocabulary:
    ```bash
    python data_preparation/tokenizer.py
    ```

-   **`dataset_creator.py`**: Creates PyTorch `Dataset` and `DataLoader` objects from raw text data, handling tokenization and batching.

    Example usage (see `if __name__ == '__main__':` block in the script):
    ```bash
    python data_preparation/dataset_creator.py
    ```

### Pre-training

The `pretraining/pretrain.py` script handles the initial training of the LLM on a large text corpus. It now supports an optional context compression mode.

To pre-train a standard Transformer:
```bash
python pretraining/pretrain.py
```

To pre-train a `CompressedTransformer` with context compression (e.g., chunk size 8, 2 compressor layers):
```bash
python pretraining/pretrain.py --use_compression --chunk_size 8 --compressor_layers 2
```

**Multi-GPU Training (DeepSpeed/FSDP)**:

For multi-GPU training with DeepSpeed, you would typically modify the `pretrain.py` script to integrate DeepSpeed and then launch it using the DeepSpeed launcher:

1.  **Install DeepSpeed:** Add `deepspeed` to `requirements.txt` and install.
2.  **Modify `pretrain.py`**: Wrap your model, optimizer, and dataloader with `deepspeed.initialize`.
3.  **Launch with DeepSpeed:**
    ```bash
    deepspeed --num_gpus=X pretraining/pretrain.py --deepspeed_config ds_config.json
    ```
    (where `X` is the number of GPUs and `ds_config.json` is your DeepSpeed configuration file).

### Fine-tuning

The `fine_tuning/finetune.py` script allows for further training of the pre-trained LLM on a smaller, task-specific dataset. It includes a placeholder for LoRA (Low-Rank Adaptation) integration.

```bash
python fine_tuning/finetune.py
```

**LoRA (Low-Rank Adaptation)**:

The `finetune.py` includes a `LoRALayer` and `inject_lora` function as a conceptual example. For a full LoRA implementation, you would:

1.  **Freeze base model parameters.**
2.  **Inject LoRA layers** into the linear layers of the Transformer.
3.  **Train only the LoRA parameters** during fine-tuning.

### Inference

Use `inference/generate.py` to generate text using a trained LLM. It also supports inference with the `CompressedTransformer`.

To generate text with a standard Transformer:
```bash
python inference/generate.py
```

To generate text with a `CompressedTransformer` (ensure `chunk_size` matches training):
```bash
python inference/generate.py --use_compression --chunk_size 8 --model_path pretrained_llm_compressed.pth
```

### Configuration

Model and training configurations are managed in the `configs/` directory. Separate files are provided for different model sizes (small, medium, large). These now include parameters for context compression:

-   `use_compression`: Boolean, whether to use the `CompressedTransformer`.
-   `chunk_size`: Integer, the size of the input token chunks to be compressed.
-   `compressor_layers`: Integer, the number of transformer blocks in the `ContextCompressor`.

To use a specific configuration, you would import it into your training or inference script:

```python
# Example: in pretrain.py
from configs.medium_config import MODEL_CONFIG, TRAINING_CONFIG

# Then use MODEL_CONFIG and TRAINING_CONFIG to set up your model and training parameters
model = Transformer(..., embed_size=MODEL_CONFIG["embed_size"], ...)
optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
```

### Monitoring

The `monitoring/train_monitor.py` script provides basic integration with [Weights & Biases (wandb)](https://wandb.ai/) for logging training metrics.

To enable wandb logging:

1.  **Install wandb:** Add `wandb` to `requirements.txt` and install.
2.  **Set `WANDB_API_KEY`**: Set your Weights & Biases API key as an environment variable.
3.  **Integrate into training scripts**: Use `init_wandb`, `log_metrics`, and `finish_wandb` as shown in the example within `train_monitor.py`.

## DeepSpeed/FSDP (Multi-GPU Training)

While the provided scripts are designed for single-GPU execution, they are structured to facilitate integration with multi-GPU training frameworks like DeepSpeed or PyTorch's Fully Sharded Data Parallel (FSDP).

**Key steps for integration:**

1.  **Installation**: Install `deepspeed` or ensure PyTorch is built with FSDP support.
2.  **Initialization**: Use `deepspeed.initialize` or `torch.distributed.init_process_group` to set up the distributed environment.
3.  **Model/Optimizer Wrapping**: Wrap your model and optimizer with the respective DeepSpeed or FSDP wrappers.
4.  **Data Loading**: Use `DistributedSampler` with your `DataLoader` to ensure data is sharded correctly across processes.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: LICENSE file is not created yet, but can be added.)
