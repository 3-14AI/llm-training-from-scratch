import torch
import torch.nn as nn
import sys
import os

# Добавляем корень репозитория в PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_architecture.transformer import Transformer, CompressedTransformer

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(cfg: dict, vocab_size: int, device: torch.device) -> nn.Module:
    """Создаёт модель по конфигу."""
    common = dict(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=cfg["embed_size"],
        num_layers=cfg["num_layers"],
        forward_expansion=cfg.get("forward_expansion", 4),
        heads=cfg["heads"],
        dropout=cfg.get("dropout", 0.1),
        device=device,
        max_length=cfg["max_length"],
    )
    if cfg.get("use_compression", False):
        model = CompressedTransformer(
            **common,
            chunk_size=cfg.get("chunk_size", 8),
            compressor_layers=cfg.get("compressor_layers", 1),
        )
    else:
        model = Transformer(**common)
    return model.to(device)
