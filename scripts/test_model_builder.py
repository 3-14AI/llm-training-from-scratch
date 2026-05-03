import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath("scripts"))
from model_builder import count_params, build_model

def test_count_params():
    model = nn.Linear(10, 5)
    assert count_params(model) == 55

def test_build_model_standard():
    cfg = {
        "embed_size": 32,
        "num_layers": 2,
        "heads": 2,
        "max_length": 64
    }
    vocab_size = 100
    device = torch.device("cpu")
    model = build_model(cfg, vocab_size, device)

    assert not hasattr(model, "compressor")

def test_build_model_compressed():
    cfg = {
        "embed_size": 32,
        "num_layers": 2,
        "heads": 2,
        "max_length": 64,
        "use_compression": True,
        "chunk_size": 8,
        "compressor_layers": 1
    }
    vocab_size = 100
    device = torch.device("cpu")
    model = build_model(cfg, vocab_size, device)

    assert hasattr(model, "compressor")
    assert model.compressor.chunk_size == 8
