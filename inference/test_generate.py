import pytest
import torch
import torch.nn as nn
from inference.generate import generate_text

class DummyTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens):
        return " ".join([str(t) for t in tokens])

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure we have a fake position embedding for generate_text to read max_tokens
        self.encoder = type('Encoder', (), {'position_embedding': type('PosEmb', (), {'num_embeddings': 128})()})()

    def forward(self, src, trg):
        # Always output logits for 10 tokens where token 5 has high probability
        batch_size, seq_len = src.shape
        logits = torch.randn(batch_size, seq_len, 10)
        # Make token 5 the most likely
        logits[:, :, 5] = 10.0
        return logits

def test_generate_text_basic():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    device = torch.device("cpu")

    # Test basic generation
    generated = generate_text(model, tokenizer, "hello", 5, device, temperature=0.1)
    # The dummy model always favors token 5
    assert generated == "5 5 5 5 5"

def test_generate_text_repetition_penalty():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    device = torch.device("cpu")

    # We pass a high repetition penalty to see if the probability distribution shifts
    # The dummy model outputs random logits where token 5 is the largest (10.0).
    # If we penalize 5, it should output another token eventually.
    # We will test multiple tokens
    # With repetition_penalty=100.0, the logit 10.0 becomes 10.0 / 100.0 = 0.1
    # Other random logits could be larger than 0.1, making token 5 not the only token.
    generated = generate_text(model, tokenizer, "hello", 10, device, temperature=0.1, repetition_penalty=100.0)
    tokens = generated.split()

    # Check if the model generated tokens other than 5
    assert len(set(tokens)) > 1

def test_generate_text_top_k():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    device = torch.device("cpu")

    # Setting top_k=1 should only consider the top token
    generated = generate_text(model, tokenizer, "hello", 5, device, temperature=1.0, top_k=1)
    assert generated == "5 5 5 5 5"

def test_generate_text_top_p():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    device = torch.device("cpu")

    # Setting top_p=0.1 should only consider the most probable tokens
    generated = generate_text(model, tokenizer, "hello", 5, device, temperature=1.0, top_p=0.1)
    assert generated == "5 5 5 5 5"
