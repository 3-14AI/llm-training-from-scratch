import unittest
import torch
import torch.nn as nn
from monitoring.evaluation import evaluate_model

class DummyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, src, trg):
        batch_size, seq_len = trg.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size)

class TestEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        vocab_size = 10
        model = DummyModel(vocab_size)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        data = [
            (torch.randint(0, vocab_size, (5,)), torch.randint(0, vocab_size, (5,)))
            for _ in range(3)
        ]
        dataloader = torch.utils.data.DataLoader(data, batch_size=2)

        avg_loss, perplexity = evaluate_model(model, dataloader, criterion, device)

        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(perplexity, float)
        self.assertTrue(avg_loss >= 0.0)
        self.assertTrue(perplexity >= 1.0)

if __name__ == '__main__':
    unittest.main()
