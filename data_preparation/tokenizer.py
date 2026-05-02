import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self._tokenizer.pre_tokenizer = Whitespace()
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

    def train(self, text):
        trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"], vocab_size=self.vocab_size)
        # BpeTrainer takes an iterator of strings
        self._tokenizer.train_from_iterator([text], trainer=trainer)

    def encode(self, text):
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self._tokenizer.decode(token_ids)

    def save_vocab(self, path):
        # We save as tokenizer.json, not PyTorch dict.
        # But we must respect the `path` argument which might have a .pt extension.
        # To maintain compatibility, we save as JSON.
        self._tokenizer.save(path)

    def load_vocab(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary not found at {path}")
        self._tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self._tokenizer.get_vocab_size()

    @property
    def current_id(self):
        return self._tokenizer.get_vocab_size()
