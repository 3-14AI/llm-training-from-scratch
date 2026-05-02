import unittest
import os
from data_preparation.tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def test_train_encode_decode(self):
        tokenizer = BPETokenizer(vocab_size=100)
        text = "hello world this is a test hello world"
        tokenizer.train(text)

        encoded = tokenizer.encode("hello world")
        self.assertTrue(len(encoded) > 0)

        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "hello world")

    def test_save_load(self):
        tokenizer = BPETokenizer(vocab_size=100)
        text = "hello world this is a test hello world"
        tokenizer.train(text)

        path = "test_vocab.json"
        tokenizer.save_vocab(path)
        self.assertTrue(os.path.exists(path))

        new_tokenizer = BPETokenizer()
        new_tokenizer.load_vocab(path)

        encoded_orig = tokenizer.encode("hello world")
        encoded_new = new_tokenizer.encode("hello world")
        self.assertEqual(encoded_orig, encoded_new)

        os.remove(path)

if __name__ == '__main__':
    unittest.main()
