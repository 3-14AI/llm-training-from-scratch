
import torch

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.current_id = 0

    def _add_word(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.current_id
            self.id_to_word[self.current_id] = word
            self.current_id += 1

    def train(self, text):
        words = text.lower().split()
        for word in words:
            self._add_word(word)

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id.get('<unk>', 0)) for word in words]

    def decode(self, token_ids):
        return ' '.join([self.id_to_word.get(token_id, '<unk>') for token_id in token_ids])

    def save_vocab(self, path):
        torch.save({
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'current_id': self.current_id
        }, path)

    def load_vocab(self, path):
        vocab = torch.load(path)
        self.word_to_id = vocab['word_to_id']
        self.id_to_word = vocab['id_to_word']
        self.current_id = vocab['current_id']

if __name__ == '__main__':
    tokenizer = SimpleTokenizer()
    sample_text = "Hello world, this is a simple tokenizer example. Hello again."
    tokenizer.train(sample_text)
    
    encoded_text = tokenizer.encode("Hello, this is a test.")
    print(f"Encoded: {encoded_text}")
    print(f"Decoded: {tokenizer.decode(encoded_text)}")

    tokenizer.save_vocab("simple_tokenizer_vocab.pt")
    new_tokenizer = SimpleTokenizer()
    new_tokenizer.load_vocab("simple_tokenizer_vocab.pt")
    print(f"Loaded vocab and encoded: {new_tokenizer.encode('Hello world')}")
