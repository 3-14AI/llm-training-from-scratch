import torch

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.current_id = 4

    def _add_word(self, word):
        if word not in self.word_to_id and self.current_id < self.vocab_size:
            self.word_to_id[word] = self.current_id
            self.id_to_word[self.current_id] = word
            self.current_id += 1

    def train(self, text):
        words = text.lower().split()
        for word in words:
            self._add_word(word)

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id['<unk>']) for word in words]

    def decode(self, token_ids):
        return ' '.join([self.id_to_word.get(token_id, '<unk>') for token_id in token_ids])

    def save_vocab(self, path):
        torch.save({
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'current_id': self.current_id,
            'vocab_size': self.vocab_size
        }, path)

    def load_vocab(self, path):
        vocab = torch.load(path)
        self.word_to_id = vocab['word_to_id']
        self.id_to_word = vocab['id_to_word']
        self.current_id = vocab['current_id']
        self.vocab_size = vocab.get('vocab_size', self.vocab_size)
