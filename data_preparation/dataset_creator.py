import torch
from torch.utils.data import Dataset, DataLoader
from data_preparation.tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    def __init__(self, tokenized_data, block_size):
        self.tokenized_data = tokenized_data
        self.block_size = block_size

    def __len__(self):
        if len(self.tokenized_data) <= self.block_size:
            return 0
        return len(self.tokenized_data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokenized_data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def create_dataset(text_file_path, vocab_path, block_size=128, batch_size=32):
    # Load or train tokenizer
    tokenizer = SimpleTokenizer()
    try:
        tokenizer.load_vocab(vocab_path)
        print(f"Loaded vocabulary from {vocab_path}")
    except (FileNotFoundError, RuntimeError):
        print(f"Vocabulary not found or invalid at {vocab_path}. Training new tokenizer...")
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.train(text)
        tokenizer.save_vocab(vocab_path)
        print(f"Trained and saved vocabulary to {vocab_path}")

    # Tokenize the entire text
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenized_text = tokenizer.encode(text)

    # Create dataset and dataloader
    dataset = TextDataset(tokenized_text, block_size)
    if len(dataset) == 0:
        # Fallback for very short text
        print("Warning: Text is shorter than block_size. Padding...")
        tokenized_text = tokenized_text + [0] * (block_size + 1 - len(tokenized_text))
        dataset = TextDataset(tokenized_text, block_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, tokenizer.current_id
