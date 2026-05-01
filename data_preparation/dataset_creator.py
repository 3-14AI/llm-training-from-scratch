
import torch
from torch.utils.data import Dataset, DataLoader
from data_preparation.tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    def __init__(self, tokenized_data, block_size):
        self.tokenized_data = tokenized_data
        self.block_size = block_size

    def __len__(self):
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
    except FileNotFoundError:
        print(f"Vocabulary not found at {vocab_path}. Training new tokenizer...")
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, tokenizer.vocab_size

if __name__ == '__main__':
    # Create a dummy text file for testing
    dummy_text = "This is a sample text for testing the dataset creation. It contains multiple sentences to simulate real data." * 10
    with open("dummy_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    vocab_file = "dummy_vocab.pt"
    dataloader, vocab_size = create_dataset("dummy_data.txt", vocab_file)

    print(f"Vocabulary size: {vocab_size}")
    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        print(f"x: {x}")
        print(f"y: {y}")
        if i == 0: # Print only the first batch for brevity
            break

    # Clean up dummy files
    import os
    os.remove("dummy_data.txt")
    os.remove("dummy_vocab.pt")
