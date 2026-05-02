
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.dataset_creator import create_dataset
import os
import argparse

def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} finished, Average Loss: {total_loss / len(dataloader):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Pre-train the LLM with optional context compression.")
    parser.add_argument("--use_compression", action="store_true", help="Use context compression transformer")
    parser.add_argument("--chunk_size", type=int, default=8, help="Size of context chunks to compress")
    parser.add_argument("--compressor_layers", type=int, default=2, help="Number of layers in the compressor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data creation for demonstration
    dummy_text = "This is a sample text for pre-training the LLM. It should be a large corpus of text data. " * 1000
    with open("dummy_pretrain_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    vocab_file = "pretrain_vocab.pt"
    block_size = 128
    batch_size = 16
    dataloader, vocab_size = create_dataset("dummy_pretrain_data.txt", vocab_file, block_size, batch_size)

    # Model parameters
    src_vocab_size = vocab_size
    trg_vocab_size = vocab_size
    src_pad_idx = 0 # Assuming 0 is padding index
    trg_pad_idx = 0 # Assuming 0 is padding index
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = block_size

    if args.use_compression:
        print(f"Initializing CompressedTransformer with chunk_size={args.chunk_size}...")
        model = CompressedTransformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            embed_size, num_layers, forward_expansion, heads, dropout, device, max_length,
            chunk_size=args.chunk_size, compressor_layers=args.compressor_layers
        ).to(device)
    else:
        print("Initializing standard Transformer...")
        model = Transformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            embed_size, num_layers, forward_expansion, heads, dropout, device, max_length
        ).to(device)

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    print("Starting pre-training...")
    train_model(model, dataloader, optimizer, criterion, device, epochs=5)
    print("Pre-training finished.")

    # Save the pre-trained model
    save_path = "pretrained_llm_compressed.pth" if args.use_compression else "pretrained_llm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pre-trained model saved to {save_path}")

    # Clean up dummy data
    os.remove("dummy_pretrain_data.txt")
    os.remove("pretrain_vocab.pt")

if __name__ == '__main__':
    main()
