import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from scripts.model_builder import build_model, count_params
from data_preparation.dataset_creator import create_dataset
import time
import json

from scripts.experiment_configs import RUN_EXPERIMENTS_CONFIGS as EXPERIMENTS

def train_model(model, dataloader, optimizer, criterion, device, epochs=1):
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            
            output = output.reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx >= 5:
                break

        avg_loss = total_loss / min(len(dataloader), 6)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")
    return history

def run_experiment(exp_name, config, data_file, vocab_file, device):
    print(f"\n{'='*50}\nRunning experiment: {exp_name}\nConfig: {config}\n{'='*50}")
    
    block_size = config["max_length"]
    batch_size = 4
    
    dataloader, vocab_size = create_dataset(data_file, vocab_file, block_size, batch_size)
    
    src_vocab_size = vocab_size
    trg_vocab_size = vocab_size
    src_pad_idx = 0
    trg_pad_idx = 0

    start_time = time.time()

    model = build_model(config, vocab_size, device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    history = train_model(model, dataloader, optimizer, criterion, device, epochs=1)
    
    end_time = time.time()
    
    return {
        "loss": history[-1] if history else float('inf'),
        "time_seconds": end_time - start_time,
        "params": count_params(model)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_file = "multilingual_corpus.txt"
    vocab_file = "multilingual_vocab.pt"
    
    results = {}
    for exp_name, config in EXPERIMENTS.items():
        try:
            res = run_experiment(exp_name, config, data_file, vocab_file, device)
            results[exp_name] = res
            print(f"Result for {exp_name}: {res}")
        except Exception as e:
            print(f"Error in {exp_name}: {e}")
            results[exp_name] = {"error": str(e)}
            
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments finished. Results saved to experiment_results.json")

if __name__ == '__main__':
    main()
