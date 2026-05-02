import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.dataset_creator import create_dataset
import time
import json

# Конфигурации для экспериментов
# Ограничение 12 ГБ VRAM (для e2e теста на CPU используем маленькие параметры, но структура конфигов сохраняется)
# Для реального запуска на GPU параметры можно увеличить.

# 1. Зависимость от размера модели (малые, средние, большие)
# 2. Зависимость от размера компрессора (на средней модели)
# 3. Зависимость от длины последовательности

EXPERIMENTS = {
    # 1. Зависимость от размера модели
    "small_baseline": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": False, "max_length": 64},
    "small_compressed": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": True, "chunk_size": 8, "compressor_layers": 1, "max_length": 64},
    
    "medium_baseline": {"embed_size": 128, "num_layers": 3, "heads": 4, "use_compression": False, "max_length": 64},
    "medium_compressed": {"embed_size": 128, "num_layers": 3, "heads": 4, "use_compression": True, "chunk_size": 8, "compressor_layers": 2, "max_length": 64},
    
    "large_baseline": {"embed_size": 256, "num_layers": 4, "heads": 8, "use_compression": False, "max_length": 64},
    "large_compressed": {"embed_size": 256, "num_layers": 4, "heads": 8, "use_compression": True, "chunk_size": 8, "compressor_layers": 2, "max_length": 64},

    # 2. Зависимость от размера компрессора (на средней модели)
    "medium_comp_small": {"embed_size": 128, "num_layers": 3, "heads": 4, "use_compression": True, "chunk_size": 8, "compressor_layers": 1, "max_length": 64},
    "medium_comp_large": {"embed_size": 128, "num_layers": 3, "heads": 4, "use_compression": True, "chunk_size": 8, "compressor_layers": 4, "max_length": 64},

    # 3. Зависимость от длины последовательности (на малой модели)
    "seq_128_baseline": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": False, "max_length": 128},
    "seq_128_compressed": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": True, "chunk_size": 16, "compressor_layers": 1, "max_length": 128},
    
    "seq_256_baseline": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": False, "max_length": 256},
    "seq_256_compressed": {"embed_size": 64, "num_layers": 2, "heads": 4, "use_compression": True, "chunk_size": 32, "compressor_layers": 1, "max_length": 256},
}

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
            
            # Для e2e теста ограничиваем количество батчей
            if batch_idx >= 5:
                break

        avg_loss = total_loss / min(len(dataloader), 6)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")
    return history

def run_experiment(exp_name, config, data_file, vocab_file, device):
    print(f"\n{'='*50}\nRunning experiment: {exp_name}\nConfig: {config}\n{'='*50}")
    
    block_size = config["max_length"]
    batch_size = 4 # Маленький батч для CPU
    
    dataloader, vocab_size = create_dataset(data_file, vocab_file, block_size, batch_size)
    
    src_vocab_size = vocab_size
    trg_vocab_size = vocab_size
    src_pad_idx = 0
    trg_pad_idx = 0
    forward_expansion = 4
    dropout = 0.1

    start_time = time.time()

    if config["use_compression"]:
        model = CompressedTransformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            config["embed_size"], config["num_layers"], forward_expansion, config["heads"], dropout, device, config["max_length"],
            chunk_size=config["chunk_size"], compressor_layers=config["compressor_layers"]
        ).to(device)
    else:
        model = Transformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            config["embed_size"], config["num_layers"], forward_expansion, config["heads"], dropout, device, config["max_length"]
        ).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    history = train_model(model, dataloader, optimizer, criterion, device, epochs=1)
    
    end_time = time.time()
    
    return {
        "loss": history[-1],
        "time_seconds": end_time - start_time,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad)
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
