
# configs/medium_config.py

MODEL_CONFIG = {
    "embed_size": 512,
    "num_layers": 4,
    "heads": 8,
    "forward_expansion": 4,
    "dropout": 0.1,
    "max_length": 256,
    "use_compression": False, # New parameter
    "chunk_size": 16,         # New parameter
    "compressor_layers": 3,   # New parameter
}

TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 0.00005,
}
