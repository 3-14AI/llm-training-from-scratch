
# configs/large_config.py

MODEL_CONFIG = {
    "embed_size": 768,
    "num_layers": 6,
    "heads": 12,
    "forward_expansion": 4,
    "dropout": 0.1,
    "max_length": 512,
    "use_compression": False, # New parameter
    "chunk_size": 32,         # New parameter
    "compressor_layers": 4,   # New parameter
}

TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 0.00001,
}
