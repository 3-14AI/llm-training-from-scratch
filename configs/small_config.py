
# configs/small_config.py

MODEL_CONFIG = {
    "embed_size": 256,
    "num_layers": 2,
    "heads": 8,
    "forward_expansion": 4,
    "dropout": 0.1,
    "max_length": 128,
    "use_compression": False, # New parameter
    "chunk_size": 8,          # New parameter
    "compressor_layers": 2,   # New parameter
}

TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.0001,
}
