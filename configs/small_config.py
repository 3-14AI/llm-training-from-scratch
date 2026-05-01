
# configs/small_config.py

MODEL_CONFIG = {
    "embed_size": 256,
    "num_layers": 2,
    "heads": 8,
    "forward_expansion": 4,
    "dropout": 0.1,
    "max_length": 128,
}

TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.0001,
}
