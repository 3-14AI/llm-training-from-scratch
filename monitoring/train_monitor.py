
import torch
import wandb # Assuming wandb is installed for logging
import os

def init_wandb(project_name, run_name, config):
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(project=project_name, name=run_name, config=config)
        print("WandB initialized.")
    else:
        print("WANDB_API_KEY not found. Skipping WandB initialization.")

def log_metrics(metrics):
    if wandb.run:
        wandb.log(metrics)

def finish_wandb():
    if wandb.run:
        wandb.finish()

# Example usage in a training loop (conceptual)
# from monitoring.train_monitor import init_wandb, log_metrics, finish_wandb
# 
# config = {"learning_rate": 0.001, "epochs": 10, "model_size": "small"}
# init_wandb("llm-training", "small-model-run", config)
# 
# for epoch in range(config["epochs"]):
#     # ... training code ...
#     loss = 0.1 # dummy loss
#     accuracy = 0.9 # dummy accuracy
#     log_metrics({"epoch": epoch, "loss": loss, "accuracy": accuracy})
# 
# finish_wandb()
