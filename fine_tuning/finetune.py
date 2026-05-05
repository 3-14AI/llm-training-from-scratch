
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer
from data_preparation.dataset_creator import create_dataset
import os
import argparse

from monitoring.train_monitor import init_wandb, log_metrics, finish_wandb
from monitoring.evaluation import evaluate_model


# LoRA implementation (simplified for demonstration)
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

# Function to inject LoRA layers into a model
def inject_lora(model, rank=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace original linear layer with a LoRA-enabled one
            # This is a simplified example; a full implementation would carefully merge weights
            # or create a parallel path for LoRA. For now, we'll just add LoRA as a parallel branch.
            # In a real scenario, you'd modify the forward pass of the original Linear layer.
            # For demonstration, we'll just print that LoRA is being applied.
            print(f"Injecting LoRA into {name}")
            # A more robust implementation would involve modifying the Linear layer's forward method
            # or replacing it with a custom layer that incorporates LoRA.
            # For this example, we'll assume the user will adapt this for actual integration.
            # module.add_module("lora_layer", LoRALayer(module.in_features, module.out_features, rank, alpha))
    return model

def train_model(model, dataloader, optimizer, criterion, device, scheduler=None, epochs=10, start_epoch=0, global_step=0, use_lora=False, val_dataloader=None, eval_steps=100, checkpoint_dir="checkpoints_finetune"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_lora:
        print("Fine-tuning with LoRA enabled.")
        # In a real LoRA setup, only LoRA parameters would be trainable
        # For this simplified example, we'll assume the model is already prepared for LoRA training
        # by freezing base model parameters and enabling LoRA parameters.

    for epoch in range(start_epoch, epochs):
        model.train()
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
            global_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                log_metrics({"train/loss": loss.item(), "train/epoch": epoch + 1, "train/step": epoch * len(dataloader) + batch_idx})

            if val_dataloader is not None and batch_idx > 0 and batch_idx % eval_steps == 0:
                val_loss, val_perplexity = evaluate_model(model, val_dataloader, criterion, device)
                print(f"Validation - Epoch {epoch+1}, Step {batch_idx}, Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
                log_metrics({"val/loss": val_loss, "val/perplexity": val_perplexity, "val/epoch": epoch + 1, "val/step": epoch * len(dataloader) + batch_idx})
                model.train() # Set back to train mode
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")
        log_metrics({"train/avg_loss": avg_loss, "train/epoch": epoch + 1})

        if scheduler is not None:
            scheduler.step()

        # Evaluate at the end of each epoch
        if val_dataloader is not None:
            val_loss, val_perplexity = evaluate_model(model, val_dataloader, criterion, device)
            print(f"End of Epoch Validation - Epoch {epoch+1}, Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
            log_metrics({"val/epoch_loss": val_loss, "val/epoch_perplexity": val_perplexity, "val/epoch": epoch + 1})
            model.train()

        # Save Checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_finetune_epoch_{epoch+1}.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the LLM.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data creation for demonstration
    dummy_text = "This is a sample text for fine-tuning the LLM. It should be a smaller, task-specific corpus." * 500
    with open("dummy_finetune_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    dummy_val_text = "This is a sample text for fine-tuning validation. " * 100
    with open("dummy_val_finetune_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_val_text)

    vocab_file = "finetune_vocab.pt"
    block_size = 128
    batch_size = 16
    dataloader, vocab_size = create_dataset("dummy_finetune_data.txt", vocab_file, block_size, batch_size)
    val_dataloader, _ = create_dataset("dummy_val_finetune_data.txt", vocab_file, block_size, batch_size)

    # Model parameters (should match pre-trained model)
    src_vocab_size = vocab_size
    trg_vocab_size = vocab_size
    src_pad_idx = 0 
    trg_pad_idx = 0 
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = block_size

    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
        embed_size, num_layers, forward_expansion, heads, dropout, device, max_length
    ).to(device)

    # Load pre-trained weights (assuming 'pretrained_llm.pth' exists from pre-training)
    if os.path.exists("pretrained_llm.pth"):
        model.load_state_dict(torch.load("pretrained_llm.pth", map_location=device))
        print("Loaded pre-trained model weights.")
    else:
        print("No pre-trained model found. Starting fine-tuning from scratch.")

    use_lora = True # Set to True to enable LoRA (simplified)
    if use_lora:
        model = inject_lora(model) # This call is illustrative; actual LoRA integration is more complex.
        # Freeze base model parameters and unfreeze LoRA parameters here for actual LoRA training.

    optimizer = Adam(model.parameters(), lr=0.00001) # Smaller learning rate for fine-tuning
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    start_epoch = 0
    global_step = 0
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            print(f"Resuming from checkpoint {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
        else:
            print(f"Checkpoint {args.resume_from} not found. Starting from scratch.")

    # Init WandB
    config = {
        "learning_rate": 0.00001,
        "epochs": 3,
        "batch_size": batch_size,
        "block_size": block_size,
        "embed_size": embed_size,
        "num_layers": num_layers,
        "heads": heads,
        "use_lora": use_lora,
    }
    init_wandb("llm-finetuning", "finetune-run", config)

    print("Starting fine-tuning...")
    train_model(model, dataloader, optimizer, criterion, device, scheduler=scheduler, epochs=3, start_epoch=start_epoch, global_step=global_step, use_lora=use_lora, val_dataloader=val_dataloader, eval_steps=100)
    print("Fine-tuning finished.")

    finish_wandb()

    # Save the fine-tuned model
    torch.save(model.state_dict(), "finetuned_llm.pth")
    print("Fine-tuned model saved to finetuned_llm.pth")

    # Clean up dummy data
    os.remove("dummy_finetune_data.txt")
    os.remove("dummy_val_finetune_data.txt")
    os.remove("finetune_vocab.pt")

if __name__ == '__main__':
    main()
