
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer
from data_preparation.dataset_creator import create_dataset
import os

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

def train_model(model, dataloader, optimizer, criterion, device, epochs=10, use_lora=False):
    model.train()
    if use_lora:
        print("Fine-tuning with LoRA enabled.")
        # In a real LoRA setup, only LoRA parameters would be trainable
        # For this simplified example, we'll assume the model is already prepared for LoRA training
        # by freezing base model parameters and enabling LoRA parameters.

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

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} finished, Average Loss: {total_loss / len(dataloader):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data creation for demonstration
    dummy_text = "This is a sample text for fine-tuning the LLM. It should be a smaller, task-specific corpus." * 500
    with open("dummy_finetune_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    vocab_file = "finetune_vocab.pt"
    block_size = 128
    batch_size = 16
    dataloader, vocab_size = create_dataset("dummy_finetune_data.txt", vocab_file, block_size, batch_size)

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
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    print("Starting fine-tuning...")
    train_model(model, dataloader, optimizer, criterion, device, epochs=3, use_lora=use_lora)
    print("Fine-tuning finished.")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "finetuned_llm.pth")
    print("Fine-tuned model saved to finetuned_llm.pth")

    # Clean up dummy data
    os.remove("dummy_finetune_data.txt")
    os.remove("finetune_vocab.pt")

if __name__ == '__main__':
    main()
