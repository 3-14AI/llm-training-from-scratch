
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.dataset_creator import create_dataset
import os
import argparse
import deepspeed
from monitoring.train_monitor import init_wandb, log_metrics, finish_wandb
from monitoring.evaluation import evaluate_model


def train_model(model, dataloader, optimizer, criterion, device, scheduler=None, epochs=10, start_epoch=0, global_step=0, val_dataloader=None, eval_steps=100, is_deepspeed=False, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        # If using DistributedSampler, we need to set the epoch
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)

            if not is_deepspeed:
                optimizer.zero_grad()

            output = model(src, trg[:, :-1])
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            if is_deepspeed:
                model.backward(loss)
                model.step()
            else:
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

        # Evaluate at the end of each epoch as well
        if val_dataloader is not None:
            val_loss, val_perplexity = evaluate_model(model, val_dataloader, criterion, device)
            print(f"End of Epoch Validation - Epoch {epoch+1}, Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
            log_metrics({"val/epoch_loss": val_loss, "val/epoch_perplexity": val_perplexity, "val/epoch": epoch + 1})
            model.train()

        # Save Checkpoint
        if not is_deepspeed or (hasattr(model, 'module') and deepspeed.comm.get_rank() == 0) or (not hasattr(model, 'module')):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            state_dict = model.module.state_dict() if is_deepspeed and hasattr(model, 'module') else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train the LLM with optional context compression.")
    parser.add_argument("--use_compression", action="store_true", help="Use context compression transformer")
    parser.add_argument("--chunk_size", type=int, default=8, help="Size of context chunks to compress")
    parser.add_argument("--compressor_layers", type=int, default=2, help="Number of layers in the compressor")

    # DeepSpeed args
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed config file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    # Optional deepspeed args parsing natively
    args, unknown = parser.parse_known_args()

    is_deepspeed = args.deepspeed

    if is_deepspeed:
        deepspeed.init_distributed()
        device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"DeepSpeed initialized on device: {device}, local_rank: {args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Dummy data creation for demonstration
    dummy_text = "This is a sample text for pre-training the LLM. It should be a large corpus of text data. " * 1000
    with open("dummy_pretrain_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    dummy_val_text = "This is a sample text for validating the LLM. " * 200
    with open("dummy_val_data.txt", "w", encoding="utf-8") as f:
        f.write(dummy_val_text)

    vocab_file = "pretrain_vocab.pt"
    block_size = 128
    batch_size = 16
    dataloader, vocab_size = create_dataset("dummy_pretrain_data.txt", vocab_file, block_size, batch_size, is_distributed=is_deepspeed)
    val_dataloader, _ = create_dataset("dummy_val_data.txt", vocab_file, block_size, batch_size, is_distributed=is_deepspeed)

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

    if is_deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=args.deepspeed_config if args.deepspeed_config else None
        )

    # Init WandB
    config = {
        "learning_rate": 0.0001,
        "epochs": 5,
        "batch_size": batch_size,
        "block_size": block_size,
        "embed_size": embed_size,
        "num_layers": num_layers,
        "heads": heads,
        "use_compression": args.use_compression,
    }
    if args.use_compression:
        config.update({"chunk_size": args.chunk_size, "compressor_layers": args.compressor_layers})

    init_wandb("llm-pretraining", "pretrain-run", config)

    print("Starting pre-training...")
    train_model(model, dataloader, optimizer, criterion, device, scheduler=scheduler, epochs=5, start_epoch=start_epoch, global_step=global_step, val_dataloader=val_dataloader, eval_steps=100, is_deepspeed=is_deepspeed)
    print("Pre-training finished.")

    finish_wandb()

    # Save the pre-trained model
    if not is_deepspeed or args.local_rank <= 0:
        save_path = "pretrained_llm_compressed.pth" if args.use_compression else "pretrained_llm.pth"

        # When using deepspeed, model might be DeepSpeedEngine, use module.state_dict() if so
        state_dict = model.module.state_dict() if is_deepspeed and hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, save_path)
        print(f"Pre-trained model saved to {save_path}")

    # Clean up dummy data
    os.remove("dummy_pretrain_data.txt")
    os.remove("dummy_val_data.txt")
    os.remove("pretrain_vocab.pt")

if __name__ == '__main__':
    main()
