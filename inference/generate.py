
import torch
import torch.nn.functional as F
from model_architecture.transformer import Transformer
from data_preparation.tokenizer import SimpleTokenizer
import os

def generate_text(model, tokenizer, prompt, max_new_tokens, device, temperature=1.0):
    model.eval()
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = []
    for _ in range(max_new_tokens):
        # If the context window is full, truncate it
        if context.shape[1] > model.encoder.position_embedding.num_embeddings:
            context = context[:, -model.encoder.position_embedding.num_embeddings:]

        # Get predictions
        with torch.no_grad():
            output = model(context, context)

        # Focus only on the last token's prediction
        logits = output[:, -1, :]

        # Apply temperature for sampling
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to the list of generated tokens and update context
        generated_tokens.append(next_token.item())
        context = torch.cat((context, next_token), dim=1)

        # Optional: break if an end-of-sequence token is generated
        # if next_token.item() == tokenizer.word_to_id.get('<eos>', -1):
        #     break

    return tokenizer.decode(generated_tokens)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer (assuming it was saved during pre-training or fine-tuning)
    vocab_file = "pretrain_vocab.pt" # Or finetune_vocab.pt
    tokenizer = SimpleTokenizer()
    if os.path.exists(vocab_file):
        tokenizer.load_vocab(vocab_file)
        print(f"Loaded vocabulary from {vocab_file}")
    else:
        print(f"Error: Vocabulary file {vocab_file} not found. Please run pre-training or fine-tuning first.")
        return

    # Model parameters (must match the trained model)
    src_vocab_size = tokenizer.current_id
    trg_vocab_size = tokenizer.current_id
    src_pad_idx = 0 
    trg_pad_idx = 0 
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 128 # Should match block_size used during training

    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
        embed_size, num_layers, forward_expansion, heads, dropout, device, max_length
    ).to(device)

    # Load trained model weights
    model_path = "pretrained_llm.pth" # Or finetuned_llm.pth
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Error: Model weights file {model_path} not found. Please run pre-training or fine-tuning first.")
        return

    prompt = "Hello world, this is a"
    max_new_tokens = 50
    temperature = 0.7

    print(f"\nPrompt: {prompt}")
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens, device, temperature)
    print(f"Generated text: {prompt} {generated_text}")

if __name__ == '__main__':
    main()
