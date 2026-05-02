
import torch
import torch.nn.functional as F
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.tokenizer import BPETokenizer
import os
import argparse

def generate_text(model, tokenizer, prompt, max_new_tokens, device, temperature=1.0):
    model.eval()
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = []
    for _ in range(max_new_tokens):
        # Determine the effective max length
        if isinstance(model, CompressedTransformer):
            # For compressed transformer, the main transformer sees num_chunks
            # But the compressor sees the raw tokens.
            max_tokens = model.transformer.encoder.position_embedding.num_embeddings * model.chunk_size
        else:
            max_tokens = model.encoder.position_embedding.num_embeddings

        # If the context window is full, truncate it
        if context.shape[1] > max_tokens:
            context = context[:, -max_tokens:]

        # Get predictions
        with torch.no_grad():
            output = model(context, context)

        # Focus only on the last token's prediction
        logits = output[:, -1, :]

        # Apply temperature for sampling
        logits = logits / (temperature + 1e-8)
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to the list of generated tokens and update context
        generated_tokens.append(next_token.item())
        context = torch.cat((context, next_token), dim=1)

    return tokenizer.decode(generated_tokens)

def main():
    parser = argparse.ArgumentParser(description="Generate text using the trained LLM.")
    parser.add_argument("--use_compression", action="store_true", help="Use context compression transformer")
    parser.add_argument("--chunk_size", type=int, default=8, help="Size of context chunks used during training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    vocab_file = "pretrain_vocab.pt"
    tokenizer = BPETokenizer()
    if os.path.exists(vocab_file):
        tokenizer.load_vocab(vocab_file)
        print(f"Loaded vocabulary from {vocab_file}")
    else:
        # Fallback for testing
        print(f"Warning: Vocabulary file {vocab_file} not found. Training a dummy one.")
        tokenizer.train("this is a sample text for pre-training the llm. it should be a large corpus of text data.")

    # Model parameters
    src_vocab_size = tokenizer.current_id
    trg_vocab_size = tokenizer.current_id
    src_pad_idx = 0 
    trg_pad_idx = 0 
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 128

    if args.use_compression:
        model = CompressedTransformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            embed_size, num_layers, forward_expansion, heads, dropout, device, max_length,
            chunk_size=args.chunk_size
        ).to(device)
        default_path = "pretrained_llm_compressed.pth"
    else:
        model = Transformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            embed_size, num_layers, forward_expansion, heads, dropout, device, max_length
        ).to(device)
        default_path = "pretrained_llm.pth"

    model_path = args.model_path if args.model_path else default_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model weights file {model_path} not found. Using uninitialized model.")

    prompt = "Hello world, this is a"
    max_new_tokens = 20
    temperature = 0.7

    print(f"\nPrompt: {prompt}")
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens, device, temperature)
    print(f"Generated text: {prompt} {generated_text}")

if __name__ == '__main__':
    main()
