import sys
import os

# Add the parent directory to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import json
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.tokenizer import BPETokenizer
from inference.generate import generate_text

def main():
    parser = argparse.ArgumentParser(description="Run inference and output JSON.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model checkpoint")
    parser.add_argument("--use_compression", action="store_true", help="Use context compression transformer")
    parser.add_argument("--chunk_size", type=int, default=8, help="Size of context chunks used during training")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    # We should search for it in standard locations or just use the current directory
    vocab_file = "pretrain_vocab.pt"
    if not os.path.exists(vocab_file):
        # Try to find it in the parent directory
        vocab_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "pretrain_vocab.pt")

    tokenizer = BPETokenizer()
    if os.path.exists(vocab_file):
        tokenizer.load_vocab(vocab_file)
    else:
        # Fallback for testing
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
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), model_path)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    try:
        generated_text = generate_text(model, tokenizer, args.prompt, args.max_tokens, device, args.temperature, args.top_k, args.top_p, args.repetition_penalty)
        result = {
            "prompt": args.prompt,
            "generated_text": generated_text,
            "status": "success"
        }
    except Exception as e:
        result = {
            "status": "error",
            "error": str(e)
        }

    # Output only JSON to standard output
    print(json.dumps(result))

if __name__ == '__main__':
    main()
