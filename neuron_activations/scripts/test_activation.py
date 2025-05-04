"""
Test neural activation capture on a sample tweet
Shows how to use hooks to capture neuron activations right before first token output
"""

import os
import sys
import torch
import tiktoken

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from model import GPT, GPTConfig

# Import from local files
from tweet_activations import ActivationCapturer

# Sample tweet
SAMPLE_TWEET = "i feel so excited for college"
SAMPLE_LABEL = "joy"


def test_activations():
    # Create output directory
    os.makedirs("test_output", exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create GPT-2 config for the 124M parameter model
    config_args = dict(n_layer=12, n_head=12, n_embd=768)
    config_args["vocab_size"] = 50257  # GPT-2 vocab size
    config_args["block_size"] = 1024  # GPT-2 context window

    # Create model
    gptconf = GPTConfig(**config_args)
    model = GPT(gptconf)
    model.eval()
    model.to(device)

    # Get tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # Tokenize the sample tweet
    tokens = encode(SAMPLE_TWEET)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    print(f"Input tweet: '{SAMPLE_TWEET}'")

    # Create activation capturer and register hooks
    capturer = ActivationCapturer(model)
    capturer.register_hooks()

    try:
        # Generate one token with activation capture
        with torch.no_grad():
            # Enable activation capture
            capturer.start_capture()

            # Forward pass to get logits
            logits, _ = model(input_ids)

            # Get next token probabilities (we care about the activations, not the output)
            next_token_logits = logits[:, -1, :] / 0.8
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop capturing
            capturer.stop_capture()

            # Get activations
            activations = capturer.get_activations()
    finally:
        # Always remove hooks, even if there's an error
        capturer.remove_hooks()

    # Save activations
    gen_dir = os.path.join("test_output", "generation_activations")
    os.makedirs(gen_dir, exist_ok=True)

    # Create metadata
    metadata = {
        "tweet_ids": ["test_0"],
        "tweets": [SAMPLE_TWEET],
        "labels": [SAMPLE_LABEL],
        "generated_text": [decode(next_token.cpu().tolist()[0])],
        "token_lengths": [len(tokens)],
    }
    torch.save(metadata, os.path.join(gen_dir, "gen_metadata.pt"))

    # Save activations
    for layer_name, activation_data in activations.items():
        layer_data = [
            {"data": activation_data, "label": SAMPLE_LABEL, "tweet_id": "test_0"}
        ]
        torch.save(layer_data, os.path.join(gen_dir, f"{layer_name}_activations.pt"))

    print(f"Captured activations for {len(activations)} layers")

    # Print layer shapes
    for layer_name, activation_data in activations.items():
        print(f"Layer {layer_name}: shape {activation_data.shape}")


if __name__ == "__main__":
    test_activations()
