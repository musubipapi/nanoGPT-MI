"""
Capture neural activations from GPT-2 when processing labeled tweet datasets
Using hooks for more precise and efficient activation capture
"""

import os
import sys
import json
import torch
import argparse
import tiktoken
from tqdm import tqdm
import gc

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
# Import model from parent directory
from model import GPTConfig, GPT


def load_model(model_path, device):
    """Load a GPT model from checkpoint"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    # Fix for checkpoint keys with unwanted prefix
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_encoder():
    """Get the tokenizer/encoder"""
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


class ActivationCapturer:
    """Class to capture activations using hooks"""

    def __init__(self, model, layer_names=None):
        """
        Initialize activation capturer with specified layers

        Args:
            model: The GPT model
            layer_names: List of layer names to capture. If None, captures all transformer layers,
                       final layer norm, and logits
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        self.capture_enabled = False

        # Default layers to capture if none specified
        if layer_names is None:
            # Create default layer names: input_embeds, layer_0 to layer_11, final_layer
            n_layers = model.config.n_layer
            layer_names = (
                ["input_embeds"]
                + [f"layer_{i}" for i in range(n_layers)]
                + ["final_layer"]
            )

        self.layer_names = layer_names

    def _capture_input_hook(self, module, inputs, outputs):
        """Hook for capturing input embeddings"""
        if self.capture_enabled:
            # Only get the last token position during generation
            self.activations["input_embeds"] = outputs[:, -1, :].detach().cpu()

    def _capture_layer_hook(self, name):
        """Create a hook function for a specific layer"""

        def hook(module, inputs, outputs):
            if self.capture_enabled:
                # Only get the last token position during generation
                self.activations[name] = outputs[:, -1, :].detach().cpu()

        return hook

    def register_hooks(self):
        """Register all hooks on the model"""
        # Input embeddings
        self.hooks.append(
            self.model.transformer.drop.register_forward_hook(self._capture_input_hook)
        )

        # Transformer layers
        for i in range(self.model.config.n_layer):
            layer_name = f"layer_{i}"
            if layer_name in self.layer_names:
                self.hooks.append(
                    self.model.transformer.h[i].register_forward_hook(
                        self._capture_layer_hook(layer_name)
                    )
                )

        # Final layer norm
        if "final_layer" in self.layer_names:
            self.hooks.append(
                self.model.transformer.ln_f.register_forward_hook(
                    self._capture_layer_hook("final_layer")
                )
            )

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def start_capture(self):
        """Enable activation capturing"""
        self.activations = {}
        self.capture_enabled = True

    def stop_capture(self):
        """Disable activation capturing"""
        self.capture_enabled = False

    def get_activations(self):
        """Get the captured activations"""
        return self.activations


def capture_activations(
    model,
    tweets_file,
    output_dir,
    device="cuda",
    max_tweets=None,
    batch_size=32,
):
    """
    Process tweets and capture activations during first token generation using hooks

    Args:
        model: GPT model
        tweets_file: JSON file with tweets and their labels
        output_dir: Directory to save activations
        device: Device to run model on
        max_tweets: Maximum number of tweets to process (None for all)
        batch_size: Batch size for processing tweets
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tweets
    with open(tweets_file, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    if max_tweets is not None:
        tweets = tweets[:max_tweets]

    encode, decode = get_encoder()

    # Prepare storage for all metadata
    all_tweet_ids = []
    all_tweets = []
    all_labels = []
    all_generated = []
    all_token_lengths = []

    # Initialize dictionaries to store activations by layer
    activation_data = {}

    # Create activation capturer and register hooks
    capturer = ActivationCapturer(model)
    capturer.register_hooks()

    try:
        # Process tweets one by one
        for i in tqdm(range(len(tweets)), desc="Processing tweets"):
            tweet_data = tweets[i]
            tweet_id = str(i)
            tweet = tweet_data["text"]
            label = tweet_data["label"]

            # Tokenize tweet
            input_ids = encode(tweet)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Generate the first token with activation capture
            with torch.no_grad():
                # Enable activation capture
                capturer.start_capture()

                # If sequence is too long, crop it
                if input_tensor.size(1) > model.config.block_size:
                    input_tensor = input_tensor[:, -model.config.block_size :]

                # Forward pass
                logits, _ = model(input_tensor)

                # Get logits for the last position and sample the next token
                next_token_logits = logits[:, -1, :] / 0.8  # temperature 0.8

                # Sample from the distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Get the generated token as text
                generated_text = decode(next_token.cpu().tolist()[0])

                # Stop capturing
                capturer.stop_capture()

                # Get activations
                activations = capturer.get_activations()

                # Store activations
                for layer_name, activation in activations.items():
                    if layer_name not in activation_data:
                        activation_data[layer_name] = []

                    activation_data[layer_name].append(
                        {
                            "data": activation,  # Already detached and on CPU
                            "label": label,
                            "tweet_id": tweet_id,
                        }
                    )

            # Store metadata
            all_tweet_ids.append(tweet_id)
            all_tweets.append(tweet)
            all_labels.append(label)
            all_generated.append(generated_text)
            all_token_lengths.append(len(input_ids))

            # Print progress sample occasionally
            if i % 50 == 0:
                print(f"Sample {i}: '{tweet[:30]}...' → '{generated_text}' ({label})")

    finally:
        # Always remove hooks, even if there's an error
        capturer.remove_hooks()

    # Save metadata
    torch_metadata = {
        "tweet_ids": all_tweet_ids,
        "tweets": all_tweets,
        "labels": all_labels,
        "generated_text": all_generated,
        "token_lengths": all_token_lengths,
    }
    torch.save(torch_metadata, os.path.join(output_dir, "gen_metadata.pt"))

    # Save each layer's activations separately
    for layer_name, layer_data in activation_data.items():
        torch.save(layer_data, os.path.join(output_dir, f"{layer_name}_activations.pt"))

    # Clean up
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Generation processing complete! Saved activations to {output_dir}")


def process_multi_emotion_dataset(
    model,
    tweets_file,
    output_dir,
    device="cuda",
    batch_size=32,
):
    """
    Process a multi-emotion dataset and capture activations in generation mode

    Args:
        model: GPT model
        tweets_file: JSON file with tweets and their labels
        output_dir: Directory to save activations
        device: Device to run model on
        batch_size: Batch size for processing tweets
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create directory for generation activations
    gen_dir = os.path.join(output_dir, "generation_activations")
    os.makedirs(gen_dir, exist_ok=True)

    # Process generation activations
    print(f"\n{'-' * 40}\nProcessing generation mode activations\n{'-' * 40}")
    capture_activations(
        model,
        tweets_file,
        gen_dir,
        device,
        max_tweets=None,
        batch_size=batch_size,
    )

    # Return paths
    return {"tweets_file": tweets_file, "generate_dir": gen_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Capture GPT-2 activations on labeled tweets using hooks"
    )
    parser.add_argument(
        "--tweets_file",
        type=str,
        required=True,
        help="JSON file containing tweets with labels",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tweet_activations",
        help="Directory to save activations",
    )
    parser.add_argument(
        "--model_path", type=str, default="out/gpt2.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run model on"
    )
    parser.add_argument(
        "--max_tweets",
        type=int,
        default=None,
        help="Maximum number of tweets to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing tweets",
    )

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    # Process tweets with hooks
    capture_activations(
        model,
        args.tweets_file,
        args.output_dir,
        device,
        args.max_tweets,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
