#!/usr/bin/env python3
"""
Capture detailed neural activations from GPT-2, including internal transformer components
This version captures attention and MLP activations separately
"""

import os
import sys
import json
import torch
import argparse
import tiktoken
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from model import GPTConfig, GPT

# For loading datasets
try:
    from datasets import load_dataset
except ImportError:
    print(
        "Warning: datasets library not installed. Direct loading from Hugging Face will not work."
    )
    print("Install with: pip install datasets")


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


def load_tweets(tweet_source, samples_per_emotion=100, emotions=None):
    """
    Load tweets from either a local JSON file or the dair-ai/emotion dataset

    Args:
        tweet_source: Path to JSON file or "dair" to load from HuggingFace
        samples_per_emotion: Number of samples per emotion to include
        emotions: List of emotions to include or None for all

    Returns:
        List of tweet dictionaries with 'text' and 'label' fields
    """
    if tweet_source.lower() == "dair":
        print("Loading tweets from dair-ai/emotion dataset")

        try:
            # Load the dataset
            dataset = load_dataset("dair-ai/emotion", split="train")
            label_names = dataset.features["label"].names

            print(f"Loaded dataset with {len(dataset)} samples")
            print(f"Available emotions: {label_names}")

            # Filter for specific emotions if requested
            if emotions:
                # Convert emotion names to IDs
                emotion_ids = [
                    label_names.index(emotion)
                    for emotion in emotions
                    if emotion in label_names
                ]

                # Filter dataset
                dataset = dataset.filter(
                    lambda example: example["label"] in emotion_ids
                )
                print(
                    f"Filtered to {len(dataset)} samples with emotions: {[label_names[id] for id in emotion_ids]}"
                )

            # Group samples by emotion for balanced sampling
            samples_by_emotion = {}
            for sample in dataset:
                emotion = label_names[sample["label"]]
                if emotion not in samples_by_emotion:
                    samples_by_emotion[emotion] = []
                samples_by_emotion[emotion].append(sample)

            # Sample balanced dataset
            balanced_samples = []
            for emotion, samples in samples_by_emotion.items():
                num_samples = min(samples_per_emotion, len(samples))
                print(f"Including {num_samples} samples for emotion: {emotion}")

                # Use deterministic sampling for reproducibility
                indices = np.random.RandomState(42).choice(
                    len(samples), num_samples, replace=False
                )

                for idx in indices:
                    balanced_samples.append(
                        {"text": samples[idx]["text"], "label": emotion}
                    )

            # Shuffle the final set
            np.random.RandomState(42).shuffle(balanced_samples)
            return balanced_samples

        except Exception as e:
            print(f"Error loading from dair-ai/emotion: {e}")
            raise

    else:
        # Load from local JSON file
        print(f"Loading tweets from local file: {tweet_source}")
        with open(tweet_source, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        # Filter by emotions if specified
        if emotions:
            tweets = [t for t in tweets if t["label"] in emotions]
            print(f"Filtered to {len(tweets)} samples with emotions: {emotions}")

        return tweets


class DetailedActivationCapturer:
    """Class to capture detailed activations using hooks, including internal transformer components"""

    def __init__(self, model):
        """
        Initialize detailed activation capturer to get internal transformer components

        Args:
            model: The GPT model
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        self.capture_enabled = False

    def _capture_hook(self, name):
        """Create a hook function for a specific component"""

        def hook(module, inputs, outputs):
            if self.capture_enabled:
                # Only get the last token position during generation
                if isinstance(outputs, tuple):
                    # Some modules return tuples, take first element
                    output = outputs[0]
                else:
                    output = outputs

                # Handle different output shapes
                if len(output.shape) == 3:  # [batch, seq, hidden]
                    self.activations[name] = output[:, -1, :].detach().cpu()
                else:
                    # Try to capture anyway
                    self.activations[name] = output.detach().cpu()

        return hook

    def register_hooks(self):
        """Register hooks on model components including internal transformer parts"""
        # Input embeddings
        self.hooks.append(
            self.model.transformer.drop.register_forward_hook(
                self._capture_hook("input_embeds")
            )
        )

        # Register hooks for each transformer layer and its components
        for i in range(self.model.config.n_layer):
            # Full layer output
            self.hooks.append(
                self.model.transformer.h[i].register_forward_hook(
                    self._capture_hook(f"layer_{i}_output")
                )
            )

            # Layer norm before attention
            self.hooks.append(
                self.model.transformer.h[i].ln_1.register_forward_hook(
                    self._capture_hook(f"layer_{i}_ln_1")
                )
            )

            # Attention component
            self.hooks.append(
                self.model.transformer.h[i].attn.register_forward_hook(
                    self._capture_hook(f"layer_{i}_attn")
                )
            )

            # Attention projections
            self.hooks.append(
                self.model.transformer.h[i].attn.c_attn.register_forward_hook(
                    self._capture_hook(f"layer_{i}_attn_proj")
                )
            )

            # Layer norm before MLP
            self.hooks.append(
                self.model.transformer.h[i].ln_2.register_forward_hook(
                    self._capture_hook(f"layer_{i}_ln_2")
                )
            )

            # MLP component
            self.hooks.append(
                self.model.transformer.h[i].mlp.register_forward_hook(
                    self._capture_hook(f"layer_{i}_mlp")
                )
            )

            # MLP intermediate (after first projection and activation)
            self.hooks.append(
                self.model.transformer.h[i].mlp.c_fc.register_forward_hook(
                    self._capture_hook(f"layer_{i}_mlp_fc")
                )
            )

        # Final layer norm
        self.hooks.append(
            self.model.transformer.ln_f.register_forward_hook(
                self._capture_hook("final_layer")
            )
        )

        # # Logits
        # self.hooks.append(
        #     self.model.lm_head.register_forward_hook(self._capture_hook("logits"))
        # )

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


def capture_detailed_activations(
    model,
    tweet_source,
    output_dir,
    device="cuda",
    max_tweets=None,
    batch_size=32,
    samples_per_emotion=100,
    emotions=None,
):
    """
    Process tweets and capture detailed activations during first token generation using hooks

    Args:
        model: GPT model
        tweet_source: Path to JSON file or "dair" to load from HuggingFace
        output_dir: Directory to save activations
        device: Device to run model on
        max_tweets: Maximum number of tweets to process (None for all)
        batch_size: Batch size for processing tweets
        samples_per_emotion: Number of samples per emotion (only for dair dataset)
        emotions: List of emotions to include (only used if loading from dair)
    """
    output_dir = os.path.join(output_dir, "detailed_activations")
    os.makedirs(output_dir, exist_ok=True)

    # Load tweets from either local file or dair dataset
    tweets = load_tweets(tweet_source, samples_per_emotion, emotions)

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
    capturer = DetailedActivationCapturer(model)
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
                            "data": activation,
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

            # Print sample occasionally
            if i % 20 == 0:
                print(f"Sample {i} ({label}): '{tweet[:30]}...' → '{generated_text}'")

            # Clear any unused memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    finally:
        # Always remove hooks, even if there's an error
        capturer.remove_hooks()

    # Save all activations
    for layer_name, data in activation_data.items():
        torch.save(data, os.path.join(output_dir, f"{layer_name}_activations.pt"))
        print(f"Saved {len(data)} activations for {layer_name}")

    # Save metadata
    metadata = {
        "tweet_ids": all_tweet_ids,
        "tweets": all_tweets,
        "labels": all_labels,
        "generated_text": all_generated,
        "token_lengths": all_token_lengths,
    }
    torch.save(metadata, os.path.join(output_dir, "gen_metadata.pt"))

    print(f"Saved metadata for {len(all_tweets)} tweets")

    # Also save as JSON for easier inspection
    with open(os.path.join(output_dir, "samples.json"), "w", encoding="utf-8") as f:
        samples_json = []
        for i in range(len(all_tweets)):
            samples_json.append(
                {
                    "id": all_tweet_ids[i],
                    "text": all_tweets[i],
                    "label": all_labels[i],
                    "generated": all_generated[i],
                }
            )
        json.dump(samples_json, f, indent=2, ensure_ascii=False)

    print(f"Saved samples to {os.path.join(output_dir, 'samples.json')}")


def list_available_components(model_path, device="cpu"):
    """List all available components in the model that can be captured"""
    model = load_model(model_path, device)

    print("\nAvailable Transformer Components:")
    print("=" * 50)

    # Embeddings
    print("\nEmbeddings:")
    print("  - input_embeds (token + position)")

    # Layer components
    for i in range(model.config.n_layer):
        print(f"\nLayer {i}:")
        print(f"  - layer_{i}_output (full layer output)")
        print(f"  - layer_{i}_ln_1 (layer norm before attention)")
        print(f"  - layer_{i}_attn (attention output)")
        print(f"  - layer_{i}_attn_proj (attention projections)")
        print(f"  - layer_{i}_ln_2 (layer norm before MLP)")
        print(f"  - layer_{i}_mlp (MLP output)")
        print(f"  - layer_{i}_mlp_fc (MLP intermediate)")

    # Final components
    print("\nFinal Components:")
    print("  - final_layer (final layer norm)")
    print("  - logits (token predictions)")


def analyze_component_sizes(output_dir):
    """Analyze the sizes of captured components"""
    output_dir = os.path.join(output_dir, "detailed_activations")

    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist")
        return

    # Get all activation files
    activation_files = [
        f for f in os.listdir(output_dir) if f.endswith("_activations.pt")
    ]

    print(f"\nFound {len(activation_files)} component activation files")
    print("=" * 50)

    # Collect info for each component
    component_info = []

    for file_name in activation_files:
        if file_name == "gen_metadata.pt":
            continue

        component_name = file_name.replace("_activations.pt", "")
        file_path = os.path.join(output_dir, file_name)

        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB

        # Load the first sample to get shape
        try:
            data = torch.load(file_path)
            if data and len(data) > 0 and "data" in data[0]:
                shape = data[0]["data"].shape
                component_info.append(
                    {
                        "component": component_name,
                        "size_mb": file_size,
                        "shape": shape,
                        "elements": np.prod(shape),
                        "samples": len(data),
                    }
                )
        except Exception as e:
            print(f"Error loading {component_name}: {e}")

    # Sort by component name for better organization
    component_info.sort(key=lambda x: x["component"])

    # Print information
    print(f"{'Component':<25} {'Shape':<15} {'Elements':<10} {'Size (MB)':<10}")
    print("-" * 60)

    for info in component_info:
        print(
            f"{info['component']:<25} {str(info['shape']):<15} {info['elements']:<10} {info['size_mb']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Capture detailed neural activations from GPT-2, including internal transformer components"
    )

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Capture command
    capture_parser = subparsers.add_parser(
        "capture", help="Capture detailed activations"
    )
    capture_parser.add_argument(
        "--model_path",
        type=str,
        default="../out/gpt2.pt",
        help="Path to model checkpoint",
    )
    capture_parser.add_argument(
        "--tweets_file",
        type=str,
        help="Path to JSON file with tweets and labels, or 'dair' to load from HuggingFace",
    )
    capture_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/captures",
        help="Directory for output files",
    )
    capture_parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run model on"
    )
    capture_parser.add_argument(
        "--max_tweets",
        type=int,
        default=None,
        help="Maximum number of tweets to process",
    )
    capture_parser.add_argument(
        "--samples_per_emotion",
        type=int,
        default=30,
        help="Number of samples per emotion (only for dair dataset)",
    )
    capture_parser.add_argument(
        "--emotions",
        type=str,
        nargs="+",
        default=["joy", "sadness", "anger"],
        help="Emotions to include (default: joy sadness anger)",
    )

    # List components command
    list_parser = subparsers.add_parser(
        "list", help="List available components in the model"
    )
    list_parser.add_argument(
        "--model_path",
        type=str,
        default="../out/gpt2.pt",
        help="Path to model checkpoint",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze captured component sizes"
    )
    analyze_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/captures",
        help="Directory containing captures",
    )

    args = parser.parse_args()

    if args.command == "capture":
        if not args.tweets_file:
            parser.error(
                "--tweets_file is required (use a path to JSON file or 'dair' for HuggingFace dataset)"
            )

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = load_model(args.model_path, device)

        capture_detailed_activations(
            model,
            args.tweets_file,
            args.output_dir,
            device=args.device,
            max_tweets=args.max_tweets,
            samples_per_emotion=args.samples_per_emotion,
            emotions=args.emotions,
        )

    elif args.command == "list":
        device = torch.device("cpu")  # Always use CPU for listing
        list_available_components(args.model_path, device)

    elif args.command == "analyze":
        analyze_component_sizes(args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
