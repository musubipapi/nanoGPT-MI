#!/usr/bin/env python3
"""
Script to inspect the shapes of all activation tensors
"""

import os
import torch
import numpy as np

# Directory containing the activation files
activations_dir = "output/captures/detailed_activations"  # Current directory

# Get all activation files
activation_files = [
    f for f in os.listdir(activations_dir) if f.endswith("_activations.pt")
]

if not activation_files:
    print("No activation files found")
else:
    print(f"Found {len(activation_files)} activation files")
    print(
        f"{'Layer':<30} {'Shape':<20} {'Type':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Size (MB)':<10}"
    )
    print("-" * 100)

    for file_name in sorted(activation_files):
        if file_name == "gen_metadata.pt":
            continue

        layer_name = file_name.replace("_activations.pt", "")
        file_path = os.path.join(activations_dir, file_name)

        # Get file size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        try:
            # Load the activation data
            data = torch.load(file_path)

            if not data or not isinstance(data, list) or len(data) == 0:
                print(f"{layer_name:<30} Empty or invalid data")
                continue

            if "data" in data[0]:
                # Get the tensor from the first sample
                tensor = data[0]["data"]
                if isinstance(tensor, torch.Tensor):
                    # Get statistics
                    tensor_np = tensor.numpy()
                    min_val = np.min(tensor_np)
                    max_val = np.max(tensor_np)
                    mean_val = np.mean(tensor_np)

                    # Print information
                    print(
                        f"{layer_name:<30} {str(tensor.shape):<20} {str(tensor.dtype):<10} {min_val:<10.4f} {max_val:<10.4f} {mean_val:<10.4f} {file_size_mb:<10.2f}"
                    )
                else:
                    print(
                        f"{layer_name:<30} {str(tensor.shape):<20} Not a tensor {file_size_mb:<10.2f}"
                    )
            else:
                print(
                    f"{layer_name:<30} No 'data' field in sample {file_size_mb:<10.2f}"
                )

        except Exception as e:
            print(f"{layer_name:<30} ERROR: {str(e)} {file_size_mb:<10.2f}")

# Group components by type
print("\nComponent Types:")
component_types = {
    "input": [f for f in activation_files if "input" in f],
    "layer_output": [f for f in activation_files if "_output_" in f],
    "attention": [f for f in activation_files if "_attn" in f and not "_proj" in f],
    "attention_proj": [f for f in activation_files if "_attn_proj" in f],
    "mlp": [f for f in activation_files if "_mlp_" in f and not "_fc" in f],
    "mlp_fc": [f for f in activation_files if "_mlp_fc" in f],
    "layer_norm": [f for f in activation_files if "_ln_" in f],
    "final": [f for f in activation_files if "final_" in f],
}

for comp_type, files in component_types.items():
    if files:
        print(f"{comp_type}: {len(files)} components")

# Calculate total size of activation files
total_size_mb = sum(
    os.path.getsize(os.path.join(activations_dir, f)) / (1024 * 1024)
    for f in activation_files
)
print(f"\nTotal size of activation files: {total_size_mb:.2f} MB")

# Count total number of samples
if activation_files:
    # Get the first file to count samples
    first_file = os.path.join(activations_dir, activation_files[0])
    try:
        data = torch.load(first_file)
        if isinstance(data, list):
            print(f"Total number of samples: {len(data)}")
    except Exception as e:
        print(f"Could not count samples: {str(e)}")

# Sample command to run tweet activation script with 1000 tweets:
# python ../../scripts/tweet_activations.py --tweets_file path/to/your/tweets.json --output_dir . --model_path ../../../out/gpt2.pt --max_tweets 1000
