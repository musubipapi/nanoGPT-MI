#!/usr/bin/env python3
"""
Utility script to view the contents of a layer's activation file (.pt)
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


def view_layer_activations(activations_file):
    """View the contents of a layer's activation file"""
    print(f"Loading activations from {activations_file}")

    # Load the activation data
    activation_data = torch.load(activations_file)

    if not activation_data:
        print("No activation data found")
        return

    print(f"Found {len(activation_data)} samples in activation file")

    # Display information about the first few samples
    for i, sample in enumerate(activation_data[:5]):
        print(f"\nSample {i}:")
        print(f"  Label: {sample['label']}")
        print(f"  Tweet ID: {sample['tweet_id']}")

        # Shape information
        activation = sample["data"]
        print(f"  Activation shape: {activation.shape}")

        # Display basic statistics
        if isinstance(activation, torch.Tensor):
            activation_np = activation.numpy()
        else:
            activation_np = np.array(activation)

        print(f"  Min value: {activation_np.min():.4f}")
        print(f"  Max value: {activation_np.max():.4f}")
        print(f"  Mean value: {activation_np.mean():.4f}")
        print(f"  Std deviation: {activation_np.std():.4f}")

    # Find neurons with highest activation
    all_activations = []
    for sample in activation_data:
        act = sample["data"]
        if isinstance(act, torch.Tensor):
            act = act.numpy()
        all_activations.append(act)

    # Stack activations if possible
    try:
        stacked = np.vstack(all_activations)
        mean_activations = np.mean(stacked, axis=0)

        # Get top 10 neurons by mean activation
        top_neuron_indices = np.argsort(-mean_activations)[:10]
        print("\nTop 10 neurons by mean activation:")
        for i, idx in enumerate(top_neuron_indices):
            print(f"  Neuron {idx}: {mean_activations[idx]:.4f}")
    except:
        print("\nCould not stack activations - inconsistent shapes")

    # Create a visualization if activations are 1D or 2D
    try:
        # Get first sample for visualization
        first_sample = activation_data[0]["data"]
        if isinstance(first_sample, torch.Tensor):
            first_sample = first_sample.numpy()

        # Plot based on shape
        if len(first_sample.shape) == 1:
            # 1D vector - plot directly
            plt.figure(figsize=(12, 6))
            plt.plot(first_sample)
            plt.title(f"Activation values for sample 0 ({activation_data[0]['label']})")
            plt.xlabel("Neuron Index")
            plt.ylabel("Activation Value")
            plt.grid(alpha=0.3)

            # Save plot
            plot_file = os.path.splitext(activations_file)[0] + "_plot.png"
            plt.savefig(plot_file)
            print(f"\nPlot saved to {plot_file}")

        elif len(first_sample.shape) == 2:
            # 2D - create heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(first_sample, cmap="viridis", aspect="auto")
            plt.colorbar(label="Activation Value")
            plt.title(
                f"Activation heatmap for sample 0 ({activation_data[0]['label']})"
            )
            plt.xlabel("Feature Dimension")
            plt.ylabel("Position")

            # Save plot
            plot_file = os.path.splitext(activations_file)[0] + "_heatmap.png"
            plt.savefig(plot_file)
            print(f"\nHeatmap saved to {plot_file}")
    except Exception as e:
        print(f"\nError creating visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="View the contents of a layer's activation file (.pt)"
    )
    parser.add_argument(
        "activations_file", type=str, help="Path to the layer's activation file (.pt)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.activations_file):
        print(f"Error: File {args.activations_file} not found")
        sys.exit(1)

    view_layer_activations(args.activations_file)


if __name__ == "__main__":
    main()
