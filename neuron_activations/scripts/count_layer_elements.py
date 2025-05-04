#!/usr/bin/env python3
"""
Script to count and analyze the number of elements in each activation layer file
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def count_layer_elements(activations_dir, detailed=False, save_plots=False):
    """Count elements in each layer's activation file"""
    print(f"Analyzing activations in {activations_dir}")

    # Get all activation files
    activation_files = [
        f for f in os.listdir(activations_dir) if f.endswith("_activations.pt")
    ]

    if not activation_files:
        print("No activation files found")
        return

    print(f"Found {len(activation_files)} activation files")

    # Create output directory for plots if needed
    plots_dir = os.path.join(activations_dir, "activation_plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    # Analyze each layer
    layer_stats = {}

    for file_name in sorted(activation_files):
        file_path = os.path.join(activations_dir, file_name)
        layer_name = file_name.replace("_activations.pt", "")

        try:
            # Load the activation data
            print(f"Loading {layer_name}...")
            activation_data = torch.load(file_path)

            if not activation_data:
                print(f"  No data in {layer_name}")
                continue

            # Count samples
            num_samples = len(activation_data)

            # Get shapes
            shapes = defaultdict(int)
            total_elements = 0

            # For detailed analysis
            all_values = []
            activations_by_label = defaultdict(list)

            for sample in activation_data:
                if "data" not in sample:
                    continue

                activation = sample["data"]
                label = sample.get("label", "unknown")
                shape_str = str(activation.shape)
                shapes[shape_str] += 1

                # Convert to numpy if needed
                if isinstance(activation, torch.Tensor):
                    activation_np = activation.numpy()
                else:
                    activation_np = np.array(activation)

                # Count total elements (neurons × features)
                total_elements += activation_np.size

                # For detailed analysis
                if detailed:
                    # Flatten for distribution analysis
                    flat_values = activation_np.flatten()
                    all_values.append(flat_values)

                    # Store by label for analysis
                    activations_by_label[label].append(activation_np)

            # Store stats
            layer_stats[layer_name] = {
                "samples": num_samples,
                "shapes": dict(shapes),
                "total_elements": total_elements,
                "elements_per_sample": total_elements / num_samples
                if num_samples > 0
                else 0,
            }

            print(
                f"  {layer_name}: {num_samples} samples, {total_elements} total elements"
            )
            for shape, count in shapes.items():
                print(f"    Shape {shape}: {count} samples")

            # Detailed analysis if requested
            if detailed and all_values:
                # Concatenate all values for distribution analysis
                all_values_array = np.concatenate(all_values)

                # Calculate distribution stats
                min_val = np.min(all_values_array)
                max_val = np.max(all_values_array)
                mean_val = np.mean(all_values_array)
                std_val = np.std(all_values_array)

                # Add to stats
                layer_stats[layer_name].update(
                    {
                        "min_val": min_val,
                        "max_val": max_val,
                        "mean_val": mean_val,
                        "std_val": std_val,
                    }
                )

                print(f"    Value range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"    Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")

                # Find most active neurons (for layers where it makes sense)
                if (
                    any(s.endswith("]") and "768" in s for s in shapes.keys())
                    and len(activation_np.shape) <= 2
                ):
                    # Stack samples by label
                    label_means = {}
                    for label, label_activations in activations_by_label.items():
                        # For each label, average across samples
                        try:
                            # Reshape to handle both [1, 768] and [768] shapes
                            reshaped = [
                                a.reshape(-1) if len(a.shape) > 1 else a
                                for a in label_activations
                            ]
                            stacked = np.vstack(reshaped)
                            label_means[label] = np.mean(stacked, axis=0)
                        except Exception as e:
                            print(f"    Error processing {label} samples: {e}")

                    # Show most active neurons by label
                    if label_means:
                        print("\n    Most active neurons by emotion:")
                        for label, means in label_means.items():
                            # Get top 5 neurons
                            top_indices = np.argsort(-means)[:5]
                            print(
                                f"      {label}: {', '.join([f'neuron {i} ({means[i]:.4f})' for i in top_indices])}"
                            )

                # Plot distribution if requested
                if save_plots:
                    plt.figure(figsize=(10, 6))

                    # Histogram of all values
                    plt.hist(all_values_array, bins=50, alpha=0.7)
                    plt.title(f"Activation Distribution - {layer_name}")
                    plt.xlabel("Activation Value")
                    plt.ylabel("Frequency")
                    plt.grid(alpha=0.3)

                    # Add vertical line at mean
                    plt.axvline(
                        mean_val,
                        color="r",
                        linestyle="--",
                        label=f"Mean: {mean_val:.4f}",
                    )
                    plt.legend()

                    # Save plot
                    plot_path = os.path.join(
                        plots_dir, f"{layer_name}_distribution.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"    Saved distribution plot to {plot_path}")

                    # For layers with 768 features, also plot neuron activations across samples
                    if any("768" in s for s in shapes.keys()):
                        plt.figure(figsize=(12, 6))

                        # Plot average activation per neuron across all samples
                        neuron_means = np.mean(
                            [a.reshape(-1) for a in all_values], axis=0
                        )
                        neuron_indices = np.arange(len(neuron_means))

                        plt.bar(neuron_indices[:50], neuron_means[:50], alpha=0.7)
                        plt.title(f"Top 50 Neuron Activations - {layer_name}")
                        plt.xlabel("Neuron Index")
                        plt.ylabel("Mean Activation")
                        plt.grid(alpha=0.3)

                        # Save plot
                        plot_path = os.path.join(
                            plots_dir, f"{layer_name}_top_neurons.png"
                        )
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"    Saved neuron activation plot to {plot_path}")

        except Exception as e:
            print(f"Error processing {layer_name}: {e}")

    # Print summary
    print("\nSUMMARY:")
    print("=" * 80)
    print(
        f"{'Layer':<20} {'Samples':<10} {'Elements/Sample':<20} {'Total Elements':<15} {'Shape(s)'}"
    )
    print("-" * 80)

    for layer_name, stats in sorted(layer_stats.items()):
        shapes_str = ", ".join(f"{s}({c})" for s, c in stats["shapes"].items())
        print(
            f"{layer_name:<20} {stats['samples']:<10} {stats['elements_per_sample']:<20.0f} {stats['total_elements']:<15} {shapes_str}"
        )

    # Print value distribution summary if detailed analysis was performed
    if detailed:
        print("\nVALUE DISTRIBUTION SUMMARY:")
        print("=" * 80)
        print(f"{'Layer':<20} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std Dev':<10}")
        print("-" * 80)

        for layer_name, stats in sorted(layer_stats.items()):
            if "min_val" in stats:
                print(
                    f"{layer_name:<20} {stats['min_val']:<10.4f} {stats['max_val']:<10.4f} {stats['mean_val']:<10.4f} {stats['std_val']:<10.4f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Count and analyze elements in each layer's activation file"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="output/captures/generation_activations",
        help="Directory containing activation files",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Perform detailed analysis of activation values",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save distribution plots for each layer",
    )

    args = parser.parse_args()

    if not os.path.exists(args.activations_dir):
        print(f"Error: Directory {args.activations_dir} not found")
        sys.exit(1)

    count_layer_elements(
        args.activations_dir, detailed=args.detailed, save_plots=args.save_plots
    )


if __name__ == "__main__":
    main()
