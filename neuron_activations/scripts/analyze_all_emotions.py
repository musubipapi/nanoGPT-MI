#!/usr/bin/env python3
"""
Analyze and visualize neural activations for each emotion present in the data
without tuning toward any specific emotion
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from sklearn.decomposition import PCA
import pandas as pd


def load_activations(activations_dir):
    """Load activations from PyTorch files"""
    # Load metadata
    metadata = torch.load(
        os.path.join(activations_dir, "gen_metadata.pt"), weights_only=False
    )

    # List all activation files
    activation_files = [
        f for f in os.listdir(activations_dir) if f.endswith("_activations.pt")
    ]

    # Extract layer names from filenames
    layers = [
        os.path.basename(f).replace("_activations.pt", "") for f in activation_files
    ]

    # For analysis, focus on these key layers
    key_layers = ["final_layer", "layer_11", "layer_10", "layer_9"]
    key_layers = [
        layer for layer in key_layers if f"{layer}_activations.pt" in activation_files
    ]

    # Load activations for key layers
    activations_by_layer = {}
    activations_by_emotion = defaultdict(lambda: defaultdict(list))
    neuron_importance_by_emotion = defaultdict(dict)

    for layer in key_layers:
        layer_file = os.path.join(activations_dir, f"{layer}_activations.pt")
        layer_data = torch.load(layer_file, weights_only=False)

        # Extract activations by emotion
        all_activations = []
        all_labels = []

        for i, item in enumerate(layer_data):
            # Skip entries that don't match our expected format
            if i >= len(metadata["labels"]):
                continue

            try:
                # Extract the data for the last token position
                if isinstance(item, dict) and "data" in item:
                    # Get the activation data
                    if isinstance(item["data"], torch.Tensor):
                        # Convert to numpy array
                        neuron_activations = item["data"].numpy()
                    else:
                        neuron_activations = item["data"]

                    # Handle different tensor shapes
                    orig_shape = neuron_activations.shape

                    # For 3D tensors like (1, 1, 50257) commonly seen in logits
                    if (
                        len(neuron_activations.shape) == 3
                        and neuron_activations.shape[0] == 1
                        and neuron_activations.shape[1] == 1
                    ):
                        neuron_activations = neuron_activations[0, 0, :]
                    # For 3D tensors like (1, sequence_length, hidden_size)
                    elif (
                        len(neuron_activations.shape) == 3
                        and neuron_activations.shape[0] == 1
                    ):
                        # Average across sequence dimension or take the last token
                        neuron_activations = np.mean(neuron_activations[0], axis=0)
                    # For 2D tensors like (sequence_length, hidden_size)
                    elif len(neuron_activations.shape) == 2:
                        # Average across sequence dimension
                        neuron_activations = np.mean(neuron_activations, axis=0)
                    # Already 1D, no change needed
                    elif len(neuron_activations.shape) == 1:
                        pass
                    else:
                        # Flatten any other shape completely
                        print(f"Flattening unusual tensor shape {orig_shape}")
                        neuron_activations = neuron_activations.flatten()

                    label = metadata["labels"][i]

                    # Store by emotion
                    activations_by_emotion[label][layer].append(neuron_activations)

                    # Store all activations
                    all_activations.append(neuron_activations)
                    all_labels.append(label)
            except (KeyError, IndexError) as e:
                print(f"Error processing item {i} for layer {layer}: {e}")
                continue

        # Skip if no valid data
        if not all_activations:
            print(f"No valid activations found for layer {layer}")
            continue

        # Process activations together - no filtering by dimension now since we've standardized the shapes
        try:
            activations_array = np.stack(all_activations)
            activations_by_layer[layer] = {
                "activations": activations_array,
                "labels": np.array(all_labels),
            }

            print(
                f"Layer {layer}: Loaded {len(all_activations)} activations with dimension {all_activations[0].shape}"
            )
        except ValueError as e:
            print(f"Error creating activation array for {layer}: {e}")
            print(f"Shapes of activations: {[a.shape for a in all_activations]}")

            # Try to find the most common shape and use only those activations
            from collections import Counter

            shapes = [a.shape for a in all_activations]
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            print(f"Using only activations with shape {most_common_shape}")

            filtered_activations = []
            filtered_labels = []
            for act, lbl in zip(all_activations, all_labels):
                if act.shape == most_common_shape:
                    filtered_activations.append(act)
                    filtered_labels.append(lbl)

            try:
                activations_array = np.stack(filtered_activations)
                activations_by_layer[layer] = {
                    "activations": activations_array,
                    "labels": np.array(filtered_labels),
                }
                print(
                    f"Layer {layer}: Loaded {len(filtered_activations)} activations after filtering"
                )
            except ValueError:
                print(f"Still unable to process layer {layer}, skipping")
                continue

        # Find important neurons for each emotion
        unique_emotions = np.unique(all_labels)

        for emotion in unique_emotions:
            # Get indices for this emotion
            emotion_indices = np.where(np.array(all_labels) == emotion)[0]
            other_indices = np.where(np.array(all_labels) != emotion)[0]

            if len(emotion_indices) > 0 and len(other_indices) > 0:
                # Calculate difference in activations
                emotion_activations = activations_array[emotion_indices]
                other_activations = activations_array[other_indices]

                emotion_mean = np.mean(emotion_activations, axis=0)
                other_mean = np.mean(other_activations, axis=0)

                # Calculate difference
                diff = emotion_mean - other_mean

                # Get top neurons (limited to the dimension of the vector)
                top_n = min(20, diff.shape[0])
                top_neurons = np.argsort(-diff)[:top_n]

                neuron_importance_by_emotion[emotion][layer] = {
                    "neurons": top_neurons,
                    "scores": diff[top_neurons],
                }

                print(
                    f"Layer {layer}, Emotion {emotion}: Found {top_n} important neurons"
                )

    return (
        metadata,
        activations_by_layer,
        activations_by_emotion,
        neuron_importance_by_emotion,
        key_layers,
    )


def plot_emotion_neuron_heatmap(activations_by_layer, layer, output_dir):
    """Create a heatmap showing how neurons activate for different emotions"""
    layer_data = activations_by_layer[layer]
    activations = layer_data["activations"]
    labels = layer_data["labels"]

    # Get unique emotions
    unique_emotions = np.unique(labels)

    # Create activation matrix (emotions x neurons)
    num_neurons = min(
        50, activations.shape[1]
    )  # Limit to first 50 neurons for visibility
    emotion_neuron_matrix = np.zeros((len(unique_emotions), num_neurons))

    for i, emotion in enumerate(unique_emotions):
        emotion_indices = np.where(labels == emotion)[0]
        if len(emotion_indices) > 0:
            emotion_activations = activations[emotion_indices]
            emotion_neuron_matrix[i] = np.mean(
                emotion_activations[:, :num_neurons], axis=0
            )

    # Plot heatmap
    plt.figure(figsize=(18, len(unique_emotions) * 0.6 + 2))
    sns.heatmap(
        emotion_neuron_matrix,
        cmap="coolwarm",
        center=0,
        xticklabels=[f"N{i}" for i in range(num_neurons)],
        yticklabels=unique_emotions,
    )
    plt.title(f"Neuron Activations by Emotion for {layer}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Emotion")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        os.path.join(output_dir, f"{layer}_emotion_neuron_heatmap.png"), dpi=300
    )
    plt.close()


def plot_top_neurons_by_emotion(neuron_importance, emotions, layer, output_dir):
    """Plot the top neurons for each emotion"""
    plt.figure(figsize=(15, len(emotions) * 3))

    for i, emotion in enumerate(emotions):
        if emotion not in neuron_importance or layer not in neuron_importance[emotion]:
            continue

        data = neuron_importance[emotion][layer]
        neurons = data["neurons"][:10]  # Top 10 neurons
        scores = data["scores"][:10]

        plt.subplot(len(emotions), 1, i + 1)
        plt.bar(range(len(neurons)), scores)
        plt.xticks(range(len(neurons)), [f"N{n}" for n in neurons])
        plt.title(f"Top Neurons for {emotion.capitalize()} in {layer}")
        plt.xlabel("Neuron")
        plt.ylabel("Importance Score")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{layer}_top_neurons_by_emotion.png"), dpi=300
    )
    plt.close()


def plot_pca_by_emotion(activations_by_layer, layer, output_dir):
    """Create PCA plot colored by emotion"""
    layer_data = activations_by_layer[layer]
    activations = layer_data["activations"]
    labels = layer_data["labels"]

    # Apply PCA
    if activations.shape[0] > 1:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(activations)

        # Create DataFrame for plotting
        df = pd.DataFrame(
            {"x": reduced_data[:, 0], "y": reduced_data[:, 1], "emotion": labels}
        )

        # Plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x="x", y="y", hue="emotion", s=80, alpha=0.7)
        plt.title(f"PCA of {layer} Activations by Emotion")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_dir, f"{layer}_pca_by_emotion.png"), dpi=300)
        plt.close()


def generate_emotion_report(neuron_importance, emotions, key_layers, output_dir):
    """Generate a summary report of the most important neurons for each emotion"""
    report_file = os.path.join(output_dir, "emotion_neuron_report.html")

    with open(report_file, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Emotion Neuron Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        h3 { color: #2980b9; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .emotion { font-weight: bold; }
        .neuron { font-family: monospace; }
        .score { text-align: right; }
    </style>
</head>
<body>
    <h1>Emotion Neuron Analysis Report</h1>
""")

        # Write overview
        f.write(
            f"<p>This report analyzes neurons that activate distinctly for specific emotions across {len(key_layers)} layers.</p>"
        )
        f.write("<h2>Top Neurons by Emotion</h2>")

        # For each emotion, show the top neurons across layers
        for emotion in sorted(emotions):
            f.write(f'<h3 class="emotion">{emotion.capitalize()}</h3>')
            f.write("<table>")
            f.write(
                "<tr><th>Layer</th><th>Top Neurons</th><th>Activation Scores</th></tr>"
            )

            for layer in key_layers:
                if emotion in neuron_importance and layer in neuron_importance[emotion]:
                    data = neuron_importance[emotion][layer]
                    top_5_neurons = data["neurons"][:5]  # Show top 5 neurons
                    top_5_scores = data["scores"][:5]

                    # Format neuron list and scores
                    neurons_str = ", ".join([f"N{n}" for n in top_5_neurons])
                    scores_str = ", ".join([f"{s:.3f}" for s in top_5_scores])

                    f.write(
                        f'<tr><td>{layer}</td><td class="neuron">{neurons_str}</td><td class="score">{scores_str}</td></tr>'
                    )
                else:
                    f.write(
                        f'<tr><td>{layer}</td><td colspan="2">No significant neurons found</td></tr>'
                    )

            f.write("</table>")

        # Write recommendations section
        f.write("<h2>Tuning Recommendations</h2>")
        f.write(
            "<p>Based on the analysis, here are neurons you might consider focusing on for emotion tuning:</p>"
        )

        for emotion in sorted(emotions):
            f.write(f'<h3 class="emotion">{emotion.capitalize()}</h3>')
            f.write("<ul>")

            # Find the layer with the strongest activation for this emotion
            best_layer = None
            top_score = 0
            top_neuron = None

            for layer in key_layers:
                if emotion in neuron_importance and layer in neuron_importance[emotion]:
                    data = neuron_importance[emotion][layer]
                    if len(data["scores"]) > 0 and data["scores"][0] > top_score:
                        top_score = data["scores"][0]
                        top_neuron = data["neurons"][0]
                        best_layer = layer

            if best_layer:
                f.write(f"<li>Best layer: <strong>{best_layer}</strong></li>")
                f.write(
                    f"<li>Primary neuron: <strong>Neuron {top_neuron}</strong> (score: {top_score:.3f})</li>"
                )

                # Get top 3 neurons from the best layer
                data = neuron_importance[emotion][best_layer]
                top_3_neurons = data["neurons"][:3]
                top_3_scores = data["scores"][:3]

                neuron_info = ", ".join(
                    [f"N{n} ({s:.3f})" for n, s in zip(top_3_neurons, top_3_scores)]
                )
                f.write(f"<li>Recommended neurons to amplify: {neuron_info}</li>")
            else:
                f.write("<li>No strong distinctive neurons found</li>")

            f.write("</ul>")

        f.write("</body>\n</html>")

    print(f"Emotion report saved to {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neural activations for each emotion"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory containing activation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="emotion_analysis",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load activations
    (
        metadata,
        activations_by_layer,
        activations_by_emotion,
        neuron_importance,
        key_layers,
    ) = load_activations(args.activations_dir)

    # Get all unique emotions
    emotions = list(activations_by_emotion.keys())
    print(f"Found {len(emotions)} emotions: {emotions}")

    # Create visualizations for each key layer
    for layer in key_layers:
        if layer in activations_by_layer:
            # Heatmap of neuron activations by emotion
            plot_emotion_neuron_heatmap(activations_by_layer, layer, args.output_dir)

            # Top neurons for each emotion
            plot_top_neurons_by_emotion(
                neuron_importance, emotions, layer, args.output_dir
            )

            # PCA plot by emotion
            plot_pca_by_emotion(activations_by_layer, layer, args.output_dir)

    # Generate summary report
    generate_emotion_report(neuron_importance, emotions, key_layers, args.output_dir)

    print(f"All analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
