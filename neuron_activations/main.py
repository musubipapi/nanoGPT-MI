#!/usr/bin/env python3
"""
Main entry point for Neural Activation Analysis toolkit
Capture and visualize detailed neural activations from language models
"""

import os
import argparse
import subprocess

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


def run_command(cmd, description=None):
    """Run a command and print its output"""
    if description:
        print(f"\n{'-' * 80}\n{description}\n{'-' * 80}")

    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Stream output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for process to complete and get return code
    process.wait()
    return process.returncode


def test():
    """Run a simple test to verify the installation"""
    script_path = os.path.join(SCRIPTS_DIR, "test_activation.py")
    cmd = ["python", script_path]
    return run_command(cmd, "Running simple test on a sample tweet")


def capture(args):
    """Capture detailed activations from tweets with emotions"""
    script_path = os.path.join(SCRIPTS_DIR, "detailed_activations.py")

    cmd = [
        "python",
        script_path,
        "capture",
        "--model_path",
        args.model_path,
        "--tweets_file",
        "dair",  # Using dair dataset as default
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
        "--max_tweets",
        str(args.max_tweets) if args.max_tweets is not None else "None",
        "--samples_per_emotion",
        str(args.samples_per_emotion),
        "--emotions",
    ] + args.emotions

    return run_command(
        cmd, "Capturing detailed activations from emotion-labeled tweets"
    )


def list_components(args):
    """List available neural components that can be captured"""
    script_path = os.path.join(SCRIPTS_DIR, "detailed_activations.py")

    cmd = [
        "python",
        script_path,
        "list",
        "--model_path",
        args.model_path,
    ]

    return run_command(cmd, "Listing available neural components")


def analyze_components(args):
    """Analyze captured component sizes"""
    script_path = os.path.join(SCRIPTS_DIR, "detailed_activations.py")

    cmd = [
        "python",
        script_path,
        "analyze",
        "--output_dir",
        args.output_dir,
    ]

    return run_command(cmd, "Analyzing detailed component sizes")


def visualize_by_emotion(args):
    """Visualize activations across emotions with detailed analysis"""
    script_path = os.path.join(SCRIPTS_DIR, "analyze_all_emotions.py")

    cmd = [
        "python",
        script_path,
        "--activations_dir",
        os.path.join(args.input_dir, "detailed_activations"),
        "--output_dir",
        args.output_dir,
    ]

    return run_command(cmd, "Analyzing and visualizing activations across emotions")


def view_layer(args):
    """View the contents and visualization of a single layer's activations"""
    script_path = os.path.join(SCRIPTS_DIR, "view_layer_activations.py")

    activations_file = os.path.join(
        args.input_dir, "detailed_activations", f"{args.layer_name}_activations.pt"
    )

    cmd = [
        "python",
        script_path,
        activations_file,
    ]

    return run_command(cmd, f"Viewing activations for layer: {args.layer_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural Activation Analysis: Analyze neural activations in language models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Capture command
    capture_parser = subparsers.add_parser(
        "capture", help="Capture detailed activations from tweets"
    )
    capture_parser.add_argument(
        "--model_path",
        type=str,
        default="../out/gpt2.pt",
        help="Path to model checkpoint",
    )
    capture_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/captures",
        help="Directory for outputs",
    )
    capture_parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cuda or cpu)"
    )
    capture_parser.add_argument(
        "--max_tweets",
        type=int,
        default=None,
        help="Maximum number of tweets to process",
    )
    capture_parser.add_argument(
        "--samples_per_emotion", type=int, default=30, help="Samples per emotion"
    )
    capture_parser.add_argument(
        "--emotions",
        nargs="+",
        default=["joy", "sadness", "anger"],
        help="Emotions to analyze",
    )

    # List components command
    list_parser = subparsers.add_parser("list", help="List available neural components")
    list_parser.add_argument(
        "--model_path",
        type=str,
        default="../out/gpt2.pt",
        help="Path to model checkpoint",
    )

    # Analyze components command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze captured component sizes"
    )
    analyze_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/captures",
        help="Directory containing captures",
    )

    # Visualization by emotion command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Create visualizations of activations by emotion"
    )
    visualize_parser.add_argument(
        "--input_dir",
        type=str,
        default="output/captures",
        help="Input directory with captures",
    )
    visualize_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/results",
        help="Directory for visualization results",
    )

    # View single layer command
    view_parser = subparsers.add_parser(
        "view", help="View and visualize a single layer's activations"
    )
    view_parser.add_argument(
        "--input_dir",
        type=str,
        default="output/captures",
        help="Input directory with captures",
    )
    view_parser.add_argument(
        "--layer_name",
        type=str,
        required=True,
        help="Name of the layer to view (e.g., 'layer_0_attn', 'final_layer')",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create required directories
    os.makedirs("output/captures", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)

    # Execute the appropriate command
    if args.command == "test":
        test()
    elif args.command == "capture":
        capture(args)
    elif args.command == "list":
        list_components(args)
    elif args.command == "analyze":
        analyze_components(args)
    elif args.command == "visualize":
        visualize_by_emotion(args)
    elif args.command == "view":
        view_layer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
