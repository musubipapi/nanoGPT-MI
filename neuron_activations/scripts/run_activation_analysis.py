#!/usr/bin/env python3
"""
Run the complete GPT-2 activation analysis pipeline for emotion-labeled tweets
"""

import os
import argparse
import subprocess


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


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete GPT-2 activation analysis pipeline"
    )

    # General arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="activation_results",
        help="Directory to save all output files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../out/gpt2.pt",
        help="Path to GPT-2 model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=["joy", "sadness", "anger"],
        help="List of emotions to analyze",
    )
    parser.add_argument(
        "--samples_per_emotion",
        type=int,
        default=100,
        help="Number of samples to process per emotion",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Process the dataset and capture activations
    batch_cmd = [
        "python",
        "batch_process_tweets.py",
        "--model_path",
        args.model_path,
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--samples_per_label",
        str(args.samples_per_emotion),
        "--emotions",
    ] + args.emotions

    run_command(batch_cmd, "Capturing activations for emotion tweets")

    # Step 2: Analyze the activations across all emotions
    analysis_cmd = [
        "python",
        "analyze_all_emotions.py",
        "--activations_dir",
        os.path.join(args.output_dir, "generation_activations"),
        "--output_dir",
        os.path.join(args.output_dir, "all_emotions"),
    ]

    run_command(analysis_cmd, "Analyzing activations across all emotions")

    print(f"\nAnalysis complete! Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
