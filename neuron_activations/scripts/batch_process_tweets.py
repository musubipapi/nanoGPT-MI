#!/usr/bin/env python3
"""
Batch process tweets from the emotion dataset, capturing activations for emotion analysis
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from datasets import load_dataset

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from model import GPTConfig, GPT

# Local imports
from tweet_activations import (
    load_model,
    process_multi_emotion_dataset,
)


def prepare_emotion_dataset(
    output_file, samples_per_emotion=100, emotion_filter=None, random_seed=42
):
    """
    Download and prepare the emotion dataset from Hugging Face
    Saves a balanced subset with specified number of samples per emotion
    """
    print("Loading emotion dataset...")
    rng = np.random.RandomState(random_seed)

    # Load the dataset
    dataset = load_dataset("dair-ai/emotion", split="train")
    label_names = dataset.features["label"].names

    # Print dataset info
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Labels: {label_names}")

    # Filter for specific emotions if requested
    if emotion_filter:
        # Convert emotion names to IDs
        emotion_ids = [
            label_names.index(emotion)
            for emotion in emotion_filter
            if emotion in label_names
        ]
        # Print which emotions were found and which weren't
        found_emotions = [label_names[id] for id in emotion_ids]
        missing_emotions = [e for e in emotion_filter if e not in found_emotions]
        print(f"Found emotions: {found_emotions}")
        if missing_emotions:
            print(
                f"Warning: Some requested emotions were not found: {missing_emotions}"
            )

        # Filter dataset to only include these emotion IDs
        dataset = dataset.filter(lambda example: example["label"] in emotion_ids)
        print(f"Filtered dataset has {len(dataset)} samples")

    # Group samples by emotion
    samples_by_emotion = {}
    for sample in dataset:
        emotion = label_names[sample["label"]]
        if emotion not in samples_by_emotion:
            samples_by_emotion[emotion] = []
        samples_by_emotion[emotion].append(sample)

    # Print sample counts
    for emotion, samples in samples_by_emotion.items():
        print(f"{emotion}: {len(samples)} samples")

    # Sample balanced dataset
    balanced_samples = []
    for emotion, samples in samples_by_emotion.items():
        num_samples = min(samples_per_emotion, len(samples))
        indices = rng.choice(len(samples), num_samples, replace=False)
        for idx in indices:
            balanced_samples.append({"text": samples[idx]["text"], "label": emotion})

    # Shuffle the final set
    rng.shuffle(balanced_samples)

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(balanced_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(balanced_samples)} balanced samples to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch process tweets with emotion labels for activation analysis"
    )
    # Common arguments
    parser.add_argument(
        "--model_path", type=str, default="out/gpt2.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="emotion_analysis",
        help="Directory for output files",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run model on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing tweets",
    )
    parser.add_argument(
        "--samples_per_label",
        type=int,
        default=100,
        help="Number of samples per emotion to include",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help="List of emotions to include (default: all)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for dataset creation"
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the emotions dataset
    print(
        f"Processing emotions dataset with {args.samples_per_label} samples per emotion"
    )

    # Create dataset dir
    dataset_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    # Prepare dataset
    output_file = os.path.join(dataset_dir, "emotion_tweets.json")
    prepare_emotion_dataset(
        output_file,
        samples_per_emotion=args.samples_per_label,
        emotion_filter=args.emotions,
        random_seed=args.random_seed,
    )

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    # Process the dataset
    process_multi_emotion_dataset(
        model,
        output_file,
        args.output_dir,
        device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
