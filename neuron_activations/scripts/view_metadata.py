#!/usr/bin/env python3
"""
Utility script to view the contents of the gen_metadata.pt file
"""

import os
import sys
import torch
import argparse
import pandas as pd
from collections import Counter


def view_metadata(metadata_file):
    """View the contents of the gen_metadata.pt file"""
    print(f"Loading metadata from {metadata_file}")

    # Load the metadata
    metadata = torch.load(metadata_file)

    if not metadata:
        print("No metadata found")
        return

    # Display available keys
    print(f"Metadata contains the following keys: {list(metadata.keys())}")

    # Display information about each key
    for key, value in metadata.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            print(f"  Type: list with {len(value)} elements")
            if value:
                print(f"  First element type: {type(value[0]).__name__}")
                # Show a few examples
                if len(str(value[0])) < 100:  # Only show if not too long
                    print(f"  Examples: {value[:3]}")
                else:
                    print(f"  First element preview: {str(value[0])[:100]}...")
        else:
            print(f"  Type: {type(value).__name__}")
            print(f"  Value: {value}")

    # If we have tweets and labels, create a summary
    if "tweets" in metadata and "labels" in metadata:
        tweets = metadata["tweets"]
        labels = metadata["labels"]

        # Count occurrences of each label
        label_counts = Counter(labels)
        print("\nLabel distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} tweets")

        # Create a DataFrame for easier viewing
        if len(tweets) == len(labels):
            df = pd.DataFrame({"tweet": tweets, "label": labels})

            if "generated_text" in metadata:
                df["generated_text"] = metadata["generated_text"]

            # Display a few examples
            print("\nSample tweets with labels:")
            pd.set_option("display.max_colwidth", 50)  # Limit display width
            print(df.head())

    # If there's generated text, summarize it
    if "generated_text" in metadata:
        generated = metadata["generated_text"]

        # Count the most common first generated tokens
        token_counts = Counter(generated)
        print("\nMost common generated tokens:")
        for token, count in token_counts.most_common(10):
            # Clean up the token for display
            display_token = token.replace("\n", "\\n")
            print(f"  '{display_token}': {count} occurrences")


def main():
    parser = argparse.ArgumentParser(
        description="View the contents of the gen_metadata.pt file"
    )
    parser.add_argument(
        "metadata_file", type=str, help="Path to the gen_metadata.pt file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.metadata_file):
        print(f"Error: File {args.metadata_file} not found")
        sys.exit(1)

    view_metadata(args.metadata_file)


if __name__ == "__main__":
    main()
