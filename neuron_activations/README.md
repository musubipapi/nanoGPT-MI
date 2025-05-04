# Neuron Activations

This toolkit analyzes neural activations in a GPT-2 model when processing tweets associated with different emotions. It helps identify neurons that respond to specific emotions like joy, sadness, and anger.

## Overview

The project is based on nanoGPT and focuses on:

1. **Capturing neural activations** right before token generation for emotion-labeled tweets
2. **Analyzing activation patterns** to identify neurons associated with specific emotions

## Quick Start

### 1. Test Installation

Verify that everything is working properly:

```bash
python main.py test
```

### 2. Capture Activations

Process tweets and capture neural activations:

```bash
python main.py capture --model_path ../out/gpt2.pt --emotions joy sadness anger --samples_per_label 50
```

### 3. Analyze Activations

Analyze the captured activations:

```bash
python main.py analyze
```

## Command Reference

### Capture Command

```bash
python main.py capture [options]
```

Options:
- `--model_path`: Path to GPT-2 model checkpoint (default: ../out/gpt2.pt)
- `--output_dir`: Directory for output files (default: output/captures)
- `--device`: Device to run on (cuda or cpu) (default: cpu)
- `--batch_size`: Batch size for processing (default: 16)
- `--samples_per_label`: Number of samples per emotion (default: 100)
- `--emotions`: List of emotions to analyze (default: joy sadness anger)

### Analyze Command

```bash
python main.py analyze [options]
```

Options:
- `--input_dir`: Directory containing captured activations (default: output/captures)
- `--output_dir`: Directory for results (default: output/results)

## Utility Scripts

### View Layer Activations

Examine the contents of a specific layer's activation file:

```bash
python scripts/view_layer_activations.py output/captures/generation_activations/layer_11_activations.pt
```

This script:
- Shows basic statistics for the activations
- Identifies neurons with highest mean activation
- Creates a visualization of the activation pattern
- Saves the visualization as a PNG file

### View Metadata

Examine the captured metadata for all processed tweets:

```bash
python scripts/view_metadata.py output/captures/generation_activations/gen_metadata.pt
```

This script:
- Shows the distribution of emotion labels
- Displays sample tweets and their generated tokens
- Shows the most common generated tokens

## Project Structure

```
neuron_activations/
├── main.py              # Main entry point
├── data/                # Data storage
├── output/              # Output files
│   ├── captures/        # Captured activations
│   └── results/         # Analysis results
├── scripts/             # Core scripts
│   ├── tweet_activations.py       # Activation capture
│   ├── batch_process_tweets.py    # Batch processing
│   ├── analyze_all_emotions.py    # Cross-emotion analysis 
│   ├── run_activation_analysis.py # Full pipeline
│   ├── test_activation.py         # Simple test
│   ├── view_layer_activations.py  # View layer activations
│   └── view_metadata.py           # View metadata
└── docs/                # Documentation
``` 