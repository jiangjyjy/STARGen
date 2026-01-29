# STARGen：Generative RNA Inverse Folding via Discrete Geometric Representations

STARGen is a structure-conditioned generative framework for RNA inverse folding. It formulates RNA sequence design as an autoregressive generation problem guided by discretized geometric representations of RNA backbones.

# Repository Structure

**./code**
Contains the core implementation of STARGen.

    train_STARGen.py: Training script for STARGen.
    inference_STARGen.py: Inference and evaluation script.

    The scripts can be executed via the provided shell wrappers:

    run_train.sh
    run_inference.sh

**./dataset**
Contains preprocessed datasets used for training, validation, and testing. 
All datasets are ready for direct use without additional preprocessing.

# Training

**run_train.sh** is an automated training launcher for STARGen with the RIGA optimization module.

All essential training hyperparameters and file paths are specified within the script. Users should modify these configurations according to their local environment before execution.

# Inference and Evaluation

**run_inference.sh** is an automated inference and evaluation launcher for STARGen with the BCD decoding module.

Model checkpoints, test datasets, and output directories are specified in the script and can be adjusted as needed.

# Dataset

The datasets are stored in JSONL (JSON Lines) format, where each line corresponds to an independent RNA structure-conditioned design task.

All samples have been fully preprocessed, including: Sequence tokenization, Positional and structural feature encoding Format normalization.

No additional data processing is required. The datasets are directly compatible with the provided training and inference scripts, enabling efficient batch loading and large-scale experimentation.
