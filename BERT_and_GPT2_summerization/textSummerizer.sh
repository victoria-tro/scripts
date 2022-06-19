#!/bin/bash

# Create and activate virtualenv
conda create --name ml_env python=3.8
conda activate ml_env

# SpaCy
pip install -U spacy

# BERT and GPT-2
echo "Installing ML stuff and other packages"
pip install transformers
pip install bert-extractive-summarizer
#pip install tensorflow
pip install torch torchvision torchaudio
pip install sacremoses

# Run python script textSummerizer.py 
echo "Running Python script"
python summarize_text.py

# Close virtual environment
conda deactivate