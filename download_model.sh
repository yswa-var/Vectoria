#!/bin/bash

# Create models directory
mkdir -p models

# Download BERT tokenizer
curl -L https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json -o models/bert-base-uncased-tokenizer.json

# Download BERT model weights
curl -L https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin -o models/pytorch_model.bin

# Download BERT config
curl -L https://huggingface.co/bert-base-uncased/resolve/main/config.json -o models/config.json 