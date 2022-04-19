#!/bin/bash

echo "Training the model..."
python3 train.py

echo "Evaluating the model..."
python3 train.py --eval True
