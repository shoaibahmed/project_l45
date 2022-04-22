#!/bin/bash

for dataset_size in 10 20 30 40 50 60 70 80 90 100 250 500 750 1000; do
    for k in 2 3 4 5 6 7 8 9; do
        echo "Training the model..."
        python3 train.py --exp_name "exp_01" --num-training-examples ${dataset_size} --num-test-examples 10000 --k ${k}

        echo "Evaluating the model..."
        python3 train.py --exp_name "exp_01" --num-training-examples ${dataset_size} --num-test-examples 10000 --k ${k} --eval True
    done
done
