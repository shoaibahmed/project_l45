#!/bin/bash

# scm=scm_easy
scm=scm_difficult

for dataset_size in 10 20 30 40 50 60 70 80 90 100 250 500 750 1000 2500 5000 7500 10000; do
    for k in 2 3 4 5 6 7 8 9 10; do
        echo "Training the model..."
        python3 train.py --exp_name "exp_01" --num-training-examples ${dataset_size} --num-test-examples 10000 --k ${k} --scm ${scm}

        echo "Evaluating the model..."
        python3 train.py --exp_name "exp_01" --num-training-examples ${dataset_size} --num-test-examples 10000 --k ${k} --scm ${scm} --eval True
    done
done
