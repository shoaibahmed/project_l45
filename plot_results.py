import os
import glob

import json
import natsort

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True


relevant_files = "./*/*/*/training_dynamics.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])

output_dict = {}
for f in files:
    print("Reading file:", f)
    df = pd.read_csv(f)
    final_results = df.tail(1)
    train_mse, test_mse = float(final_results["train_loss"]), float(final_results["test_loss"])
    print(final_results, train_mse, test_mse)
    
    # Get config from the model file
    file_parts = f.split(os.sep)
    config = file_parts[-3].split("_")  # file, model, config, outputs
    print(config)
    
    assert config[-1] == "pos"
    assert config[-3] == "k"
    k = int(config[-2])
    
    assert config[-6] == "train"
    assert config[-5] == "ex"
    num_examples = int(config[-4])
    print(f"k: {k} / num examples: {num_examples}")
    
    if num_examples not in output_dict:
        output_dict[num_examples] = {}
    assert k not in output_dict[num_examples]
    output_dict[num_examples][k] = (train_mse, test_mse)

# print(json.dumps(output_dict, indent=4, sort_keys=True))

# Plot the dictionary here
num_training_examples = natsort.natsorted(list(output_dict.keys()))
print("Number of training examples:", num_training_examples)

k_settings = natsort.natsorted(list(output_dict[num_training_examples[0]].keys()))
print("k-NN configurations:", k_settings)

# Validate that the k settings are same for every run
assert all([all([natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) == k_settings]) for i in range(len(num_training_examples))])

# Create the bar chart
# bar_width = 0.2

# # The x position of bars
# r1 = np.arange(len(num_training_examples))
# r2 = [x-(0.5*bar_width) for x in r1]
# r3 = [x+(0.5*bar_width) for x in r1]
# r4 = [x+(1.5*bar_width) for x in r1]
# r1 = [x +(2.5*bar_width) for x in r1]

# plt.figure(figsize=(12, 8))

# plt.bar(r2, acc, width=bar_width, color = 'gray', edgecolor = 'black', capsize=7, label='Normal training')
# plt.bar(r1, acc_early_stop, width=bar_width, color = 'red', edgecolor = 'black', capsize=7, label='Early stopping')
# plt.bar(r3, acc_flooding, width=bar_width, color = 'purple', edgecolor = 'black', capsize=7, label='Flooding')
# plt.bar(r4, acc_flooding_ex, width=bar_width, color = 'orange', edgecolor = 'black', capsize=7, label='Flooding per ex.')

# # general layout
# plt.xticks([r + bar_width for r in range(len(num_training_examples))], num_training_examples)
# plt.ylabel(r'MSE \Huge{$ \leftarrow $}', fontsize=14)
# plt.xlabel('Number of training examples', fontsize=14)
# plt.legend(fontsize=14)

# plt.tight_layout()
# plt.savefig("results.png", dpi=300)
# plt.show()

for training_examples in num_training_examples:
    plt.figure(figsize=(6, 6))
    
    bar_width = 0.5
    k_nn_mse = [output_dict[training_examples][k][1] for k in k_settings]
    plt.bar(range(len(k_settings)), k_nn_mse, width=bar_width, color = 'orange', edgecolor = 'black')
    
    plt.xticks(np.arange(len(k_settings)), k_settings)
    # plt.ylabel(r'MSE \Huge{$ \leftarrow $}', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('k for k-NN', fontsize=14)
    plt.title(f"Impact of k with {training_examples} training examples", fontsize=14)
    # plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"results_k_nn_{training_examples}_train_ex.png", dpi=300)
    # plt.show()

# TODO: Plot the intervention results here
