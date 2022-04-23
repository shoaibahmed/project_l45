import os
import glob

import json
import natsort

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def get_k_train_ex_from_file(file_name):
    # Get config from the model file
    file_parts = file_name.split(os.sep)
    config = file_parts[-3].split("_")  # file, model, config, outputs
    print(config)
    
    assert config[-1] == "pos"
    assert config[-3] == "k"
    k = int(config[-2])
    
    assert config[-6] == "train"
    assert config[-5] == "ex"
    num_examples = int(config[-4])
    
    return k, num_examples


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
    
    k, num_examples = get_k_train_ex_from_file(f)
    print(f"k: {k} / num examples: {num_examples}")
    
    if num_examples not in output_dict:
        output_dict[num_examples] = {}
    assert k not in output_dict[num_examples]
    output_dict[num_examples][k] = {"train": train_mse, "test": test_mse}

# print(json.dumps(output_dict, indent=4, sort_keys=True))

# Plot the dictionary here
num_training_examples = natsort.natsorted(list(output_dict.keys()))
print("Number of training examples:", num_training_examples)

k_settings = natsort.natsorted(list(output_dict[num_training_examples[0]].keys()))
print("k-NN configurations:", k_settings)

# Validate that the k settings are same for every run
assert all([all([natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) == k_settings]) for i in range(len(num_training_examples))])

# Compute the same limits
max_val = None
for training_examples in num_training_examples:
    k_nn_mse = [output_dict[training_examples][k]["test"] for k in k_settings]
    if max_val is None:
        max_val = max(k_nn_mse)
    else:
        max_val = max(max_val, max(k_nn_mse))

output_dir = "plots/"
if not os.path.exists(output_dir):
    print("Created output directory...")
    os.mkdir(output_dir)

for training_examples in num_training_examples:
    plt.figure(figsize=(6, 6))
    
    bar_width = 0.5
    k_nn_mse = [output_dict[training_examples][k]["test"] for k in k_settings]
    plt.bar(range(len(k_settings)), k_nn_mse, width=bar_width, color = 'orange', edgecolor = 'black')
    
    plt.xticks(np.arange(len(k_settings)), k_settings)
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('k for k-NN', fontsize=14)
    plt.title(f"Impact of k with {training_examples} training examples", fontsize=14)
    plt.ylim(0, max_val)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"results_k_nn_{training_examples}_train_ex.png"), dpi=300)

# TODO: Plot the intervention results here
relevant_files = "./*/*/*/intervention_test.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])

output_dir = "plots/intervention_plots/"
if not os.path.exists(output_dir):
    print("Created output directory...")
    os.mkdir(output_dir)

output_dict = {}
for f in files:
    print("Reading file:", f)
    df = pd.read_csv(f)
    print(df)
    
    k, num_examples = get_k_train_ex_from_file(f)
    print(f"k: {k} / num examples: {num_examples}")
    
    if num_examples not in output_dict:
        output_dict[num_examples] = {}
    assert k not in output_dict[num_examples]
    output_dict[num_examples][k] = df
    
    plt.figure(figsize=(10, 6))
    
    intervention_nodes = df["intervention_node"].to_numpy().tolist()
    all_nodes = [col.replace("diff_", "") for col in df.columns if col != "intervention_node"]
    del df["intervention_node"]
    
    print("All nodes:", all_nodes)
    print("Intervention nodes:", intervention_nodes)
    
    intervention_results = df.to_numpy()
    print(intervention_results)
    
    ax = sns.heatmap(intervention_results, annot=True, cmap='Blues', cbar=False)

    ax.set_title(f'Impact of intervention with model trained with k={k} and # training examples={num_examples}\n');
    ax.set_xlabel('\n MSE between expected output and the model prediction')
    ax.set_ylabel('Intervention on variable');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(all_nodes)
    ax.yaxis.set_ticklabels(intervention_nodes)

    ## Display the visualization of the Confusion Matrix.
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"results_k_{k}_{training_examples}_train_ex.png"), dpi=300, bbox_inches='tight', pad_inches=0.04)

# TODO: Plot the attention results here

