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
    
    offset = 0
    if config[-1] == "norm":
        offset = 1
    assert config[-1-offset] == "pos"
    assert config[-3-offset] == "k"
    k = int(config[-2-offset])
    
    assert config[-6-offset] == "train"
    assert config[-5-offset] == "ex"
    num_examples = int(config[-4-offset])
    
    return k, num_examples


root_dir = "./outputs"
relevant_files = f"{root_dir}/*/*/training_dynamics.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])
assert len(files) > 0

output_dir = "plots/"
if not os.path.exists(output_dir):
    print("Created output directory...")
    os.mkdir(output_dir)

dynamics_output_dir = "plots/dynamics_plots/"
if not os.path.exists(dynamics_output_dir):
    print("Created output directory...")
    os.mkdir(dynamics_output_dir)

plot_knn_results = True
if plot_knn_results:
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

        # Generate the dynamics plot
        train_mse, test_mse = df["train_loss"].to_numpy(), df["test_loss"].to_numpy()
        print(len(train_mse), len(test_mse))

        plt.figure(figsize=(12, 8))
        
        plt.plot(train_mse, label='Train loss', color='b')
        plt.plot(test_mse, label='Test loss', color='r')

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.title(f"DGCNN trained on synthetic realizations of an SCM with k={k} and # training examples={num_examples}", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()

        output_file = os.path.join(dynamics_output_dir, f"results_dynamics_k_{k}_{num_examples}_train_ex.png")
        plt.savefig(output_file, dpi=300)

    # Plot the dictionary here
    num_training_examples = natsort.natsorted(list(output_dict.keys()))
    print("Number of training examples:", num_training_examples)

    k_settings = natsort.natsorted(list(output_dict[num_training_examples[0]].keys()))
    print("k-NN configurations:", k_settings)

    # Validate that the k settings are same for every run
    assert all([all([natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) == k_settings]) for i in range(len(num_training_examples))]), \
        {num_training_examples[i]: natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) for i in range(len(num_training_examples))}

    # Compute the same limits
    max_val = None
    for training_examples in num_training_examples:
        k_nn_mse = [output_dict[training_examples][k]["test"] for k in k_settings]
        if max_val is None:
            max_val = max(k_nn_mse)
        else:
            max_val = max(max_val, max(k_nn_mse))

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

# Plot the intervention results
relevant_files = f"{root_dir}/*/*/intervention_test.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])
assert len(files) > 0

output_dir = "plots/intervention_plots/"
if not os.path.exists(output_dir):
    print("Created output directory...")
    os.mkdir(output_dir)

plot_intervention_results = True
if plot_intervention_results:
    output_dict = {}
    for f in files:
        print("Reading file:", f)
        df = pd.read_csv(f)
        print(df)
        
        k, num_examples = get_k_train_ex_from_file(f)
        print(f"k: {k} / num examples: {num_examples}")
        
        plt.figure(figsize=(10, 6))
        
        intervention_nodes = df["intervention_node"].to_numpy().tolist()
        all_nodes = [col.replace("diff_", "") for col in df.columns if col != "intervention_node"]
        del df["intervention_node"]
        
        print("All nodes:", all_nodes)
        print("Intervention nodes:", intervention_nodes)
        
        intervention_results = df.to_numpy()
        print(intervention_results)
        
        if num_examples not in output_dict:
            output_dict[num_examples] = {}
        assert k not in output_dict[num_examples]
        output_dict[num_examples][k] = intervention_results.sum()  # Sum over all nodes
        
        ax = sns.heatmap(intervention_results, annot=True, cmap='Blues', cbar=False)

        ax.set_title(f'Impact of intervention with model trained with k={k} and # training examples={num_examples}\n')
        ax.set_xlabel('\n MSE between expected output and the model prediction')
        ax.set_ylabel('Intervention on variable')

        ax.xaxis.set_ticklabels(all_nodes)
        ax.yaxis.set_ticklabels(intervention_nodes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"results_intervention_k_{k}_{num_examples}_train_ex.png"), dpi=300, bbox_inches='tight', pad_inches=0.04)
    
    # Plot the dictionary here
    num_training_examples = natsort.natsorted(list(output_dict.keys()))
    print("Number of training examples:", num_training_examples)

    k_settings = natsort.natsorted(list(output_dict[num_training_examples[0]].keys()))
    print("k-NN configurations:", k_settings)

    # Validate that the k settings are same for every run
    assert all([all([natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) == k_settings]) for i in range(len(num_training_examples))]), \
        {num_training_examples[i]: natsort.natsorted(list(output_dict[num_training_examples[i]].keys())) for i in range(len(num_training_examples))}

    # Compute the same limits
    max_val = None
    for training_examples in num_training_examples:
        k_nn_mse = [output_dict[training_examples][k] for k in k_settings]
        if max_val is None:
            max_val = max(k_nn_mse)
        else:
            max_val = max(max_val, max(k_nn_mse))

    for training_examples in num_training_examples:
        plt.figure(figsize=(6, 6))
        
        bar_width = 0.5
        k_nn_mse = [output_dict[training_examples][k] for k in k_settings]
        plt.bar(range(len(k_settings)), k_nn_mse, width=bar_width, color = 'orange', edgecolor = 'black')
        
        plt.xticks(np.arange(len(k_settings)), k_settings)
        plt.ylabel('Aggregated MSE after interventions', fontsize=14)
        plt.xlabel('k for k-NN', fontsize=14)
        plt.title(f"Impact of k with {training_examples} training examples", fontsize=14)
        plt.ylim(0, max_val)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"results_aggregated_intervention_k_nn_{training_examples}_train_ex.png"), dpi=300)

# Plot the attention results
relevant_files = f"{root_dir}/*/*/attention_stats_layer_[0-4].csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])
assert len(files) > 0

output_dir = "plots/attention_plots/"
if not os.path.exists(output_dir):
    print("Created output directory...")
    os.mkdir(output_dir)

plot_attention_results = True
if plot_attention_results:
    for f in files:
        print("Reading file:", f)
        df = pd.read_csv(f)
        print(df)
        
        k, num_examples = get_k_train_ex_from_file(f)
        layer_idx = int(os.path.split(f)[1].replace(".csv", "").replace("attention_stats_layer_", ""))
        print(f"k: {k} / num examples: {num_examples} / layer idx: {layer_idx}")
        
        plt.figure(figsize=(10, 6))
        
        main_node = df["node"].to_numpy().tolist()
        all_nodes = [col.replace("connected_to_", "") for col in df.columns if col != "node"]
        assert len(main_node) == len(all_nodes)
        assert all([main_node[i] == all_nodes[i] for i in range(len(all_nodes))])
        del df["node"]
        print("All nodes:", all_nodes)
        
        connection_matrix = df.to_numpy().astype(np.int64)
        print(connection_matrix)
        num_elements = connection_matrix.diagonal()
        assert all([num_elements[0] == num_elements[i] for i in range(len(num_elements))])
        bs = num_elements[0]
        connection_matrix_norm = connection_matrix.astype(np.float32) / bs
        
        ax = sns.heatmap(connection_matrix_norm, annot=True, cmap='Blues', cbar=False)

        ax.set_title(f'Nearest neighbor matrix at layer {layer_idx+1} with model trained with k={k} and # training examples={num_examples}\n');
        ax.set_xlabel('\n Nearest neighbor connection')
        ax.set_ylabel('Node')

        ax.xaxis.set_ticklabels(all_nodes)
        ax.yaxis.set_ticklabels(all_nodes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"results_attention_k_{k}_{num_examples}_train_ex_layer_{layer_idx}.png"), dpi=300, bbox_inches='tight', pad_inches=0.04)
