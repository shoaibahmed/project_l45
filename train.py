"""
Code adapted from: https://github.com/AnTao97/dgcnn.pytorch
"""

from __future__ import print_function
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from dgcnn import DGCNN_Reg

import shutil
import graphviz
import pandas as pd
from colour import Color
import matplotlib.pyplot as plt

import natsort
import networkx as nx
from causalgraphicalmodels import StructuralCausalModel


class SCM:
    def __init__(self):
        # Define the SCM
        # self.scm_dict = {
        #     "x1": lambda     n_samples: np.random.normal(loc=1, scale=0.1, size=n_samples),
        #     "x2": lambda     n_samples: np.random.normal(loc=0, scale=0.2, size=n_samples),
        #     "x3": lambda x1, x2, n_samples: (0.5 * x1) + x2,
        #     "x4": lambda     n_samples: np.random.normal(loc=-1, scale=0.1, size=n_samples),
        #     "x5": lambda x3, x4, n_samples: x3 + x4,
        #     "x6": lambda     n_samples: np.random.normal(loc=0, scale=0.5, size=n_samples),
        #     "x7": lambda x5, x6, n_samples: x5 + x6,
        #     "x8": lambda     n_samples: np.random.normal(loc=-1, scale=0.2, size=n_samples),
        #     "x9": lambda     n_samples: np.random.normal(loc=1, scale=0.2, size=n_samples),
        #     "x10": lambda x7, x8, x9, n_samples: x7 + x8 + x9,
        # }
        
        self.scm_dict = {
            "x1": None,
            "x2": None,
            "x3": lambda x1, x2, n_samples: (0.5 * x1) + x2,
            "x4": None,
            "x5": lambda x3, x4, n_samples: x3 + x4,
            "x6": None,
            "x7": lambda x5, x6, n_samples: x5 + x6,
            "x8": None,
            "x9": None,
            "x10": lambda x7, x8, x9, n_samples: x7 + x8 + x9,
        }
        
        self.scm = StructuralCausalModel(self.scm_dict)
        # self.plot_scm()

        self.nodes = natsort.natsorted(self.scm.cgm.dag.nodes())
        print("Nodes:", self.nodes)

        # Observed nodes have no parents
        self.observed_nodes = [x for x in self.nodes if not nx.ancestors(self.scm.cgm.dag, x)]
        print("Observed nodes:", self.observed_nodes)

        self.computed_nodes = [x for x in self.nodes if x not in self.observed_nodes]
        print("Computed nodes:", self.computed_nodes)
        
        # Define the generator dict that can be used to sample values
        self.generator_dict = {
            "x1": lambda num_samples: np.random.normal(loc=1, scale=0.1, size=num_samples),
            "x2": lambda num_samples: np.random.normal(loc=0, scale=0.2, size=num_samples),
            "x4": lambda num_samples: np.random.normal(loc=-1, scale=0.1, size=num_samples),
            "x6": lambda num_samples: np.random.normal(loc=0, scale=0.5, size=num_samples),
            "x8": lambda num_samples: np.random.normal(loc=-1, scale=0.2, size=num_samples),
            "x9": lambda num_samples: np.random.normal(loc=1, scale=0.2, size=num_samples)
        }
        assert all([x in self.observed_nodes for x in list(self.generator_dict.keys())])
        assert all([x not in self.computed_nodes for x in list(self.generator_dict.keys())])
    
    def plot_scm(self, output_file='out'):
        dot = self.scm.cgm.draw()
        print(dot)
        dot.render(output_file, format='jpg', cleanup=True)
    
    def get_nodes(self):
        return self.nodes
    
    def get_observed_nodes(self):
        return self.observed_nodes
    
    def get_computed_nodes(self):
        return self.computed_nodes

    def get_samples(self, num_samples):
        # Set values for all the observed nodes
        set_dict = {k: self.generator_dict[k](num_samples) for k in self.observed_nodes}
        return self.scm.sample(n_samples=num_samples, set_values=set_dict)
    
    def get_all_parent_observed_nodes(self, node):
        # Simply assume that all dependencies are sequential i.e. the variable names are sorted
        assert node in self.observed_nodes
        idx = [i for i in range(len(self.observed_nodes)) if self.observed_nodes[i] == node]
        assert len(idx) == 1
        idx = idx[0]
        return self.observed_nodes[:idx]
    
    def get_unaffected_nodes(self, node):
        # Simply assume that all dependencies are sequential i.e. the variable names are sorted
        assert node in self.nodes
        preceding_nodes = [x for x in self.computed_nodes if int(x.replace("x", "")) < int(node.replace("x", ""))]
        unaffected_nodes = [x for x in self.observed_nodes if x != node] + preceding_nodes
        return unaffected_nodes
    
    def intervention_at_x(self, node, num_samples):
        """
        1. Changing the value at any node should not change the value of any other observed nodes
        2. Changing the value of x should only change all the following unobserved nodes, but not anything which is not a child of the given node
        """
        assert node in self.observed_nodes
        
        # Get a single sample for reference
        sample = self.get_samples(num_samples=1)
        print("Sample:", sample)
        
        # Get all the parent nodes
        unaffected_nodes = self.get_unaffected_nodes(node)
        print(f"Unaffected nodes for {node}: {unaffected_nodes}")

        # Set values for all the parent nodes
        set_dict = {k: [float(sample[k]) for _ in range(num_samples)] for k in unaffected_nodes}
        set_dict[node] = self.generator_dict[node](num_samples)
        print("Using fixed values for unaffected nodes:", set_dict)

        return self.scm.sample(n_samples=num_samples, set_values=set_dict)


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp dgcnn.py outputs'+'/'+args.exp_name+'/'+'dgcnn.py.backup')
    os.system('cp train.py outputs' + '/' + args.exp_name + '/' + 'train.py.backup')


def get_data(num_examples, device, inject_positional_features=False):
    # dataset = pd.read_csv(file_name)
    scm = SCM()
    dataset = scm.get_samples(num_samples=num_examples)
    dataset = dataset.reindex(natsort.natsorted(dataset.columns), axis=1)
    dataset_np = dataset.to_numpy()

    # Convert the train and test set to tensors
    dataset_tensor = torch.from_numpy(dataset_np).to(device)[:, None, :].float()  # Add synthetic channel dim
    dataset_input = dataset_tensor.clone()
    observed_vars = ["x1", "x2", "x4", "x6", "x8", "x9"]
    keep_cols = [int(x.replace("x", "")) - 1 for x in observed_vars]
    print("Keeping the following columns:", keep_cols)
    num_examples = dataset_input.shape[0]
    num_nodes = dataset_input.shape[2]
    for i in range(num_nodes):
        if i not in keep_cols:
            dataset_input[:, 0, i] = 0.  # Mask all the entries in that column
    if inject_positional_features:
        positional_features = torch.arange(num_nodes)
        positional_features = positional_features.repeat(num_examples, 1, 1).to(device)
        dataset_input = torch.cat([dataset_input, positional_features], dim=1)
        print("Dataset input size after positional information:", dataset_input.shape)
    dataset_target = dataset_tensor
    return dataset_input, dataset_target


def train(args, io):
    if os.path.exists(args.model_output_file):
        print("Output file already exists:", args.model_output_file)
        print("Returning without executing the training process...")
        return
    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load the dataset
    train_set_input, train_set_target = get_data(args.num_training_examples, device, args.inject_positional_features)
    test_set_input, test_set_target = get_data(args.num_test_examples, device, args.inject_positional_features)
    print(f"Train shape: {train_set_input.shape} / Test shape: {test_set_input.shape}")
    print(f"First example: {train_set_input[0, 0, :]} / Target {train_set_target[0, 0, :]}")

    # Create the model
    input_features = train_set_input.shape[1]  # Number of features
    if args.k is None:
        args.k = train_set_input.shape[2]  # Number of nodes
    print("Setting k in latent graph inference to be:", args.k)
    model = DGCNN_Reg(args, input_features=input_features).to(device)
    # print(str(model))

    weight_decay = 1e-4
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = nn.MSELoss()

    best_test_loss = 10000.
    train_loss_list = []
    test_loss_list = []
    log_iter = 25
    if args.batch_size is not None:
        print("Using random sampling in the dataset with a batch size of:", args.batch_size)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        for data, label in [(train_set_input, train_set_target)]:
            if args.batch_size is not None:
                # Select a random batch of data
                assert isinstance(args.batch_size, int)
                selected_idx = np.random.choice(np.arange(len(data)), size=min(args.batch_size, len(data)), replace=False)
                data = torch.stack([data[i] for i in selected_idx], dim=0)
                label = torch.stack([label[i] for i in selected_idx], dim=0)

            batch_size = data.size()[0]
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_loss = train_loss * 1.0 / count
        train_loss_list.append(float(train_loss))

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        for data, label in [(test_set_input, test_set_target)]:
            batch_size = data.size()[0]
            output = model(data)
            loss = criterion(output, label)
            count += batch_size
            test_loss += loss.item() * batch_size

        test_loss = test_loss * 1.0 / count
        test_loss_list.append(float(test_loss))
        outstr = 'Epoch: %d / Train loss: %.6f / Test loss: %.6f' % (epoch, train_loss, test_loss)
        if epoch % log_iter == 0:
            io.cprint(outstr)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), args.model_output_file)
    
    assert len(train_loss_list) == args.epochs
    assert len(test_loss_list) == args.epochs

    # Plot the results in the form of a figure
    plt.figure(figsize=(12, 8))

    plt.plot(train_loss_list, label='Train loss', color='b')
    plt.plot(test_loss_list, label='Test loss', color='r')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title("DGCNN trained on synthetic realizations of an SCM")
    plt.legend()
    plt.tight_layout()

    output_file = 'outputs/%s/models/training_dynamics.png' % args.exp_name
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    
    # Write the list in a csv file to be read in later
    epoch_list = list(range(1, len(train_loss_list)+1))
    assert len(epoch_list) == len(train_loss_list) == len(test_loss_list)
    df = pd.DataFrame()
    df["epochs"] = epoch_list
    df["train_loss"] = train_loss_list
    df["test_loss"] = test_loss_list
    
    output_file = 'outputs/%s/models/training_dynamics.csv' % args.exp_name
    df.to_csv(output_file)


def plot_distance(distance, selected_idx, path_prefix, k):
    assert len(distance.shape) == 2
    assert distance.shape[0] == distance.shape[1], f"Distance should be an N x N matrix (found: {distance.shape})"
    num_vars = distance.shape[0]
    assert selected_idx.shape == (num_vars, k), f"{selected_idx.shape} != ({num_vars}, 4)"
    
    # Plot one graph for each node
    for node in range(num_vars):
        node_distance = distance[node, :]
        connected_nodes = selected_idx[node, 1:]  # The highest similarity should be with the node itself (discard idx=0)
        selected_vals = [node_distance[i] for i in connected_nodes]
        assert max(selected_vals) == selected_vals[0]
        assert min(selected_vals) == selected_vals[-1]
        
        # Normalize the distance using min-max normalization
        # Distances are usually negative -- selecting top-k from it
        dist_min, dist_max = selected_vals[-1], selected_vals[0]
        if dist_min != dist_max:
            node_distance = (node_distance - dist_min) / (dist_max - dist_min)
        else:
            node_distance = node_distance - dist_min

        dot = graphviz.Digraph()

        color = Color(rgb=(1, 1, 1))   ## full 3-tuple RGB specification
        dot.node(f"x{node+1}", f"x{node+1}", {"shape": "ellipse", "peripheries": "1", "fillcolor": color.hex_l})
    
        # Draw the edges
        for other_node in range(num_vars):
            if other_node == node:
                continue

            if other_node not in connected_nodes:
                dot.node(f"x{other_node+1}", f"x{other_node+1}", {"shape": "ellipse", "peripheries": "1"})
                continue
            
            # Red indicates maximum distance i.e. top-k, green represents minimum distance
            current_node_dist = node_distance[other_node]
            color = Color(rgb=(current_node_dist, 1-current_node_dist, 0))   ## full 3-tuple RGB specification
            dot.node(f"x{other_node+1}", f"x{other_node+1}", {"shape": "ellipse", "peripheries": "1", "fillcolor": color.hex_l, "style": "filled"})
            dot.edge(f"x{other_node+1}", f"x{node+1}", _attributes={"style": "dashed"})
        
        # Save the dot figure here
        dot.render(f'{path_prefix}x{node+1}', format='jpg', cleanup=True)


def test(args, io):
    print("Evaluating the pretrained model...")
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Load test set
    test_set_input, test_set_target = get_data(args.num_test_examples, device, args.inject_positional_features)
    print(f"Test shape: {test_set_input.shape}")
    
    # TODO: Also include an intervention test here

    # Create the model
    input_features = test_set_input.shape[1]  # Number of features
    if args.k is None:
        args.k = test_set_input.shape[2]  # Number of nodes
    print("Setting k in latent graph inference to be:", args.k)
    model = DGCNN_Reg(args, input_features=input_features, return_features=True).to(device)
    model.load_state_dict(torch.load(args.model_output_file))  # Load the checkpoint
    model = model.eval()

    criterion = nn.MSELoss()
    
    count = 0
    test_loss = 0.0
    for data, label in [(test_set_input, test_set_target)]:
        batch_size = data.size()[0]
        output, distances, features = model(data)
        loss = criterion(output, label)
        count += batch_size
        test_loss += loss.item() * batch_size
    test_loss = test_loss * 1.0 / count
    
    outstr = 'Test loss: %.6f' % (test_loss)
    io.cprint(outstr)

    # Compare the predictions on any of the examples
    idx = np.random.choice(np.arange(len(output)))
    print("Selected example:", idx)
    print("Input:", data[idx])
    print("Target:", label[idx])
    print("Prediction:", output[idx])

    plot_distances = True
    if not plot_distances:
        return

    # Plot the distances for one of the examples
    (d1, d2, d3, d4), (x1, x2, x3, x4) = distances, features
    print(f"Distance tensors: {d1[0].shape} / {d2[0].shape} / {d3[0].shape} / {d4[0].shape}")
    print(f"Feature tensors: {x1.shape} / {x2.shape} / {x3.shape} / {x4.shape}")
    for i, d in enumerate([d1, d2, d3, d4]):
        print("Layer #", i)
        print("Distances:", d[0][0])
        print("Selected idx:", d[1][0])

        output_prefix = 'outputs/%s/attention_plots/layer_%d/' % (args.exp_name, i)
        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)
        os.makedirs(output_prefix)
        plot_distance(d[0][0, :, :], d[1][0, :, :], output_prefix, args.k)  # Only plot the first example


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='SCM training test')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num-training-examples', type=int, default=100, metavar='N',
                        help='Num of training examples to use')
    parser.add_argument('--num-test-examples', type=int, default=1000, metavar='N',
                        help='Num of test examples to use')
    parser.add_argument('--batch_size', type=int, default=None, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=4, metavar='N',
                        help='Num of nearest neighbors to use')
    
    args = parser.parse_args()

    args.inject_positional_features = True
    args.exp_name = f"{args.exp_name}_train_ex_{args.num_training_examples}_{('_k_' + str(args.k)) if args.k is not None else '_fc'}{'_pos' if args.inject_positional_features else ''}"
    
    # Create the required directories
    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    args.model_output_file = 'outputs/%s/models/model.pth' % args.exp_name
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # scm = SCM()
    # samples = scm.get_samples(num_samples=args.num_training_samples)
    # samples = samples.reindex(natsort.natsorted(samples.columns), axis=1)
    # print(samples.shape)

    # node = "x4"
    # samples = scm.intervention_at_x(node=node, num_samples=10)
    # samples = samples.reindex(natsort.natsorted(samples.columns), axis=1)
    # samples.to_csv("test.csv")
    # print(f"Samples after intervention at {node}: {samples}")
    # exit()

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
