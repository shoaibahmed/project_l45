"""
Code adapted from: https://github.com/AnTao97/dgcnn.pytorch
"""

from __future__ import print_function
import os
import argparse
from matplotlib import markers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

from dgcnn import DGCNN_Reg

import pandas as pd
import matplotlib.pyplot as plt


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


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
    os.system('cp generate_data_scm.py outputs' + '/' + args.exp_name + '/' + 'generate_data_scm.py.backup')


def train(args, io):
    if os.path.exists(args.model_output_file):
        print("Output file already exists:", args.model_output_file)
        print("Returning without executing the training process...")
        return
    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load train set
    train_set_df = pd.read_csv("train_dataset.csv")
    train_set_np = train_set_df.to_numpy()

    # Load test set
    test_set_df = pd.read_csv("test_dataset.csv")
    test_set_np = test_set_df.to_numpy()

    # Convert the train and test set to tensors
    train_set_tensor = torch.from_numpy(train_set_np).to(device)[:, None, :].float()  # Add synthetic channel dim
    train_set_input = train_set_tensor.clone()
    train_set_input[:, 1:] = 0.  # Mask all other entries except x1
    train_set_target = train_set_tensor

    test_set_tensor = torch.from_numpy(test_set_np).to(device)[:, None, :].float()  # Add synthetic channel dim
    test_set_input = test_set_tensor.clone()
    test_set_input[:, 1:] = 0.  # Mask all other entries except x1
    test_set_target = test_set_tensor
    print(f"Train shape: {train_set_tensor.shape} / Test shape: {test_set_tensor.shape}")

    # Create the model
    model = DGCNN_Reg(args).to(device)
    # print(str(model))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = nn.MSELoss()

    best_test_loss = 10000.
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        for data, label in [(train_set_input, train_set_target)]:
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
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss)
        io.cprint(outstr)
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
        outstr = 'Test %d, loss: %.6f' % (epoch, test_loss)
        test_loss_list.append(float(test_loss))

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
    plt.show()
    plt.close('all')


def test(args, io):
    print("Evaluating the pretrained model...")
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Load test set
    test_set_df = pd.read_csv("test_dataset.csv")
    test_set_np = test_set_df.to_numpy()

    test_set_tensor = torch.from_numpy(test_set_np).to(device)[:, None, :].float()  # Add synthetic channel dim
    test_set_input = test_set_tensor.clone()
    test_set_input[:, 1:] = 0.  # Mask all other entries except x1
    test_set_target = test_set_tensor
    print(f"Test shape: {test_set_tensor.shape}")

    # Create the model
    model = DGCNN_Reg(args, return_features=True).to(device)
    model.load_state_dict(torch.load(args.model_output_file))  # Load the checkpoint
    model = model.eval()

    criterion = nn.MSELoss()
    
    count = 0
    test_loss = 0.0
    for data, label in [(test_set_input, test_set_target)]:
        batch_size = data.size()[0]
        output, distances = model(data)
        loss = criterion(output, label)
        count += batch_size
        test_loss += loss.item() * batch_size
    test_loss = test_loss * 1.0 / count
    
    outstr = 'Test loss: %.6f' % (test_loss)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
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
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=4, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()

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

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
