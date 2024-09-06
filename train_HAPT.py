"""
Name   : train_rnn.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
from model.simple_rnn import HumanActivityGRU
from data.data_utils import HAPT


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default='HAPT', help='name of dataset')
parser.add_argument('--train_dir', dest='train_dir', default='./file/data/HAPT/Train/', help='path to data dir')
parser.add_argument('--out_path', dest='out_path', default='./file/checkpoints/', help='output path')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--num_epochs', dest='num_epochs', default=40, type=int, help='num of epochs')
parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_layers', dest='num_layers', default=2, type=int, help='num of rnn layers')
parser.add_argument('--rnn_size', dest='rnn_size', default=256, type=int, help='hidden state dim')
parser.add_argument('--dense_hidden_dim', dest='dense_hidden_dim', default=[64], type=list, help='hidden state dim')
parser.add_argument('--timestamp', dest='timestamp', default=20, type=int, help='time stamp length')


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    dir_name = args.train_dir
    X_ori = pd.read_csv(dir_name + 'X_train.txt', sep=' ', header=None).to_numpy()
    y_ori = pd.read_csv(dir_name + 'y_train.txt', sep=' ', header=None).to_numpy() - 1

    X = np.array([X_ori[i:i + args.timestamp] for i in range(len(X_ori) - args.timestamp)])
    y = np.array([y_ori[i + args.timestamp] for i in range(len(y_ori) - args.timestamp)])

    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.7)
    X_train = X[idx[:split]]
    y_train = y[idx[:split]]
    X_val = X[idx[split:]]
    y_val = y[idx[split:]]
    # X_train = X.iloc[idx[:split]].to_numpy().reshape((-1, 561, 1))
    # y_train = y.iloc[idx[:split]].to_numpy()
    # X_val = X.iloc[idx[split:]].to_numpy().reshape((-1, 561, 1))
    # y_val = y.iloc[idx[split:]].to_numpy()

    train_dataset = HAPT(X_train, y_train)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

    val_dataset = HAPT(X_val, y_val)

    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    model = HumanActivityGRU(input_size=561, rnn_size=args.rnn_size, num_layers=args.num_layers, dense_hidden_dim=args.dense_hidden_dim, target_size=12)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0
        batch_num = 0
        for batch in train_dataloader:
            input_tensor, target_tensor = batch['x'].to(device).float(), batch['y'].to(device).long().view(-1)
            y_pred = model(input_tensor)
            loss = loss_function(y_pred, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()  # / len(train_loader)
            batch_num += 1
            if batch_num > -1 and batch_num % 100 == 0:
                print('Train Epoch: %d, Batch: %d/%d, Loss: %6f' % (
                epoch + 1, batch_num + 1, len(train_dataloader), avg_loss / (batch_num + 1)))
        print('Train Epoch: %d, Loss: %6f' % (epoch + 1, avg_loss / (len(train_dataloader))))
        best_loss = float('inf')
        best_acc = 0
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            correct = 0
            all_pred = 0
            for batch in val_dataloader:
                input_tensor, target_tensor = batch['x'].to(device).float(), batch['y'].to(device).long().view(-1)
                y_pred = model(input_tensor)
                loss = loss_function(y_pred, target_tensor)
                avg_loss += loss.item()  # / len(train_loader)
                y_pred = torch.argmax(y_pred, dim=1)
                correct += torch.sum(y_pred == target_tensor).item()
                all_pred += y_pred.size(0)
            acc = correct / all_pred
            print('Val Epoch: %d, Loss: %6f, Acc: %4f' % (epoch + 1, avg_loss / (len(val_dataloader)), acc))
            if (avg_loss / (len(val_dataloader))) < best_loss or acc > best_acc:
                best_loss = avg_loss / (len(val_dataloader))
                best_acc = acc
                result = {'model': model.state_dict(),
                          'loss': avg_loss / (len(val_dataloader)),
                          'acc': acc,
                          'model_args': (model.input_size, model.rnn_size, model.dense_hidden_dim, model.dropoutProb,
                                         model.num_layers, model.target_size)}
                torch.save(result, '%s%s_ckpt_best.pth' % (out_path, args.dataset_name))