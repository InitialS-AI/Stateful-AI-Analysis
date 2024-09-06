"""
Name   : train_rnn_translation.py
Author : Zhijie Wang
Time   : 2021/7/19
"""

from data.data_utils import TranslationDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import argparse
import numpy as np
from model.simple_rnn import SimpleGRUTranslation


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default='wmt', help='name of dataset')
parser.add_argument('--train_file_ori', dest='train_file_ori', default='./file/data/small_vocab_en.txt', help='path to data file')
parser.add_argument('--train_file_tar', dest='train_file_tar', default='./file/data/small_vocab_fr.txt', help='path to data file')
parser.add_argument('--out_path', dest='out_path', default='./file/checkpoints/', help='output path')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--num_epochs', dest='num_epochs', default=40, type=int, help='num of epochs')
parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_layers', dest='num_layers', default=2, type=int, help='num of rnn layers')
parser.add_argument('--word_vec_size', dest='word_vec_size', default=256, type=int, help='word embedding size')
parser.add_argument('--rnn_size', dest='rnn_size', default=256, type=int, help='hidden state dim')
parser.add_argument('--dense_hidden_dim', dest='dense_hidden_dim', default=[1024], type=list, help='hidden state dim')


def yield_tokens(lines):
    for line in lines:
        yield line.strip().split()


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path

    with open(args.train_file_ori, 'r', encoding='utf-8') as f:
        X = f.readlines()
    with open(args.train_file_tar, 'r', encoding='utf-8') as f:
        Y = f.readlines()

    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.7)
    X_train = np.array(X)[idx[:split]].tolist()
    X_val = np.array(X)[idx[split:]].tolist()
    Y_train = np.array(Y)[idx[:split]].tolist()
    Y_val = np.array(Y)[idx[split:]].tolist()

    vocab_ori = build_vocab_from_iterator(yield_tokens(X_train), specials=["<unk>"])
    vocab_ori.set_default_index(vocab_ori["<unk>"])

    vocab_tar = build_vocab_from_iterator(yield_tokens(Y_train), specials=["<unk>"])
    vocab_tar.set_default_index(vocab_tar["<unk>"])

    max_sentence_length = 30

    train_dataset = TranslationDataset(ori_text=X_train, tar_text=Y_train, ori_tokenizer=vocab_ori,
                                       tar_tokenizer=vocab_tar, max_length=max_sentence_length)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

    val_dataset = TranslationDataset(ori_text=X_val, tar_text=Y_val, ori_tokenizer=vocab_ori,
                                     tar_tokenizer=vocab_tar, max_length=max_sentence_length)

    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    model = SimpleGRUTranslation(word_vec_size=args.word_vec_size, rnn_size=args.rnn_size,
                                 embedding_size=len(vocab_ori), num_layers=args.num_layers,
                                 dense_hidden_dim=args.dense_hidden_dim, target_size=len(vocab_tar))
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0
        batch_num = 0
        for batch in train_dataloader:
            input_tensor, target_tensor = batch['ori_id'].to(device), batch['tar_id'].to(device)
            sentence_length = input_tensor.size(1)
            y_pred = model(input_tensor)
            loss = loss_function(y_pred[:, 0, :], target_tensor[:, 0])
            for di in range(1, sentence_length):
                loss += loss_function(y_pred[:, di, :], target_tensor[:, di])
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
                input_tensor, target_tensor = batch['ori_id'].to(device), batch['tar_id'].to(device)
                sentence_length = input_tensor.size(1)
                y_pred = model(input_tensor)
                loss = loss_function(y_pred[:, 0, :], target_tensor[:, 0])
                for di in range(1, sentence_length):
                    loss += loss_function(y_pred[:, di, :], target_tensor[:, di])
                avg_loss += loss.item()  # / len(train_loader)
                y_pred = torch.argmax(y_pred, dim=2)
                correct += torch.sum(y_pred == target_tensor).item()
                all_pred += y_pred.size(0) * y_pred.size(1)
            acc = correct / all_pred
            print('Val Epoch: %d, Loss: %6f, Acc: %4f' % (epoch + 1, avg_loss / (len(val_dataloader)), acc))
            if (avg_loss / (len(val_dataloader))) < best_loss or acc > best_acc:
                best_loss = avg_loss / (len(val_dataloader))
                best_acc = acc
                result = {'model': model.state_dict(),
                          'loss': avg_loss / (len(val_dataloader)),
                          'acc': acc,
                          'vocab_ori': vocab_ori,
                          'vocab_tar': vocab_tar,
                          'model_args': (model.rnn_size, model.word_vec_size, model.embedding_size,
                                         model.dense_hidden_dim, model.dropoutProb, model.num_layers,
                                         model.target_size)}
                torch.save(result, '%s%s_ckpt_best.pth' % (out_path, args.dataset_name))