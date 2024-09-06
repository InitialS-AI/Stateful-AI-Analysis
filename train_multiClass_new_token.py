import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os

from data.data_utils import text_simple, get_agnews_train_dataframe, get_agnews_test_dataframe
from data.data_utils import get_yelp_multi_test_dataframe, get_yelp_multi_train_dataframe
from data.token_pipeline import load_tokenizer, get_collate_fn
from model.simple_rnn import SimpleGRUMultiClassification


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default='agnews', choices=['agnews', 'yelp_multi'], help='name of dataset')
parser.add_argument('--cache_file', default="file/cache/", type=str, help="the cache folder")
parser.add_argument('--out_path', dest='out_path', default='./file/checkpoints/', help='output path')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', dest='num_epochs', default=40, type=int, help='num of epochs')
parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_layers', dest='num_layers', default=2, type=int, help='num of rnn layers')
parser.add_argument('--word_vec_size', dest='word_vec_size', default=200, type=int, help='word embedding size')
parser.add_argument('--rnn_size', dest='rnn_size', default=256, type=int, help='hidden state dim')
parser.add_argument('--maxlen', default=200, type=int, help="the max num of words in a sentence")
parser.add_argument('--dense_hidden_dim', dest='dense_hidden_dim', default=[64], type=list, help='hidden state dim')
parser.add_argument('--vocab_size', default=30000, type=int, help="The vocab size. Set to None for no limit")

def evaluation(model, data_loader, loss_function):
    with torch.no_grad():
        avg_loss = 0
        correct = 0
        all_pred = 0
        for batch in data_loader:
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).long()
            y_pred = model(input_tensor)
            loss = loss_function(y_pred, target_tensor)
            avg_loss += loss.item()  # / len(train_loader)
            y_pred = F.log_softmax(y_pred, dim=-1)
            y_pred = torch.argmax(y_pred, dim=-1)
            correct += torch.sum(y_pred == target_tensor).item()
            all_pred += y_pred.size(0)
        acc = correct / all_pred
        loss = avg_loss / (len(data_loader))
    return loss, acc


if __name__ == '__main__':
    args = parser.parse_args()
    collate_fn = get_collate_fn(args.maxlen)
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.dataset_name == "agnews":
        X_train_all = get_agnews_train_dataframe()
        X_test = get_agnews_test_dataframe()
        x_key = "comment_text"
        y_key = "target"
        target_size = 4
    elif args.dataset_name == "yelp_multi":
        X_train_all = get_yelp_multi_train_dataframe()
        X_test = get_yelp_multi_test_dataframe()
        x_key = "text"
        y_key = "stars"
        target_size = 5
    all_text = pd.concat([X_train_all[x_key], X_test[x_key]]).to_numpy(dtype=str)
    vocab_size, tokenizer_pipeline, tokenizer, vocab = load_tokenizer(all_text, args.dataset_name, args.cache_file, top_word_num=args.vocab_size)
    print("Vocabulary size: {}".format(vocab_size))

    idx = np.random.permutation(len(X_train_all))
    split = int(len(X_train_all) * 0.8)
    X_train = X_train_all.iloc[idx[:split]]
    X_val = X_train_all.iloc[idx[split:]]
    train_dataset = text_simple(tokenizer=tokenizer_pipeline, x_col=X_train[x_key].to_numpy(dtype=str), y_col=X_train[y_key].to_numpy())
    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataset = text_simple(tokenizer=tokenizer_pipeline, x_col=X_val[x_key].to_numpy(dtype=str), y_col=X_val[y_key].to_numpy())
    val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if args.usegpu else 'cpu')
    model = SimpleGRUMultiClassification(word_vec_size=args.word_vec_size, rnn_size=args.rnn_size, 
                                        embedding_size=vocab_size, num_layers=args.num_layers, 
                                        dense_hidden_dim=args.dense_hidden_dim, target_size=target_size)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0
        batch_num = 0
        for batch in train_dataloader:
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).long()
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
        model.eval()
        loss, acc = evaluation(model, val_dataloader, loss_function)
        print('Val Epoch: %d, Loss: %6f, Acc: %4f' % (epoch + 1, loss, acc))
        if acc > best_acc:
            best_acc = acc
            result = {'model': model.state_dict(),
                        'loss': loss,
                        'acc': acc,
                        'model_args': (model.rnn_size, model.word_vec_size, model.embedding_size,
                                        model.dense_hidden_dim, model.dropoutProb, model.num_layers,
                                        model.target_size)}
            torch.save(result, os.path.join(out_path, '%s_newToken_ckpt_best.pth' % (args.dataset_name)))
    

    test_dataset = text_simple(tokenizer=tokenizer_pipeline, x_col=X_test[x_key].to_numpy(dtype=str), y_col=X_test[y_key].to_numpy())
    test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    loss, acc = evaluation(model, test_dataloader, loss_function)
    print('Test Loss: {}, Acc: {}'.format(loss, acc))
