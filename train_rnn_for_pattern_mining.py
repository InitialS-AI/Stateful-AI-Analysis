from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import joblib
import argparse
import os
import numpy as np

class StateDataset(Dataset):
    def __init__(self, trace, label):
        self.trace = trace
        self.label = label
    
    def __getitem__(self, ix):
        return (self.trace[ix], self.label[ix], len(self.trace[ix]))
    
    def __len__(self):
        return len(self.trace)

class SimpleGRUSeq(nn.Module):
    def __init__(self, rnn_size, word_vec_size, embedding_size, padding_idx=-1, dense_hidden_dim=None, dropout_prob=0.3, num_layers=1, target_size=2):
        super(SimpleGRUSeq, self).__init__()
        self.rnn_size = rnn_size
        self.word_vec_size = word_vec_size
        self.num_layers = num_layers
        self.target_size = target_size
        self.dropoutProb = dropout_prob
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(embedding_size, word_vec_size, padding_idx)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.GRU(word_vec_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.dense_hidden_dim = dense_hidden_dim
        if dense_hidden_dim is None:
            self.dense = nn.Linear(rnn_size, target_size)
        else:
            layers = [nn.Linear(rnn_size, dense_hidden_dim[0])]
            for i in range(1, len(dense_hidden_dim)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(dense_hidden_dim[i-1], dense_hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dense_hidden_dim[-1], target_size))
            self.dense = nn.Sequential(*layers)
        self.sigomid = torch.sigmoid

    def forward(self, x, x_lengths):
        embeds = self.dropout(self.embedding(x))
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        output = self.dense(lstm_out)
        #prediction = self.sigomid(prediction)
        output = output[:, -1, :].squeeze()
        return output

def get_collate_fn(padding_value):
    def seq_collate_fn(batch):
        seq_list = [torch.tensor(i[0]) for i in batch]
        seq_list = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=padding_value)
        labels = torch.tensor([torch.tensor(i[1]).float() for i in batch])
        lengths = torch.tensor([torch.tensor(i[2]).int() for i in batch])
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_list = seq_list[perm_idx]
        labels = labels[perm_idx]
        return (seq_list, seq_lengths, labels.long())
    return seq_collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./file/profile/yelp/train.data', help='path to train data file')
    parser.add_argument('--train_prop', type=float, default=0.8, help="the training data proportion")
    parser.add_argument('--max_state_idx', type=int, default=100, help='the max state idx for state trace')
    parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--num_epochs', dest='num_epochs', default=40, type=int, help='num of epochs')
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_layers', dest='num_layers', default=2, type=int, help='num of rnn layers')
    parser.add_argument('--word_vec_size', dest='word_vec_size', default=200, type=int, help='word embedding size')
    parser.add_argument('--rnn_size', dest='rnn_size', default=128, type=int, help='hidden state dim')
    parser.add_argument('--dense_hidden_dim', dest='dense_hidden_dim', default=[32], type=list, help='hidden state dim')
    parser.add_argument('--device', type=str, default="cuda", help="Cuda or cpu")
    parser.add_argument('--out_path', type=str, default="mining_rnn", help="the output foler path")
    parser.add_argument('--dataset_name', type=str, default="yelp", help="the dataset name for traces")
    parser.add_argument('--pos_weight', type=int, default=9, help="The weight for positive class")
    args = parser.parse_args()
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    train_data = joblib.load(args.train_data)
    device = args.device
    train_trace = train_data['trace']
    train_gt = train_data['groundtruth']
    train_seq_labels = train_data['seq_labels']
    train_model_pred = [i[-1][0] for i in train_seq_labels]

    whole_dataset = StateDataset(train_trace, np.array(train_model_pred) != np.array(train_gt))
    train_size = math.ceil(len(train_trace) * args.train_prop)
    eval_size = len(train_trace) - train_size
    train_set, val_set = torch.utils.data.random_split(whole_dataset, [train_size, eval_size])
    seq_collate_fn = get_collate_fn(args.max_state_idx+1)
    train_dataloader = DataLoader(train_set, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_dataloader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn=seq_collate_fn)

    model = SimpleGRUSeq(word_vec_size=args.word_vec_size, rnn_size=args.rnn_size,
                        embedding_size=args.max_state_idx+2, num_layers=args.num_layers, 
                        dense_hidden_dim=args.dense_hidden_dim)

    model = model.to(device)
    pos_weight = torch.tensor([1, args.pos_weight]).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0
        batch_num = 0
        for batch in train_dataloader:
            input_tensor, input_length, target_tensor = batch[0].to(device), batch[1], batch[2].to(device)
            output = model(input_tensor, input_length)
            loss = loss_function(output, F.one_hot(target_tensor, num_classes=2).float())
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
                input_tensor, input_length, target_tensor = batch[0].to(device), batch[1], batch[2].to(device)
                out = model(input_tensor, input_length)
                loss = loss_function(out, F.one_hot(target_tensor, num_classes=2).float())
                avg_loss += loss.item()  # / len(train_loader)
                y_pred = torch.sigmoid(out)
                y_pred = torch.argmax(y_pred, dim=-1)
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
                          'model_args': (model.rnn_size, model.word_vec_size, model.embedding_size, model.padding_idx,
                                         model.dense_hidden_dim, model.dropoutProb, model.num_layers,
                                         model.target_size)}
                torch.save(result, os.path.join(args.out_path, 'maxid{}_posWeight{}_{}_ckpt_best.pth'.format(args.max_state_idx, args.pos_weight, args.dataset_name)))