from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
import joblib
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import tqdm

torch.manual_seed(10)


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
    parser.add_argument('--checkpoint', type=str, default="mining_rnn/maxid100_yelp_ckpt_best.pth", help="model checkpoint")
    parser.add_argument('--test_data', type=str, default='./file/profile/yelp/test.data', help='path to test data file')
    parser.add_argument('--max_state_idx', type=int, default=100, help='the max state idx for state trace')
    parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--device', type=str, default="cuda", help="Cuda or cpu")
    parser.add_argument('--dataset_name', type=str, default="yelp", help="the dataset name for traces")
    args = parser.parse_args()
    
    test_data = joblib.load(args.test_data)
    device = args.device
    test_trace = test_data['trace']
    test_gt = test_data['groundtruth']
    test_seq_labels = test_data['seq_labels']
    test_model_pred = [i[-1][0] for i in test_seq_labels]

    whole_dataset = StateDataset(test_trace, np.array(test_model_pred) != np.array(test_gt))
    seq_collate_fn = get_collate_fn(args.max_state_idx+1)
    test_dataloader = DataLoader(whole_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, collate_fn=seq_collate_fn)

    ckpt = torch.load(args.checkpoint)
    #model = SimpleGRUSeq(word_vec_size=200, rnn_size=128,
    #                    embedding_size=100+2, num_layers=2, 
    #                    dense_hidden_dim=[32])
    model = SimpleGRUSeq(*ckpt['model_args'])
    print("model acc: {}".format(ckpt['acc']))
    model = model.to(device)
    model.eval()
    print(model.training)
    #pos_weight = torch.tensor([1, 10]).to(device)
    #loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #best_loss = float('inf')
    best_acc = 0
    y_pred_list = np.array([])
    label_list = np.array([])
    with torch.no_grad():
        avg_loss = 0
        correct = 0
        all_pred = 0
        for batch in tqdm.tqdm(test_dataloader):
            input_tensor, input_length, target_tensor = batch[0].to(device), batch[1], batch[2].to(device)
            output = model(input_tensor, input_length)
            y_pred = torch.sigmoid(output)
            y_pred = torch.argmax(y_pred, dim=-1)
            correct += torch.sum(y_pred == target_tensor).item()
            all_pred += y_pred.size(0)
            y_pred_list = np.concatenate([y_pred_list, y_pred.cpu().numpy()], axis=0)
            label_list = np.concatenate([label_list, target_tensor.cpu().numpy()], axis=0)
        acc = correct / all_pred
        print('Acc: %4f' %(acc))
        print('Precision: %4f' % (precision_score(label_list, y_pred_list)))
        print('Recall_score: {}'.format(recall_score(label_list, y_pred_list)))
        print("F1 score {}".format(f1_score(label_list, y_pred_list)))
