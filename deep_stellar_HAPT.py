import argparse
import pandas as pd
import torch
import numpy as np
import joblib
import os
import joblib
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from abstraction.profiling import DeepStellar
from model.simple_rnn import HumanActivityGRU
from data.data_utils import HAPT


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default='HAPT', help='name of dataset')
parser.add_argument('--train_dir', dest='train_dir', default='./file/data/HAPT/Train/', help='path to data dir')
parser.add_argument('--test_dir', dest='test_dir', default='./file/data/HAPT/Test/', help='path to data dir')
parser.add_argument('--out_path', dest='out_path', default='./file/profile/HAPT/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/HAPT_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')
parser.add_argument('--timestamp', dest='timestamp', default=20, type=int, help='time stamp length')


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    dir_name = args.train_dir
    X_ori = pd.read_csv(dir_name + 'X_train.txt', sep=' ', header=None).to_numpy()
    y_ori = pd.read_csv(dir_name + 'y_train.txt', sep=' ', header=None).to_numpy() - 1

    X = np.array([X_ori[i:i + args.timestamp] for i in range(len(X_ori) - args.timestamp)])
    y = np.array([y_ori[i + args.timestamp] for i in range(len(y_ori) - args.timestamp)])

    train_dataset = HAPT(X, y)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = HumanActivityGRU(*ckpt['model_args'])
    model = model.to(device)
    model.eval()

    model.load_state_dict(ckpt['model'])

    pca_data_path = '%spca_%d.ptr' % (args.out_path, args.pca_components)
    deep_stellar_path = '%sdeep_stellar_p_%d_s_%d.profile' % (args.out_path, args.pca_components, args.state_num)

    if os.path.exists(pca_data_path) and os.path.exists(deep_stellar_path) and not args.reprofiling:
        (pca_data, seq_labels, label, pred, seq_prob) = joblib.load(pca_data_path)
        deep_stellar_model = joblib.load(deep_stellar_path)
    else:
        state_vec = []
        label = []
        seq_labels = []
        pred = []
        seq_prob = []

        for batch in tqdm(train_dataloader):
            input_tensor, target_tensor = batch['x'].to(device).float(), batch['y'].to(device).long().view(-1)
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                label_ = batch['y'][i].cpu().numpy()
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = torch.argmax(pred_tensor[i].cpu(), dim=1).numpy()
                prediction_ = prediction_.astype(int)
                state_vec.append(state_)
                label.append(label_)
                seq_labels.append(prediction_)
                pred.append(seq_labels[-1][-1])
                seq_prob.append(torch.max(pred_tensor[i].cpu(), dim=1)[0].numpy())

        deep_stellar_model = DeepStellar(args.pca_components, args.state_num, state_vec)
        pca_data = deep_stellar_model.pca.do_reduction(state_vec)
        joblib.dump((pca_data, seq_labels, label, pred, seq_prob), pca_data_path)
        joblib.dump(deep_stellar_model, deep_stellar_path)

    train_trace = deep_stellar_model.get_trace(pca_data)

    train_trace_path = '%strain.trace' % args.out_path
    if os.path.exists(train_trace_path):
        train_trace = joblib.load(train_trace_path)
    else:
        train_trace = deep_stellar_model.get_trace(pca_data)
        joblib.dump(train_trace, train_trace_path)

    train_data = {'trace': train_trace, 'seq_labels': seq_labels, 'groundtruth': label, 'seq_prob': seq_prob}
    joblib.dump(train_data, args.out_path + 'train.data')

    dir_name = args.test_dir
    X_ori = pd.read_csv(dir_name + 'X_test.txt', sep=' ', header=None).to_numpy()
    y_ori = pd.read_csv(dir_name + 'y_test.txt', sep=' ', header=None).to_numpy() - 1

    X = np.array([X_ori[i:i + args.timestamp] for i in range(len(X_ori) - args.timestamp)])
    y = np.array([y_ori[i + args.timestamp] for i in range(len(y_ori) - args.timestamp)])

    test_dataset = HAPT(X, y)

    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    test_pca_data_path = '%spca_%d_test.ptr' % (args.out_path, args.pca_components)

    if os.path.exists(test_pca_data_path) and not args.reprofiling:
        (pca_data_test, seq_labels_test, label_test, pred_test, seq_prob_test) = joblib.load(test_pca_data_path)
    else:
        state_vec_test = []
        label_test = []
        pred_test = []
        seq_labels_test = []
        seq_prob_test = []

        for batch in tqdm(test_dataloader):
            input_tensor, target_tensor = batch['x'].to(device).float(), batch['y'].to(device).long().view(-1)
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                label_ = batch['y'][i].cpu().numpy()
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = torch.argmax(pred_tensor[i].cpu(), dim=1).numpy()
                prediction_ = prediction_.astype(int)
                state_vec_test.append(state_)
                label_test.append(label_)
                seq_labels_test.append(prediction_)
                pred_test.append(seq_labels[-1][-1])
                seq_prob_test.append(torch.max(pred_tensor[i].cpu(), dim=1)[0].numpy())

        pca_data_test = deep_stellar_model.pca.do_reduction(state_vec_test)
        joblib.dump((pca_data_test, seq_labels_test, label_test, pred_test, seq_prob_test), test_pca_data_path)

    test_trace_path = '%stest.trace' % args.out_path
    if os.path.exists(test_trace_path):
        test_trace = joblib.load(test_trace_path)
    else:
        test_trace = deep_stellar_model.get_trace(pca_data_test)
        joblib.dump(test_trace, test_trace_path)

    test_data = {'trace': test_trace, 'seq_labels': seq_labels_test, 'groundtruth': label_test, 'seq_prob': seq_prob_test}
    joblib.dump(test_data, args.out_path + 'test.data')
    pass