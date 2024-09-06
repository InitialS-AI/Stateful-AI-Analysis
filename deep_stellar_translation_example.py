"""
Name   : deep_stellar_translation_example.py
Author : Zhijie Wang
Time   : 2021/7/19
"""

import argparse
import pandas as pd
import torch
import numpy as np
import joblib
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from data.data_utils import TranslationDataset
from model.simple_rnn import SimpleGRUTranslation
from abstraction.profiling import DeepStellar


parser = argparse.ArgumentParser()
parser.add_argument('--train_file_ori', dest='train_file_ori', default='./file/data/small_vocab_en.txt', help='path to data file')
parser.add_argument('--train_file_tar', dest='train_file_tar', default='./file/data/small_vocab_fr.txt', help='path to data file')
parser.add_argument('--out_path', dest='out_path', default='./file/profile/wmt/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/wmt_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=29, type=int, help='num of abstract states')
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')


def yield_tokens(lines):
    for line in lines:
        yield line.strip().split()


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    with open(args.train_file_ori, 'r', encoding='utf-8') as f:
        X = f.readlines()
    with open(args.train_file_tar, 'r', encoding='utf-8') as f:
        Y = f.readlines()

    max_sentence_length = 30

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    vocab_ori = ckpt['vocab_ori']
    vocab_tar = ckpt['vocab_tar']

    vocab_ori_itos = vocab_ori.get_itos()
    vocab_tar_itos = vocab_tar.get_itos()

    train_dataset = TranslationDataset(ori_text=X, tar_text=Y, ori_tokenizer=vocab_ori,
                                       tar_tokenizer=vocab_tar, max_length=max_sentence_length)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    model = SimpleGRUTranslation(*ckpt['model_args'])
    model = model.to(device)
    model.eval()

    model.load_state_dict(ckpt['model'])

    pca_data_path = '%spca_%d.ptr' % (args.out_path, args.pca_components)
    deep_stellar_path = '%sdeep_stellar_p_%d_s_%d.profile' % (args.out_path, args.pca_components, args.state_num)

    if os.path.exists(pca_data_path) and os.path.exists(deep_stellar_path) and not args.reprofiling:
        (pca_data, embedding, text, seq_labels, label) = joblib.load(pca_data_path)
        deep_stellar_model = joblib.load(deep_stellar_path)
    else:
        state_vec = []
        embedding = []
        text = []
        label = []
        seq_labels = []

        for batch in tqdm(train_dataloader):
            input_tensor, target_tensor = batch['ori_id'].to(device), batch['tar_id'].to(device)
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                embedding_ = batch['ori_id'][i].cpu().numpy()
                mask_ = batch['ori_mask'][i].cpu().numpy()
                text_ = np.array([vocab_ori_itos[v] for v in embedding_.tolist()])
                label_ = batch['tar_id'][i].cpu().numpy()
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = torch.argmax(pred_tensor[i].cpu(), dim=1).numpy()
                prediction_ = prediction_.astype(int)
                state_vec.append(state_[mask_ == 1.])
                embedding.append(embedding_[mask_ == 1.])
                text.append(text_[mask_ == 1.])
                label.append(label_)
                seq_labels.append(prediction_[mask_ == 1.])

        deep_stellar_model = DeepStellar(args.pca_components, args.state_num, state_vec)
        pca_data = deep_stellar_model.pca.do_reduction(state_vec)
        joblib.dump((pca_data, embedding, text, seq_labels, label), pca_data_path)
        joblib.dump(deep_stellar_model, deep_stellar_path)

    train_trace = deep_stellar_model.get_trace(pca_data)