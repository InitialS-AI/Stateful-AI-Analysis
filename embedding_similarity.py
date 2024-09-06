"""
Name   : embedding_similarity.py
Author : Zhijie Wang
Time   : 2021/8/6
"""

import argparse
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast

from model.simple_rnn import SimpleGRU

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', dest='train_file', default='./file/data/yelp_train.csv', help='path to data file')
parser.add_argument('--test_file', dest='test_file', default='./file/data/yelp_test.csv', help='path to test file')
parser.add_argument('--out_path', dest='out_path', default='./file/profile/yelp/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/yelp_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')


if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = SimpleGRU(*ckpt['model_args'])
    model = model
    model.eval()

    model.load_state_dict(ckpt['model'])

    embeddings = np.zeros((ckpt['model_args'][2], ckpt['model_args'][1]))

    for i in range(ckpt['model_args'][2]):
        embeddings[i] = model.embedding(torch.tensor([i])).detach().numpy()

    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(embeddings, embeddings)

    sims_index = np.argsort(sims, axis=1)
    sims_index = np.flip(sims_index, axis=1)[:, 1:]
    np.save('./file/profile/embedding_sims.npy', sims_index)
    pass