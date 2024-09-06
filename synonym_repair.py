"""
Name   : synonym_repair.py
Author : Zhijie Wang
Time   : 2021/8/6
"""

import argparse
import pandas as pd
import torch
import numpy as np
import joblib
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from data.data_utils import CommentDataset
from model.simple_rnn import SimpleGRU
from abstraction.profiling import DeepStellar
from utils.state_acquisition import gather_word_state_text, gather_state_labels


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


def find_similar_pattern(target_trace, querry_trace):
    """
    find similar pattern in a trace
    :param target_trace: [1, 2, 3, 4, 5, 6, ...]
    :param querry_trace: [2, 3, 4]
    :return:
    """
    # TODO
    result = []
    querry_trace_len = len(querry_trace)
    this_trace_len = len(target_trace)
    if this_trace_len < querry_trace_len:
        return False
    if querry_trace[0] in target_trace:
        i = target_trace.index(querry_trace[0])
    else:
        return False
    while i < (this_trace_len - querry_trace_len):
        while i < (this_trace_len - querry_trace_len) and target_trace[i] != querry_trace[0]:
            i += 1
        j = 0
        k = 0
        id = []
        while (i + j) < this_trace_len:
            if target_trace[i + j] == querry_trace[k]:
                id.append(i + j)
                k += 1
            j += 1
            if k == querry_trace_len:
                if result == [] or id[0] > result[-1][1]:
                    result.append(id)
                break
        i = i + 1 if len(id) != querry_trace_len else id[0] + 1
    return result


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_sentence_length = 200

    file_name = args.test_file
    X_test = pd.read_csv(file_name)

    test_dataset = CommentDataset(df=X_test, tokenizer=tokenizer, max_length=max_sentence_length,
                                  data_col="comment_text", target="target", is_testing=False)

    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = SimpleGRU(*ckpt['model_args'])
    model = model.to(device)
    model.eval()

    model.load_state_dict(ckpt['model'])

    embedding_sims = np.load('./file/profile/embedding_sims.npy')

    test_data = joblib.load('./file/profile/yelp/test.data')

    deep_stellar_model = joblib.load('./file/profile/yelp/deep_stellar_p_20_s_39.profile')

    test_trace = test_data['trace']
    test_text = test_data['text']
    test_seq_labels = test_data['seq_labels']
    test_embedding = test_data['embedding']
    test_gt = test_data['groundtruth']

    best_split = {}

    with open('./file/cache/yelp/yelp_my/best_split.txt', 'r') as f:
        for line in f.readlines():
            key = tuple([int(v.strip()) for v in line.split(':')[0][1:-1].split(',')])
            best_split[key] = int(line.split(':')[1].strip().strip('\n'))

    for (mining_support_rate, min_pattern_size) in [(40, 1), (40, 2), (40, 3), (40, 4), (40, 5), (50, 1), (50, 2)]:
        with open('./file/cache/yelp/yelp_my/mined_results_%d_%d.txt' % (mining_support_rate, min_pattern_size + 1), 'r') as f:
            mined_patterns = f.readlines()
        mined_patterns = [[int(p.strip()) for p in v.strip('\n')[1:-1].split(')')[0][1:].split(',')] for v in mined_patterns]
        topk = best_split[(mining_support_rate, min_pattern_size + 1)]
        success_fixed = set()
        failed_fixed = set()
        for i in range(len(test_trace)):
            trace = test_trace[i].tolist()
            this_embedding = test_embedding[i]
            suc = 0
            fai = 0
            # print('------- Starting repair %d ------' % i)
            for j in range(topk):
                fixed_embedding = this_embedding
                # print('------- Top %d pattern ------' % (j + 1))
                search_result = find_similar_pattern(trace, mined_patterns[j])
                if search_result:
                    fix_pos = []
                    for buggy_pattern in search_result:
                        for index in buggy_pattern:
                            fixed_embedding[index] = embedding_sims[fixed_embedding[index]][0]
                            fix_pos.append(index)
                    fixed_embedding = fixed_embedding.tolist() + [0] * (max_sentence_length - len(fixed_embedding))
                    hidden_states, pred_tensor = model.profile(torch.tensor(fixed_embedding).view(1, max_sentence_length).to(device).long())
                    prediction_ = pred_tensor[0].cpu().numpy()
                    prediction_ = prediction_ >= 0.5
                    prediction_ = prediction_.astype(int)
                    prediction_ = prediction_[:len(this_embedding)]
                    y_pred = int(prediction_[-1][0])
                    state_ = hidden_states[0].cpu().numpy()
                    state_ = state_[:len(this_embedding)]
                    pca = deep_stellar_model.pca.do_reduction([state_])
                    new_trace = deep_stellar_model.get_trace(pca)[0]
                    if sum([new_trace[v] == trace[v] for v in fix_pos]) == len(fix_pos):
                        continue
                    else:
                        if (y_pred != test_seq_labels[i][-1][0]) and (y_pred == test_gt[i]):
                            suc = 1
                        elif (y_pred != test_seq_labels[i][-1][0]) and (y_pred != test_gt[i]):
                            fai = 1
                        else:
                            fai = 0
            if suc == 1:
                success_fixed.add(i)
            elif fai == 1:
                failed_fixed.add(i)
        print((mining_support_rate, min_pattern_size + 1))
        print(len(success_fixed))
        print(success_fixed)
        print(len(failed_fixed))
        pass
