"""
Name   : data_utils.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from torchtext.datasets import AG_NEWS
import os
import numpy as np

class HAPT(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': torch.from_numpy(self.X[idx]), 'y': torch.from_numpy(self.y[idx])}


class TranslationDataset(Dataset):
    def __init__(self, ori_text, tar_text, ori_tokenizer, tar_tokenizer, max_length=50):
        self.ori_text = ori_text
        self.tar_text = tar_text
        self.max_length = max_length
        self.ori_tokenizer = ori_tokenizer
        self.tar_tokenizer = tar_tokenizer

    def __len__(self):
        return len(self.ori_text)

    def __getitem__(self, ix):
        ori_sentence = self.ori_text[ix]
        tar_sentence = self.tar_text[ix]
        ori_ids = self.ori_tokenizer(ori_sentence.strip('\n').split())
        tar_ids = self.tar_tokenizer(tar_sentence.strip('\n').split())
        len_ori = len(ori_ids)
        len_tar = len(tar_ids)
        padded_ori_ids = ori_ids + [0] * (self.max_length - len_ori)
        padded_tar_ids = tar_ids + [0] * (self.max_length - len_tar)
        attention_mask_ori = [1] * len_ori + [0] * (self.max_length - len_ori)
        attention_mask_tar = [1] * len_tar + [0] * (self.max_length - len_tar)
        padded_ori_ids_tensor, padded_tar_ids_tensor, attention_mask_ori_tensor, attention_mask_tar_tensor = map(
            torch.LongTensor, [padded_ori_ids, padded_tar_ids, attention_mask_ori, attention_mask_tar]
        )
        encoded_dict = {'ori_id': padded_ori_ids_tensor,
                        'tar_id': padded_tar_ids_tensor,
                        'ori_mask': attention_mask_ori_tensor,
                        'tar_mask': attention_mask_tar_tensor,
                        }
        return encoded_dict


class text_simple(Dataset):
    def __init__(self, tokenizer, x_col, y_col):
        self.tokenizer = tokenizer
        self.x_col = x_col
        self.y_col = y_col
    
    def __len__(self):
        return self.x_col.shape[0]
    
    def __getitem__(self, index):
        text = self.x_col[index]
        label = self.y_col[index]
        text_tensor = self.tokenizer(text)
        text_tensor = torch.LongTensor(text_tensor)
        label = torch.tensor(label).long()
        return text_tensor, label, text
        

class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, data_col="comment_text", weight_col=None, identity_cols=None,
                 target="target", is_testing=False, special_token=False):
        self.df = df
        self.data_col = data_col
        self.weight_col = weight_col
        self.identity_cols = identity_cols
        self.target = target
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing
        self.special_token = special_token

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        comment_text = self.df.iloc[ix][self.data_col]
        comment_class = self.df.iloc[ix][self.target]

        weight_loss = None
        if self.weight_col is not None:
            weight_loss = self.df.iloc[ix][self.weight_col]

        identities = None
        if self.identity_cols is not None:
            identities = self.df.iloc[ix][self.identity_cols].to_numpy().astype(int)

        encoded_text = self.tokenizer.encode_plus(comment_text, add_special_tokens=self.special_token,
                                                  return_token_type_ids=True,
                                                  return_attention_mask=True, padding='max_length',
                                                  max_length=self.max_length, truncation=True)

        input_ids, attn_mask, token_type_ids = map(
            torch.LongTensor,
            [encoded_text['input_ids'],
             encoded_text['attention_mask'],
             encoded_text['token_type_ids']]
        )

        encoded_dict = {'input_ids': input_ids,
                        'attn_mask': attn_mask,
                        'token_type_ids': token_type_ids,
                        'y': comment_class,
                        }

        if self.weight_col is not None:
            encoded_dict['loss_w'] = weight_loss

        if self.identity_cols is not None:
            encoded_dict['identities'] = torch.from_numpy(identities)

        if not self.is_testing:
            target = encoded_dict['y']

        return encoded_dict

def get_agnews_train_dataframe():
    train_iter = AG_NEWS(root='./file/data/', split='train')
    X_train_all = {'comment_text': [], 'target': []}
    for label, line in train_iter:
        X_train_all['comment_text'] += [line]
        X_train_all['target'] += [label - 1]
    X_train_all = pd.DataFrame(data=X_train_all)
    return X_train_all

def get_agnews_test_dataframe():
    X_test = {'comment_text': [], 'target': []}
    test_iter = AG_NEWS(root='./file/data/', split='test')
    for label, line in test_iter:
        X_test['comment_text'] += [line]
        X_test['target'] += [label - 1]
    X_test = pd.DataFrame(data=X_test)
    return X_test

def split_yelp_multi_data(data_file, train_file, test_file, prop=0.8):
    all_data = pd.read_csv(data_file)
    split = int(len(all_data) * prop)
    idx = np.random.permutation(len(all_data))
    X_train = all_data.iloc[idx[:split]]
    X_test = all_data.iloc[idx[split:]]
    X_train.to_csv(train_file)
    X_test.to_csv(test_file)

def get_yelp_multi_train_dataframe():
    all_data_file = os.path.join("file/data", "yelp_training_set_review.csv")
    assert os.path.exists(all_data_file)
    train_file = os.path.join("file/data", "yelp_sub_train_review.csv")
    test_file = os.path.join("file/data", "yelp_sub_test_review.csv")
    if not os.path.exists(train_file):
        split_yelp_multi_data(all_data_file, train_file, test_file)
    X_train = pd.read_csv(train_file)
    X_train['text'] = X_train['text'].astype("str")
    X_train['stars'] = X_train['stars'] - 1
    return X_train

def get_yelp_multi_test_dataframe():
    all_data_file = os.path.join("file/data", "yelp_training_set_review.csv")
    assert os.path.exists(all_data_file)
    train_file = os.path.join("file/data", "yelp_sub_train_review.csv")
    test_file = os.path.join("file/data", "yelp_sub_test_review.csv")
    if not os.path.exists(test_file):
        split_yelp_multi_data(all_data_file, train_file, test_file)
    X_test = pd.read_csv(test_file)
    X_test['text'] = X_test['text'].astype("str")
    X_test['stars'] = X_test['stars'] - 1
    return X_test
