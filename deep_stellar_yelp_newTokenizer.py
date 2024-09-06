import joblib
import os
from tqdm import tqdm
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import gc

from data.token_pipeline import load_tokenizer, get_collate_fn
from data.data_utils import text_simple
from model.simple_rnn import SimpleGRU
from abstraction.profiling import DeepStellar


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', dest='train_file', default='./file/data/yelp_train.csv', help='path to data file')
parser.add_argument('--test_file', dest='test_file', default='./file/data/yelp_test.csv', help='path to test file')
parser.add_argument('--cache_file', default='file/cache/', type=str, help="cache path for tokenizer")
parser.add_argument('--out_path', dest='out_path', default='./file/profile/yelp_newToken/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/yelp_newToken_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=256, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--ds_batch_size', default=None, type=int, help="The batch size for deepstellar model")
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')
parser.add_argument('--maxlen', default=200, type=int, help="the max num of words in a sentence")
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')
parser.add_argument('--vocab_size', default=30000, type=int, help="The vocab size. Set to None for no limit")


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    collate_fn = get_collate_fn(args.maxlen)
    os.makedirs(out_path, exist_ok=True)
    file_name = args.train_file
    X_train = pd.read_csv(file_name)

    vocab_size, tokenizer_pipeline, tokenizer, vocab = load_tokenizer(None, "yelp", args.cache_file, top_word_num=args.vocab_size)

    train_dataset = text_simple(tokenizer=tokenizer_pipeline, x_col=X_train['comment_text'].to_numpy(), y_col=X_train['target'].to_numpy())
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if args.usegpu else 'cpu')
    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])
    model = SimpleGRU(*ckpt['model_args'])
    model = model.to(device)
    model.eval()
    model.load_state_dict(ckpt['model'])

    pca_data_path = os.path.join(args.out_path, "pca_{}_newTok.prt".format(args.pca_components))
    deep_stellar_path = os.path.join(args.out_path, "deep_stellar_p_{}_s_{}_newTok.profile".format(args.pca_components, args.state_num))

    if os.path.exists(deep_stellar_path) and not args.reprofiling:
        deep_stellar_model = joblib.load(deep_stellar_path)
        deep_stellar_model.reload()
        print("Load deep stellar model..")
    else:
        deep_stellar_model = None
    if os.path.exists(pca_data_path) and not args.reprofiling:
        (pca_data, embedding, text, seq_labels, label, pred, pred_pro) = joblib.load(pca_data_path)
    else:
        state_vec = []
        embedding = []
        text = []
        label = []
        pred = []
        seq_labels = []
        pred_pro = []

        for batch in train_dataloader:
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                embedding_ = batch['input_ids'][i].cpu().numpy()
                mask_ = batch['attn_mask'][i].cpu().numpy()
                text_ = np.array(vocab.lookup_tokens(embedding_))
                label_ = batch['y'][i].cpu().numpy() >= 0.5
                label_ = label_.astype(int)
                state_ = hidden_states[i].cpu().numpy()
                if state_[mask_ == 1.].shape[0] == 0:
                    continue
                prediction_ = pred_tensor[i].cpu().numpy()
                prediction_ = prediction_ >= 0.5
                prediction_ = prediction_.astype(int)
                state_vec.append(state_[mask_ == 1.])
                embedding.append(embedding_[mask_ == 1.])
                text.append(text_[mask_ == 1.])
                label.append(label_)
                seq_labels.append(prediction_[mask_ == 1.])
                pred.append(seq_labels[-1][-1][0])
                pred_pro.append(pred_tensor[i].cpu().numpy()[mask_ == 1.])
        
        #temp_saved_data = (embedding, text, seq_labels, label, pred, pred_pro)
        #joblib.dump(temp_saved_data, pca_data_path)
        #for item in temp_saved_data:
        #    del item
        #del temp_saved_data
        del X_train
        del train_dataloader
        del train_dataset
        gc.collect()
        if deep_stellar_model is None:
            print("Begin fitting deepstellar models")
            deep_stellar_model = DeepStellar(args.pca_components, args.state_num, state_vec, batch_size=args.ds_batch_size)
            print("Finish building deep stellar model")
            joblib.dump(deep_stellar_model, deep_stellar_path)
        pca_data = deep_stellar_model.pca.do_reduction(state_vec)
        #(embedding, text, seq_labels, label, pred, pred_pro) = joblib.load(pca_data_path)
        joblib.dump((pca_data, embedding, text, seq_labels, label, pred, pred_pro), pca_data_path)
        joblib.dump(deep_stellar_model, deep_stellar_path)
    
    train_trace_path = os.path.join(args.out_path, 'train_newTok.trace')
    if os.path.exists(train_trace_path):
        train_trace = joblib.load(train_trace_path)
    else:
        train_trace = deep_stellar_model.get_trace(pca_data)
        joblib.dump(train_trace, train_trace_path)

    train_data = {'trace': train_trace, 'text': text, 'seq_labels': seq_labels,
            'embedding': embedding, 'groundtruth': label, "pred_pro": pred_pro}
    joblib.dump(train_data, os.path.join(args.out_path, 'train_newTok.data'))

    for item in (pca_data, embedding, text, seq_labels, label, pred, pred_pro):
        del item
    gc.collect()
    print("Begin profiling test data")
    file_name = args.test_file
    X_test = pd.read_csv(file_name)

    test_dataset = text_simple(tokenizer=tokenizer_pipeline, x_col=X_test['comment_text'].to_numpy(), y_col=X_test['target'].to_numpy())
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    test_pca_data_path = os.path.join(args.out_path, "pca_{}_test_newTok.ptr".format(args.pca_components))

    if os.path.exists(test_pca_data_path) and not args.reprofiling:
        (pca_data_test, embedding_test, text_test, seq_labels_test, label_test, pred_test, pred_pro_test) = joblib.load(test_pca_data_path)
    else:
        state_vec_test = []
        embedding_test = []
        text_test = []
        label_test = []
        pred_test = []
        seq_labels_test = []
        pred_pro_test = []

        for batch in test_dataloader:
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                embedding_ = batch['input_ids'][i].cpu().numpy()
                mask_ = batch['attn_mask'][i].cpu().numpy()
                text_ = np.array(vocab.lookup_tokens(embedding_))
                label_ = batch['y'][i].cpu().numpy() >= 0.5
                label_ = label_.astype(int)
                state_ = hidden_states[i].cpu().numpy()
                if state_[mask_ == 1.].shape[0] == 0:
                    continue
                prediction_ = pred_tensor[i].cpu().numpy()
                prediction_ = prediction_ >= 0.5
                prediction_ = prediction_.astype(int)
                state_vec_test.append(state_[mask_ == 1.])
                embedding_test.append(embedding_[mask_ == 1.])
                text_test.append(text_[mask_ == 1.])
                label_test.append(label_)
                seq_labels_test.append(prediction_[mask_ == 1.])
                pred_test.append(seq_labels[-1][-1][0])
                pred_pro_test.append(pred_tensor[i].cpu().numpy()[mask_ == 1.])

        pca_data_test = deep_stellar_model.pca.do_reduction(state_vec_test)
        joblib.dump((pca_data_test, embedding_test, text_test, seq_labels_test, label_test, pred_test, pred_pro_test), test_pca_data_path)

    test_trace_path = os.path.join(args.out_path, "test_newTok.trace")
    if os.path.exists(test_trace_path):
        test_trace = joblib.load(test_trace_path)
    else:
        test_trace = deep_stellar_model.get_trace(pca_data_test)
        joblib.dump(test_trace, test_trace_path)

    test_data = {'trace': test_trace, 'text': text_test, 'seq_labels': seq_labels_test,
            'embedding': embedding_test, 'groundtruth': label_test, "pred_pro":pred_pro_test}
    joblib.dump(test_data, os.path.join(args.out_path, 'test_newTok.data'))
