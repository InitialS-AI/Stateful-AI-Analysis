from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import tqdm
import nltk
import torchtext
from collections import Counter, OrderedDict

class tokenizer_pipeline():
    def __init__(self, tokenizer, voc):
        self.tokenizer = tokenizer
        self.voc = voc

    def __call__(self, text):
        text = text.strip().lower()
        text = self.tokenizer.tokenize(text)
        return self.voc(text)

def load_tokenizer(data, dataset_name, cache_folder, tokenizer_name="basic_english", top_word_num=None):
    """
    Get tokenizer pipeline for NLP data processing.
    data: better to be training data + test data. If it is already loaded, you can pass None to data.
    dataset_name: the name of the dataset
    cache_folder: the folder for saving tokenizer and vocabulary
    tokenizer_name: the name of tokenizer to be load.
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    if not (top_word_num is None):
        vocab_path = "token{}_dataset{}_vocab_top{}.pt".format(tokenizer_name, dataset_name, top_word_num)
    else:
        vocab_path = "token{}_dataset{}_vocab.pt".format(tokenizer_name, dataset_name)
    if os.path.exists(os.path.join(cache_folder, vocab_path)):
        print("Find existing vocabulary, loading...")
        vocab = torch.load(os.path.join(cache_folder,vocab_path))
        token_pipeline = tokenizer_pipeline(tokenizer, vocab)
        return len(vocab), token_pipeline, tokenizer, vocab
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    print("Building new tokenizer pipeline...")
    def get_iter(data):
        for text in data:
            text = text.strip().lower()
            text = tokenizer.tokenize(text)
            yield text
    if top_word_num is None:
        vocab = build_vocab_from_iterator(get_iter(data), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab["<unk>"])
    else:
        freq_count = Counter([])
        for tokenized_word in get_iter(data):
            freq_count += Counter(tokenized_word)
        freq_count = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)
        voc_dict = OrderedDict(freq_count[:top_word_num])
        vocab = torchtext.vocab.vocab(voc_dict)
        unk_token = '<unk>'
        default_index = 0
        if unk_token not in vocab: vocab.insert_token(unk_token, 0)
        vocab.set_default_index(default_index)
        pad_token = '<pad>'
        vocab.insert_token(pad_token, 1)
        

    torch.save(vocab, os.path.join(cache_folder, vocab_path))
    token_pipeline = tokenizer_pipeline(tokenizer, vocab)
    return len(vocab), token_pipeline, tokenizer, vocab

def get_collate_fn(maxlen):
    print("choose maxlen: {}".format(maxlen))
    def collate_fn(batch):
        text_tensor = [i[0] for i in batch]
        label = [i[1] for i in batch]
        label = torch.tensor(label).float()
        text = [i[2] for i in batch]
        # truncating remove values from sequences larger than maxlen at the beginning
        text_tensor = [i[-maxlen:] for i in text_tensor]
        batch_size = len(text_tensor)
        text_tensor_length = [len(i) for i in text_tensor]
        max_text_length = max(text_tensor_length)
        attention_mask = torch.ones(batch_size, max_text_length)
        # build_vocab_from_iterator <pad> vaule default to 1
        text_tensor = pad_sequence(text_tensor, batch_first=True, padding_value=1)
        for idx in range(batch_size):
            attention_mask[idx, text_tensor_length[idx]:] = 0
        return_batch = {"input_ids": text_tensor, "y":label, "text":text, "attn_mask":attention_mask}
        return return_batch
    return collate_fn

