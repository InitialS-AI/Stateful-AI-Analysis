# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input, Reshape
from keras import Model
from keras.layers.embeddings import Embedding
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from tqdm import tqdm

def original_data_process(file_name="../file/data/yelp_training_set_review.csv"):
    df = pd.read_csv(file_name, delimiter = ',', error_bad_lines=False)

    text = df['text'].tolist()
    binstars = df['stars'].tolist()
    binstars = [1 if star>3 else 0 for star in binstars ]

    return text, binstars

def balanced_data(text, binstars, record_amount):
    balanced_texts = []  
    balanced_labels = []     
    neg_pos = [0, 0]
    for i in range(len(text)):
        sentiment = binstars[i]
        if neg_pos[sentiment] < record_amount:
            balanced_texts.append(text[i])
            balanced_labels.append(binstars[i])
            neg_pos[sentiment] += 1

    return balanced_texts, balanced_labels

def tokenization_and_padding(max_words, maxlen, balanced_texts, balanced_labels):

    tokenizer = Tokenizer(num_words = max_words)
    balanced_texts = [str(x) for x in balanced_texts]
    tokenizer.fit_on_texts(balanced_texts)
    sequences = tokenizer.texts_to_sequences(balanced_texts)

    data_input = pad_sequences(sequences, maxlen = maxlen)
    labels = np.asarray(balanced_labels)
    
    return data_input, labels

def data_split(data_input, labels, training_samples_count, test_sample_count):

    indices = np.arange(data_input.shape[0])
    np.random.shuffle(indices)
    data_input = data_input[indices]
    labels = labels[indices]

    x_train = data_input[:training_samples_count]
    y_train = labels[:training_samples_count]
    x_test = data_input[training_samples_count: training_samples_count + test_sample_count]
    y_test = labels[training_samples_count: training_samples_count + test_sample_count]

    return x_train, y_train, x_test, y_test


def build_model(maxlen, max_words, embed_dim=64, lstm_hidden_dim=0.2, dropout_rate=0.2):

    inputs = Input(shape=(maxlen,))
    input_embed = Embedding(max_words, embed_dim)(inputs)
    lstm_output, state_h, state_c  =  LSTM(lstm_hidden_dim, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_state=True)(input_embed)
    #lstm_output = lstm_output[-1]
    #lstm_output = tf.reshape(lstm_output, (1, lstm_hidden_dim))
    fc_0_output = Dense(100, activation="tanh")(lstm_output)
    pred_output = Dense(1, activation='sigmoid')(fc_0_output)

    model = Model(inputs, pred_output)
    get_lstm_state = Model(inputs, [pred_output, state_h, state_c])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc'])

    return model, get_lstm_state

def convert_to_str(x, lstm_hidden_emb):

    x =x.tolist()
    str_set = []
    
    
    for i in range(len(x)):
        if x[i] != 0.:
            hidden_state_at_i =[str(x) for x in lstm_hidden_emb[i].tolist()]
            hidden_state_at_i = ",".join(hidden_state_at_i)
            str_set.append(hidden_state_at_i)
    
    return str_set


def extract_error_state_sequence(get_lstm_state_func, x_record, labels, output_file_name):

    size = x_record.shape[0]
    labels = labels.tolist()
    
    count = 0
    with open(output_file_name, "w") as fw:
        
        for i in range(size):
            pred_result = get_lstm_state_func.predict(x_record[i])
            pred_prob = pred_result[0].reshape(-1).tolist()[-1]
            if round(pred_prob) != labels[i]:
                state_seq_set = convert_to_str(x_record[i], pred_result[1])
                state_seq_set = "\t".join(state_seq_set)
                fw.write("%s\n" % state_seq_set)
            else:
                count+=1
    print("accuracy = %s" %(count*1./size))


def get_state(x, lstm_hidden_emb):
    x = x.tolist()
    state_set = []

    for i in range(len(x)):
        if x[i] != 0.:
            hidden_state_at_i = [x for x in lstm_hidden_emb[i].tolist()]
            state_set.append(hidden_state_at_i)

    return state_set


def extract_state_vectors(get_lstm_state_func, x_record, labels,):
    size = x_record.shape[0]
    labels = labels.tolist()
    state_set = []
    pred = []
    label = []

    for i in tqdm(range(size)):
        pred_result = get_lstm_state_func.predict(x_record[i])
        pred_prob = pred_result[0].reshape(-1).tolist()[-1]
        h_state = get_state(x_record[i], pred_result[1])
        if len(h_state) > 0:
            state_set.append(np.array(h_state))
            seq_pred = pred_result[0].reshape(-1).tolist()
            pred.append(seq_pred)
            label.append(labels[i])

    return state_set, label, pred


def extract_pred(get_lstm_state_func, x_record, labels,):
    size = x_record.shape[0]
    labels = labels.tolist()
    pred = []
    label = []

    for i in tqdm(range(size)):
        pred_result = get_lstm_state_func.predict(x_record[i])
        pred_prob = pred_result[0].reshape(-1).tolist()[-1]
        h_state = get_state(x_record[i], pred_result[1])
        if len(h_state) > 0:
            seq_pred = pred_result[0].reshape(-1)
            pred.append(seq_pred.tolist())
            label.append(labels[i])

    return label, pred


def run(Train=True):
    
    maxlen = 200 
    max_words = 30000  
    training_samples = 150000 
    test_samples = 50000

    text, binstars = original_data_process()
    balanced_texts, balanced_labels = balanced_data(text, binstars, record_amount=100000)
    data_input, labels = tokenization_and_padding(max_words, maxlen, balanced_texts, balanced_labels)
    x_train, y_train, x_test, y_test = data_split(data_input, labels, training_samples, test_samples)
    lstm_model, get_lstm_state = build_model(maxlen, max_words, embed_dim=64, lstm_hidden_dim=128, dropout_rate=0.2)
    print("x_train.shape = %s, x_test.shape=%s,y_test.shape=%s" %(x_train.shape, x_test.shape, y_test.shape))

    if Train:
        lstm_model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=2046,
                            validation_split=0.2)
        lstm_model.save_weights("../file/checkpoints/my_checkpoint")
    else:
        lstm_model.load_weights("../file/checkpoints/my_checkpoint")
    # (state_vec, labels, pred) = extract_state_vectors(get_lstm_state, x_train, y_train)
    # joblib.dump((state_vec, labels, pred), '../file/yelp_train.data')
    (state_vec, labels, pred) = extract_state_vectors(get_lstm_state, x_test, y_test)
    joblib.dump((state_vec, labels, pred), '../file/yelp_test.data')
    (labels, pred) = extract_pred(get_lstm_state, x_train, y_train)
    joblib.dump((labels, pred), '../file/yelp_train.y')

run(False)


