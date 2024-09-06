# Online RNN Repair

## Preliminaries

This is the online repair code for Recurrent Neural Networks, a kind of stateful deep neural network. The underlying technique is DeepStellar, which was published at ESEC/FSE 2019 (DeepStellar: Model-Based Quantitative Analysis of Stateful Deep Learning Systems)

## Yelp deepstellar model
```bash
deepstellar_model = joblib.load('yelp.profile')
```

## Get Started

### Environments

Basic environments: `Python >= 3.6`.

Required packages:
```bash
pip install -r requirements.txt
```

Organize your dataset as follows (put in `file/`):
```
├── abstraction
├── data
├── file
│   ├── checkpoints
│   │   │── toxic_ckpt_best.pth
│   ├── data
│   │   │── train.csv
│   ├── profile
├── model
├── utils
```

### Datasets & Pre-trained Models

Check datasets and checkpoints [here](https://drive.google.com/drive/folders/1Im4MkZNxfdd9onEI3hxyYZGYLENSnphs?usp=sharing).

The google drive folder is organized as follows:
```
├── derui yelp model (pre-trained model / dataset / deep stellar model)
├── zhijie yelp model (pre-trained model / dataset / deep stellar model)
├── toxic model
```

### Step 1 Train an RNN model

```bash
python train_rnn.py --train_file=../file/data/train.csv
```

### Step 2 Do profiling on training set

This script also contain some simple graph plotting.

```bash
python deep_stellar_example.py --train_file=./file/data/train.csv
```


### Get profiling result based on new tokenizer

#### yelp

First train the model:

```bash
python train_yelp_new_token.py
```

For trained models, please check: https://drive.google.com/drive/u/2/folders/1AMn3vO-zJby9Fu5CphFGQosNUDXi2Sna

After training, the tokenizer (and vocabulary) is built and stored at `file/cache/token*` by default.

tokenizer is kept at both github and google drive

The trained model is by default kept at `file/checkpoints/yelp_newToken_ckpt_best.pth`

And then you can do profiling by:

```bash
python deep_stellar_yelp_newTokenizer.py
```

Or use partial fit for lower memory usage:

```bash
nohup python3 -u -W ignore deep_stellar_yelp_newTokenizer.py --out_path ./file/profile/yelp_30000/ --state_num 99 --ds_batch_size 4096 --checkpoint ./file/checkpoints/vocab30000/yelp_newToken_ckpt_best.pth > yelp100_30000.log 2>&1 &

python3 profiling_as_dataframe.py --pro_data file/profile/yelp_30000/train_newTok.data --output_folder file/profile/yelp_30000/ --output_name train # output the result as csv format. optional

python3 profiling_as_dataframe.py --pro_data file/profile/yelp_30000/test_newTok.data --output_folder file/profile/yelp_30000/ --output_name test # output the result as csv format. optional
```

`profiling_as_dataframe.py` will transfer the profiling data into dataframe. It is just a kind of format transferring. Example result could be checked at https://drive.google.com/drive/u/2/folders/1KUjXxFaID89ooq2bsRU56rdTd5V-qjZY

For yelp multi-label classification, we use data at https://drive.google.com/drive/u/2/folders/18_rTQDjAOnvjMNT4Ii21eKCVK7hgOhwY

If you want to train the model, you can run 

```bash
python3 train_multiClass_new_token.py --dataset_name yelp_multi
```

For profiling:

```bash
python3 deep_stellar_multiClass_newTokenizer.py --dataset_name yelp_multi --checkpoint file/checkpoints/yelp_multi_newToken_ckpt_best.pth --out_path './file/profile/yel_pmulti_newToken/'
```

Or use partial fit for lower memory usage:

```bash
nohup python3 -u -W ignore deep_stellar_multiClass_newTokenizer.py --dataset_name yelp_multi --out_path ./file/profile/yelp_multi_30000/ --state_num 99 --ds_batch_size 4096 --checkpoint ./file/checkpoints/vocab30000/yelp_multi_newToken_ckpt_best.pth > yelp_multi100_30000.log 2>&1 &
```

#### agnews

Basically the same as yelp, except that you should run

```bash
python train_agnews_new_token.py

python deep_stellar_agnews_newTokenizer.py
```

For downloading trained model: https://drive.google.com/drive/u/2/folders/11t3319tsJPUtBbhX8XDDHVWho7OQMGcw

Or use partial fit for lower memory usage:

```bash
nohup python3 -u -W ignore deep_stellar_multiClass_newTokenizer.py --dataset_name agnews --out_path ./file/profile/agnews_30000/ --state_num 99 --ds_batch_size 4096 --checkpoint ./file/checkpoints/vocab30000/agnews_newToken_ckpt_best.pth > agnews100_30000.log 2>&1 &

python3 profiling_as_dataframe.py --pro_data file/profile/agnews_30000/train_newTok.data --output_folder file/profile/agnews_30000/ --output_name train # output the result as csv format. optional

python3 profiling_as_dataframe.py --pro_data file/profile/agnews_30000/test_newTok.data --output_folder file/profile/agnews_30000/ --output_name test # output the result as csv format. optional
```
