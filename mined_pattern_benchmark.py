import joblib
import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score

def find_similar_pattern(target_trace, querry_trace, max_gap):
    """
    find similar pattern in a trace
    :param target_trace: [1, 2, 3, 4, 5, 6, ...]
    :param querry_trace: [2, 3, 4]
    :return:
    """
    # TODO
    querry_trace_len = len(querry_trace)
    this_trace_len = len(target_trace)
    if this_trace_len < querry_trace_len:
        return False
    if querry_trace[0] in target_trace:
        i = target_trace.index(querry_trace[0])
    else:
        return False
    while i < (this_trace_len - querry_trace_len + 1):
        while i < (this_trace_len - querry_trace_len + 1) and target_trace[i] != querry_trace[0]:
            i += 1
        j = 0
        k = 0
        id = []
        gap = 0
        while (i + j) < this_trace_len and gap <= max_gap:
            if target_trace[i + j] == querry_trace[k]:
                id.append(i + j)
                k += 1
                gap = -1
            j += 1
            gap += 1
            if k == querry_trace_len:
                return True
        i += 1
    return False

def get_mined_patterns(correct_path, fault_path, min_pattern_size):
    with open(correct_path, 'r') as f:
        mined_data = f.readlines()
    mined_patterns_correct = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) <= min_pattern_size:
            continue
        support = int(line_tmp[1][5:])
        mined_patterns_correct[key] = support
    del mined_data

    with open(fault_path, 'r') as f:
        mined_data = f.readlines()
    mined_patterns = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) <= min_pattern_size or key in mined_patterns_correct.keys():
            continue
        support = int(line_tmp[1][5:])
        mined_patterns[key] = support
    del mined_data

    if len(mined_patterns.keys()) < 1:
        raise Exception("No mined patterns found")

    mined_patterns = sorted(mined_patterns.items(), key=lambda item: item[1], reverse=True)
    return [i[0] for i in mined_patterns] # return list that doesn't contain # of support 

# text,pred_pro,label,is_bug,trace

def get_result_dataframe(topk, mined_patterns, test_data, match_max_gap):
    test_trace = test_data['trace']
    test_seq_labels = test_data['seq_labels']
    test_gt = test_data['groundtruth']
    test_text = test_data['text']
    test_pred_pro = test_data['pred_pro']
    result_dict = {"text":[], "pred_pro":[], "label":[], "model_pred":[], "trace":[], "is_bug":[]}
    print("# of patterns: {}".format(len(mined_patterns)))
    for i in tqdm(range(len(test_trace))):
        result_dict['text'].append(test_text[i])
        if not isinstance(test_pred_pro[i][-1], float):
            pred_pro_temp = test_pred_pro[i][-1][0]
        else:
            pred_pro_temp = test_pred_pro[i][-1]
        if not isinstance(test_seq_labels[i][-1], int):
            model_pred_temp = test_seq_labels[i][-1][0]
        else:
            model_pred_temp = test_seq_labels[i][-1]
        result_dict['pred_pro'].append(pred_pro_temp)
        result_dict['model_pred'].append(model_pred_temp)
        result_dict['label'].append(test_gt[i])
        #if test_seq_labels[i][-1] != test_gt[i]:
        #    result_dict['is_bug'].append(True)
        #else:
        #    result_dict['is_bug'].append(False)
        result_dict["trace"].append(test_trace[i])
        is_bug = False
        for j in range(topk):
            if find_similar_pattern(list(test_trace[i]), list(mined_patterns[j]), match_max_gap):
                is_bug = True
                break
        result_dict["is_bug"].append(is_bug)
    result = pd.DataFrame(result_dict)
    return result

def save_result(args, idx, result, max_gap):
    if args.mode == "r":
        if args.fina_patt != None:
            input_pattern_name = args.fina_patt[idx].split('/')[-1].split('.')[0]
        else:
            input_pattern_name = args.corr_patt[idx].split('/')[-1].split('.')[0]
        result_name = "benchmark_result_topk{}_maxG{}_pattern_{}.csv".format(args.topk[idx], max_gap, input_pattern_name)
        result.to_csv(os.path.join(args.output_folder, result_name))
        return None
    elif args.mode == "b":
        y_true = (result['label'] != result['model_pred'])
        y_pred = result["is_bug"]
        re_score = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("Recall:{}, precision:{}, f1_score:{} ".format(re_score, prec, f1))
        return {"recall":re_score, "precision":prec, "f1_score":f1}
        
        
if __name__ == "__main__": 
    match_max_gap = 0 # FIXME: We only consider continuous situations.
    parser = argparse.ArgumentParser()
    parser.add_argument('--corr_patt', nargs='+', default=None, help='the path for correct patterns')
    parser.add_argument('--faul_patt', nargs='+', default=None, help='the path for faulty patterns')
    parser.add_argument('--fina_patt', nargs='+', default=None, 
                        help="the path for faulty pattern set intersect with non-correct patterns. If given, corr_patt and \
                            faul_patt is not needed")
    parser.add_argument('--test_data', default='./file/profile/yelp/test.data', help="the path for test data")
    parser.add_argument('--min_pattern_size', default=3, help="the minimum length of patterns to be considered.\
                            if fina_patt is given, this value will not be used.")
    parser.add_argument("--topk", nargs='+', default=[10], type=int, help="the top k patterns to be considerd.")
    parser.add_argument("--output_folder", default="./", help="the path of the folder for output")
    parser.add_argument("--mode", default="b", choices=['r', 'b'], type=str,
                        help="r for recording, output csv file. b for benchmark, output score.")
    args = parser.parse_args()
    
    
    test_data = joblib.load(args.test_data)
    if args.fina_patt == None:
        for topk in args.topk:
            for idx in range(len(args.corr_patt)):
                print("Evaluating {} at top{} ...".format(args.corr_patt[idx]), topk)
                mined_patterns = get_mined_patterns(args.corr_patt[idx], args.faul_patt[idx], args.min_pattern_size)
                result = get_result_dataframe(topk, mined_patterns, test_data, match_max_gap)
                saved_result = save_result(args, idx, result, match_max_gap)
    else:
        result_dict = dict()
        for idx in range(len(args.fina_patt)):
            result_dict[args.fina_patt[idx]] = dict()
            for topk in args.topk:
                print("Evaluating {} at top{}...".format(args.fina_patt[idx], topk))
                with open(args.fina_patt[idx], 'r') as f:
                    mined_patterns = f.readlines()
                mined_patterns = [[int(p.strip()) for p in v.strip('\n')[1:-1].split(')')[0][1:].split(',')] for v in mined_patterns]
                result = get_result_dataframe(topk, mined_patterns, test_data, match_max_gap)
                saved_result = save_result(args, idx, result, match_max_gap)
                result_dict[args.fina_patt[idx]][topk] = saved_result
        joblib.dump(result_dict, "benchmark_result.dict")
        