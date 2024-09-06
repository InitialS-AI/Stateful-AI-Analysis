"""
Name   : deep_stellar_yelp.py
Author : Zhijie Wang
Time   : 2021/7/30
"""
import numpy as np
import joblib
import os
from tqdm import tqdm

from abstraction.profiling import DeepStellar


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
    while i < (this_trace_len - querry_trace_len + 1):
        while i < (this_trace_len - querry_trace_len + 1) and target_trace[i] != querry_trace[0]:
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
                return True
        i += 1
    return False


if __name__ == '__main__':
    # (state_vec, labels, pred) = joblib.load('./file/yelp_train.data')
    # deep_stellar_model = DeepStellar(20, 39, state_vec)
    # pca_data = deep_stellar_model.pca.do_reduction(state_vec)
    # train_trace = deep_stellar_model.get_trace(pca_data)
    (labels, pred) = joblib.load('./file/yelp_train.y')
    deep_stellar_model = joblib.load('./file/yelp.profile')
    train_trace = joblib.load('./file/yelp_train.trace')
    train_faults = []
    for i in range(len(train_trace)):
        if round(pred[i][-1]) != labels[i]:
            train_faults += [i]
    print('Training faults: [%d, %d]', (len(train_faults), len(train_trace)))
    os.makedirs('./file/cache/yelp/', exist_ok=True)
    with open('./file/cache/yelp/train_trace.txt', 'w') as f:
        for fault in train_faults:
            trace = train_trace[fault]
            tmp = [str(v) + ' -1' for v in trace]
            tmp = ' '.join(tmp) + ' -2\n'
            f.writelines(tmp)

    mining_support_rate = 70
    if not os.path.exists('./file/cache/yelp/mined_%d_bide.txt' % (mining_support_rate)):
        command = 'java -jar ./file/java/spmf.jar run BIDE+ ./file/cache/yelp/train_trace.txt'
        command += ' ./file/cache/yelp/mined_%d_bide.txt %d%%' % (mining_support_rate, mining_support_rate)
        os.system(command)

    min_pattern_size = 3

    with open('./file/cache/yelp/mined_%d_bide.txt' % (mining_support_rate), 'r') as f:
        mined_data = f.readlines()
    mined_patterns = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) <= min_pattern_size:
            continue
        support = int(line_tmp[1][5:])
        mined_patterns[key] = support
    del mined_data

    mined_patterns = sorted(mined_patterns.items(), key=lambda item: item[1], reverse=True)

    test_faults = []
    if os.path.exists('./file/yelp_test.trace'):
        (labels_test, pred_test) = joblib.load('./file/yelp_test.y')
        test_trace = joblib.load('./file/yelp_test.trace')
    else:
        (state_vec_test, labels_test, pred_test) = joblib.load('./file/yelp_test.data')
        pca_data_test = deep_stellar_model.pca.do_reduction(state_vec_test)
        test_trace = deep_stellar_model.get_trace(pca_data_test)
        joblib.dump(test_trace, './file/yelp_test.trace')
        joblib.dump((labels_test, pred_test), './file/yelp_test.y')

    for i in range(len(test_trace)):
        if round(pred_test[i][-1]) != labels_test[i]:
            test_faults.append(i)

    import matplotlib.pyplot as plt

    fault_coverage = []
    total_coverage = []

    for topk in range(1, 101):
        cov = 0
        for fault in test_faults:
            trace = test_trace[fault].tolist()
            for i in range(topk):
                # if find_exact_pattern([v for i, v in enumerate(trace) if i == 0 or v != trace[i - 1]], list(mined_patterns[i][0])):
                if find_similar_pattern(trace, list(mined_patterns[i][0])):
                    cov += 1
                    break
        print('Top %d, Coverage on failed cases: [%d, %d]' % (topk, cov, len(test_faults)))
        fault_coverage += [cov / len(test_faults)]
    print('==========================================================')
    for topk in range(1, 101):
        cov = 0
        for case in range(len(test_trace)):
            trace = test_trace[case].tolist()
            for i in range(topk):
                # if find_exact_pattern([v for i, v in enumerate(trace) if i == 0 or v != trace[i - 1]], list(mined_patterns[i][0])):
                if find_similar_pattern(trace, list(mined_patterns[i][0])):
                    cov += 1
                    break
        print('Top %d, Coverage on all cases: [%d, %d]' % (topk, cov, len(test_trace)))
        total_coverage += [cov / len(test_trace)]

    plt.figure()
    plt.plot(range(1, len(fault_coverage) + 1), fault_coverage)
    plt.plot(range(1, len(total_coverage) + 1), total_coverage)
    plt.legend(['faults', 'all_cases'])
    plt.xlabel('Top K patterns')
    plt.ylabel('Coverage rate')
    plt.title('Mining support rate %d%%, min pattern size %d' % (mining_support_rate, min_pattern_size + 1))
    plt.show()