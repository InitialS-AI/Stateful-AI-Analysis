"""
Name   : sequential_mining.py
Author : Zhijie Wang
Time   : 2021/7/27
"""

import joblib
import numpy as np
import os


def find_exact_pattern(target_trace, querry_trace):
    querry_trace_len = len(querry_trace)
    this_trace_len = len(target_trace)
    if this_trace_len < querry_trace_len:
        return False
    if querry_trace[0] in target_trace:
        i = target_trace.index(querry_trace[0])
    else:
        return False
    while i < (this_trace_len - querry_trace_len + 1):
        k = 0
        while k < querry_trace_len and target_trace[i + k] == querry_trace[k]:
            k += 1
        if k == querry_trace_len:
            return True
        i += 1
    return False


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
    train_data = joblib.load('./file/data/train.data')
    test_data = joblib.load('./file/data/test.data')
    train_trace = train_data['trace']
    train_text = train_data['text']
    train_seq_labels = train_data['seq_labels']
    train_embedding = train_data['embedding']
    train_gt = train_data['groundtruth']
    train_faults = []
    for i in range(len(train_trace)):
        if train_seq_labels[i][-1] != train_gt[i]:
            train_faults += [i]
    os.makedirs('./file/cache/old/', exist_ok=True)

    with open('./file/cache/old/train_trace.txt', 'w') as f:
        for fault in train_faults:
            trace = train_trace[fault]
            tmp = [str(v) + ' -1' for v in trace]
            tmp = ' '.join(tmp) + ' -2\n'
            f.writelines(tmp)
    with open('./file/cache/old/train_trace_unique.txt', 'w') as f:
        for fault in train_faults:
            trace = train_trace[fault]
            trace = [v for i, v in enumerate(trace) if i == 0 or v != trace[i - 1]]
            tmp = [str(v) + ' -1' for v in trace]
            tmp = ' '.join(tmp) + ' -2\n'
            f.writelines(tmp)

    mining_support_rate = 10
    command = 'java -jar ./file/java/spmf.jar run BIDE+ ./file/cache/old/train_trace.txt'
    command += ' ./file/cache/old/mined_%d_bide.txt %d%%' % (mining_support_rate, mining_support_rate)
    os.system(command)

    test_trace = test_data['trace']
    test_text = test_data['text']
    test_seq_labels = test_data['seq_labels']
    test_embedding = test_data['embedding']
    test_gt = test_data['groundtruth']

    min_pattern_size = 1
    # with open('./file/cache/old/mined_tks_top50.txt', 'r') as f:
    with open('./file/cache/old/mined_%d_bide.txt' % (mining_support_rate), 'r') as f:
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
    for i in range(len(test_trace)):
        if test_seq_labels[i][-1] != test_gt[i]:
            test_faults.append(i)

    import matplotlib.pyplot as plt

    fault_coverage = []
    total_coverage = []

    for topk in range(1, len(mined_patterns) + 1):
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
    for topk in range(1, len(mined_patterns) + 1):
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
    plt.plot(range(1, len(mined_patterns) + 1), fault_coverage)
    plt.plot(range(1, len(mined_patterns) + 1), total_coverage)
    plt.legend(['faults', 'all_cases'])
    plt.xlabel('Top K patterns')
    plt.ylabel('Coverage rate')
    plt.title('Mining support rate %d%%, min pattern size %d' % (mining_support_rate, min_pattern_size + 1))
    plt.show()