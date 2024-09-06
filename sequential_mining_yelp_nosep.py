"""
Name   : sequential_mining_yelp_nosep.py
Author : Zhijie Wang
Time   : 2021/8/7
"""

import joblib
import numpy as np
import os
from tensorboardX import SummaryWriter


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


if __name__ == '__main__':
    train_data = joblib.load('./file/profile/yelp/train.data')
    test_data = joblib.load('./file/profile/yelp/test.data')
    train_trace = train_data['trace']
    train_text = train_data['text']
    train_seq_labels = train_data['seq_labels']
    train_embedding = train_data['embedding']
    train_gt = train_data['groundtruth']
    train_faults = []
    log_dir = './file/cache/yelp_nosep/log/'
    os.makedirs(log_dir, exist_ok=True)
    fail_writer = SummaryWriter(log_dir + 'failed')
    all_writer = SummaryWriter(log_dir + 'all')
    for i in range(len(train_trace)):
        if train_seq_labels[i][-1] != train_gt[i]:
            train_faults += [i]
    os.makedirs('./file/cache/yelp_nosep/', exist_ok=True)

    with open('./file/cache/yelp_nosep/train_trace.txt', 'w') as f:
        for fault in train_faults:
            trace = train_trace[fault]
            tmp = [str(v) + ' -1' for v in trace]
            tmp = ' '.join(tmp) + ' -2\n'
            f.writelines(tmp)
    for maxgap in [0, 1, 2, 3, 4, 5]:
        for mining_support_rate in [40, 50, 60]:
            command = 'java -jar ./file/java/spmf.jar run NOSEP ./file/cache/yelp_nosep/train_trace.txt'
            command += ' ./file/cache/yelp_nosep/mined_%d_nosep.txt 1 20 0 %d %d' % (mining_support_rate, maxgap, int(len(train_faults) * mining_support_rate / 100))
            os.system(command)

            test_trace = test_data['trace']
            test_text = test_data['text']
            test_seq_labels = test_data['seq_labels']
            test_embedding = test_data['embedding']
            test_gt = test_data['groundtruth']

            for min_pattern_size in [1, 2, 3, 4, 5]:
                with open('./file/cache/yelp_nosep/mined_%d_nosep.txt' % (mining_support_rate), 'r') as f:
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

                if len(mined_patterns.keys()) < 1:
                    continue

                mined_patterns = sorted(mined_patterns.items(), key=lambda item: item[1], reverse=True)

                with open('./file/cache/yelp_nosep/mined_results_%d_%d_%d.txt' % (mining_support_rate, min_pattern_size + 1, maxgap),
                          'w') as f:
                    rs = [str(v) + '\n' for v in mined_patterns]
                    f.writelines(rs)

                test_faults = []
                for i in range(len(test_trace)):
                    if test_seq_labels[i][-1] != test_gt[i]:
                        test_faults.append(i)

                fault_coverage = np.zeros((len(test_faults), len(mined_patterns)), dtype=bool)

                for i in range(len(test_faults)):
                    fault = test_faults[i]
                    trace = test_trace[fault].tolist()
                    for j in range(len(mined_patterns)):
                        if find_similar_pattern(trace, list(mined_patterns[j][0]), maxgap):
                            fault_coverage[i][j] = True
                fault_coverage = np.cumsum(fault_coverage, axis=1, dtype=bool)
                for i in range(len(mined_patterns)):
                    fail_writer.add_scalar('MG_%d/SR_%d/MP_%d' % (maxgap, mining_support_rate, min_pattern_size + 1),
                                           np.sum(fault_coverage[:, i]).astype(int) / len(test_faults), i + 1)

                total_coverage = np.zeros((len(test_trace), len(mined_patterns)), dtype=bool)

                for i in range(len(test_trace)):
                    trace = test_trace[i].tolist()
                    for j in range(len(mined_patterns)):
                        if find_similar_pattern(trace, list(mined_patterns[j][0]), maxgap):
                            total_coverage[i][j] = True
                total_coverage = np.cumsum(total_coverage, axis=1, dtype=bool)
                for i in range(len(mined_patterns)):
                    all_writer.add_scalar('MG_%d/SR_%d/MP_%d' % (maxgap, mining_support_rate, min_pattern_size + 1),
                                          np.sum(total_coverage[:, i]).astype(int) / len(test_trace), i + 1)

            pass