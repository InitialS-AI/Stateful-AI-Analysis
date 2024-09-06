import joblib
import numpy as np
import os
import argparse


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./file/profile/yelp_100/train.data', help='path to train data file')
    parser.add_argument('--test_data', type=str, default='./file/profile/yelp_100/test.data', help='path to test data file')
    parser.add_argument('--base_save_path', type=str, default="./file/cache/yelp_nosep_unique_100/", help="the path for save foler.")
    parser.add_argument('--maxgap', type=int, default=0, help="max gap for mining")
    parser.add_argument('--mining_support_rate', type=int, default=40, help="support rate for mining")
    parser.add_argument('--min_pattern_size', type=int, default=3, help="the minimum pattern to be considered.")
    
    args = parser.parse_args()
    train_data = joblib.load(args.train_data)
    test_data = joblib.load(args.test_data)
    train_trace = train_data['trace']
    train_text = train_data['text']
    train_seq_labels = train_data['seq_labels']
    train_embedding = train_data['embedding']
    train_gt = train_data['groundtruth']
    train_faults = []
    os.makedirs(args.base_save_path, exist_ok=True)
    log_dir = os.path.join(args.base_save_path, "log")
    os.makedirs(log_dir, exist_ok=True)
    
    fault_trace_path = os.path.join(args.base_save_path, 'fault_trace.txt')
    correct_trace_path = os.path.join(args.base_save_path, 'correct_trace.txt')
    for i in range(len(train_trace)):
        if train_seq_labels[i][-1] != train_gt[i]:
            train_faults += [i]
    if os.path.exists(fault_trace_path) and os.path.exists(correct_trace_path):
        print("trace already collected, skip...")
    else:
        with open(fault_trace_path, 'w') as f:
            for fault in train_faults:
                trace = train_trace[fault]
                tmp = [str(v) + ' -1' for v in trace]
                tmp = ' '.join(tmp) + ' -2\n'
                f.writelines(tmp)
        
        print("fault trace recording complete")

        with open(correct_trace_path, 'w') as f:
            for i in range(len(train_trace)):
                if i not in train_faults:
                    trace = train_trace[i]
                    tmp = [str(v) + ' -1' for v in trace]
                    tmp = ' '.join(tmp) + ' -2\n'
                    f.writelines(tmp)  
        print("correct trace recording complete")

    command = 'java -jar ./file/java/spmf.jar run NOSEP {}fault_trace.txt'.format(args.base_save_path)
    command += ' {}mined_{}_nosep_fault.txt 1 20 0 {} {}'.format(
        args.base_save_path, args.mining_support_rate, args.maxgap, int(len(train_faults) * args.mining_support_rate / 100))
    os.system(command)

    command = 'java -jar ./file/java/spmf.jar run NOSEP {}correct_trace.txt'.format(args.base_save_path)
    command += ' {}mined_{}_nosep.txt 1 20 0 {} {}'.format(
        args.base_save_path, args.mining_support_rate, args.maxgap, int(len(train_faults) * args.mining_support_rate / 100))
    os.system(command)

    test_trace = test_data['trace']
    test_text = test_data['text']
    test_seq_labels = test_data['seq_labels']
    test_embedding = test_data['embedding']
    test_gt = test_data['groundtruth']
    
    with open('{}mined_{}_nosep.txt'.format(args.base_save_path, args.mining_support_rate), 'r') as f:
        mined_data = f.readlines()
    mined_patterns_correct = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) <= args.min_pattern_size:
            continue
        support = int(line_tmp[1][5:])
        mined_patterns_correct[key] = support
    del mined_data

    with open('{}mined_{}_nosep_fault.txt'.format(args.base_save_path, args.mining_support_rate), 'r') as f:
        mined_data = f.readlines()
    mined_patterns = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) <= args.min_pattern_size or key in mined_patterns_correct.keys():
            continue
        support = int(line_tmp[1][5:])
        mined_patterns[key] = support
    del mined_data

    if len(mined_patterns.keys()) < 1:
        raise Exception('No patterns found')

    mined_patterns = sorted(mined_patterns.items(), key=lambda item: item[1], reverse=True)

    with open('{}mined_results_{}_{}.txt'.format(args.base_save_path, args.mining_support_rate, args.min_pattern_size + 1),
                'w') as f:
        rs = [str(v) + '\n' for v in mined_patterns]
        f.writelines(rs)
    
    """
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
    """