"""
Name   : data_analysis.py
Author : Zhijie Wang
Time   : 2021/7/19
"""

import joblib
import numpy as np


if __name__ == '__main__':
    train_data = joblib.load('./file/data/train.data')
    test_data = joblib.load('./file/data/test.data')
    train_trace = train_data['trace']
    train_text = train_data['text']
    train_seq_labels = train_data['seq_labels']
    train_embedding = train_data['embedding']
    train_gt = train_data['groundtruth']
    state_acc = {i: [0, 0] for i in range(1, 38)}
    edge_acc = {}
    for i in range(len(train_trace)):
        this_trace = train_trace[i]
        # this_trace = [v for ii, v in enumerate(this_trace) if ii == 0 or v != this_trace[ii - 1]]
        for j in range(0, len(this_trace)):
            correctness = int(train_seq_labels[i][-1] == train_gt[i])
            # correctness = int(train_seq_labels[i][j] == train_gt[i])
            state_acc[this_trace[j]][correctness] += 1

    for key in state_acc.keys():
        acc = state_acc[key][0] / (state_acc[key][0] + state_acc[key][1])
        acc = max(acc, 1e-6)
        state_acc[key].append(acc)

    transfer_times = {i: {j: 0 for j in range(1, 38)} for i in range(1, 38)}

    for i in range(len(train_trace)):
        tr = train_trace[i]
        # tr = [v for ii, v in enumerate(tr) if ii == 0 or v != tr[ii - 1]]
        for j in range(len(tr)):
            transfer_times[tr[j - 1]][tr[j]] += 1
    transfer_prob = {i: {j: 0 for j in range(1, 38)} for i in range(1, 38)}
    for k1 in transfer_prob.keys():
        tot = sum([item[1] for item in transfer_times[k1].items()])
        for k2 in transfer_prob[k1].keys():
            transfer_prob[k1][k2] = 1 - transfer_times[k1][k2] / tot

    pass
    test_trace = test_data['trace']
    test_text = test_data['text']
    test_seq_labels = test_data['seq_labels']
    test_embedding = test_data['embedding']
    test_gt = test_data['groundtruth']
    test_acc_stat = []
    test_fault = []
    for i in range(len(test_trace)):
        if test_seq_labels[i][-1] != test_gt[i]:
            test_fault.append(i)
        this_trace = test_trace[i]
        prob = np.log(state_acc[this_trace[0]][-1])
        for j in range(1, len(this_trace)):
            prob += np.log(transfer_prob[this_trace[j - 1]][this_trace[j]])
            prob += np.log(state_acc[this_trace[j]][-1])
        prob /= len(this_trace)
        test_acc_stat.append([i, prob])
        pass
    sorted_test_acc_stat = sorted(test_acc_stat, key=lambda item: item[1], reverse=True)
    sorted_test_acc_value = [v[1] for v in sorted_test_acc_stat]
    sorted_test_acc_stat = [v[0] for v in sorted_test_acc_stat]

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    fault_labels = np.array([0] * len(test_trace))
    fault_labels[test_fault] = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    lw = 2
    for top_case in [30, 50, 100, 500, 1000, 1500]:
        fault_localization = np.array([0.] * len(test_trace))
        found_top_case = sorted_test_acc_stat[:top_case]
        found_top_prob = sorted_test_acc_value[:top_case]
        found_top_prob = [np.exp(v) for v in found_top_prob]
        found_top_prob = [v / found_top_prob[0] for v in found_top_prob]
        fault_localization[found_top_case] = found_top_prob
        fpr[top_case], tpr[top_case], thresholds = roc_curve(fault_labels, fault_localization)
        roc_auc[top_case] = auc(fpr[top_case], tpr[top_case])
        plt.plot(fpr[top_case], tpr[top_case], lw=lw, label='ROC curve (top%d, area = %0.2f)' %
                                                            (top_case, roc_auc[top_case]))
        pass
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # intersect = set(test_fault).intersection(set(sorted_test_acc_stat[:50]))
    # print(len(intersect))
    # import matplotlib.pyplot as plt
    # color = ['r' if v in test_fault else 'b' for v in range(len(test_trace))]
    # test_acc_stat = np.array(test_acc_stat)
    # fig = plt.figure(figsize=(100, 10))
    # plt.scatter(test_acc_stat[:, 0], test_acc_stat[:, 1], c=color)
    # plt.xlim((1, len(test_trace)))
    # # plt.savefig('./file/prob_analysis.png', dpi=300)
    # plt.show()
    # pass
    # sorted_edge_acc = sorted(edge_acc.items(), key=lambda item: item[1][-1], reverse=True)
    # test_trace = test_data['trace']
    # test_text = test_data['text']
    # test_seq_labels = test_data['seq_labels']
    # test_embedding = test_data['embedding']
    # test_gt = test_data['groundtruth']
    # test_acc_stat = []
    # test_fault = []
    # for i in range(len(test_trace)):
    #     if test_seq_labels[i][-1] != test_gt[i]:
    #         test_fault.append(i)
    #     this_trace = test_trace[i]
    #     this_trace = [v for ii, v in enumerate(this_trace) if ii == 0 or v != this_trace[ii - 1]]
    #     prob = np.log(state_acc[this_trace[0]][-1])
    #     for j in range(1, len(this_trace)):
    #         if (this_trace[j - 1], this_trace[j]) in edge_acc.keys():
    #             prob += np.log(edge_acc[(this_trace[j - 1], this_trace[j])][-1])
    #         else:
    #             prob += 0
    #         prob += np.log(state_acc[this_trace[j]][-1])
    #     prob /= len(this_trace)
    #     test_acc_stat.append([i, prob])
    #     pass
    # sorted_test_acc_stat = sorted(test_acc_stat, key=lambda item: item[1], reverse=True)
    # sorted_test_acc_stat = [v[0] for v in sorted_test_acc_stat]
    # sb = set(test_fault).intersection(set(sorted_test_acc_stat[:50]))
    # import matplotlib.pyplot as plt
    # color = ['r' if v in test_fault else 'b' for v in range(len(test_trace))]
    # test_acc_stat = np.array(test_acc_stat)
    # fig = plt.figure(figsize=(100, 10))
    # plt.scatter(test_acc_stat[:, 0], test_acc_stat[:, 1], c=color)
    # plt.xlim((1, len(test_trace)))
    # plt.savefig('./prob_analysis.png', dpi=300)
    # plt.show()
    # pass