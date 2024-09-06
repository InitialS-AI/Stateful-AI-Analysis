"""
Name   : parse_mining_result.py
Author : Zhijie Wang
Time   : 2021/8/7
"""

import pandas as pd
import os
import numpy as np
import json


if __name__ == '__main__':
    files = os.listdir('./file/cache/yelp/yelp_my/')
    files = [v for v in files if v.split('.')[-1] == 'csv']
    file_dict = {}
    best_split = {}
    for f in files:
        f = f.strip('.csv')
        sr = int(f.split('_')[1])
        mp = int(f.split('_')[-1])
        if (sr, mp) not in file_dict.keys():
            file_dict[(sr, mp)] = []
        file_dict[(sr, mp)].append(f + '.csv')

    for (sr, mp) in file_dict.keys():
        f = file_dict[(sr, mp)]
        if 'fail' in f[0]:
            failed = pd.read_csv('./file/cache/yelp/yelp_my/' + f[0])
            all = pd.read_csv('./file/cache/yelp/yelp_my/' + f[1])
        else:
            failed = pd.read_csv('./file/cache/yelp/yelp_my/' + f[1])
            all = pd.read_csv('./file/cache/yelp/yelp_my/' + f[0])
        area = failed['Value'].values - all['Value'].values
        best_effort = np.argmax(area) + 1
        best_split[(sr, mp)] = best_effort
        pass
    with open('./file/cache/yelp/yelp_my/best_split.txt', 'w') as f:
        for key in best_split.keys():
            f.write(str(key) + ': ' + str(best_split[key]) + '\n')