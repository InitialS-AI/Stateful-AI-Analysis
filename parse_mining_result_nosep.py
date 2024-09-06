"""
Name   : parse_mining_result_nosep.py
Author : Zhijie Wang
Time   : 2021/8/8
"""

import pandas as pd
import os
import numpy as np
import json


if __name__ == '__main__':
    files = os.listdir('./file/cache/yelp/yelp_nosep/')
    files = [v for v in files if v.split('.')[-1] == 'csv']
    file_dict = {}
    best_split = {}
    for f in files:
        f = f.strip('.csv')
        mg = int(f.split('_')[1])
        sr = int(f.split('_')[3])
        mp = int(f.split('_')[-1])
        if (mg, sr, mp) not in file_dict.keys():
            file_dict[(mg, sr, mp)] = []
        file_dict[(mg, sr, mp)].append(f + '.csv')

    for (mg, sr, mp) in file_dict.keys():
        f = file_dict[(mg, sr, mp)]
        if 'fail' in f[0]:
            failed = pd.read_csv('./file/cache/yelp/yelp_nosep/' + f[0])
            all = pd.read_csv('./file/cache/yelp/yelp_nosep/' + f[1])
        else:
            failed = pd.read_csv('./file/cache/yelp/yelp_nosep/' + f[1])
            all = pd.read_csv('./file/cache/yelp/yelp_nosep/' + f[0])
        area = failed['Value'].values - all['Value'].values
        best_effort = np.argmax(area) + 1
        best_split[(mg, sr, mp)] = best_effort
        pass
    with open('./file/cache/yelp/yelp_nosep/best_split.txt', 'w') as f:
        for key in best_split.keys():
            f.write(str(key) + ': ' + str(best_split[key]) + '\n')