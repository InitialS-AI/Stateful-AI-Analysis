import joblib
import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score

# text,pred_pro,label,is_bug,trace

def get_result_dataframe(test_data):
    test_trace = test_data['trace']
    test_seq_labels = test_data['seq_labels']
    test_gt = test_data['groundtruth']
    test_text = test_data['text']
    test_pred_pro = test_data['pred_pro']
    result_dict = {"text":[], "pred_pro":[], "label":[], "model_pred":[], "trace":[]}
    for i in tqdm(range(len(test_trace))):
        result_dict['text'].append(test_text[i].tolist())
        if not np.issubdtype(type(test_seq_labels[i][-1]), np.integer):
            # not int, not multiclass
            model_pred_temp = test_seq_labels[i][-1][0]
        else:
            model_pred_temp = test_seq_labels[i][-1]
        if not np.issubdtype(type(test_pred_pro[i][-1]), np.integer):
            if len(test_pred_pro[i][-1]) > 1:
                pred_pro_temp = test_pred_pro[i][-1][model_pred_temp]
            else:
                pred_pro_temp = test_pred_pro[i][-1][0]
        else:
            pred_pro_temp = test_pred_pro[i][-1]
        result_dict['pred_pro'].append(pred_pro_temp)
        result_dict['model_pred'].append(model_pred_temp)
        result_dict['label'].append(test_gt[i])
        #if test_seq_labels[i][-1] != test_gt[i]:
        #    result_dict['is_bug'].append(True)
        #else:
        #    result_dict['is_bug'].append(False)
        result_dict["trace"].append(test_trace[i].tolist())
    result = pd.DataFrame(result_dict)
    return result
        
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--pro_data', default='./file/profile/yelp/test.data', type=str, help="the path for test data")
    parser.add_argument('--output_folder', default="./", type=str, help="the path of the folder for output")
    parser.add_argument('--output_name', default='newToken', type=str, help="The name of the output result")
    args = parser.parse_args()
    
    
    pro_data = joblib.load(args.pro_data)
    result = get_result_dataframe(pro_data)
    result_name = "profiling_result_{}.csv".format(args.output_name)
    result_name = os.path.join(args.output_folder, result_name)
    result.to_csv(result_name)
