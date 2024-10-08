"""
Name   : utils.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

from sklearn.decomposition import PCA, IncrementalPCA
import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans as KMeansClustering, MiniBatchKMeans
import time
import math

from torch._C import NoneType


class AbstractModel(object):
    def __init__(self):
        self.m = -1
        self.clustering = None

    def get_trace(self, pca_data):
        output_trace = []
        for k, trace in enumerate(pca_data):
            labels = self.clustering.predict(trace) + 1
            output_trace.append(labels)
        return output_trace

    def update_transitions(self, trace, embedding):

        max_id = 0 if embedding is None else max([max(y) for y in embedding])

        transitions = np.zeros((self.m, max_id + 1, self.m), dtype=int)
        transition_dict = dict()

        for i, feature in enumerate(trace):
            gmm_labels = feature
            assert (np.sum(gmm_labels == 0) == 0)
            pre_gmm_label = 0

            for j in range(len(feature)):
                cur_gmm_label = int(gmm_labels[j])
                emb_num = 0 if embedding is None else embedding[i][j]
                transitions[pre_gmm_label][emb_num][cur_gmm_label] += 1
                key = (pre_gmm_label, emb_num, cur_gmm_label)
                if key in transition_dict:
                    transition_dict[key].add(i)
                else:
                    transition_dict[key] = {i}

                pre_gmm_label = cur_gmm_label
        return transitions, transition_dict


class GMM(AbstractModel):
    def __init__(self, traces, components, class_num):
        super().__init__()
        ts = traces[0]
        self.clustering = GaussianMixture(n_components=components, covariance_type='diag')

        # print(ts.shape, labels.shape)
        new_array = np.concatenate(ts, axis=0)
        gmm_labels = self.clustering.fit_predict(new_array)
        gmm_labels += 1

        self.bic = self.clustering.bic(new_array)
        self.m = components + 1
        print('converged: ', self.clustering.converged_)


class KMeans(AbstractModel):
    def __init__(self, traces, components, class_num, batch_size=None):
        super().__init__()
        ts = traces[0]
        new_array = np.concatenate(ts, axis=0)
        self.batch_size = batch_size
        if batch_size is not None:
            print("KMeans: minibatch mode")
            self.clustering = MiniBatchKMeans(components, batch_size=batch_size)
            total_batch_num = math.ceil(new_array.shape[0] / self.batch_size)
            print("Begin partial fitting of Kmeans, total_batch: {}".format(total_batch_num))
            for batch_num in range(total_batch_num):
                self.clustering.partial_fit(new_array[batch_num * self.batch_size: (batch_num+1) * self.batch_size])
                if batch_num > 0 and batch_num % (total_batch_num // 10) == 0:
                    print("Now batch: {}".format(batch_num))
        else:
            self.clustering = KMeansClustering(components)
            gmm_labels = self.clustering.fit_predict(new_array)
        self.m = components + 1

        # print(ts.shape, labels.shape)
        # gmm_labels += 1


class PCAReduction(object):
    def __init__(self, top_k, batch_size=None):
        self.top_components = top_k
        self.pca = None
        self.batch_size = batch_size

    def create_pca(self, data_list):

        # pca_path = os.path.join(dir, str(self.top_components)+'.pca')
        assert (len(data_list) > 0)
        if self.top_components >= data_list[0].shape[-1]:
            self.pca = None
            return data_list, np.amin(data_list, axis=0), np.amax(data_list, axis=0)
        else:
            data = np.concatenate(data_list, axis=0)
            if self.batch_size is not None:
                print("PCA: incremental mode")
                self.pca = IncrementalPCA(n_components=self.top_components, copy=False, batch_size=self.batch_size)
                total_batch_num = math.ceil(data.shape[0] / self.batch_size)
                print("Begin partial fitting of PCA, total_batch: {}".format(total_batch_num))
                for batch_num in range(total_batch_num):
                    self.pca.partial_fit(data[batch_num * self.batch_size: (batch_num+1) * self.batch_size])
                    if batch_num > 0 and batch_num % (total_batch_num // 10) == 0:
                        print("Now batch: {}".format(batch_num))
                ori_pca_data = np.empty((data.shape[0], self.top_components))
                for batch_num in range(total_batch_num):
                    partial_data = data[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
                    ori_pca_data[batch_num * self.batch_size: (batch_num+1) * self.batch_size] = self.pca.transform(partial_data)
            else:
                self.pca = PCA(n_components=self.top_components, copy=False)
                ori_pca_data = self.pca.fit_transform(data)
            min_val = np.amin(ori_pca_data, axis=0)
            max_val = np.amax(ori_pca_data, axis=0)
            pca_data = []
            indx = 0
            for state_vec in data_list:
                l = state_vec.shape[0]
                pca_data.append(ori_pca_data[indx: (indx + l)])
                indx += l
            return pca_data, min_val, max_val

    def do_reduction(self, data_list):
        if self.pca is None:
            return data_list
        else:
            data = np.concatenate(data_list, axis=0)
            if self.batch_size is not None:
                total_batch_num = math.ceil(data.shape[0] / self.batch_size)
                pca_data = np.empty((data.shape[0], self.top_components))
                for batch_num in range(total_batch_num):
                    partial_data = data[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
                    pca_data[batch_num * self.batch_size: (batch_num+1) * self.batch_size] = self.pca.transform(partial_data)
            else:
                pca_data = self.pca.transform(data)

            new_data = []
            indx = 0
            for state_vec in data_list:
                l = state_vec.shape[0]
                new_data.append(pca_data[indx: (indx + l)])
                indx += l

            return new_data
