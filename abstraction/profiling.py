"""
Name   : profiling.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import numpy as np
import torch
import os
from tqdm import tqdm
from abstraction.utils import PCAReduction, GMM, KMeans
import joblib


class DeepStellar(object):
    def __init__(self, pca_dimension, abstract_state, state_vec, class_num=2, batch_size=None, method=None):
        """

        :param pca_dimension: reduced dimension
        :param abstract_state: # of abstract states
        :param state_vec: e.g. state_vec = [np.array(text_length, pca_dimension), ...]
        :param class_num: # of classes when classification
        """
        self.batch_size = batch_size
        if method is None:
            if batch_size is None:
                method = "GMM" # default to GMM
            else:
                method = "KMeans"
        self.pca_dimesion = pca_dimension
        self.abstract_state = abstract_state
        self.pca = PCAReduction(pca_dimension, batch_size=batch_size)
        pca_data, _, _ = self.pca.create_pca(state_vec)
        if method == 'GMM':
            if batch_size != None:
                raise Exception("GMM not support MINI-batch")
            self.ast_model = GMM([pca_data], abstract_state, class_num)
        elif method == 'KMeans':
            self.ast_model = KMeans([pca_data], abstract_state, class_num, batch_size=batch_size)
        else:
            raise NotImplementedError('Unknown clustering method!')

    def get_trace(self, pca_data):
        """

        :param pca_data: pca_data = self.pca.do_reduction(state_vec)
        :return: trace: e.g. tr = [(1,2,3), (4,2,1,5,7), ...]
        """
        return self.ast_model.get_trace(pca_data)
    
    def reload(self):
        """
        Compatible with old version deepstellar.
        """
        if not hasattr(self, "batch_size"):
            self.batch_size = None
        if hasattr(self.pca, "batch_size"):
            self.batch_size = self.pca.batch_size
        self.pca.batch_size = self.batch_size
        #new_pca = PCAReduction(self.pca.top_components, batch_size=self.batch_size)
        #for key in self.pca.__dict__():
        #    new_pca[key] = self.pca.__dict__()[key]
        #self.pca = new_pca
        
        # We do not do reload on ast_model yet, as
        # for now init of ast_model involves model.fit.
        # However this could be changed future.
        self.ast_model.batch_size = self.batch_size

class old_DeepStellar(object):
    def __init__(self, pca_dimension, abstract_state, state_vec, class_num=2, method='GMM'):
        """
        :param pca_dimension: reduced dimension
        :param abstract_state: # of abstract states
        :param state_vec: e.g. state_vec = [np.array(text_length, pca_dimension), ...]
        :param class_num: # of classes when classification
        """
        self.pca_dimesion = pca_dimension
        self.abstract_state = abstract_state
        self.pca = PCAReduction(pca_dimension)
        pca_data, _, _ = self.pca.create_pca(state_vec)
        if method == 'GMM':
            self.ast_model = GMM([pca_data], abstract_state, class_num)
        elif method == 'KMeans':
            self.ast_model = KMeans([pca_data], abstract_state, class_num)
        else:
            raise NotImplementedError('Unknown clustering method!')

    def get_trace(self, pca_data):
        """
        :param pca_data: pca_data = self.pca.do_reduction(state_vec)
        :return: trace: e.g. tr = [(1,2,3), (4,2,1,5,7), ...]
        """
        return self.ast_model.get_trace(pca_data)