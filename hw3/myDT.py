import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as sps
from sympy import *
import copy
from matplotlib import cm
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, accuracy_score 

class DecisionTree(BaseEstimator):
    '''
    Simple decision tree regressor with MSE rule and random best split by one feature,
    X and y -Features and Targets, must have numpy.ndarray class
    n_partitions - Number of partitions in each split, n_partitions = number of values if None 
    err_f - H(R) - Inhomogeneity function
    '''
    class Node(object):
        def __init__(self, X, y, err_f, n_partitions, max_depth, n_features):
            self.isLeaf = True
            self.err_f = err_f
            self.max_depth = max_depth
            self.n_partitions = n_partitions
            self.n_features = n_features
            if self.n_partitions:
                self.n_partitions = max(2, self.n_partitions)
                self.quants = np.arange(self.n_partitions)/(self.n_partitions - 1)
            self.X = X
            self.y = y
            self.split()
            
        def bin_err(self, B):
            return self.err_f(B, np.ones_like(B)*np.mean(B))
        
        def total_err(self,err_L,err_R,L,R,Q):
            return L/Q*err_L + R/Q*err_R
            
        def partition_err(self, bins):
            Q = len(self.y)
            R = np.sum(bins)
            L = Q - R
            R_bin = self.y[bins > 0]
            L_bin = self.y[bins < 1]
            err_L = self.bin_err(L_bin)
            err_R = self.bin_err(R_bin)
            return self.total_err(err_L,err_R,L,R,Q)
                
        def split(self):
            #bound conditions for exit
            if self.max_depth == 0 or len(self.X) <= 1:
                self.predict = np.mean(self.y)
                return
            
            self.split_err = np.inf
            fts = np.arange(len(self.X[0]))
            np.random.shuffle(fts)
            if self.n_features is not None:
                fts = fts[:self.n_features]
            for ft in fts:
                X_feature_slice = self.X[:,ft]
                
                good_slice = np.sort(np.unique(X_feature_slice))
                good_slice = good_slice[good_slice < np.max(good_slice)]
                
                if len(good_slice) >= 1:

                    if  self.n_partitions:
                        partitions = good_slice[np.array(self.quants * (len(good_slice)-1),dtype=int)]
                    else:
                        partitions = good_slice

                    for bound in partitions:
                        bins =  (X_feature_slice > bound)
                        error = self.partition_err(bins)
                        
                        # update best split
                        if error < self.split_err:
                            self.split_err = error
                            self.split_bins = bins
                            self.split_feature = ft
                            self.split_bound = bound
            
            #suitable partitions aren't founded
            if self.split_err == np.inf:
                self.predict = np.mean(self.y)
                return
            
            self.isLeaf = False
            
            #depth for childs
            md = self.max_depth
            if md:
                md -= 1
                
            self.Left = DecisionTree.Node(self.X[self.split_bins < 1], self.y[self.split_bins < 1], 
                                          self.err_f, self.n_partitions, md, self.n_features)
            
            self.Right = DecisionTree.Node(self.X[self.split_bins > 0], self.y[self.split_bins > 0], 
                                           self.err_f, self.n_partitions, md, self.n_features) 
            
    def __init__(self, err_f = mean_squared_error, n_partitions=None,max_depth=None,n_features=None):
        self.err_f = err_f
        self.max_depth = max_depth
        self.n_partitions = n_partitions
        self.n_features = n_features
        
    def fit(self, X, y):
        self.root = self.Node(X,y,self.err_f, self.n_partitions, self.max_depth, self.n_features)
    
    def find(self,x):
        cur = self.root
        while not cur.isLeaf:
            if x[cur.split_feature] > cur.split_bound:
                cur = cur.Right
            else:
                cur = cur.Left
        return cur.predict
        
    def predict(self, X):
        return np.array([self.find(x) for x in X])