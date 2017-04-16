#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from IO_util import Raw_to_NetCDF

N_clusters = 8
N_samples = 20000
N_features = 30
N_repeat = 20

X, y = make_blobs(n_samples=N_samples, centers=N_clusters,
                  n_features=N_features,random_state=0,
                  cluster_std=1.0)

initial_ind = np.zeros([N_repeat,N_clusters],dtype=np.int32)

for i in range(N_repeat):
    initial_ind[i,:] = np.random.choice(np.arange(N_samples),
                                   N_clusters,replace=False)

dirname = "../test_data/"
filename = "Blobs_smp{0}_fea{1}_cls{2}.nc".format(N_samples,N_features,N_clusters)

Raw_to_NetCDF(X,initial_ind,dirname+filename,y_true=y)