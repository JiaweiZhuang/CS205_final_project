#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from netCDF4  import Dataset
from timeit import default_timer as timer
from sklearn.cluster import KMeans

dirname = "../test_data/"
filename = "Blobs_smp20000_fea30_cls8.nc"

# read data from nc file
start1 = timer()
with xr.open_dataset(dirname+filename) as ds: 
    n_clusters = ds.dims["N_clusters"]
    n_features = ds.dims["N_features"]
    n_repeat = ds.dims["N_repeat"]
    X = ds["X"].values
    GUESS = ds["GUESS"].values
del ds

elapse1 = timer()-start1

# apply Kmeans
start2 = timer()
inert_best = np.inf
for i_repeat in range(n_repeat):
    # manually guess initial clusters (to compare with C)
    initial_idx = GUESS[i_repeat,:]
    initial_position = X[initial_idx,:]
    kmeans = KMeans(n_clusters=n_clusters,n_init=1,init=initial_position,
                    algorithm='full',tol=1e-4) 
    kmeans.fit(X)
    
    if kmeans.inertia_ < inert_best:
        inert_best = kmeans.inertia_
        y_kmeans = kmeans.labels_

elapse2 = timer()-start2

# write results  back
with Dataset(dirname+filename,mode='r+') as dset:
    dset["Y_Py"][:] = y_kmeans
    dset["INERT_Py"][:] = inert_best

# summary        
print("final inertia:",inert_best)
print("Kmean time use (ms):",elapse2*1e3)
print("I/O time use (ms):",elapse1*1e3)