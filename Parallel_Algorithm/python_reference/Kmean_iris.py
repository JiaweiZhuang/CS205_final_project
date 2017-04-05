#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:11:36 2017

@author: desnow
"""

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# load iris data set from sklearn package
iris = load_iris()

# extract the data to numpy array
X =  np.float32(iris['data'])
y_true = np.int32(iris['target'])

'''
# convert to pandas and print
df_iris = pd.DataFrame(data= np.c_[iris['target'],iris['data']],
                     columns= ['target']+iris['feature_names'] )
print(df_iris.head())
'''


# apply K-mean
kmeans = KMeans(n_clusters=3,n_init=10,init='random',
                algorithm='full',tol=1e-2) 
kmeans.fit(X)
y_kmeans = kmeans.labels_
#y_kmeans = kmeans.predict(X) # this can predict new points
print("final inertia:",kmeans.inertia_)

'''
# print results
for i in range(len(y_kmeans)): 
    print(i+1,y_kmeans[i],y_true[i])
'''

# convert to xarray Dataset and write into nc files.
ds = xr.Dataset({'X': (['N_samples', 'N_features'], X),
                 'y_true': (['N_samples'],        y_true),
                 'y_kmeans_python': (['N_samples'],        y_kmeans),
                 'inertia_python': np.float32(kmeans.inertia_) ,
                 'y_kmeans_C': (['N_samples'], np.zeros_like(y_kmeans)),
                 'inertia_C': np.float32(0.0)
                 },
                  coords={'samples': (['N_samples'],np.arange(y_true.size,dtype=np.float32)+1),
                          'features': (['N_features'], iris['feature_names'])}
              )
ds.to_netcdf('../test_data/iris_data_Kmean.nc')
