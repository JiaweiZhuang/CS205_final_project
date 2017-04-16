import numpy as np
import xarray as xr

def Raw_to_NetCDF(X,ind,filename,y_true=None,feature_names=None):
    
    N_samples,N_features = X.shape
    label_zero = np.zeros(N_samples,dtype=np.int32)
    if feature_names is None:
        feature_names = np.arange(N_features,dtype=np.int32)
    if y_true is None:
        y_true = label_zero
    
    ds = xr.Dataset()
    ds['X'] = (['N_samples', 'N_features'], np.float32(X) )
    ds['X'].attrs["long_name"]="data points"
    
    ds['GUESS'] = (['N_repeat', 'N_clusters'], ind)
    ds['GUESS'].attrs["long_name"]="indices of data points as initial guess of cluster centers"
    ds['GUESS'].attrs["purpose"]="make sure that C and python use the same initial starting points"
    
    ds['Y_TRUE']=(['N_samples'], np.int32(y_true) )
    ds['Y_TRUE'].attrs["long_name"]="(optional) true label of each data point"
    
    ds['Y_Py']=(['N_samples'], label_zero)
    ds['Y_Py'].attrs["long_name"]="labels predicted by python Kmean function"
    
    ds['Y_C']=(['N_samples'], label_zero)
    ds['Y_C'].attrs["long_name"] = "labels predicted by C implementation"
    ds['Y_C'].attrs["purpose"] = "make sure that C implementation gives the same result as python"
    
    ds['INERT_Py'] = np.float32(0.0)
    ds['INERT_Py'].attrs["long_name"] = "kmeans.inertia_ in python code, "+\
        "i.e. sum of distances between data points and cluster centers"
    
    ds['INERT_C'] = np.float32(0.0)
    ds['INERT_C'].attrs["long_name"] = "the C version of kmeans.inertia_"
    ds['INERT_C'].attrs["purpose"] = "make sure that C implementation gives the same result as python"
    
    ds['FEATURES']=(['N_features'], feature_names)
    ds['FEATURES'].attrs["long_name"] = "(optional) the meaning of each feature"
                     
    ds.to_netcdf(filename)
    ds.close()
