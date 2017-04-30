import sys 
import numpy as np
from IO_util import Raw_to_NetCDF
import xarray as xr

dirname = '../../Data_Analysis/data/'
filename='SSWdata.nc'

ds = xr.open_dataset(dirname+filename)

print('total data size',ds["Y_TRUE"].size)
print('size of 2nd cluster by MATLAB',ds["Y_TRUE"].sum())
print('size of 2nd cluster by C',ds["Y_C"].sum())

mismatch = (ds["Y_TRUE"].values != ds["Y_C"].values) 
print("inconsistent labels: ",mismatch.sum())

#ds.close()
