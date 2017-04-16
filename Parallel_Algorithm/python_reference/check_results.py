import xarray as xr

dirname = "../test_data/"
filename = "Blobs_smp20000_fea30_cls8.nc"

with xr.open_dataset(dirname+filename) as ds:
    mismatch = (ds["Y_Py"].values != ds["Y_C"].values) 
   
print("total number of samples: ",mismatch.size)
print("inconsistent labels: ",mismatch.sum())
