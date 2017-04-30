import sys 
import numpy as np
from IO_util import Raw_to_NetCDF

ndata = 17878
nfeatures = 252 

dirname = '../../Data_Analysis/data/'
# Read data points
file1=open(dirname+'SSWdata.bin','rb')
X=np.fromfile(file1)
if sys.byteorder=='little':
    X.byteswap(True)
X=X.reshape(ndata,nfeatures)

# Read python label
file1=open(dirname+'Label_py.bin','rb')
Y_py=np.fromfile(file1,np.int32)
if sys.byteorder=='little':
    Y_py.byteswap(True)

# Read matlab label
file1=open(dirname+'Label_matlab.bin','rb')
Y_matlab=np.fromfile(file1,np.int32)
Y_matlab -= 1 # 1~2 to 0~1

# ========================
# convert the NetCDF format
# ========================
N_clusters = 2
N_samples = ndata 
N_features = nfeatures 
N_repeat = 20

initial_ind = np.zeros([N_repeat,N_clusters],dtype=np.int32)
for i in range(N_repeat):
    initial_ind[i,:] = np.random.choice(np.arange(N_samples),
                                   N_clusters,replace=False)

filename='SSWdata.nc'
Raw_to_NetCDF(X,initial_ind,dirname+filename,y_true=Y_matlab)

