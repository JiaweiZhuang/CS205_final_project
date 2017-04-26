import sys
import numpy as np
ndata = 17878
nfeatures = 252
# Read data points
file1=open('data/SSWdata.bin','rb')
X=np.fromfile(file1)
if sys.byteorder=='little':
    X.byteswap(True)
X=X.reshape(ndata,nfeatures)

# Read python label
file1=open('data/Label_py.bin','rb')
Y_py=np.fromfile(file1,np.int32)
if sys.byteorder=='little':
    Y_py.byteswap(True)

# Read matlab label
file1=open('data/Label_matlab.bin','rb')
Y_matlab=np.fromfile(file1,np.int32)

