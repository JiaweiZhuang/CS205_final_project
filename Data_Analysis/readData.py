import sys
ndata = 17878
nfeatures = 252
# Read data points
file1=open('./SSWdata.bin','rb')
X=np.fromfile(file1)
if sys.byteorder=='little':
    X.byteswap(True)
X=X.reshape(ndata,nfeatures)

# Read python label
file1=open('./Label_py.bin','rb')
Y_py=np.fromfile(file1,np.int32)
if sys.byteorder=='little':
    Y_py.byteswap(True)

# Read matlab label
file1=open('./Label_matlab.bin','rb')
Y_matlab=np.fromfile(file1,np.int32)

