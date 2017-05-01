export PATH=/usr/local/cuda-7.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

nvcc -lnetcdf kmeans_cdf.cu
