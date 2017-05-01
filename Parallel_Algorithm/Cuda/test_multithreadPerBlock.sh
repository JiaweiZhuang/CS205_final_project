#!/bin/bash
export PATH=/usr/local/cuda-7.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

thread_list='1 2 4 8 16 32 64'

for thread in $thread_list
do
    echo " "
    echo =========================================
    echo =========================================
    echo testing with $thread threads per block on device
    ./a.out $thread ../test_data/Blobs_smp20000_fea30_cls8.nc
done
