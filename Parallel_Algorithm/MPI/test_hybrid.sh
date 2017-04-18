#!/bin/bash

np_list='1 2 4 8'
thread_list='1 2'

for np in $np_list
do
    for thread in $thread_list
    do
    echo " "
    echo =========================================
    echo =========================================
        echo testing with $np processes, $thread threads
        export OMP_NUM_THREADS=$thread
        mpiexec -np $np ./Kmean_mpi.out
    done
done
