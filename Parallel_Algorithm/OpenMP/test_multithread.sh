#!/bin/bash

thread_list='1 2 4 8'

for thread in $thread_list
do
    echo " "
    echo =========================================
    echo =========================================
    echo testing with $thread threads
    export OMP_NUM_THREADS=$thread
    ./Kmean_omp.out 
done
