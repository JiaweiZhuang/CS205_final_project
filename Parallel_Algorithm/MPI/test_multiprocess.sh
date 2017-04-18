#!/bin/bash

np_list='1 2 4 8'

for np in $np_list
do
    echo " "
    echo =========================================
    echo =========================================
    echo testing with $np processes
    mpiexec -np $np ./Kmean_mpi.out 
done
