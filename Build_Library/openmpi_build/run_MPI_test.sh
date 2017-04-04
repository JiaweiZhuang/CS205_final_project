#!/bin/bash

mpicc --version
mpicc MPI_test.c -o MPI_test.out
mpirun -np 4 ./MPI_test.out
