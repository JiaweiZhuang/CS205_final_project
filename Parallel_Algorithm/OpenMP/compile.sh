gcc -O2 -std=c99 -lnetcdf -fopenmp -lpthread Kmean_omp.c -o Kmean_omp.out

#For debugging with gdb
#gcc -g -O0 -std=c99 -lnetcdf Kmean_seq.c -o Kmean_seq.out
