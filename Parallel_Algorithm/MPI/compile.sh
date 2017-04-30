mpicc -o Kmean_mpi.out -O2 -std=c99 -lm -lnetcdf -fopenmp -lpthread ../shared/make_2D_array.c ../shared/ncdf_util.c ../shared/math_util.c Kmean_mpi.c 
