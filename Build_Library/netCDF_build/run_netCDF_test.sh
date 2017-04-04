# all of them should pass (a single build should work for different compilers)
gcc -lnetcdf netCDF_test.c -o netCDF_test_by_gcc.out
./netCDF_test_by_gcc.out

nvcc -lnetcdf netCDF_test.c -o netCDF_test_by_nvcc.out
./netCDF_test_by_nvcc.out

pgcc -lnetcdf netCDF_test.c -o netCDF_test_by_pgcc.out
./netCDF_test_by_pgcc.out
