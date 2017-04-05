#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>

/* This is the name of the data file we will read. */
#define FILE_NAME "../test_data/iris_data_Kmean.nc"

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

/* For dynamically allocating 2D array in pure-C environment.
Unlke in HW2, the array here is contagious!
See:
http://stackoverflow.com/questions/33794657/how-to-pass-a-2d-array-to-a-function-in-c-when-the-array-is-formatted-like-this
http://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
*/
float** Make2DFloatArray(int rows, int cols) {
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array= (float **)malloc(rows*sizeof(float*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

/* Read the input data from NetCDF file. 
 * Dynamically allocate the array based on the data size.
 * 
 * Why need 3-levels of pointers:
 * The first two levels are for 2D dynamic array, 
 * the last level is for modifying function arguments in place.
 * (need to pass the address)
 */
int readX(float*** p_X,size_t* p_N_samples,size_t* p_N_features) {
   int ncid, varid,dimid;
   int retval;

   /* Open the file. NC_NOWRITE tells netCDF we want read-only access
    * to the file.*/
   if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
      ERR(retval);

   /* Get the size of the data for dynamical allocation*/
   nc_inq_dimid(ncid,"N_samples",&dimid);
   nc_inq_dimlen(ncid,dimid,p_N_samples);
   printf("Number of samples: %d \n",*p_N_samples);

   nc_inq_dimid(ncid,"N_features",&dimid);
   nc_inq_dimlen(ncid,dimid,p_N_features);
   printf("Number of features: %d \n",*p_N_features);

   /* Get the varid of the data variable, based on its name. */
   if ((retval = nc_inq_varid(ncid, "X", &varid)))
      ERR(retval);

   /* Read the data. */
   *p_X = Make2DFloatArray(*p_N_samples,*p_N_features);
   if ((retval = nc_get_var_float(ncid, varid, (*p_X)[0])))
      ERR(retval);

   /*close the netcdf file*/
   if ((retval = nc_close(ncid) ))
      ERR(retval);

   return 0;
}

int main() {

    size_t N_samples,N_features;
    float** X;
    
    readX(&X,&N_samples,&N_features);

    // check the input data
    for (int i=0; i<N_samples; i=i+N_samples-1){
    printf("no.%d  ",i+1);
    for (int j=0; j<N_features; j++)
        printf("%f ",X[i][j]);
    printf("\n");
    }

    return 0;
}
