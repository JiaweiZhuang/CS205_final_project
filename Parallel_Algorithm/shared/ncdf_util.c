#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include "make_2D_array.h"
#include "ncdf_util.h"
// including <stdlib.h> at last leads to "error: unknown type name ‘size_t’"
// no idea why?

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

/* Read the input data from NetCDF file. 
 * Dynamically allocate the array based on the data size.
 * 
 * Why need 3-levels of pointers:
 * The first two levels are for 2D dynamic array, 
 * the last level is for modifying function arguments in place.
 * (need to pass the address)
 */
int readX(char* FILE_NAME, float*** p_X,int*** p_GUESS,
          int* p_N_samples,int* p_N_features,
          int* p_N_clusters,int* p_N_repeat ) {
   int ncid, varid,dimid;
   int retval;
   size_t N_temp;

    printf("reading data \n");

   /* Open the file. NC_NOWRITE tells netCDF we want read-only access
    * to the file.*/
   if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
      ERR(retval);

   /* Get the size of the data for dynamical allocation*/
   nc_inq_dimid(ncid,"N_samples",&dimid);
   nc_inq_dimlen(ncid,dimid,&N_temp);
   *p_N_samples = (int)N_temp;
   printf("Number of samples: %d \n",*p_N_samples);

   nc_inq_dimid(ncid,"N_features",&dimid);
   nc_inq_dimlen(ncid,dimid,&N_temp);
   *p_N_features = (int)N_temp;
   printf("Number of features: %d \n",*p_N_features);

   nc_inq_dimid(ncid,"N_clusters",&dimid);
   nc_inq_dimlen(ncid,dimid,&N_temp);
   *p_N_clusters = (int)N_temp;
   printf("Number of clusters: %d \n",*p_N_clusters);

   nc_inq_dimid(ncid,"N_repeat",&dimid);
   nc_inq_dimlen(ncid,dimid,&N_temp);
   *p_N_repeat = (int)N_temp;
   printf("Number of repeated runs: %d \n",*p_N_repeat);

    /* Get the varid of the data variable, based on its name. */
   if ((retval = nc_inq_varid(ncid, "X", &varid)))
      ERR(retval);
   /* Read the data. */
   *p_X = Make2DFloatArray(*p_N_samples,*p_N_features);
   if ((retval = nc_get_var_float(ncid, varid, (*p_X)[0])))
      ERR(retval);

    /* Initial Guess*/
   if ((retval = nc_inq_varid(ncid, "GUESS", &varid)))
      ERR(retval);
   *p_GUESS = Make2DIntArray(*p_N_repeat,*p_N_clusters);
   if ((retval = nc_get_var_int(ncid, varid, (*p_GUESS)[0])))
      ERR(retval);

   /*close the netcdf file*/
   if ((retval = nc_close(ncid) ))
      ERR(retval);

    printf("=====reading data finished======\n");

   return 0;
}

int writeY(char* FILE_NAME, int* labels, float inert) {
   int ncid, varid;
   int retval;

   if ((retval = nc_open(FILE_NAME, NC_WRITE, &ncid)))
      ERR(retval);

   if ((retval = nc_inq_varid(ncid, "INERT_C", &varid)))
      ERR(retval)
   if ((retval = nc_put_var_float(ncid, varid, &inert )))
      ERR(retval);

   if ((retval = nc_inq_varid(ncid, "Y_C", &varid)))
      ERR(retval)
   if ((retval = nc_put_var_int(ncid, varid, labels )))
      ERR(retval);

   /*close the netcdf file*/
   if ((retval = nc_close(ncid) ))
      ERR(retval);

    printf("=====writting data finished======\n");

   return 0;
}
