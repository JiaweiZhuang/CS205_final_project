#include "../shared/timing.h" //for timer seconds()
#include <stdio.h>
#include <stdlib.h>
#include <float.h> //for FLT_MAX
#include <netcdf.h>


/* This is the name of the data file we will read. */
#define FILE_NAME "../test_data/Blobs_smp20000_fea30_cls8.nc"
#define TOL 0.0001 
#define MAX_ITER 100 

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
int** Make2DIntArray(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
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
int readX(float*** p_X,int*** p_GUESS,
          size_t* p_N_samples,size_t* p_N_features, 
          size_t* p_N_clusters,size_t* p_N_repeat ) {
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

   nc_inq_dimid(ncid,"N_clusters",&dimid);
   nc_inq_dimlen(ncid,dimid,p_N_clusters);
   printf("Number of clusters: %d \n",*p_N_clusters);

   nc_inq_dimid(ncid,"N_repeat",&dimid);
   nc_inq_dimlen(ncid,dimid,p_N_repeat);
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

int writeY(int* labels, float inert) { 
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

// square of the distance between x1[N_features] and x2[N_features]
float distance(int N_features,float *x1,float *x2){
    float dist=0.0;
    for (int j=0; j<N_features; j++)
        dist += (x1[j]-x2[j])*(x1[j]-x2[j]); 
    return(dist);
}


int main() {

    size_t N_samples,N_features,N_clusters,N_repeat;
    //i for samples; j for features; k for clusters (typically)
    int i,j,k;
    int k_best,initial_idx;
    float** X;
    int** GUESS;
    float dist,dist_min,dist_sum_old,dist_sum_new,inert_best=FLT_MAX;
    
    // get input data and its size
    double iStart1 = seconds();
    readX(&X,&GUESS,&N_samples,&N_features,&N_clusters,&N_repeat);
    double iElaps1 = seconds() - iStart1;

    // each data point belongs to which cluster
    // values range from 0 to N_cluster-1
    int* labels = (int *)malloc(N_samples*sizeof(int));
    int* labels_best = (int *)malloc(N_samples*sizeof(int));

    // The position of each cluster center.
    // Two arrays are needed as we are calculating the distance to the
    // old centers and accumulating the new centers in the same iteration.
    float** old_cluster_centers = Make2DFloatArray(N_clusters,N_features);
    float** new_cluster_centers = Make2DFloatArray(N_clusters,N_features);

    // how many data points in the cluster
    // needed by calculating the average position of data points in each cluster
    int* cluster_sizes = (int *)malloc(N_clusters*sizeof(int)); 

    printf("=====Applying K-mean======\n");
    double iStart2 = seconds();
    for (int i_repeat=0; i_repeat < N_repeat; i_repeat++){

    // guess initial centers
    for (k=0; k<N_clusters; k++){
        cluster_sizes[k] = 0; // for accumulating 
        // the index of data points as the initial guess for cluster centers
        initial_idx = GUESS[i_repeat][k]; 
        for (j=0; j<N_features; j++){
            old_cluster_centers[k][j]=X[initial_idx][j];
            //set the "new" array to 0 for accumulating
            new_cluster_centers[k][j] = 0.0;
    }
    }

    // K-mean stepping begins here!!
    int i_iter = 0;
    dist_sum_new = 0.0;//prevent the firt iteration error
    do {
    i_iter++;
    dist_sum_old = dist_sum_new; 
    dist_sum_new = 0.0;
    // E-Step: assign points to the nearest cluster center
    for (i = 0; i < N_samples; i++) {
    
        k_best = 0;//assume cluster no.0 is the nearest
        dist_min = distance(N_features, X[i], old_cluster_centers[k_best]); 
        for (k = 1; k < N_clusters; k++){
            dist = distance(N_features, X[i], old_cluster_centers[k]); 
            if (dist < dist_min){
                dist_min = dist;
                k_best = k; 
            }
        }
       labels[i] = k_best;
       dist_sum_new += dist_min;

    // M-Step (half): set the cluster centers to the mean
    cluster_sizes[k_best]++; // add one more points to this cluster
    // As the total number of samples in each cluster is not known yet,
    // here we are just calculating the sum, not the mean.
    for (j=0; j<N_features; j++)
        new_cluster_centers[k_best][j] += X[i][j];

    } //end if E-Step and half M-Step

    // M-Step-continued: convert the sum to the mean
    for (k=0; k<N_clusters; k++) {
            for (j=0; j<N_features; j++) {

                if (cluster_sizes[k] > 0) //avoid divide-by-zero error
                    // sum -> mean
                    old_cluster_centers[k][j] = new_cluster_centers[k][j] / cluster_sizes[k];
               
               new_cluster_centers[k][j] = 0.0;//for the next iteration
            }
            cluster_sizes[k] = 0;//for the next iteration
    }

    } while( i_iter==1 || ((dist_sum_old - dist_sum_new > TOL)&&i_iter<MAX_ITER) ); 
    //end of K-mean stepping

    //printf("Final inertia: %f, iteration: %d \n",dist_sum_new,i_iter);

    // record the best results
    if (dist_sum_new < inert_best) 
        inert_best = dist_sum_new;
        for (i = 0; i < N_samples; i++)
            labels_best[i] = labels[i];
    } //end of one repeated run
    double iElaps2 = seconds() - iStart2;


    // write data back to NetCDF file
    writeY(labels_best, inert_best);

    // print summary
    printf("Best inertia: %f \n",inert_best);
    printf("Kmean time use (ms): %f \n", iElaps2*1000.0);
    printf("I/O time use (ms): %f \n", iElaps1*1000.0);

    return 0;
}
