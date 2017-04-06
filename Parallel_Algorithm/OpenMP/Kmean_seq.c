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

// square of the distance between x1[N_features] and x2[N_features]
float distance(int N_features,float *x1,float *x2){
    float dist=0.0;
    for (int j=0; j<N_features; j++)
        dist += (x1[j]-x2[j])*(x1[j]-x2[j]); 
    return(dist);
}


int main() {

    int N_clusters=3;
    size_t N_samples,N_features;
    //i for samples; j for features; k for clusters (typically)
    int i,j,k;
    int k_best;
    float** X;
    float dist,dist_min;
    
    // get input data and its size
    readX(&X,&N_samples,&N_features);

    // check the input data
    for (i=0; i<N_samples; i=i+N_samples-1){
        printf("no.%d  ",i+1);
        for (j=0; j<N_features; j++)
          printf("%f ",X[i][j]);
        printf("\n");
    }

    // each data point belongs to which cluster
    // values range from 0 to N_cluster-1
    int* labels = (int *)malloc(N_samples*sizeof(int));

    // The position of each cluster center.
    // Two arrays are needed as we are calculating the distance to the
    // old centers and accumulating the new centers in the same iteration.
    float** old_cluster_centers = Make2DFloatArray(N_clusters,N_features);
    float** new_cluster_centers = Make2DFloatArray(N_clusters,N_features);

    // how many data points in the cluster
    // needed by calculating the average position of data points in each cluster
    int* cluster_sizes = (int *)malloc(N_clusters*sizeof(int)); 

    // guess initial centers
    // use the tops elements (random guess TBD)
    for (k=0; k<N_clusters; k++){
        cluster_sizes[k] = 0; // for accumulating 

        for (j=0; j<N_features; j++){
            old_cluster_centers[k][j]=X[k][j];
            //set the "new" array to 0 for accumulating
            new_cluster_centers[k][j] = 0.0;
    }
    }


    // K-mean stepping begins here!!
    for (int step=0; step < 10; step++){

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

    // M-Step: set the cluster centers to the mean
    cluster_sizes[k_best]++; // add one more points to this cluster
    // As the total number of samples in each cluster is not known yet,
    // here we are just calculating the sum, not the mean.
    for (j=0; j<N_features; j++)
        new_cluster_centers[k_best][j] += X[i][j];

    }

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

    // check the classification results
    printf("\n step %d, labels: \n",step);
    for (i=0; i<N_samples; i++){
        printf("%d ",labels[i]);
    }

    } //end of K-mean stepping

    return 0;
}
